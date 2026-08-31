# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 VQ2 expert artifacts with dynamic FP8 activations."""

from __future__ import annotations

import functools
import json
import math
import os
import re
from typing import Any

import torch
from safetensors import safe_open

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts, SharedExperts
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
    Fp8PerTensorOnlineLinearMethod,
    Fp8PtpcOnlineMoEMethod,
)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)

VQ2_BACKEND_ENV = "VLLM_VQ2_BACKEND"
VQ2_DECODE_FP8_BACKEND = "decode_fp8"
VQ2_DIRECT_BACKEND = "direct"


def get_vq2_backend() -> str:
    backend = os.getenv(VQ2_BACKEND_ENV, VQ2_DECODE_FP8_BACKEND).lower()
    if backend not in (VQ2_DECODE_FP8_BACKEND, VQ2_DIRECT_BACKEND):
        raise ValueError(
            f"{VQ2_BACKEND_ENV} must be {VQ2_DECODE_FP8_BACKEND!r} or "
            f"{VQ2_DIRECT_BACKEND!r}, got {backend!r}."
        )
    return backend


def extract_decoder_layer_index(prefix: str) -> int:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", prefix)
    if match is None:
        raise ValueError(f"Cannot extract decoder layer index from {prefix!r}.")
    return int(match.group(1))


def unpack_vq2_indices(
    packed: torch.Tensor,
    num_vectors: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Unpack little-endian 4-bit VQ indices stored in int32 words."""
    if num_vectors < 0:
        raise ValueError(f"num_vectors must be non-negative, got {num_vectors}.")
    words = packed.reshape(-1).to(device=device, dtype=torch.int64) & 0xFFFFFFFF
    required_words = (num_vectors + 7) // 8
    if words.numel() < required_words:
        raise ValueError(
            f"Packed VQ2 tensor has {words.numel()} words, but {required_words} "
            f"are required for {num_vectors} vectors."
        )
    positions = torch.arange(num_vectors, device=device, dtype=torch.int64)
    word_indices = torch.div(positions, 8, rounding_mode="floor")
    shifts = positions.remainder(8) * 4
    return (words[word_indices] >> shifts) & 15


def unpack_vq2_kernel_indices(packed: torch.Tensor, input_size: int) -> torch.Tensor:
    """Unpack the TP-kernel layout to [output-vector-pair, input]."""
    if packed.ndim != 2:
        raise ValueError(
            f"Kernel packed indices must be 2D, got {tuple(packed.shape)}."
        )
    positions = torch.arange(input_size, device=packed.device, dtype=torch.int64)
    words = packed.to(torch.int64) & 0xFFFFFFFF
    return (words[:, positions // 8] >> ((positions % 8) * 4)) & 15


@functools.cache
def _sylvester_hadamard(size: int) -> torch.Tensor:
    if size <= 0 or size & (size - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {size}.")
    matrix = torch.tensor([[1.0]])
    while matrix.shape[0] < size:
        matrix = torch.cat(
            [
                torch.cat([matrix, matrix], dim=1),
                torch.cat([matrix, -matrix], dim=1),
            ],
            dim=0,
        )
    return matrix / math.sqrt(size)


def _inverse_rht(
    weight: torch.Tensor,
    block_size: int,
    sign: torch.Tensor,
) -> torch.Tensor:
    width = sign.numel()
    if weight.shape[-1] != width:
        raise ValueError(
            f"RHT width mismatch: weight={weight.shape[-1]}, sign={width}."
        )
    hadamard = _sylvester_hadamard(block_size).to(
        device=weight.device, dtype=weight.dtype
    )
    shape = weight.shape
    blocks = weight.reshape(*shape[:-1], width // block_size, block_size)
    blocks = blocks @ hadamard
    blocks *= sign.to(device=weight.device, dtype=weight.dtype).reshape(
        width // block_size, block_size
    )
    return blocks.reshape(shape)


def _forward_rht_activation(
    activation: torch.Tensor,
    block_size: int,
    sign: torch.Tensor,
) -> torch.Tensor:
    """Move the inverse weight RHT to the activation side of a matmul."""
    width = sign.numel()
    if activation.shape[-1] != width:
        raise ValueError(
            f"RHT width mismatch: activation={activation.shape[-1]}, sign={width}."
        )
    hadamard = _sylvester_hadamard(block_size).to(
        device=activation.device, dtype=activation.dtype
    )
    shape = activation.shape
    blocks = activation.reshape(*shape[:-1], width // block_size, block_size)
    blocks = blocks * sign.to(device=activation.device, dtype=activation.dtype).reshape(
        width // block_size, block_size
    )
    blocks = blocks @ hadamard
    return blocks.reshape(shape)


def decode_codebook_weight(
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    device: torch.device | str,
) -> torch.Tensor:
    """Decode only VQ indices/codebooks, without perm/norm/RHT transforms."""
    rows = int(metadata["rows"])
    columns = int(metadata["cols"])
    row_tiles = int(metadata["n_row_tiles"])
    column_tiles = int(metadata["n_col_tiles"])
    row_group_size = int(metadata["row_group_size"])
    group_size = int(metadata["group_size"])
    codebook_size = int(metadata["K"])
    index_bits = int(metadata["index_bits"])
    vector_len = int(metadata["vector_len"])
    if index_bits != 4 or codebook_size != 16 or vector_len != 2:
        raise ValueError(
            "VQ2A8 requires index_bits=4, K=16, and vector_len=2, got "
            f"index_bits={index_bits}, K={codebook_size}, "
            f"vector_len={vector_len}."
        )
    if row_group_size % vector_len:
        raise ValueError(
            f"row_group_size={row_group_size} is not divisible by "
            f"vector_len={vector_len}."
        )
    if rows != row_tiles * row_group_size:
        raise ValueError(
            f"rows={rows} does not match n_row_tiles * row_group_size "
            f"({row_tiles * row_group_size})."
        )

    num_vectors = int(metadata["n_vectors"])
    indices = unpack_vq2_indices(tensors["packed_indices"], num_vectors, device)
    codebooks = tensors["codebooks"].to(device=device).float()
    expected_codebook_shape = (
        column_tiles,
        row_tiles,
        codebook_size,
        vector_len,
    )
    if tuple(codebooks.shape) != expected_codebook_shape:
        raise ValueError(
            f"Codebook shape is {tuple(codebooks.shape)}, expected "
            f"{expected_codebook_shape}."
        )

    weight = torch.empty((rows, columns), device=device, dtype=torch.float32)
    vectors_per_row_group = row_group_size // vector_len
    position = 0
    for column_tile in range(column_tiles):
        start = column_tile * group_size
        end = min(start + group_size, columns)
        width = end - start
        tile_vector_count = row_tiles * width * vectors_per_row_group
        tile_indices = indices[position : position + tile_vector_count].reshape(
            row_tiles, width * vectors_per_row_group
        )
        position += tile_vector_count
        vectors = torch.gather(
            codebooks[column_tile],
            1,
            tile_indices.unsqueeze(-1).expand(-1, -1, vector_len),
        )
        block = (
            vectors.reshape(row_tiles, width, row_group_size)
            .transpose(1, 2)
            .reshape(rows, width)
        )
        weight[:, start:end] = block
    if position != num_vectors:
        raise ValueError(
            f"Decoded {position} VQ vectors, but metadata declares {num_vectors}."
        )
    return weight


def apply_vq2_weight_transforms(
    weight: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> torch.Tensor:
    """Apply the producer's inverse perm/norm/RHT transforms to a VQ matrix."""
    if metadata["enable_perm"]:
        permutation = tensors["perm"].to(device=weight.device, dtype=torch.int64)
        weight = weight[:, torch.argsort(permutation)]

    norm_dim = int(metadata["norm_dim"])
    if metadata["enable_norm"]:
        scale = tensors["weight_scale"].to(device=weight.device).float()
        bias = tensors["weight_bias"].to(device=weight.device).float()
        weight = weight * scale.unsqueeze(norm_dim) + bias.unsqueeze(norm_dim)

    if metadata["enable_rht"]:
        weight = _inverse_rht(
            weight,
            int(metadata["rht_block_size"]),
            tensors["rht_sign"],
        )
        weight = weight[..., : int(metadata["rht_true_columns"])]
    return weight


def transform_vq2_activation(
    activation: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Move perm/norm/RHT to X for a packed-weight direct matmul."""
    if int(metadata["norm_dim"]) != 0:
        raise ValueError("VQ2A8 direct matmul currently requires norm_dim=0.")

    width = int(metadata["cols"])
    true_width = int(metadata.get("rht_true_columns", width))
    if activation.shape[-1] != true_width:
        raise ValueError(
            f"Activation width is {activation.shape[-1]}, expected {true_width}."
        )
    transformed = activation.float()
    if true_width < width:
        transformed = torch.nn.functional.pad(transformed, (0, width - true_width))
    if metadata["enable_rht"]:
        transformed = _forward_rht_activation(
            transformed,
            int(metadata["rht_block_size"]),
            tensors["rht_sign"],
        )

    bias_correction = None
    if metadata["enable_norm"]:
        scale = tensors["weight_scale"].to(
            device=transformed.device, dtype=transformed.dtype
        )
        bias = tensors["weight_bias"].to(
            device=transformed.device, dtype=transformed.dtype
        )
        bias_correction = transformed @ bias
        transformed = transformed * scale

    if metadata["enable_perm"]:
        permutation = tensors["perm"].to(device=transformed.device, dtype=torch.int64)
        transformed = transformed[..., permutation]
    return transformed, bias_correction


def vq2_matmul_reference(
    activation: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> torch.Tensor:
    """Correctness reference for direct VQ2 matmul without decoding final W."""
    codebook_weight = decode_codebook_weight(tensors, metadata, activation.device)
    transformed, bias_correction = transform_vq2_activation(
        activation, tensors, metadata
    )
    output = transformed @ codebook_weight.transpose(0, 1)
    if bias_correction is not None:
        output += bias_correction.unsqueeze(-1)
    return output


def decode_repacked_vq2_weight(
    packed_indices: torch.Tensor,
    codebooks: torch.Tensor,
    codebook_tile_ids: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
    rht_sign: torch.Tensor,
    rht_block_size: int,
) -> torch.Tensor:
    """Debug decoder for one TP-local ``vq2a8_tp_v1`` matrix."""
    input_size = codebook_tile_ids.numel()
    indices = unpack_vq2_kernel_indices(packed_indices, input_size)
    output_size = indices.shape[0] * 2
    if output_size % 32:
        raise ValueError(f"Output size {output_size} is not tile-aligned.")
    expected_codebook_shape = (
        codebooks.shape[0],
        output_size // 32,
        16,
        2,
    )
    if tuple(codebooks.shape) != expected_codebook_shape:
        raise ValueError(
            f"Codebook shape is {tuple(codebooks.shape)}, expected "
            f"{expected_codebook_shape}."
        )

    output_rows = torch.arange(output_size, device=indices.device)
    vector_rows = torch.div(output_rows, 2, rounding_mode="floor")
    row_tiles = torch.div(output_rows, 32, rounding_mode="floor")
    components = output_rows.remainder(2)
    source_tiles = codebook_tile_ids.to(torch.int64)
    weight = codebooks[
        source_tiles.unsqueeze(0),
        row_tiles.unsqueeze(1),
        indices[vector_rows],
        components.unsqueeze(1),
    ].float()
    weight = weight * weight_scale.float().unsqueeze(0)
    weight += weight_bias.float().unsqueeze(0)
    return _inverse_rht(weight, rht_block_size, rht_sign).to(torch.bfloat16)


def decode_expert_weight(
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    device: torch.device | str,
) -> torch.Tensor:
    """Decode one VPTQ vector-quantized 2-bit expert matrix to BF16."""
    weight = decode_codebook_weight(tensors, metadata, device)
    weight = apply_vq2_weight_transforms(weight, tensors, metadata)
    return weight.reshape(metadata["orig_shape"]).to(torch.bfloat16)


def shard_decoded_expert_weight(
    weight: torch.Tensor,
    name: str,
    tp_size: int,
    tp_rank: int,
) -> torch.Tensor:
    """Shard a decoded expert matrix using RoutedExperts' TP layout."""
    if tp_size < 1 or not 0 <= tp_rank < tp_size:
        raise ValueError(f"Invalid TP geometry: size={tp_size}, rank={tp_rank}.")
    if tp_size == 1:
        return weight

    if name.endswith(".gate_up"):
        if weight.shape[0] % (2 * tp_size):
            raise ValueError(
                f"gate_up rows {weight.shape[0]} are not divisible by "
                f"2 * tp_size ({2 * tp_size})."
            )
        gate, up = weight.chunk(2, dim=0)
        shard_size = gate.shape[0] // tp_size
        start = tp_rank * shard_size
        return torch.cat(
            [
                gate.narrow(0, start, shard_size),
                up.narrow(0, start, shard_size),
            ],
            dim=0,
        )

    if name.endswith(".down"):
        if weight.shape[1] % tp_size:
            raise ValueError(
                f"down columns {weight.shape[1]} are not divisible by "
                f"tp_size ({tp_size})."
            )
        shard_size = weight.shape[1] // tp_size
        return weight.narrow(1, tp_rank * shard_size, shard_size)

    raise ValueError(f"Unknown VQ2 expert tensor {name!r}.")


class VQ2A8MoEMethod(Fp8PtpcOnlineMoEMethod):
    """Overlay decoded VQ2 experts, then run per-token FP8 MoE kernels."""

    def __init__(
        self,
        *,
        layer: RoutedExperts,
        experts_path: str,
        prefix: str,
    ) -> None:
        self.experts_path = experts_path
        self.layer_index = extract_decoder_layer_index(prefix)
        super().__init__(layer=layer)

    @staticmethod
    def _payload_tensors(handle, name: str, metadata: dict[str, Any]):
        fields = ["packed_indices", "codebooks"]
        if metadata["enable_perm"]:
            fields.append("perm")
        if metadata["enable_norm"]:
            fields.extend(["weight_scale", "weight_bias"])
        if metadata["enable_rht"]:
            fields.append("rht_sign")
        return {field: handle.get_tensor(f"{name}.{field}") for field in fields}

    def _overlay_vq2_experts(self, layer: RoutedExperts) -> int:
        stem = f"experts_vq_layer_{self.layer_index}"
        tensor_path = os.path.join(self.experts_path, f"{stem}.safetensors")
        metadata_path = os.path.join(self.experts_path, f"{stem}.json")
        if not os.path.isfile(tensor_path) or not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                f"Missing VQ2 artifact for layer {self.layer_index}: {tensor_path}."
            )
        if layer.w13_weight.is_meta or layer.w2_weight.is_meta:
            raise RuntimeError(
                f"Layer {self.layer_index} VQ2 overlay ran before base weights "
                "were materialized."
            )

        with open(metadata_path, encoding="utf-8") as metadata_file:
            entries = json.load(metadata_file)

        loaded = 0
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            for name, metadata in entries.items():
                expert_id = int(name.split(".")[3])
                local_expert_id = layer._map_global_expert_id_to_local_expert_id(
                    expert_id
                )
                if local_expert_id < 0:
                    continue
                tensors = self._payload_tensors(handle, name, metadata)
                decoded = decode_expert_weight(
                    tensors, metadata, layer.w13_weight.device
                )
                parallel_config = layer.moe_config.moe_parallel_config
                decoded = shard_decoded_expert_weight(
                    decoded,
                    name,
                    parallel_config.tp_size,
                    parallel_config.tp_rank,
                )
                if name.endswith(".gate_up"):
                    target = layer.w13_weight[local_expert_id]
                elif name.endswith(".down"):
                    target = layer.w2_weight[local_expert_id]
                else:
                    raise ValueError(f"Unknown VQ2 expert tensor {name!r}.")
                if target.shape != decoded.shape:
                    raise ValueError(
                        f"Layer {self.layer_index} expert {expert_id} shape "
                        f"mismatch: target={tuple(target.shape)}, "
                        f"decoded={tuple(decoded.shape)}."
                    )
                target.copy_(decoded.to(dtype=target.dtype))
                loaded += 1
        return loaded

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        loaded = self._overlay_vq2_experts(layer)
        logger.info(
            "Loaded %d local VQ2 expert matrices for decoder layer %d",
            loaded,
            self.layer_index,
        )
        super().process_weights_after_loading(layer)


class VQ2A8DirectMoEMethod(FusedMoEMethodBase):
    """Direct packed VQ2A8 method for complete TP-local expert layers."""

    _KINDS = ("gate_up", "down")
    _FIELDS = (
        "packed_indices",
        "codebooks",
        "codebook_tile_ids",
        "weight_scale",
        "weight_bias",
        "rht_sign",
    )

    def __init__(
        self,
        *,
        layer: RoutedExperts,
        kernel_path: str,
        prefix: str,
    ) -> None:
        super().__init__(layer.moe_config)
        self.kernel_path = kernel_path
        self.layer_index = extract_decoder_layer_index(prefix)
        parallel = layer.moe_config.moe_parallel_config
        self.tp_size = parallel.tp_size
        self.tp_rank = parallel.tp_rank
        stem = f"experts_vq_layer_{self.layer_index}"
        rank_dir = os.path.join(kernel_path, f"tp{self.tp_size}", f"rank{self.tp_rank}")
        self.tensor_path = os.path.join(rank_dir, f"{stem}.safetensors")
        self.metadata_path = os.path.join(rank_dir, f"{stem}.json")

    @staticmethod
    def _discard_base_expert_weight(
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        *args,
        return_success: bool = False,
        **kwargs,
    ) -> bool | None:
        del param, loaded_weight, args, kwargs
        return True if return_success else None

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del hidden_size, intermediate_size_per_partition
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None
        attrs = dict(extra_weight_attrs)
        attrs["weight_loader"] = self._discard_base_expert_weight
        for name in ("w13_weight", "w2_weight"):
            placeholder = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter(name, placeholder)
            set_weight_attrs(placeholder, attrs)
        for kind in self._KINDS:
            for field in self._FIELDS:
                layer.register_buffer(
                    f"vq_{kind}_{field}", torch.empty(0), persistent=False
                )
        layer.register_buffer("vq_expert_ids", torch.empty(0), persistent=False)
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def get_fused_moe_quant_config(self, layer: RoutedExperts):
        del layer
        return None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        with open(self.metadata_path, encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        if metadata.get("format") != "vq2a8_tp_v1":
            raise ValueError(
                f"Unsupported VQ2 kernel format in {self.metadata_path}: "
                f"{metadata.get('format')!r}."
            )
        if not metadata.get("complete", False):
            raise ValueError(
                f"Direct VQ2 layer {self.layer_index} has incomplete experts."
            )
        if (
            metadata.get("tp_size") != self.tp_size
            or metadata.get("tp_rank") != self.tp_rank
        ):
            raise ValueError(f"TP metadata mismatch for VQ2 layer {self.layer_index}.")

        device = layer.w13_weight.device
        with safe_open(self.tensor_path, framework="pt", device="cpu") as handle:
            for kind in self._KINDS:
                for field in self._FIELDS:
                    tensor = handle.get_tensor(f"{kind}_{field}")
                    setattr(
                        layer,
                        f"vq_{kind}_{field}",
                        tensor.to(device=device, non_blocking=True).contiguous(),
                    )
        layer.vq_expert_ids = torch.tensor(
            metadata["expert_ids"], device=device, dtype=torch.int32
        )
        layer._already_called_process_weights_after_loading = True
        logger.info(
            "Loaded direct TP%d rank%d VQ2 experts for decoder layer %d",
            self.tp_size,
            self.tp_rank,
            self.layer_index,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        from vllm.model_executor.layers.fused_moe.experts.vq2a8_triton import (
            vq2a8_moe,
        )

        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "VQ2A8 direct backend does not support router weight on input."
            )
        return vq2a8_moe(layer, x, topk_weights, topk_ids)


class VQ2A8Config(QuantizationConfig):
    """Configuration for VPTQ VQ2 routed experts with FP8 activations."""

    def __init__(self, experts_path: str, kernel_path: str) -> None:
        super().__init__()
        self.experts_path = experts_path
        self.kernel_path = kernel_path

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "vq2a8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> VQ2A8Config:
        experts_path = cls.get_from_keys(config, ["experts_path"])
        kernel_path = config.get("kernel_path", "experts_vq_kernel")
        return cls(str(experts_path), str(kernel_path))

    def maybe_update_config(
        self,
        model_name: str,
        hf_config=None,
        revision: str | None = None,
    ) -> None:
        del hf_config, revision
        if not os.path.isabs(self.experts_path):
            self.experts_path = os.path.abspath(
                os.path.join(model_name, self.experts_path)
            )
        if not os.path.isabs(self.kernel_path):
            self.kernel_path = os.path.abspath(
                os.path.join(model_name, self.kernel_path)
            )
        if not os.path.isdir(self.experts_path):
            raise ValueError(f"VQ2 expert artifact not found: {self.experts_path}")

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, RoutedExperts):
            if get_vq2_backend() == VQ2_DIRECT_BACKEND:
                layer_index = extract_decoder_layer_index(prefix)
                parallel = layer.moe_config.moe_parallel_config
                metadata_path = os.path.join(
                    self.kernel_path,
                    f"tp{parallel.tp_size}",
                    f"rank{parallel.tp_rank}",
                    f"experts_vq_layer_{layer_index}.json",
                )
                if os.path.isfile(metadata_path):
                    with open(metadata_path, encoding="utf-8") as metadata_file:
                        metadata = json.load(metadata_file)
                    if metadata.get("complete", False):
                        return VQ2A8DirectMoEMethod(
                            layer=layer,
                            kernel_path=self.kernel_path,
                            prefix=prefix,
                        )
                logger.warning(
                    "Falling back to decode_fp8 for incomplete VQ2 layer %d",
                    layer_index,
                )
            return VQ2A8MoEMethod(
                layer=layer,
                experts_path=self.experts_path,
                prefix=prefix,
            )
        if isinstance(layer, LinearBase):
            # DeepSeek V4's CUDA attention output path consumes FP8 weights
            # directly. Its grouped-BMM wo_a needs 128x128 block scales and a
            # 3D DeepGEMM layout; other linears use vLLM's native online-FP8
            # policy for BF16 checkpoints.
            if prefix.endswith(".wo_a"):
                return Fp8PerBlockOnlineLinearMethod()
            return Fp8PerTensorOnlineLinearMethod()
        return None
