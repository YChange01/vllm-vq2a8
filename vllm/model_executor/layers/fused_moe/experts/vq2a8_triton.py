# SPDX-License-Identifier: Apache-2.0
"""Triton kernels for TP-local VQ2A8 expert matrices."""

from __future__ import annotations

import functools
import math
import os

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)

FUSED_GATE_UP_MAX_ASSIGNMENTS = 16
GROUPED_GEMM_MIN_ASSIGNMENTS = 192
CUDA_DOWN_REDUCE_MAX_ASSIGNMENTS = 12
VQ2_PREP_BACKEND_ENV = "VLLM_VQ2_PREP_BACKEND"
VQ2_GEMM_BACKEND_ENV = "VLLM_VQ2_GEMM_BACKEND"


@functools.cache
def _hadamard_tensor(block_size: int, device: torch.device) -> torch.Tensor:
    matrix = torch.tensor([[1.0]], device=device, dtype=torch.bfloat16)
    while matrix.shape[0] < block_size:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            )
        )
    return matrix / math.sqrt(block_size)


def _normalized_fwht(activation: torch.Tensor, block_size: int) -> torch.Tensor:
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError(f"RHT block size must be a power of two, got {block_size}.")
    if activation.shape[-1] % block_size:
        raise ValueError(
            f"Activation width {activation.shape[-1]} is not divisible by "
            f"RHT block size {block_size}."
        )
    output = activation.reshape(*activation.shape[:-1], -1, block_size)
    step = 1
    while step < block_size:
        shape = output.shape
        pairs = output.reshape(*shape[:-1], -1, 2, step)
        left, right = pairs.unbind(dim=-2)
        output = torch.cat((left + right, left - right), dim=-1).reshape(shape)
        step *= 2
    return output.reshape(activation.shape) / math.sqrt(block_size)


def prepare_vq2a8_activation(
    activation: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
    rht_sign: torch.Tensor,
    rht_block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply expert-specific transforms and dynamic per-token FP8 quantization."""
    if activation.ndim != 2:
        raise ValueError(f"Expected a 2D activation, got {activation.shape}.")
    input_size = activation.shape[1]
    for name, value in (
        ("weight_scale", weight_scale),
        ("weight_bias", weight_bias),
        ("rht_sign", rht_sign),
    ):
        if value.numel() != input_size:
            raise ValueError(
                f"{name} has {value.numel()} values, expected {input_size}."
            )

    rotated = _normalized_fwht(
        activation.float() * rht_sign.float().unsqueeze(0), rht_block_size
    )
    bias_correction = rotated @ weight_bias.float()
    transformed = rotated * weight_scale.float().unsqueeze(0)
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
    amax = transformed.abs().amax(dim=-1)
    activation_scale = torch.clamp(amax / fp8_max, min=1e-12)
    quantized = torch.clamp(
        transformed / activation_scale.unsqueeze(-1), -fp8_max, fp8_max
    ).to(fp8_dtype)
    return quantized, activation_scale, bias_correction


def prepare_vq2a8_batched_activation(
    activation: torch.Tensor,
    expert_ids: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
    rht_sign: torch.Tensor,
    rht_block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized activation preparation for token-expert assignments."""
    expert_ids = expert_ids.to(torch.int64)
    signs = rht_sign[expert_ids].float()
    scales = weight_scale[expert_ids].float()
    biases = weight_bias[expert_ids].float()
    rotated = _normalized_fwht(activation.float() * signs, rht_block_size)
    bias_correction = torch.sum(rotated * biases, dim=-1)
    transformed = rotated * scales
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
    amax = transformed.abs().amax(dim=-1)
    activation_scale = torch.clamp(amax / fp8_max, min=1e-12)
    quantized = torch.clamp(
        transformed / activation_scale.unsqueeze(-1), -fp8_max, fp8_max
    ).to(fp8_dtype)
    return quantized, activation_scale, bias_correction


@triton.jit
def _vq2a8_rht_kernel(
    activation_ptr,
    expert_ids_ptr,
    weight_scale_ptr,
    weight_bias_ptr,
    rht_sign_ptr,
    hadamard_ptr,
    transformed_ptr,
    partial_amax_ptr,
    partial_bias_ptr,
    size_k: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_se: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_hm: tl.constexpr,
    stride_hk: tl.constexpr,
    stride_tm: tl.constexpr,
    stride_tk: tl.constexpr,
    stride_pm: tl.constexpr,
    stride_pb: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_RHT: tl.constexpr,
):
    assignment = tl.program_id(0)
    rht_block = tl.program_id(1)
    expert_id = tl.load(expert_ids_ptr + assignment).to(tl.int64)
    offsets_k = rht_block * BLOCK_RHT + tl.arange(0, BLOCK_RHT)
    activation_vector = tl.load(
        activation_ptr + assignment * stride_am + offsets_k * stride_ak
    )
    signs = tl.load(rht_sign_ptr + expert_id * stride_se + offsets_k * stride_sk)
    signed = (activation_vector * signs).to(tl.bfloat16)
    activation_matrix = tl.broadcast_to(signed[None, :], (BLOCK_M, BLOCK_RHT))
    hadamard_offsets = (
        tl.arange(0, BLOCK_RHT)[:, None] * stride_hm
        + tl.arange(0, BLOCK_RHT)[None, :] * stride_hk
    )
    hadamard = tl.load(hadamard_ptr + hadamard_offsets)
    rotated_matrix = tl.dot(activation_matrix, hadamard)
    rotated = tl.sum(rotated_matrix * (tl.arange(0, BLOCK_M)[:, None] == 0), axis=0)
    rotated = tl.reshape(rotated, (BLOCK_RHT,))
    scales = tl.load(weight_scale_ptr + expert_id * stride_se + offsets_k * stride_sk)
    biases = tl.load(weight_bias_ptr + expert_id * stride_se + offsets_k * stride_sk)
    transformed = rotated * scales
    tl.store(
        transformed_ptr + assignment * stride_tm + offsets_k * stride_tk,
        transformed,
    )
    tl.store(
        partial_amax_ptr + assignment * stride_pm + rht_block * stride_pb,
        tl.max(tl.abs(transformed), axis=0),
    )
    tl.store(
        partial_bias_ptr + assignment * stride_pm + rht_block * stride_pb,
        tl.sum(rotated * biases, axis=0),
    )


@triton.jit
def _vq2a8_quantize_rht_kernel(
    transformed_ptr,
    partial_amax_ptr,
    partial_bias_ptr,
    quantized_ptr,
    activation_scale_ptr,
    bias_correction_ptr,
    size_k: tl.constexpr,
    num_rht_blocks: tl.constexpr,
    stride_tm: tl.constexpr,
    stride_tk: tl.constexpr,
    stride_pm: tl.constexpr,
    stride_pb: tl.constexpr,
    BLOCK_REDUCE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    assignment = tl.program_id(0)
    if num_rht_blocks == 1:
        amax = tl.load(partial_amax_ptr + assignment * stride_pm)
        bias_correction = tl.load(partial_bias_ptr + assignment * stride_pm)
    else:
        offsets_block = tl.arange(0, BLOCK_REDUCE)
        block_mask = offsets_block < num_rht_blocks
        partial_amax = tl.load(
            partial_amax_ptr + assignment * stride_pm + offsets_block * stride_pb,
            mask=block_mask,
            other=0.0,
        )
        partial_bias = tl.load(
            partial_bias_ptr + assignment * stride_pm + offsets_block * stride_pb,
            mask=block_mask,
            other=0.0,
        )
        amax = tl.max(partial_amax, axis=0)
        bias_correction = tl.sum(partial_bias, axis=0)
    scale = tl.maximum(amax / 448.0, 1e-12)
    tl.store(activation_scale_ptr + assignment, scale)
    tl.store(bias_correction_ptr + assignment, bias_correction)

    for start_k in range(0, size_k, BLOCK_K):
        offsets_k = start_k + tl.arange(0, BLOCK_K)
        transformed = tl.load(
            transformed_ptr + assignment * stride_tm + offsets_k * stride_tk,
            mask=offsets_k < size_k,
            other=0.0,
        )
        quantized = tl.minimum(tl.maximum(transformed / scale, -448.0), 448.0)
        tl.store(
            quantized_ptr + assignment * stride_tm + offsets_k * stride_tk,
            quantized,
            mask=offsets_k < size_k,
        )


def prepare_vq2a8_batched_activation_triton(
    activation: torch.Tensor,
    expert_ids: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
    rht_sign: torch.Tensor,
    rht_block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused SM90 activation RHT, norm, bias correction, and A8 quantization."""
    if rht_block_size != 128 or activation.shape[1] % rht_block_size:
        return prepare_vq2a8_batched_activation(
            activation,
            expert_ids,
            weight_scale,
            weight_bias,
            rht_sign,
            rht_block_size,
        )
    size_m, size_k = activation.shape
    num_rht_blocks = size_k // rht_block_size
    transformed = torch.empty(
        (size_m, size_k), device=activation.device, dtype=torch.float32
    )
    partial_amax = torch.empty(
        (size_m, num_rht_blocks), device=activation.device, dtype=torch.float32
    )
    partial_bias = torch.empty_like(partial_amax)
    quantized = torch.empty(
        (size_m, size_k),
        device=activation.device,
        dtype=torch.float8_e4m3fn,
    )
    activation_scale = torch.empty(
        size_m, device=activation.device, dtype=torch.float32
    )
    bias_correction = torch.empty_like(activation_scale)
    hadamard = _hadamard_tensor(rht_block_size, activation.device)
    block_m = 16
    _vq2a8_rht_kernel[(size_m, num_rht_blocks)](
        activation,
        expert_ids,
        weight_scale,
        weight_bias,
        rht_sign,
        hadamard,
        transformed,
        partial_amax,
        partial_bias,
        size_k,
        activation.stride(0),
        activation.stride(1),
        weight_scale.stride(0),
        weight_scale.stride(1),
        hadamard.stride(0),
        hadamard.stride(1),
        transformed.stride(0),
        transformed.stride(1),
        partial_amax.stride(0),
        partial_amax.stride(1),
        BLOCK_M=block_m,
        BLOCK_RHT=rht_block_size,
        num_warps=8,
    )
    _vq2a8_quantize_rht_kernel[(size_m,)](
        transformed,
        partial_amax,
        partial_bias,
        quantized,
        activation_scale,
        bias_correction,
        size_k,
        num_rht_blocks,
        transformed.stride(0),
        transformed.stride(1),
        partial_amax.stride(0),
        partial_amax.stride(1),
        BLOCK_REDUCE=triton.next_power_of_2(num_rht_blocks),
        BLOCK_K=256,
        num_warps=4,
    )
    return quantized, activation_scale, bias_correction


def prepare_vq2a8_batched_activation_dispatch(
    activation: torch.Tensor,
    expert_ids: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
    rht_sign: torch.Tensor,
    rht_block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch activation preparation to CUDA C++ or the Triton fallback."""
    if use_vq2a8_cuda_preparation():
        from .vq2a8_cuda import prepare_vq2a8_activation_cuda

        return prepare_vq2a8_activation_cuda(
            activation,
            expert_ids,
            weight_scale,
            weight_bias,
            rht_sign,
            rht_block_size,
        )
    return prepare_vq2a8_batched_activation_triton(
        activation,
        expert_ids,
        weight_scale,
        weight_bias,
        rht_sign,
        rht_block_size,
    )


def use_vq2a8_cuda_preparation() -> bool:
    """Return whether the optional CUDA C++ activation kernel is selected."""
    backend = os.getenv(VQ2_PREP_BACKEND_ENV, "auto").lower()
    if backend not in ("auto", "cuda", "triton"):
        raise ValueError(
            f"{VQ2_PREP_BACKEND_ENV} must be 'auto', 'cuda', or 'triton', "
            f"got {backend!r}."
        )
    if backend == "triton":
        return False
    from .vq2a8_cuda import load_vq2a8_cuda_extension

    available = load_vq2a8_cuda_extension()
    if not available and backend == "cuda":
        raise RuntimeError(
            "VLLM_VQ2_PREP_BACKEND=cuda was requested, but the CUDA "
            "extension is unavailable."
        )
    return available


def use_vq2a8_cuda_down_reduce(num_assignments: int) -> bool:
    """Select CUDA down-reduce only in its measured low-latency range."""
    backend = os.getenv(VQ2_GEMM_BACKEND_ENV, "auto").lower()
    if backend not in ("auto", "cuda", "triton"):
        raise ValueError(
            f"{VQ2_GEMM_BACKEND_ENV} must be 'auto', 'cuda', or 'triton', "
            f"got {backend!r}."
        )
    if backend == "triton":
        return False
    from .vq2a8_cuda import has_vq2a8_cuda_down_reduce

    available = has_vq2a8_cuda_down_reduce()
    if not available and backend == "cuda":
        raise RuntimeError(
            "VLLM_VQ2_GEMM_BACKEND=cuda was requested, but the CUDA "
            "down-reduce operator is unavailable."
        )
    return available and (
        backend == "cuda" or num_assignments <= CUDA_DOWN_REDUCE_MAX_ASSIGNMENTS
    )


@triton.jit
def _vq2a8_gemm_kernel(
    activation_ptr,
    activation_scale_ptr,
    packed_indices_ptr,
    codebooks_ptr,
    codebook_tile_ids_ptr,
    bias_correction_ptr,
    output_ptr,
    size_m: tl.constexpr,
    size_n: tl.constexpr,
    size_k: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_pn: tl.constexpr,
    stride_pk: tl.constexpr,
    stride_cbt: tl.constexpr,
    stride_cbr: tl.constexpr,
    stride_cbi: tl.constexpr,
    stride_cbc: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    program_m = tl.program_id(0)
    program_n = tl.program_id(1)
    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for start_k in range(0, size_k, BLOCK_K):
        offsets_k = start_k + tl.arange(0, BLOCK_K)
        activation = tl.load(
            activation_ptr
            + offsets_m[:, None] * stride_am
            + offsets_k[None, :] * stride_ak,
            mask=offsets_m[:, None] < size_m,
            other=0.0,
        )
        vector_rows = offsets_n // 2
        words = tl.load(
            packed_indices_ptr
            + vector_rows[:, None] * stride_pn
            + (offsets_k[None, :] // 8) * stride_pk
        )
        codes = (words >> ((offsets_k[None, :] % 8) * 4)) & 15
        source_tiles = tl.load(codebook_tile_ids_ptr + offsets_k).to(tl.int32)
        output_tiles = offsets_n // 32
        components = offsets_n % 2
        codebook_offsets = (
            source_tiles[None, :] * stride_cbt
            + output_tiles[:, None] * stride_cbr
            + codes * stride_cbi
            + components[:, None] * stride_cbc
        )
        weights = tl.load(codebooks_ptr + codebook_offsets)
        accumulator += tl.dot(activation, tl.trans(weights))

    activation_scale = tl.load(
        activation_scale_ptr + offsets_m,
        mask=offsets_m < size_m,
        other=0.0,
    )
    bias_correction = tl.load(
        bias_correction_ptr + offsets_m,
        mask=offsets_m < size_m,
        other=0.0,
    )
    output = accumulator * activation_scale[:, None]
    output += bias_correction[:, None]
    tl.store(
        output_ptr + offsets_m[:, None] * stride_om + offsets_n[None, :] * stride_on,
        output,
        mask=offsets_m[:, None] < size_m,
    )


def vq2a8_gemm(
    activation: torch.Tensor,
    packed_indices: torch.Tensor,
    codebooks: torch.Tensor,
    codebook_tile_ids: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
    rht_sign: torch.Tensor,
    rht_block_size: int = 128,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run one expert GEMM directly from packed VQ2 codes and FP8 codebooks."""
    if not activation.is_cuda:
        raise ValueError("The VQ2A8 Triton kernel requires CUDA tensors.")
    if not packed_indices.is_contiguous() or not codebooks.is_contiguous():
        raise ValueError("Packed indices and codebooks must be contiguous.")
    size_m, size_k = activation.shape
    size_n = packed_indices.shape[0] * 2
    if packed_indices.shape[1] * 8 < size_k:
        raise ValueError("Packed VQ2 indices do not cover the activation K size.")
    if size_n % 32 or size_k % 64:
        raise ValueError(f"Unsupported VQ2A8 shape MxNxK={size_m}x{size_n}x{size_k}.")

    quantized, activation_scale, bias_correction = prepare_vq2a8_activation(
        activation,
        weight_scale,
        weight_bias,
        rht_sign,
        rht_block_size,
    )
    output = torch.empty((size_m, size_n), device=activation.device, dtype=output_dtype)
    block_m, block_n, block_k = 16, 32, 64
    grid = (triton.cdiv(size_m, block_m), triton.cdiv(size_n, block_n))
    _vq2a8_gemm_kernel[grid](
        quantized,
        activation_scale,
        packed_indices,
        codebooks,
        codebook_tile_ids,
        bias_correction,
        output,
        size_m,
        size_n,
        size_k,
        quantized.stride(0),
        quantized.stride(1),
        packed_indices.stride(0),
        packed_indices.stride(1),
        codebooks.stride(0),
        codebooks.stride(1),
        codebooks.stride(2),
        codebooks.stride(3),
        output.stride(0),
        output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return output


@triton.jit
def _vq2a8_batched_gemm_kernel(
    activation_ptr,
    activation_scale_ptr,
    expert_ids_ptr,
    packed_indices_ptr,
    codebooks_ptr,
    codebook_tile_ids_ptr,
    bias_correction_ptr,
    output_ptr,
    size_n: tl.constexpr,
    size_k: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_pe: tl.constexpr,
    stride_pn: tl.constexpr,
    stride_pk: tl.constexpr,
    stride_cbe: tl.constexpr,
    stride_cbt: tl.constexpr,
    stride_cbr: tl.constexpr,
    stride_cbi: tl.constexpr,
    stride_cbc: tl.constexpr,
    stride_te: tl.constexpr,
    stride_tk: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    assignment = tl.program_id(0)
    program_n = tl.program_id(1)
    expert_id = tl.load(expert_ids_ptr + assignment).to(tl.int64)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for start_k in range(0, size_k, BLOCK_K):
        offsets_k = start_k + tl.arange(0, BLOCK_K)
        activation_vector = tl.load(
            activation_ptr + assignment * stride_am + offsets_k * stride_ak
        )
        activation = tl.broadcast_to(activation_vector[None, :], (BLOCK_M, BLOCK_K))
        vector_rows = offsets_n // 2
        words = tl.load(
            packed_indices_ptr
            + expert_id * stride_pe
            + vector_rows[:, None] * stride_pn
            + (offsets_k[None, :] // 8) * stride_pk
        )
        codes = (words >> ((offsets_k[None, :] % 8) * 4)) & 15
        source_tiles = tl.load(
            codebook_tile_ids_ptr + expert_id * stride_te + offsets_k * stride_tk
        ).to(tl.int64)
        output_tiles = offsets_n // 32
        components = offsets_n % 2
        codebook_offsets = (
            expert_id * stride_cbe
            + source_tiles[None, :] * stride_cbt
            + output_tiles[:, None] * stride_cbr
            + codes * stride_cbi
            + components[:, None] * stride_cbc
        )
        weights = tl.load(codebooks_ptr + codebook_offsets)
        accumulator += tl.dot(activation, tl.trans(weights))

    activation_scale = tl.load(activation_scale_ptr + assignment)
    bias_correction = tl.load(bias_correction_ptr + assignment)
    first_row = tl.sum(accumulator * (tl.arange(0, BLOCK_M)[:, None] == 0), axis=0)
    first_row = tl.reshape(first_row, (BLOCK_N,))
    output = first_row * activation_scale + bias_correction
    tl.store(output_ptr + assignment * stride_om + offsets_n * stride_on, output)


def vq2a8_batched_gemm(
    activation: torch.Tensor,
    expert_ids: torch.Tensor,
    packed_indices: torch.Tensor,
    codebooks: torch.Tensor,
    codebook_tile_ids: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
    rht_sign: torch.Tensor,
    rht_block_size: int = 128,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run packed GEMMs for a batch of token-expert assignments."""
    size_m, size_k = activation.shape
    size_n = packed_indices.shape[1] * 2
    quantized, activation_scale, bias_correction = (
        prepare_vq2a8_batched_activation_dispatch(
            activation,
            expert_ids,
            weight_scale,
            weight_bias,
            rht_sign,
            rht_block_size,
        )
    )
    output = torch.empty((size_m, size_n), device=activation.device, dtype=output_dtype)
    block_m, block_n, block_k = 16, 32, 64
    grid = (size_m, triton.cdiv(size_n, block_n))
    _vq2a8_batched_gemm_kernel[grid](
        quantized,
        activation_scale,
        expert_ids,
        packed_indices,
        codebooks,
        codebook_tile_ids,
        bias_correction,
        output,
        size_n,
        size_k,
        quantized.stride(0),
        quantized.stride(1),
        packed_indices.stride(0),
        packed_indices.stride(1),
        packed_indices.stride(2),
        codebooks.stride(0),
        codebooks.stride(1),
        codebooks.stride(2),
        codebooks.stride(3),
        codebooks.stride(4),
        codebook_tile_ids.stride(0),
        codebook_tile_ids.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return output


@triton.jit
def _vq2a8_batched_gate_up_kernel(
    activation_ptr,
    activation_scale_ptr,
    expert_ids_ptr,
    packed_indices_ptr,
    codebooks_ptr,
    codebook_tile_ids_ptr,
    bias_correction_ptr,
    output_ptr,
    swiglu_limit,
    size_n: tl.constexpr,
    size_k: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_pe: tl.constexpr,
    stride_pn: tl.constexpr,
    stride_pk: tl.constexpr,
    stride_cbe: tl.constexpr,
    stride_cbt: tl.constexpr,
    stride_cbr: tl.constexpr,
    stride_cbi: tl.constexpr,
    stride_cbc: tl.constexpr,
    stride_te: tl.constexpr,
    stride_tk: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
):
    assignment = tl.program_id(0)
    program_n = tl.program_id(1)
    expert_id = tl.load(expert_ids_ptr + assignment).to(tl.int64)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    gate_offsets_n = offsets_n
    up_offsets_n = offsets_n + size_n
    gate_accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for start_k in range(0, size_k, BLOCK_K):
        offsets_k = start_k + tl.arange(0, BLOCK_K)
        activation_vector = tl.load(
            activation_ptr + assignment * stride_am + offsets_k * stride_ak
        )
        activation = tl.broadcast_to(activation_vector[None, :], (BLOCK_M, BLOCK_K))
        source_tiles = tl.load(
            codebook_tile_ids_ptr + expert_id * stride_te + offsets_k * stride_tk
        ).to(tl.int64)

        gate_vector_rows = gate_offsets_n // 2
        gate_words = tl.load(
            packed_indices_ptr
            + expert_id * stride_pe
            + gate_vector_rows[:, None] * stride_pn
            + (offsets_k[None, :] // 8) * stride_pk
        )
        gate_codes = (gate_words >> ((offsets_k[None, :] % 8) * 4)) & 15
        gate_codebook_offsets = (
            expert_id * stride_cbe
            + source_tiles[None, :] * stride_cbt
            + (gate_offsets_n // 32)[:, None] * stride_cbr
            + gate_codes * stride_cbi
            + (gate_offsets_n % 2)[:, None] * stride_cbc
        )
        gate_weights = tl.load(codebooks_ptr + gate_codebook_offsets)
        gate_accumulator += tl.dot(activation, tl.trans(gate_weights))

        up_vector_rows = up_offsets_n // 2
        up_words = tl.load(
            packed_indices_ptr
            + expert_id * stride_pe
            + up_vector_rows[:, None] * stride_pn
            + (offsets_k[None, :] // 8) * stride_pk
        )
        up_codes = (up_words >> ((offsets_k[None, :] % 8) * 4)) & 15
        up_codebook_offsets = (
            expert_id * stride_cbe
            + source_tiles[None, :] * stride_cbt
            + (up_offsets_n // 32)[:, None] * stride_cbr
            + up_codes * stride_cbi
            + (up_offsets_n % 2)[:, None] * stride_cbc
        )
        up_weights = tl.load(codebooks_ptr + up_codebook_offsets)
        up_accumulator += tl.dot(activation, tl.trans(up_weights))

    first_row_mask = tl.arange(0, BLOCK_M)[:, None] == 0
    gate = tl.sum(gate_accumulator * first_row_mask, axis=0)
    up = tl.sum(up_accumulator * first_row_mask, axis=0)
    gate = tl.reshape(gate, (BLOCK_N,))
    up = tl.reshape(up, (BLOCK_N,))
    activation_scale = tl.load(activation_scale_ptr + assignment)
    bias_correction = tl.load(bias_correction_ptr + assignment)
    gate = gate * activation_scale + bias_correction
    up = up * activation_scale + bias_correction

    # Match the previous BF16 gate/up materialization before the activation.
    gate = gate.to(tl.bfloat16).to(tl.float32)
    up = up.to(tl.bfloat16).to(tl.float32)
    if HAS_LIMIT:
        gate = tl.minimum(gate, swiglu_limit)
        up = tl.minimum(tl.maximum(up, -swiglu_limit), swiglu_limit)
    silu = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
    intermediate = (silu * up).to(tl.bfloat16)
    tl.store(
        output_ptr + assignment * stride_om + offsets_n * stride_on,
        intermediate,
    )


def vq2a8_batched_gate_up(
    activation: torch.Tensor,
    expert_ids: torch.Tensor,
    packed_indices: torch.Tensor,
    codebooks: torch.Tensor,
    codebook_tile_ids: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
    rht_sign: torch.Tensor,
    swiglu_limit: float | None,
    rht_block_size: int = 128,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run packed gate/up GEMMs and fuse the SwiGLU epilogue."""
    size_m, size_k = activation.shape
    gate_up_size = packed_indices.shape[1] * 2
    if gate_up_size % 2:
        raise ValueError(f"Gate/up size must be even, got {gate_up_size}.")
    size_n = gate_up_size // 2
    if size_n % 32:
        raise ValueError(f"Intermediate size must be tile-aligned, got {size_n}.")
    quantized, activation_scale, bias_correction = (
        prepare_vq2a8_batched_activation_dispatch(
            activation,
            expert_ids,
            weight_scale,
            weight_bias,
            rht_sign,
            rht_block_size,
        )
    )
    output = torch.empty((size_m, size_n), device=activation.device, dtype=output_dtype)
    block_m, block_n, block_k = 16, 32, 64
    grid = (size_m, triton.cdiv(size_n, block_n))
    has_limit = swiglu_limit is not None and swiglu_limit > 0
    _vq2a8_batched_gate_up_kernel[grid](
        quantized,
        activation_scale,
        expert_ids,
        packed_indices,
        codebooks,
        codebook_tile_ids,
        bias_correction,
        output,
        float(swiglu_limit) if has_limit else 0.0,
        size_n,
        size_k,
        quantized.stride(0),
        quantized.stride(1),
        packed_indices.stride(0),
        packed_indices.stride(1),
        packed_indices.stride(2),
        codebooks.stride(0),
        codebooks.stride(1),
        codebooks.stride(2),
        codebooks.stride(3),
        codebooks.stride(4),
        codebook_tile_ids.stride(0),
        codebook_tile_ids.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        HAS_LIMIT=has_limit,
        num_warps=4,
    )
    return output


@triton.jit
def _vq2a8_batched_down_reduce_kernel(
    activation_ptr,
    activation_scale_ptr,
    expert_ids_ptr,
    token_ids_ptr,
    routing_weights_ptr,
    packed_indices_ptr,
    codebooks_ptr,
    codebook_tile_ids_ptr,
    bias_correction_ptr,
    output_ptr,
    size_n: tl.constexpr,
    size_k: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_pe: tl.constexpr,
    stride_pn: tl.constexpr,
    stride_pk: tl.constexpr,
    stride_cbe: tl.constexpr,
    stride_cbt: tl.constexpr,
    stride_cbr: tl.constexpr,
    stride_cbi: tl.constexpr,
    stride_cbc: tl.constexpr,
    stride_te: tl.constexpr,
    stride_tk: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    assignment = tl.program_id(0)
    program_n = tl.program_id(1)
    expert_id = tl.load(expert_ids_ptr + assignment).to(tl.int64)
    token_id = tl.load(token_ids_ptr + assignment).to(tl.int64)
    routing_weight = tl.load(routing_weights_ptr + assignment)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for start_k in range(0, size_k, BLOCK_K):
        offsets_k = start_k + tl.arange(0, BLOCK_K)
        activation_vector = tl.load(
            activation_ptr + assignment * stride_am + offsets_k * stride_ak
        )
        activation = tl.broadcast_to(activation_vector[None, :], (BLOCK_M, BLOCK_K))
        vector_rows = offsets_n // 2
        words = tl.load(
            packed_indices_ptr
            + expert_id * stride_pe
            + vector_rows[:, None] * stride_pn
            + (offsets_k[None, :] // 8) * stride_pk
        )
        codes = (words >> ((offsets_k[None, :] % 8) * 4)) & 15
        source_tiles = tl.load(
            codebook_tile_ids_ptr + expert_id * stride_te + offsets_k * stride_tk
        ).to(tl.int64)
        codebook_offsets = (
            expert_id * stride_cbe
            + source_tiles[None, :] * stride_cbt
            + (offsets_n // 32)[:, None] * stride_cbr
            + codes * stride_cbi
            + (offsets_n % 2)[:, None] * stride_cbc
        )
        weights = tl.load(codebooks_ptr + codebook_offsets)
        accumulator += tl.dot(activation, tl.trans(weights))

    first_row = tl.sum(accumulator * (tl.arange(0, BLOCK_M)[:, None] == 0), axis=0)
    first_row = tl.reshape(first_row, (BLOCK_N,))
    activation_scale = tl.load(activation_scale_ptr + assignment)
    bias_correction = tl.load(bias_correction_ptr + assignment)
    expert_output = first_row * activation_scale + bias_correction
    # Preserve the old BF16 expert-output boundary before route accumulation.
    routed = expert_output.to(tl.bfloat16).to(tl.float32) * routing_weight
    tl.atomic_add(
        output_ptr + token_id * stride_om + offsets_n * stride_on,
        routed,
    )


def vq2a8_batched_down_reduce(
    activation: torch.Tensor,
    expert_ids: torch.Tensor,
    token_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    num_tokens: int,
    packed_indices: torch.Tensor,
    codebooks: torch.Tensor,
    codebook_tile_ids: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
    rht_sign: torch.Tensor,
    rht_block_size: int = 128,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run packed down GEMMs and fuse route weighting and token reduction."""
    size_m, size_k = activation.shape
    if expert_ids.numel() != size_m or token_ids.numel() != size_m:
        raise ValueError("Assignment, expert, and token counts must match.")
    if routing_weights.numel() != size_m:
        raise ValueError("Routing weight count must match assignments.")
    size_n = packed_indices.shape[1] * 2
    if size_n % 32:
        raise ValueError(f"Output size must be tile-aligned, got {size_n}.")
    quantized, activation_scale, bias_correction = (
        prepare_vq2a8_batched_activation_dispatch(
            activation,
            expert_ids,
            weight_scale,
            weight_bias,
            rht_sign,
            rht_block_size,
        )
    )
    if use_vq2a8_cuda_down_reduce(size_m):
        from .vq2a8_cuda import vq2a8_down_reduce_cuda

        return vq2a8_down_reduce_cuda(
            quantized,
            activation_scale,
            expert_ids,
            token_ids,
            routing_weights,
            num_tokens,
            packed_indices,
            codebooks,
            codebook_tile_ids,
            bias_correction,
        ).to(output_dtype)
    output = torch.zeros(
        (num_tokens, size_n), device=activation.device, dtype=torch.float32
    )
    block_m, block_n, block_k = 16, 32, 64
    grid = (size_m, triton.cdiv(size_n, block_n))
    _vq2a8_batched_down_reduce_kernel[grid](
        quantized,
        activation_scale,
        expert_ids,
        token_ids,
        routing_weights,
        packed_indices,
        codebooks,
        codebook_tile_ids,
        bias_correction,
        output,
        size_n,
        size_k,
        quantized.stride(0),
        quantized.stride(1),
        packed_indices.stride(0),
        packed_indices.stride(1),
        packed_indices.stride(2),
        codebooks.stride(0),
        codebooks.stride(1),
        codebooks.stride(2),
        codebooks.stride(3),
        codebooks.stride(4),
        codebook_tile_ids.stride(0),
        codebook_tile_ids.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return output.to(output_dtype)


@triton.jit
def _vq2a8_grouped_gemm_kernel(
    activation_ptr,
    activation_scale_ptr,
    packed_indices_ptr,
    codebooks_ptr,
    codebook_tile_ids_ptr,
    bias_correction_ptr,
    routing_weights_ptr,
    sorted_token_ids_ptr,
    block_expert_ids_ptr,
    num_tokens_post_padded_ptr,
    output_ptr,
    num_assignments: tl.constexpr,
    size_n: tl.constexpr,
    size_k: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_pe: tl.constexpr,
    stride_pn: tl.constexpr,
    stride_pk: tl.constexpr,
    stride_cbe: tl.constexpr,
    stride_cbt: tl.constexpr,
    stride_cbr: tl.constexpr,
    stride_cbi: tl.constexpr,
    stride_cbc: tl.constexpr,
    stride_te: tl.constexpr,
    stride_tk: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MUL_ROUTING: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_programs_n = tl.cdiv(size_n, BLOCK_N)
    program_m = program_id // num_programs_n
    program_n = program_id % num_programs_n
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if program_m * BLOCK_M >= num_tokens_post_padded:
        return

    sorted_offsets = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    assignment_ids = tl.load(sorted_token_ids_ptr + sorted_offsets).to(tl.int64)
    assignment_mask = assignment_ids < num_assignments
    expert_id = tl.load(block_expert_ids_ptr + program_m).to(tl.int64)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for start_k in range(0, size_k, BLOCK_K):
        offsets_k = start_k + tl.arange(0, BLOCK_K)
        activation = tl.load(
            activation_ptr
            + assignment_ids[:, None] * stride_am
            + offsets_k[None, :] * stride_ak,
            mask=assignment_mask[:, None],
            other=0.0,
        )
        vector_rows = offsets_n // 2
        words = tl.load(
            packed_indices_ptr
            + expert_id * stride_pe
            + vector_rows[:, None] * stride_pn
            + (offsets_k[None, :] // 8) * stride_pk
        )
        codes = (words >> ((offsets_k[None, :] % 8) * 4)) & 15
        source_tiles = tl.load(
            codebook_tile_ids_ptr + expert_id * stride_te + offsets_k * stride_tk
        ).to(tl.int64)
        codebook_offsets = (
            expert_id * stride_cbe
            + source_tiles[None, :] * stride_cbt
            + (offsets_n // 32)[:, None] * stride_cbr
            + codes * stride_cbi
            + (offsets_n % 2)[:, None] * stride_cbc
        )
        weights = tl.load(codebooks_ptr + codebook_offsets)
        accumulator += tl.dot(activation, tl.trans(weights))

    activation_scale = tl.load(
        activation_scale_ptr + assignment_ids,
        mask=assignment_mask,
        other=0.0,
    )
    bias_correction = tl.load(
        bias_correction_ptr + assignment_ids,
        mask=assignment_mask,
        other=0.0,
    )
    output = accumulator * activation_scale[:, None]
    output += bias_correction[:, None]
    if MUL_ROUTING:
        # Preserve the original BF16 down output before FP32 route weighting.
        output = output.to(tl.bfloat16).to(tl.float32)
        routing_weights = tl.load(
            routing_weights_ptr + assignment_ids,
            mask=assignment_mask,
            other=0.0,
        )
        output *= routing_weights[:, None]
    tl.store(
        output_ptr
        + assignment_ids[:, None] * stride_om
        + offsets_n[None, :] * stride_on,
        output,
        mask=assignment_mask[:, None],
    )


def vq2a8_grouped_gemm(
    activation: torch.Tensor,
    expert_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    block_expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    packed_indices: torch.Tensor,
    codebooks: torch.Tensor,
    codebook_tile_ids: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
    rht_sign: torch.Tensor,
    routing_weights: torch.Tensor | None = None,
    rht_block_size: int = 128,
    output_dtype: torch.dtype = torch.bfloat16,
    block_m: int = 16,
) -> torch.Tensor:
    """Run one real-M tile per aligned expert block for large routed batches."""
    size_m, size_k = activation.shape
    size_n = packed_indices.shape[1] * 2
    if block_m != 16:
        raise ValueError(f"VQ2A8 grouped GEMM requires block_m=16, got {block_m}.")
    if size_n % 32:
        raise ValueError(f"Output size must be tile-aligned, got {size_n}.")
    if expert_ids.numel() != size_m:
        raise ValueError("Expert count must match grouped GEMM assignments.")
    if routing_weights is not None and routing_weights.numel() != size_m:
        raise ValueError("Routing weight count must match assignments.")
    quantized, activation_scale, bias_correction = (
        prepare_vq2a8_batched_activation_dispatch(
            activation,
            expert_ids,
            weight_scale,
            weight_bias,
            rht_sign,
            rht_block_size,
        )
    )
    output = torch.empty((size_m, size_n), device=activation.device, dtype=output_dtype)
    block_n, block_k = 32, 64
    grid = (
        triton.cdiv(sorted_token_ids.shape[0], block_m) * triton.cdiv(size_n, block_n),
    )
    _vq2a8_grouped_gemm_kernel[grid](
        quantized,
        activation_scale,
        packed_indices,
        codebooks,
        codebook_tile_ids,
        bias_correction,
        routing_weights if routing_weights is not None else activation_scale,
        sorted_token_ids,
        block_expert_ids,
        num_tokens_post_padded,
        output,
        size_m,
        size_n,
        size_k,
        quantized.stride(0),
        quantized.stride(1),
        packed_indices.stride(0),
        packed_indices.stride(1),
        packed_indices.stride(2),
        codebooks.stride(0),
        codebooks.stride(1),
        codebooks.stride(2),
        codebooks.stride(3),
        codebooks.stride(4),
        codebook_tile_ids.stride(0),
        codebook_tile_ids.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        MUL_ROUTING=routing_weights is not None,
        num_warps=4,
    )
    return output


def vq2a8_moe(
    layer: torch.nn.Module,
    activation: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Top-k MoE runner with fused packed gate/up and routed down kernels."""
    if activation.ndim != 2 or topk_ids.ndim != 2:
        raise ValueError("VQ2A8 MoE expects 2D activation and routing tensors.")
    if topk_ids.shape != topk_weights.shape:
        raise ValueError("topk_ids and topk_weights must have identical shapes.")
    if activation.shape[0] != topk_ids.shape[0]:
        raise ValueError("Routing token count does not match activation rows.")
    if layer.moe_config.moe_parallel_config.use_ep:
        raise NotImplementedError("VQ2A8 direct backend currently requires EP=1.")

    top_k = topk_ids.shape[1]
    token_ids = torch.arange(
        activation.shape[0], device=activation.device
    ).repeat_interleave(top_k)
    expert_ids = topk_ids.reshape(-1).to(torch.int64)
    expert_input = activation[token_ids]
    use_grouped = expert_ids.numel() >= GROUPED_GEMM_MIN_ASSIGNMENTS
    if use_grouped:
        block_m = 16
        sorted_token_ids, block_expert_ids, num_tokens_post_padded = (
            moe_align_block_size(
                topk_ids,
                block_m,
                layer.vq_gate_up_packed_indices.shape[0],
            )
        )
        gate_up = vq2a8_grouped_gemm(
            expert_input,
            expert_ids,
            sorted_token_ids,
            block_expert_ids,
            num_tokens_post_padded,
            layer.vq_gate_up_packed_indices,
            layer.vq_gate_up_codebooks,
            layer.vq_gate_up_codebook_tile_ids,
            layer.vq_gate_up_weight_scale,
            layer.vq_gate_up_weight_bias,
            layer.vq_gate_up_rht_sign,
            output_dtype=activation.dtype,
            block_m=block_m,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        limit = layer.swiglu_limit
        if limit is not None and limit > 0:
            gate = torch.clamp(gate, max=limit)
            up = torch.clamp(up, min=-limit, max=limit)
        intermediate = torch.nn.functional.silu(gate) * up
    elif (
        expert_ids.numel() <= FUSED_GATE_UP_MAX_ASSIGNMENTS
        and not use_vq2a8_cuda_preparation()
    ):
        intermediate = vq2a8_batched_gate_up(
            expert_input,
            expert_ids,
            layer.vq_gate_up_packed_indices,
            layer.vq_gate_up_codebooks,
            layer.vq_gate_up_codebook_tile_ids,
            layer.vq_gate_up_weight_scale,
            layer.vq_gate_up_weight_bias,
            layer.vq_gate_up_rht_sign,
            layer.swiglu_limit,
            output_dtype=activation.dtype,
        )
    else:
        # Two accumulators increase register pressure for larger assignment
        # batches, where the original GEMM plus PyTorch epilogue is faster.
        gate_up = vq2a8_batched_gemm(
            expert_input,
            expert_ids,
            layer.vq_gate_up_packed_indices,
            layer.vq_gate_up_codebooks,
            layer.vq_gate_up_codebook_tile_ids,
            layer.vq_gate_up_weight_scale,
            layer.vq_gate_up_weight_bias,
            layer.vq_gate_up_rht_sign,
            output_dtype=activation.dtype,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        limit = layer.swiglu_limit
        if limit is not None and limit > 0:
            gate = torch.clamp(gate, max=limit)
            up = torch.clamp(up, min=-limit, max=limit)
        intermediate = torch.nn.functional.silu(gate) * up
    routing_weights = topk_weights.reshape(-1).float()
    if use_grouped:
        routed_output = vq2a8_grouped_gemm(
            intermediate,
            expert_ids,
            sorted_token_ids,
            block_expert_ids,
            num_tokens_post_padded,
            layer.vq_down_packed_indices,
            layer.vq_down_codebooks,
            layer.vq_down_codebook_tile_ids,
            layer.vq_down_weight_scale,
            layer.vq_down_weight_bias,
            layer.vq_down_rht_sign,
            routing_weights=routing_weights,
            output_dtype=torch.float32,
            block_m=block_m,
        )
        output = (
            routed_output.view(activation.shape[0], top_k, -1)
            .sum(dim=1)
            .to(activation.dtype)
        )
    else:
        output = vq2a8_batched_down_reduce(
            intermediate,
            expert_ids,
            token_ids,
            routing_weights,
            activation.shape[0],
            layer.vq_down_packed_indices,
            layer.vq_down_codebooks,
            layer.vq_down_codebook_tile_ids,
            layer.vq_down_weight_scale,
            layer.vq_down_weight_bias,
            layer.vq_down_rht_sign,
            output_dtype=activation.dtype,
        )
    if output.shape[1] != layer.moe_config.hidden_dim_unpadded:
        raise ValueError(
            f"VQ2A8 down output width is {output.shape[1]}, expected "
            f"{layer.moe_config.hidden_dim_unpadded}."
        )
    return output
