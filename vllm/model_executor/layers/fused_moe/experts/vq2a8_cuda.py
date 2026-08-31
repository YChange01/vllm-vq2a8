# SPDX-License-Identifier: Apache-2.0
"""Loader and Python wrapper for the optional VQ2A8 CUDA extension."""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

VQ2_CUDA_LIBRARY_ENV = "VLLM_VQ2_CUDA_LIBRARY"


def _default_library_path() -> Path:
    return Path(__file__).resolve().parents[4] / "_vq2a8_cuda_ext.so"


@functools.cache
def cuda_extension_path() -> Path:
    override = os.getenv(VQ2_CUDA_LIBRARY_ENV)
    return Path(override) if override else _default_library_path()


@functools.cache
def load_vq2a8_cuda_extension() -> bool:
    path = cuda_extension_path()
    if not path.is_file():
        return False
    torch.ops.load_library(str(path))
    return hasattr(torch.ops.vq2a8_cuda, "prepare_activation")


def has_vq2a8_cuda_down_reduce() -> bool:
    """Return whether the loaded extension exposes fused down reduction."""
    return load_vq2a8_cuda_extension() and hasattr(torch.ops.vq2a8_cuda, "down_reduce")


def prepare_vq2a8_activation_cuda(
    activation: torch.Tensor,
    expert_ids: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
    rht_sign: torch.Tensor,
    rht_block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not load_vq2a8_cuda_extension():
        raise RuntimeError(
            "VQ2A8 CUDA extension is unavailable. Build it with "
            "tools/build_vq2a8_cuda.py or set VLLM_VQ2_CUDA_LIBRARY."
        )
    return torch.ops.vq2a8_cuda.prepare_activation.default(
        activation,
        expert_ids,
        weight_scale,
        weight_bias,
        rht_sign,
        rht_block_size,
    )


def vq2a8_down_reduce_cuda(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    token_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    num_tokens: int,
    packed_indices: torch.Tensor,
    codebooks: torch.Tensor,
    codebook_tile_ids: torch.Tensor,
    bias_correction: torch.Tensor,
) -> torch.Tensor:
    """Run the fused packed VQ2 GEMM, route weighting, and token reduction."""
    if not has_vq2a8_cuda_down_reduce():
        raise RuntimeError(
            "The VQ2A8 CUDA extension has no fused down-reduce operator. "
            "Rebuild it with tools/build_vq2a8_cuda.py."
        )
    return torch.ops.vq2a8_cuda.down_reduce.default(
        activation,
        activation_scale,
        expert_ids,
        token_ids,
        routing_weights,
        num_tokens,
        packed_indices,
        codebooks,
        codebook_tile_ids,
        bias_correction,
    )
