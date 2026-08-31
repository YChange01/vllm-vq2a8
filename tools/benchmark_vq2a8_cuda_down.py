# The benchmark invokes each closure before advancing its enclosing loop.
# ruff: noqa: B023

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from safetensors import safe_open

from vllm.model_executor.layers.fused_moe.experts.vq2a8_triton import (
    vq2a8_batched_down_reduce,
)

DEFAULT_LIBRARY = Path(__file__).resolve().parents[1] / "vllm" / "_vq2a8_cuda_ext.so"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark fused CUDA VQ2A8 down projection."
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=Path(os.environ.get("VLLM_VQ2_CUDA_LIBRARY", DEFAULT_LIBRARY)),
        help="Path to the built VQ2A8 CUDA extension.",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=os.environ.get("VLLM_VQ2_BENCHMARK_ARTIFACT"),
        help=(
            "Path to one TP-local experts_vq_layer_*.safetensors file. "
            "May also be set with VLLM_VQ2_BENCHMARK_ARTIFACT."
        ),
    )
    args = parser.parse_args()
    if args.artifact is None:
        parser.error("--artifact or VLLM_VQ2_BENCHMARK_ARTIFACT is required")
    return args


def elapsed_ms(function, warmup: int = 10, iterations: int = 100) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    args = parse_args()
    torch.ops.load_library(str(args.library))
    fields = (
        "packed_indices",
        "codebooks",
        "codebook_tile_ids",
        "weight_scale",
        "weight_bias",
        "rht_sign",
    )
    with safe_open(args.artifact, framework="pt", device="cpu") as handle:
        artifact = {
            field: handle.get_tensor(f"down_{field}").cuda() for field in fields
        }

    torch.manual_seed(17)
    for assignments in (6, 12, 16, 30, 96):
        activation = torch.randn(assignments, 512, device="cuda", dtype=torch.bfloat16)
        expert_ids = torch.arange(assignments, device="cuda") % 256
        token_ids = torch.arange(assignments, device="cuda") // 6
        num_tokens = int(token_ids[-1].item()) + 1
        routing_weights = torch.rand(assignments, device="cuda")

        def triton_full():
            return vq2a8_batched_down_reduce(
                activation,
                expert_ids,
                token_ids,
                routing_weights,
                num_tokens,
                **artifact,
            )

        def cuda_full():
            quantized, scale, bias = torch.ops.vq2a8_cuda.prepare_activation.default(
                activation,
                expert_ids,
                artifact["weight_scale"],
                artifact["weight_bias"],
                artifact["rht_sign"],
                128,
            )
            return torch.ops.vq2a8_cuda.down_reduce.default(
                quantized,
                scale,
                expert_ids,
                token_ids,
                routing_weights,
                num_tokens,
                artifact["packed_indices"],
                artifact["codebooks"],
                artifact["codebook_tile_ids"],
                bias,
            )

        expected = triton_full()
        actual = cuda_full()
        torch.cuda.synchronize()
        difference = (expected.float() - actual.float()).abs()
        triton_ms = elapsed_ms(triton_full)
        cuda_ms = elapsed_ms(cuda_full)
        print(
            f"assignments={assignments} num_tokens={num_tokens} "
            f"triton_ms={triton_ms:.6f} cuda_ms={cuda_ms:.6f} "
            f"speedup={triton_ms / cuda_ms:.3f} "
            f"max_abs={difference.max().item():.6f} "
            f"mean_abs={difference.mean().item():.6f}"
        )


if __name__ == "__main__":
    main()
