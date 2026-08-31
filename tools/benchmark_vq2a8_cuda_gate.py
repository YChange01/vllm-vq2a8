# The benchmark invokes each closure before advancing its enclosing loop.
# ruff: noqa: B023

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from safetensors import safe_open

from vllm.model_executor.layers.fused_moe.experts.vq2a8_triton import (
    vq2a8_batched_gate_up,
    vq2a8_batched_gemm,
)

DEFAULT_LIBRARY = Path(__file__).resolve().parents[1] / "vllm" / "_vq2a8_cuda_ext.so"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark fused CUDA VQ2A8 gate/up projection."
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
            field: handle.get_tensor(f"gate_up_{field}").cuda() for field in fields
        }

    torch.manual_seed(19)
    for assignments in (6, 12, 16, 30, 96):
        activation = torch.randn(assignments, 4096, device="cuda", dtype=torch.bfloat16)
        expert_ids = torch.arange(assignments, device="cuda") % 256

        def triton_generic():
            gate_up = vq2a8_batched_gemm(activation, expert_ids, **artifact)
            gate, up = gate_up.chunk(2, dim=-1)
            gate = torch.clamp(gate, max=10.0)
            up = torch.clamp(up, min=-10.0, max=10.0)
            return torch.nn.functional.silu(gate) * up

        def triton_fused():
            return vq2a8_batched_gate_up(
                activation, expert_ids, swiglu_limit=10.0, **artifact
            )

        def cuda_fused():
            quantized, scale, bias = torch.ops.vq2a8_cuda.prepare_activation.default(
                activation,
                expert_ids,
                artifact["weight_scale"],
                artifact["weight_bias"],
                artifact["rht_sign"],
                128,
            )
            return torch.ops.vq2a8_cuda.gate_up.default(
                quantized,
                scale,
                expert_ids,
                artifact["packed_indices"],
                artifact["codebooks"],
                artifact["codebook_tile_ids"],
                bias,
                10.0,
            )

        expected = triton_generic()
        triton_actual = triton_fused()
        cuda_actual = cuda_fused()
        torch.cuda.synchronize()
        triton_difference = (expected.float() - triton_actual.float()).abs()
        cuda_difference = (expected.float() - cuda_actual.float()).abs()
        generic_ms = elapsed_ms(triton_generic)
        triton_ms = elapsed_ms(triton_fused)
        cuda_ms = elapsed_ms(cuda_fused)
        print(
            f"assignments={assignments} generic_ms={generic_ms:.6f} "
            f"triton_fused_ms={triton_ms:.6f} cuda_ms={cuda_ms:.6f} "
            f"cuda_speedup={generic_ms / cuda_ms:.3f} "
            f"triton_max_abs={triton_difference.max().item():.6f} "
            f"cuda_max_abs={cuda_difference.max().item():.6f} "
            f"cuda_mean_abs={cuda_difference.mean().item():.6f}"
        )


if __name__ == "__main__":
    main()
