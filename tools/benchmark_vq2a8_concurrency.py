from __future__ import annotations

import argparse
import json
import os
import statistics
import time

os.environ.setdefault("VLLM_ALLREDUCE_USE_FLASHINFER", "0")
os.environ.setdefault("VLLM_ALLREDUCE_USE_SYMM_MEM", "0")
os.environ.setdefault("VLLM_VQ2_BACKEND", "direct")
os.environ.setdefault("VLLM_VQ2_PREP_BACKEND", "auto")
os.environ.setdefault("VLLM_VQ2_GEMM_BACKEND", "auto")

from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationMode, CUDAGraphMode


def csv_ints(value: str) -> list[int]:
    values = sorted({int(item) for item in value.split(",")})
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark VQ2A8 CUDA Graph decode throughput by concurrency."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument(
        "--concurrency",
        type=csv_ints,
        default=csv_ints("1,2,4,8,16,32"),
    )
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--measure-tokens", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture_sizes = args.concurrency
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=max(capture_sizes),
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
        kv_cache_dtype="fp8",
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        compilation_config={
            "mode": CompilationMode.NONE,
            "cudagraph_mode": CUDAGraphMode.FULL_DECODE_ONLY,
            "cudagraph_capture_sizes": capture_sizes,
            "max_cudagraph_capture_size": max(capture_sizes),
            "pass_config": {"fuse_allreduce_rms": False},
        },
        disable_log_stats=True,
    )

    summaries: list[dict[str, float | int]] = []
    for concurrency in args.concurrency:
        prompts = [args.prompt] * concurrency
        for _ in range(args.warmup_iterations):
            llm.generate(
                prompts,
                SamplingParams(
                    temperature=0.0,
                    max_tokens=args.warmup_tokens,
                    ignore_eos=True,
                    seed=1234,
                ),
            )

        elapsed_values: list[float] = []
        rates: list[float] = []
        for iteration in range(args.iterations):
            start = time.perf_counter()
            result = llm.generate(
                prompts,
                SamplingParams(
                    temperature=0.0,
                    max_tokens=args.measure_tokens,
                    ignore_eos=True,
                    seed=1234,
                ),
            )
            elapsed = time.perf_counter() - start
            generated = sum(len(item.outputs[0].token_ids) for item in result)
            rate = generated / elapsed
            elapsed_values.append(elapsed)
            rates.append(rate)
            print(
                "VQ2A8_CONCURRENCY_ITER",
                json.dumps(
                    {
                        "concurrency": concurrency,
                        "iteration": iteration,
                        "tokens": generated,
                        "elapsed_seconds": elapsed,
                        "total_tokens_per_second": rate,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        summary: dict[str, float | int] = {
            "concurrency": concurrency,
            "iterations": args.iterations,
            "tokens_per_request": args.measure_tokens,
            "mean_elapsed_seconds": statistics.mean(elapsed_values),
            "median_elapsed_seconds": statistics.median(elapsed_values),
            "mean_total_tokens_per_second": statistics.mean(rates),
            "median_total_tokens_per_second": statistics.median(rates),
            "min_total_tokens_per_second": min(rates),
            "max_total_tokens_per_second": max(rates),
        }
        summaries.append(summary)
        print(
            "VQ2A8_CONCURRENCY_SUMMARY",
            json.dumps(summary, sort_keys=True),
            flush=True,
        )

    print(
        "VQ2A8_CONCURRENCY_ALL",
        json.dumps(summaries, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
