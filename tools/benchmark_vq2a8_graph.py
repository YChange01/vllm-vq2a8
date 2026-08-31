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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark steady single-stream VQ2A8 CUDA Graph decode."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--warmup-tokens", type=int, default=4)
    parser.add_argument("--measure-tokens", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument(
        "--disable-custom-all-reduce",
        action="store_true",
        help="Disable vLLM custom all-reduce and use the conservative NCCL path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
        kv_cache_dtype="fp8",
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        compilation_config={
            # DeepSeek-V4 does not yet advertise Dynamo/Inductor support. Full
            # decode CUDA Graph capture is independent of torch.compile.
            "mode": CompilationMode.NONE,
            "cudagraph_mode": CUDAGraphMode.FULL_DECODE_ONLY,
            "cudagraph_capture_sizes": [1],
            "max_cudagraph_capture_size": 1,
            "pass_config": {"fuse_allreduce_rms": False},
        },
        disable_log_stats=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.measure_tokens,
        ignore_eos=True,
        seed=1234,
    )
    llm.generate(
        [args.prompt],
        SamplingParams(
            temperature=0.0,
            max_tokens=args.warmup_tokens,
            ignore_eos=True,
            seed=1234,
        ),
    )

    elapsed_values: list[float] = []
    rates: list[float] = []
    outputs: list[str] = []
    for iteration in range(args.iterations):
        start = time.perf_counter()
        result = llm.generate([args.prompt], sampling_params)
        elapsed = time.perf_counter() - start
        generated = len(result[0].outputs[0].token_ids)
        rate = generated / elapsed
        elapsed_values.append(elapsed)
        rates.append(rate)
        outputs.append(result[0].outputs[0].text)
        print(
            "VQ2A8_GRAPH_ITER",
            json.dumps(
                {
                    "iteration": iteration,
                    "tokens": generated,
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": rate,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summary = {
        "iterations": args.iterations,
        "measure_tokens": args.measure_tokens,
        "mean_elapsed_seconds": statistics.mean(elapsed_values),
        "median_elapsed_seconds": statistics.median(elapsed_values),
        "mean_tokens_per_second": statistics.mean(rates),
        "median_tokens_per_second": statistics.median(rates),
        "min_tokens_per_second": min(rates),
        "max_tokens_per_second": max(rates),
        "outputs": outputs,
    }
    print("VQ2A8_GRAPH_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
