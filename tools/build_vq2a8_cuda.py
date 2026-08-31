from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument(
        "--cuda-home",
        type=Path,
        help=(
            "CUDA toolkit root. When omitted, use CUDA_HOME or PyTorch's "
            "automatic toolkit discovery."
        ),
    )
    args = parser.parse_args()

    if args.cuda_home is not None:
        os.environ["CUDA_HOME"] = str(args.cuda_home.resolve())
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    os.environ.setdefault("MAX_JOBS", "8")
    from torch.utils.cpp_extension import load

    args.build_dir.mkdir(parents=True, exist_ok=True)
    cutlass_include = (
        args.source.resolve().parents[2]
        / "vllm"
        / "third_party"
        / "deep_gemm"
        / "include"
    )
    library = Path(
        load(
            name="vq2a8_cuda_ext",
            sources=[str(args.source.resolve())],
            extra_include_paths=[str(cutlass_include)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "--threads=4",
                "-lineinfo",
            ],
            build_directory=str(args.build_dir.resolve()),
            verbose=True,
            is_python_module=False,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    shutil.copy2(library, temporary)
    temporary.replace(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
