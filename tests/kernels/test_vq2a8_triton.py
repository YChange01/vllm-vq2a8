# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import types

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts.vq2a8_triton import (
    _normalized_fwht,
    prepare_vq2a8_batched_activation_dispatch,
    prepare_vq2a8_batched_activation_triton,
    vq2a8_batched_down_reduce,
    vq2a8_batched_gate_up,
    vq2a8_batched_gemm,
    vq2a8_gemm,
    vq2a8_grouped_gemm,
    vq2a8_moe,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="VQ2A8 requires an SM90+ CUDA device",
)


def _pack_kernel_indices(indices: torch.Tensor) -> torch.Tensor:
    values = indices.reshape(*indices.shape[:-1], -1, 8).to(torch.int64)
    shifts = torch.arange(8, dtype=torch.int64).reshape(1, 1, 1, 8) * 4
    return torch.sum(values << shifts, dim=-1).to(torch.int32)


def _artifact(
    num_experts: int, output_size: int, input_size: int
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(output_size * 1000 + input_size)
    indices = torch.randint(
        0,
        16,
        (num_experts, output_size // 2, input_size),
        generator=generator,
    )
    codebooks = torch.randn(
        num_experts,
        1,
        output_size // 32,
        16,
        2,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    signs = (
        torch.randint(0, 2, (num_experts, input_size), generator=generator)
        .mul(2)
        .sub(1)
    )
    return {
        "packed_indices": _pack_kernel_indices(indices).cuda(),
        "codebooks": codebooks.cuda(),
        "codebook_tile_ids": torch.zeros(
            num_experts, input_size, dtype=torch.uint8, device="cuda"
        ),
        "weight_scale": (
            torch.rand(num_experts, input_size, generator=generator) + 0.5
        ).cuda(),
        "weight_bias": (
            torch.randn(num_experts, input_size, generator=generator) * 0.05
        ).cuda(),
        "rht_sign": signs.to(torch.int8).cuda(),
    }


def test_normalized_fwht_matches_sylvester_matrix() -> None:
    activation = torch.randn(3, 64, device="cuda")
    hadamard = torch.tensor([[1.0]], device="cuda")
    while hadamard.shape[0] < 64:
        hadamard = torch.cat(
            (
                torch.cat((hadamard, hadamard), dim=1),
                torch.cat((hadamard, -hadamard), dim=1),
            )
        )
    expected = activation @ (hadamard / 8)
    actual = _normalized_fwht(activation, 64)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_cuda_activation_preparation_matches_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe.experts.vq2a8_cuda import (
        load_vq2a8_cuda_extension,
    )

    if not load_vq2a8_cuda_extension():
        pytest.skip("VQ2A8 CUDA extension is not built")
    artifact = _artifact(2, 128, 128)
    activation = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    expert_ids = torch.tensor([0, 1, 1, 0], device="cuda")
    arguments = (
        activation,
        expert_ids,
        artifact["weight_scale"],
        artifact["weight_bias"],
        artifact["rht_sign"],
    )
    expected_q, expected_scale, expected_bias = prepare_vq2a8_batched_activation_triton(
        *arguments
    )
    monkeypatch.setenv("VLLM_VQ2_PREP_BACKEND", "cuda")
    actual_q, actual_scale, actual_bias = prepare_vq2a8_batched_activation_dispatch(
        *arguments
    )
    torch.testing.assert_close(actual_q.float(), expected_q.float(), rtol=0, atol=0)
    torch.testing.assert_close(actual_scale, expected_scale, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_bias, expected_bias, rtol=1e-4, atol=2e-3)


@pytest.mark.parametrize("output_size", [32, 128])
def test_batched_gemm_matches_single_expert_kernel(output_size: int) -> None:
    artifact = _artifact(2, output_size, 128)
    expert_ids = torch.tensor([0, 1, 1, 0], device="cuda")
    activation = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    actual = vq2a8_batched_gemm(activation, expert_ids, **artifact)
    expected = torch.cat(
        [
            vq2a8_gemm(
                activation[row : row + 1],
                **{field: tensor[expert_id] for field, tensor in artifact.items()},
            )
            for row, expert_id in enumerate(expert_ids.tolist())
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.05)


def test_fused_gate_up_matches_unfused_epilogue() -> None:
    artifact = _artifact(2, 256, 128)
    expert_ids = torch.tensor([0, 1, 1, 0], device="cuda")
    activation = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    gate_up = vq2a8_batched_gemm(activation, expert_ids, **artifact)
    gate, up = gate_up.chunk(2, dim=-1)
    gate = torch.clamp(gate, max=10.0)
    up = torch.clamp(up, min=-10.0, max=10.0)
    expected = torch.nn.functional.silu(gate) * up
    actual = vq2a8_batched_gate_up(
        activation, expert_ids, swiglu_limit=10.0, **artifact
    )
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.05)


def test_fused_down_route_reduce_matches_unfused_path() -> None:
    artifact = _artifact(2, 128, 128)
    expert_ids = torch.tensor([0, 1, 1, 0, 0, 1], device="cuda")
    token_ids = torch.tensor([0, 0, 1, 1, 2, 2], device="cuda")
    routing_weights = torch.tensor([0.6, 0.4, 0.75, 0.25, 0.55, 0.45], device="cuda")
    activation = torch.randn(6, 128, device="cuda", dtype=torch.bfloat16)
    expert_output = vq2a8_batched_gemm(activation, expert_ids, **artifact)
    expected = torch.zeros(3, 128, device="cuda", dtype=torch.float32)
    expected.index_add_(
        0, token_ids, expert_output.float() * routing_weights.unsqueeze(1)
    )
    actual = vq2a8_batched_down_reduce(
        activation,
        expert_ids,
        token_ids,
        routing_weights,
        num_tokens=3,
        **artifact,
    )
    torch.testing.assert_close(
        actual, expected.to(torch.bfloat16), rtol=0.03, atol=0.05
    )


def test_cuda_down_reduce_matches_triton(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.model_executor.layers.fused_moe.experts.vq2a8_cuda import (
        has_vq2a8_cuda_down_reduce,
    )

    if not has_vq2a8_cuda_down_reduce():
        pytest.skip("VQ2A8 CUDA down-reduce operator is not built")
    artifact = _artifact(2, 128, 128)
    expert_ids = torch.tensor([0, 1, 1, 0, 0, 1], device="cuda")
    token_ids = torch.tensor([0, 0, 1, 1, 2, 2], device="cuda")
    routing_weights = torch.tensor([0.6, 0.4, 0.75, 0.25, 0.55, 0.45], device="cuda")
    activation = torch.randn(6, 128, device="cuda", dtype=torch.bfloat16)
    arguments = (
        activation,
        expert_ids,
        token_ids,
        routing_weights,
        3,
    )
    monkeypatch.setenv("VLLM_VQ2_GEMM_BACKEND", "triton")
    expected = vq2a8_batched_down_reduce(*arguments, **artifact)
    monkeypatch.setenv("VLLM_VQ2_GEMM_BACKEND", "cuda")
    actual = vq2a8_batched_down_reduce(*arguments, **artifact)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_grouped_gemm_matches_assignment_gemm() -> None:
    artifact = _artifact(3, 128, 128)
    topk_ids = torch.tensor(
        [
            [0, 1],
            [2, 0],
            [1, 2],
            [0, 2],
            [1, 0],
            [2, 1],
            [0, 1],
            [2, 0],
        ],
        device="cuda",
    )
    expert_ids = topk_ids.reshape(-1).to(torch.int64)
    activation = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
    sorted_ids, block_expert_ids, num_post = moe_align_block_size(topk_ids, 16, 3)
    expected = vq2a8_batched_gemm(activation, expert_ids, **artifact)
    actual = vq2a8_grouped_gemm(
        activation,
        expert_ids,
        sorted_ids,
        block_expert_ids,
        num_post,
        **artifact,
    )
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.05)


@pytest.mark.parametrize("num_tokens", [3, 96])
def test_topk_vq2a8_moe_is_finite(num_tokens: int) -> None:
    gate_up = _artifact(2, 256, 128)
    down = _artifact(2, 128, 128)
    layer = types.SimpleNamespace(
        moe_config=types.SimpleNamespace(
            hidden_dim_unpadded=128,
            moe_parallel_config=types.SimpleNamespace(use_ep=False),
        ),
        swiglu_limit=10.0,
    )
    for kind, artifact in (("gate_up", gate_up), ("down", down)):
        for field, tensor in artifact.items():
            setattr(layer, f"vq_{kind}_{field}", tensor)
    activation = torch.randn(num_tokens, 128, device="cuda", dtype=torch.bfloat16)
    token_positions = torch.arange(num_tokens, device="cuda")
    topk_ids = torch.stack(
        [token_positions.remainder(2), (token_positions + 1).remainder(2)], dim=1
    )
    topk_weights = torch.tensor([0.6, 0.4], device="cuda").repeat(num_tokens, 1)
    output = vq2a8_moe(layer, activation, topk_weights, topk_ids)
    assert output.shape == activation.shape
    assert output.dtype == activation.dtype
    assert torch.isfinite(output).all()
