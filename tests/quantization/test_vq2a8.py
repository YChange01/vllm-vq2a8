# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization.vq2a8 import (
    apply_vq2_weight_transforms,
    decode_codebook_weight,
    decode_expert_weight,
    decode_repacked_vq2_weight,
    get_vq2_backend,
    shard_decoded_expert_weight,
    transform_vq2_activation,
    unpack_vq2_indices,
    vq2_matmul_reference,
)


def test_vq2_backend_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_VQ2_BACKEND", raising=False)
    assert get_vq2_backend() == "decode_fp8"
    monkeypatch.setenv("VLLM_VQ2_BACKEND", "direct")
    assert get_vq2_backend() == "direct"
    monkeypatch.setenv("VLLM_VQ2_BACKEND", "unknown")
    with pytest.raises(ValueError, match="VLLM_VQ2_BACKEND"):
        get_vq2_backend()


def _pack_vq2_indices(values: torch.Tensor) -> torch.Tensor:
    words = torch.zeros((values.numel() + 7) // 8, dtype=torch.int64)
    for index, value in enumerate(values.tolist()):
        words[index // 8] |= value << (index % 8 * 4)
    return words.to(torch.int32)


def _pack_kernel_indices(values: torch.Tensor) -> torch.Tensor:
    words = torch.zeros(values.shape[0], (values.shape[1] + 7) // 8, dtype=torch.int64)
    for column in range(values.shape[1]):
        words[:, column // 8] |= values[:, column] << (column % 8 * 4)
    return words.to(torch.int32)


def test_unpack_vq2_indices() -> None:
    expected = torch.arange(97, dtype=torch.int64).remainder(16)
    packed = _pack_vq2_indices(expected)
    actual = unpack_vq2_indices(packed, expected.numel(), "cpu")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_decode_expert_weight_without_transforms() -> None:
    rows, columns = 4, 3
    indices = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)
    codebooks = torch.empty((1, 2, 16, 2), dtype=torch.float32)
    for index in range(16):
        codebooks[0, 0, index] = torch.tensor([100 + index, 200 + index])
        codebooks[0, 1, index] = torch.tensor([300 + index, 400 + index])
    expected = torch.tensor(
        [
            [100, 101, 102],
            [200, 201, 202],
            [303, 304, 305],
            [403, 404, 405],
        ],
        dtype=torch.bfloat16,
    )
    metadata = {
        "rows": rows,
        "cols": columns,
        "n_row_tiles": 2,
        "n_col_tiles": 1,
        "row_group_size": 2,
        "group_size": columns,
        "K": 16,
        "index_bits": 4,
        "vector_len": 2,
        "n_vectors": indices.numel(),
        "n_elements": rows * columns,
        "orig_shape": [rows, columns],
        "norm_dim": 0,
        "enable_perm": False,
        "enable_norm": False,
        "enable_rht": False,
    }
    tensors = {
        "packed_indices": _pack_vq2_indices(indices),
        "codebooks": codebooks,
    }
    actual = decode_expert_weight(tensors, metadata, "cpu")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_activation_side_transforms_match_decoded_weight_matmul() -> None:
    rows, columns = 4, 4
    indices = torch.arange(8, dtype=torch.int64)
    codebooks = torch.empty((1, 2, 16, 2), dtype=torch.float32)
    for index in range(16):
        codebooks[0, 0, index] = torch.tensor([index + 0.25, index - 0.5])
        codebooks[0, 1, index] = torch.tensor([index + 1.0, index + 2.0])
    metadata = {
        "rows": rows,
        "cols": columns,
        "n_row_tiles": 2,
        "n_col_tiles": 1,
        "row_group_size": 2,
        "group_size": columns,
        "K": 16,
        "index_bits": 4,
        "vector_len": 2,
        "n_vectors": indices.numel(),
        "n_elements": rows * columns,
        "orig_shape": [rows, columns],
        "norm_dim": 0,
        "enable_perm": True,
        "enable_norm": True,
        "enable_rht": True,
        "rht_block_size": 4,
        "rht_true_columns": columns,
    }
    tensors = {
        "packed_indices": _pack_vq2_indices(indices),
        "codebooks": codebooks,
        "perm": torch.tensor([2, 0, 3, 1], dtype=torch.int32),
        "weight_scale": torch.tensor([1.25, 0.75, 1.5, 0.5]),
        "weight_bias": torch.tensor([0.5, -0.25, 0.125, 1.0]),
        "rht_sign": torch.tensor([1, -1, 1, 1], dtype=torch.int8),
    }
    activation = torch.tensor([[0.5, -1.0, 2.0, 0.25], [1.5, 0.75, -0.5, 2.0]])

    codebook_weight = decode_codebook_weight(tensors, metadata, "cpu")
    decoded_weight = apply_vq2_weight_transforms(codebook_weight, tensors, metadata)
    expected = activation @ decoded_weight.transpose(0, 1)
    transformed, bias_correction = transform_vq2_activation(
        activation, tensors, metadata
    )
    actual_from_parts = transformed @ codebook_weight.transpose(0, 1)
    assert bias_correction is not None
    actual_from_parts += bias_correction.unsqueeze(-1)
    actual = vq2_matmul_reference(activation, tensors, metadata)

    torch.testing.assert_close(actual_from_parts, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_decode_repacked_vq2_weight() -> None:
    input_size = 8
    output_size = 32
    indices = (
        torch.arange(output_size // 2 * input_size)
        .reshape(output_size // 2, input_size)
        .remainder(16)
    )
    codebooks = torch.empty((1, 1, 16, 2), dtype=torch.float32)
    for index in range(16):
        codebooks[0, 0, index] = torch.tensor([index + 0.25, index - 0.5])
    scale = torch.linspace(0.5, 1.25, input_size)
    bias = torch.linspace(-0.25, 0.5, input_size)
    actual = decode_repacked_vq2_weight(
        _pack_kernel_indices(indices),
        codebooks,
        torch.zeros(input_size, dtype=torch.uint8),
        scale,
        bias,
        torch.ones(input_size, dtype=torch.int8),
        rht_block_size=1,
    )
    expected = torch.empty((output_size, input_size))
    for row in range(output_size):
        expected[row] = codebooks[0, 0, indices[row // 2], row % 2] * scale + bias
    torch.testing.assert_close(actual, expected.to(torch.bfloat16), rtol=0, atol=0)


def test_shard_decoded_expert_weight_uses_routed_experts_layout() -> None:
    gate_up = torch.arange(8 * 3).reshape(8, 3)
    actual_gate_up = shard_decoded_expert_weight(gate_up, "0.gate_up", 2, 1)
    expected_gate_up = torch.cat([gate_up[2:4], gate_up[6:8]])
    torch.testing.assert_close(actual_gate_up, expected_gate_up)

    down = torch.arange(3 * 8).reshape(3, 8)
    actual_down = shard_decoded_expert_weight(down, "0.down", 2, 1)
    torch.testing.assert_close(actual_down, down[:, 4:8])
