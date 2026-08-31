# SPDX-License-Identifier: Apache-2.0
"""Repack VPTQ VQ2 expert artifacts into a TP-local direct-kernel layout.

The source artifact is read-only. The output resolves the producer permutation
offline and keeps 4-bit vector codes packed eight per int32 word. Since every
code selects two output weights, the effective storage is two bits per weight.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT = "vq2a8_tp_v1"


def unpack_indices(packed: torch.Tensor, count: int) -> torch.Tensor:
    words = packed.reshape(-1).to(torch.int64) & 0xFFFFFFFF
    positions = torch.arange(count, dtype=torch.int64)
    return (words[positions // 8] >> ((positions % 8) * 4)) & 15


def pack_kernel_indices(indices: torch.Tensor) -> torch.Tensor:
    """Pack [output-pair, K] uint4 codes along K for coalesced GEMM loads."""
    if indices.ndim != 2:
        raise ValueError(f"Expected a 2D index matrix, got {indices.shape}.")
    rows, columns = indices.shape
    padded_columns = (columns + 7) // 8 * 8
    if padded_columns != columns:
        indices = torch.nn.functional.pad(indices, (0, padded_columns - columns))
    values = indices.reshape(rows, -1, 8).to(torch.int64)
    shifts = torch.arange(8, dtype=torch.int64).reshape(1, 1, 8) * 4
    return torch.sum(values << shifts, dim=-1).to(torch.int32)


def unpack_kernel_indices(packed: torch.Tensor, columns: int) -> torch.Tensor:
    positions = torch.arange(columns, dtype=torch.int64)
    words = packed.to(torch.int64) & 0xFFFFFFFF
    return (words[:, positions // 8] >> ((positions % 8) * 4)) & 15


def serialized_index_grid(
    packed: torch.Tensor, metadata: dict[str, Any]
) -> torch.Tensor:
    """Return producer codes as [output-vector-pair, input-column]."""
    row_tiles = int(metadata["n_row_tiles"])
    columns = int(metadata["cols"])
    group_size = int(metadata["group_size"])
    vectors_per_tile = int(metadata["row_group_size"]) // int(metadata["vector_len"])
    flat = unpack_indices(packed, int(metadata["n_vectors"]))
    pieces = []
    position = 0
    for column_tile in range(int(metadata["n_col_tiles"])):
        start = column_tile * group_size
        width = min(start + group_size, columns) - start
        count = row_tiles * width * vectors_per_tile
        piece = flat[position : position + count].reshape(
            row_tiles, width, vectors_per_tile
        )
        pieces.append(piece)
        position += count
    if position != flat.numel():
        raise ValueError(
            f"Consumed {position} indices but artifact contains {flat.numel()}."
        )
    grid = torch.cat(pieces, dim=1)
    return grid.permute(0, 2, 1).reshape(-1, columns).contiguous()


def _selected_gate_up_row_tiles(
    metadata: dict[str, Any], tp_size: int, tp_rank: int
) -> torch.Tensor:
    rows = int(metadata["rows"])
    row_group_size = int(metadata["row_group_size"])
    if rows % (2 * tp_size * row_group_size):
        raise ValueError("gate_up output rows do not align to TP row tiles.")
    half_tiles = rows // 2 // row_group_size
    local_tiles = half_tiles // tp_size
    gate_start = tp_rank * local_tiles
    up_start = half_tiles + gate_start
    return torch.cat(
        [
            torch.arange(gate_start, gate_start + local_tiles),
            torch.arange(up_start, up_start + local_tiles),
        ]
    )


def _selected_output_pairs(
    row_tiles: torch.Tensor, metadata: dict[str, Any]
) -> torch.Tensor:
    pairs_per_tile = int(metadata["row_group_size"]) // int(metadata["vector_len"])
    offsets = torch.arange(pairs_per_tile)
    return (row_tiles[:, None] * pairs_per_tile + offsets).reshape(-1)


def repack_weight(
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    kind: str,
    tp_size: int,
    tp_rank: int,
    verify: bool,
) -> dict[str, torch.Tensor]:
    if (
        int(metadata["K"]) != 16
        or int(metadata["index_bits"]) != 4
        or int(metadata["vector_len"]) != 2
        or int(metadata["norm_dim"]) != 0
    ):
        raise ValueError(f"Unsupported VQ geometry for {kind}: {metadata}.")
    if not (
        metadata["enable_perm"] and metadata["enable_norm"] and metadata["enable_rht"]
    ):
        raise ValueError(f"{FORMAT} requires perm, norm, and RHT for {kind}.")

    columns = int(metadata["cols"])
    row_tiles = int(metadata["n_row_tiles"])
    group_size = int(metadata["group_size"])
    source_columns = torch.argsort(tensors["perm"].to(torch.int64))
    physical_columns = torch.arange(columns)
    if kind == "gate_up":
        selected_row_tiles = _selected_gate_up_row_tiles(metadata, tp_size, tp_rank)
    elif kind == "down":
        if columns % tp_size:
            raise ValueError("down input columns are not divisible by TP size.")
        width = columns // tp_size
        start = tp_rank * width
        physical_columns = physical_columns[start : start + width]
        source_columns = source_columns[start : start + width]
        selected_row_tiles = torch.arange(row_tiles)
    else:
        raise ValueError(f"Unknown expert weight kind {kind!r}.")

    grid = serialized_index_grid(tensors["packed_indices"], metadata)
    output_pairs = _selected_output_pairs(selected_row_tiles, metadata)
    selected_indices = grid[output_pairs][:, source_columns]
    packed_indices = pack_kernel_indices(selected_indices)
    codebooks = tensors["codebooks"][:, selected_row_tiles].contiguous()
    tile_ids = torch.div(source_columns, group_size, rounding_mode="floor").to(
        torch.uint8
    )
    scale = tensors["weight_scale"][physical_columns].contiguous()
    bias = tensors["weight_bias"][physical_columns].contiguous()
    sign = tensors["rht_sign"][physical_columns].contiguous()

    if verify:
        restored = unpack_kernel_indices(packed_indices, selected_indices.shape[1])
        torch.testing.assert_close(restored, selected_indices, rtol=0, atol=0)
        expected_tiles = torch.div(
            source_columns, group_size, rounding_mode="floor"
        ).to(torch.uint8)
        torch.testing.assert_close(tile_ids, expected_tiles, rtol=0, atol=0)
        torch.testing.assert_close(
            codebooks, tensors["codebooks"][:, selected_row_tiles], rtol=0, atol=0
        )

    return {
        "packed_indices": packed_indices,
        "codebooks": codebooks,
        "codebook_tile_ids": tile_ids,
        "weight_scale": scale,
        "weight_bias": bias,
        "rht_sign": sign,
    }


def _payload(handle: Any, name: str) -> dict[str, torch.Tensor]:
    return {
        field: handle.get_tensor(f"{name}.{field}")
        for field in (
            "packed_indices",
            "codebooks",
            "perm",
            "weight_scale",
            "weight_bias",
            "rht_sign",
        )
    }


def _parse_experts(entries: dict[str, Any]) -> tuple[list[int], dict[int, str]]:
    names: dict[int, str] = {}
    for name in entries:
        expert_id = int(name.split(".")[3])
        if name.endswith(".gate_up"):
            names[expert_id] = name.rsplit(".", 1)[0]
    return sorted(names), names


def _stack_rank_payloads(
    payloads: list[dict[str, torch.Tensor]], kind: str
) -> dict[str, torch.Tensor]:
    fields = payloads[0].keys()
    return {
        f"{kind}_{field}": torch.stack([item[field] for item in payloads])
        for field in fields
    }


def repack_layer(
    source_dir: Path,
    output_dir: Path,
    layer: int,
    tp_size: int,
    num_experts: int,
    verify: bool,
) -> None:
    stem = f"experts_vq_layer_{layer}"
    source_tensor_path = source_dir / f"{stem}.safetensors"
    source_metadata_path = source_dir / f"{stem}.json"
    with source_metadata_path.open(encoding="utf-8") as metadata_file:
        entries = json.load(metadata_file)
    expert_ids, bases = _parse_experts(entries)
    if not expert_ids:
        raise ValueError(f"Layer {layer} contains no expert payloads.")

    rank_payloads: list[dict[str, list[dict[str, torch.Tensor]]]] = [
        {"gate_up": [], "down": []} for _ in range(tp_size)
    ]
    with safe_open(source_tensor_path, framework="pt", device="cpu") as handle:
        for expert_id in expert_ids:
            base = bases[expert_id]
            for kind in ("gate_up", "down"):
                name = f"{base}.{kind}"
                metadata = entries[name]
                tensors = _payload(handle, name)
                for rank in range(tp_size):
                    rank_payloads[rank][kind].append(
                        repack_weight(
                            tensors,
                            metadata,
                            kind,
                            tp_size,
                            rank,
                            verify,
                        )
                    )

    representative = entries[f"{bases[expert_ids[0]]}.gate_up"]
    layer_metadata = {
        "format": FORMAT,
        "layer": layer,
        "tp_size": tp_size,
        "expert_ids": expert_ids,
        "num_expected_experts": num_experts,
        "complete": len(expert_ids) == num_experts,
        "index_bits": 4,
        "vector_len": 2,
        "effective_bits_per_weight": 2,
        "row_group_size": int(representative["row_group_size"]),
        "group_size": int(representative["group_size"]),
        "rht_block_size": int(representative["rht_block_size"]),
    }
    for rank, by_kind in enumerate(rank_payloads):
        tensors: dict[str, torch.Tensor] = {}
        for kind in ("gate_up", "down"):
            tensors.update(_stack_rank_payloads(by_kind[kind], kind))
        rank_dir = output_dir / f"tp{tp_size}" / f"rank{rank}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = rank_dir / f"{stem}.safetensors"
        metadata_path = rank_dir / f"{stem}.json"
        temp_tensor_path = tensor_path.with_suffix(".safetensors.tmp")
        temp_metadata_path = metadata_path.with_suffix(".json.tmp")
        save_file(tensors, temp_tensor_path)
        with temp_metadata_path.open("w", encoding="utf-8") as output_file:
            json.dump(layer_metadata | {"tp_rank": rank}, output_file, indent=2)
            output_file.write("\n")
        os.replace(temp_tensor_path, tensor_path)
        os.replace(temp_metadata_path, metadata_path)
        size_gib = tensor_path.stat().st_size / 1024**3
        print(
            f"layer={layer} rank={rank} experts={len(expert_ids)} "
            f"complete={layer_metadata['complete']} size={size_gib:.3f} GiB",
            flush=True,
        )


def parse_layers(specification: str, available: Iterable[int]) -> list[int]:
    if specification == "all":
        return sorted(available)
    layers: set[int] = set()
    for part in specification.split(","):
        match = re.fullmatch(r"(\d+)-(\d+)", part)
        if match:
            start, end = map(int, match.groups())
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    return sorted(layers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    available = [
        int(match.group(1))
        for path in args.input.glob("experts_vq_layer_*.json")
        if (match := re.search(r"layer_(\d+)\.json$", path.name))
    ]
    for layer in parse_layers(args.layers, available):
        repack_layer(
            args.input,
            args.output,
            layer,
            args.tp_size,
            args.num_experts,
            args.verify,
        )


if __name__ == "__main__":
    main()
