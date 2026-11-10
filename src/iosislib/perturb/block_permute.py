from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl

from iosislib.core.tsfn import (
    BatchTSFN,
    ColumnSignature,
    FrameSignature,
    TSFNConfig,
    TimeAxis,
)

_MASK64 = (1 << 64) - 1


def _validate_column_name(value: str, *, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")


def _splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & _MASK64
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    value ^= value >> 31
    return state, value


def _permute_order(length: int, block_length: int, seed: int) -> list[int]:
    blocks = [
        list(range(start, min(start + block_length, length)))
        for start in range(0, length, block_length)
    ]
    state = seed & _MASK64
    for position in range(len(blocks) - 1, 0, -1):
        state, draw = _splitmix64(state)
        swap_with = draw % (position + 1)
        blocks[position], blocks[swap_with] = blocks[swap_with], blocks[position]
    return [index for block in blocks for index in block]


@dataclass(frozen=True)
class BlockPermuteConfig(TSFNConfig):
    block_length: int = 1
    seed: int = 0
    timestamp_column: str = "timestamp"

    def __post_init__(self) -> None:
        if isinstance(self.block_length, bool) or not isinstance(
            self.block_length, int
        ):
            raise TypeError("block_length must be an integer")
        if self.block_length < 1:
            raise ValueError("block_length must be at least 1")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        _validate_column_name(self.timestamp_column, field_name="timestamp_column")


class BlockPermute(BatchTSFN[BlockPermuteConfig]):
    VERSION = "0.1.0"
    CONFIG_CLS = BlockPermuteConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        time_axis = TimeAxis(column=self.parameters.timestamp_column)
        return (
            FrameSignature(time=time_axis, columns=()),
            FrameSignature(time=time_axis, columns=()),
        )

    def resolve_signature(
        self,
        bound_input_columns: Mapping[str, ColumnSignature],
    ) -> tuple[FrameSignature, FrameSignature]:
        time_axis = TimeAxis(column=self.parameters.timestamp_column)
        columns = tuple(
            sorted(
                (
                    ColumnSignature(slot, column.dtype, column.shape)
                    for slot, column in bound_input_columns.items()
                ),
                key=lambda column: column.name,
            )
        )
        signature = FrameSignature(time=time_axis, columns=columns)
        return signature, signature

    def trailing_rows(self) -> int | None:
        return None

    def batch(self, frame: pl.DataFrame) -> pl.DataFrame:
        params = self.parameters
        time_column = params.timestamp_column
        payload = [name for name in frame.columns if name != time_column]
        if frame.height < 2 or not payload:
            return frame
        order = _permute_order(frame.height, params.block_length, params.seed)
        indices = pl.Series("__iosis_permute_index", order, dtype=pl.UInt32)
        return frame.with_columns(
            [pl.col(name).gather(indices).alias(name) for name in payload]
        )
