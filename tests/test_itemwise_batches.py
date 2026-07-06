from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import log, prod
from collections.abc import Mapping

import numpy as np
import polars as pl
import pytest

from src.classes import (
    BatchTSFN,
    ColumnSignature,
    FrameSignature,
    Graph,
    Node,
    NullHandler,
    NullPolicy,
    TSFN,
    TSFNConfig,
)
from src.tsfn.transforms import Logit, Ratio, Spread


def dt(minute: int) -> datetime:
    return datetime(2026, 1, 1, 0, minute)


@dataclass(frozen=True)
class VectorSourceConfig(TSFNConfig):
    output_column: str
    values: tuple[tuple[float | None, ...], ...]
    shape: tuple[int, ...] = (2,)


class VectorSource(TSFN):
    VERSION = "1.0.0"
    CONFIG_CLS = VectorSourceConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature.empty(),
            FrameSignature(
                columns=(
                    ColumnSignature(
                        self.parameters.output_column,
                        pl.Float64,
                        self.parameters.shape,
                    ),
                )
            ),
        )

    def apply(self) -> pl.LazyFrame:
        width = prod(self.parameters.shape)
        return pl.DataFrame(
            {
                "timestamp": [dt(i) for i in range(len(self.parameters.values))],
                self.parameters.output_column: list(self.parameters.values),
            },
            schema={
                "timestamp": pl.Datetime,
                self.parameters.output_column: pl.Array(pl.Float64, width),
            },
        ).lazy()


def vector_source(
    values: tuple[tuple[float | None, ...], ...],
    *,
    output_column: str = "value",
    shape: tuple[int, ...] = (2,),
) -> Node:
    return Node(
        VectorSource,
        parameters={
            "output_column": output_column,
            "values": values,
            "shape": shape,
        },
    )


@dataclass(frozen=True)
class VectorMeanConfig(TSFNConfig):
    pass


class VectorMean(BatchTSFN):
    VERSION = "1.0.0"
    CONFIG_CLS = VectorMeanConfig
    BATCH_IS_ELEMENTWISE = True

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature(columns=(ColumnSignature("value", pl.Float64, (2,)),)),
            FrameSignature(columns=(("mean", pl.Float64),)),
        )

    def batch_input_columns(self) -> tuple[str, ...]:
        return ("value",)

    def batch_output_column(self) -> str:
        return "mean"

    def batch(self, fields: Mapping[str, pl.Series]) -> pl.Series:
        tensor = self.series_to_torch(fields["value"], shape=(2,))
        return self.torch_to_series(
            "mean",
            tensor.mean(dim=1),
            dtype=pl.Float64,
        )


def fill_vector_nulls_with_ten(series: pl.Series) -> pl.Series:
    if not isinstance(series.dtype, pl.Array):
        return series.fill_null(10.0)

    fill_array = [10.0] * series.dtype.size
    return (
        series.to_frame()
        .select(
            pl.when(pl.col(series.name).is_null())
            .then(pl.lit(fill_array, dtype=series.dtype))
            .otherwise(pl.col(series.name))
            .arr.eval(pl.element().fill_null(10.0))
            .alias(series.name)
        )
        .to_series()
    )


def test_column_signature_shape_is_part_of_frame_contract() -> None:
    column = ColumnSignature("latent", pl.Float64, (2, 3))
    frame = FrameSignature(columns=(column,))

    assert frame.columns == (("latent", pl.Float64, (2, 3)),)
    assert column.physical_dtype == pl.Array(pl.Float64, 6)


def test_column_signature_rejects_invalid_shape_metadata() -> None:
    with pytest.raises(TypeError, match="tuple"):
        ColumnSignature("latent", pl.Float64, [2])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="positive integers"):
        ColumnSignature("latent", pl.Float64, (0,))

    with pytest.raises(TypeError, match="element dtype"):
        ColumnSignature("latent", pl.Array(pl.Float64, 2), (2,))


def test_unary_itemwise_logit_uses_array_shape_and_preserves_order() -> None:
    source = vector_source(((0.2, 0.5), (0.8, None)))
    logit_node = Node(Logit, bindings={"value": source.value})

    result = Graph(logit_node).execute()

    assert logit_node.outputs == {"logit": pl.Array(pl.Float64, 2)}
    assert result.schema["logit"] == pl.Array(pl.Float64, 2)
    values = result["logit"].to_list()
    assert values[0] == pytest.approx([log(0.2 / 0.8), 0.0])
    assert values[1][0] == pytest.approx(log(0.8 / 0.2))
    assert values[1][1] is None


def test_itemwise_struct_transform_operates_on_multiple_array_inputs() -> None:
    left = vector_source(((1.0, 2.0), (3.0, 4.0)), output_column="left_price")
    right = vector_source(((0.5, 1.5), (1.0, 1.0)), output_column="right_price")
    spread = Node(
        Spread,
        bindings={
            "left": left.left_price,
            "right": right.right_price,
        },
    )

    result = Graph(spread).execute()

    assert spread.outputs == {"spread": pl.Array(pl.Float64, 2)}
    assert result["spread"].to_list() == [[0.5, 0.5], [2.0, 3.0]]


def test_itemwise_struct_transform_rejects_mismatched_shapes() -> None:
    left = vector_source(((1.0, 2.0),), output_column="left_price", shape=(2,))
    right = vector_source(((1.0, 2.0, 3.0),), output_column="right_price", shape=(3,))

    with pytest.raises(ValueError, match="share the same shape"):
        Node(
            Spread,
            bindings={
                "left": left.left_price,
                "right": right.right_price,
            },
        )


def test_ratio_itemwise_struct_transform_nulls_zero_denominators_inside_arrays() -> None:
    numerator = vector_source(((0.8, 0.4), (None, 0.6)), output_column="numerator")
    denominator = vector_source(((0.2, 0.0), (0.5, None)), output_column="denominator")
    ratio = Node(
        Ratio,
        bindings={
            "left": numerator.numerator,
            "right": denominator.denominator,
        },
    )

    result = Graph(ratio).execute()

    assert ratio.outputs == {"ratio": pl.Array(pl.Float64, 2)}
    assert result["ratio"].to_list() == [[4.0, None], [None, None]]


def test_batch_tsfn_can_use_torch_helpers_for_series_chunks() -> None:
    pytest.importorskip("torch")
    source = vector_source(((1.0, 3.0), (2.0, 4.0)))
    mean = Node(VectorMean, bindings={"value": source.value})

    result = Graph(mean).execute()

    assert result["mean"].to_list() == pytest.approx([2.0, 3.0])


def test_batch_bridge_errors_loudly_on_nulls_by_default() -> None:
    pytest.importorskip("torch")
    source = vector_source(((1.0, None), (2.0, 4.0)))
    mean = Node(VectorMean, bindings={"value": source.value})

    with pytest.raises(RuntimeError, match="NullPolicy.ERROR failed.*value.*1"):
        Graph(mean).execute()


def test_node_null_policy_fill_prepares_batch_inputs_before_bridge() -> None:
    pytest.importorskip("torch")
    source = vector_source(((1.0, None), (2.0, 4.0)))
    mean = Node(
        VectorMean,
        bindings={"value": source.value},
        null_policies={"value": NullPolicy.FILL},
        null_fill_values={"value": 0.0},
    )

    result = Graph(mean).execute()

    assert result["timestamp"].to_list() == [dt(0), dt(1)]
    assert result["mean"].to_list() == pytest.approx([0.5, 3.0])


def test_node_null_policy_drop_removes_invalid_batch_rows() -> None:
    pytest.importorskip("torch")
    source = vector_source(((1.0, None), (2.0, 4.0), (None, 6.0)))
    mean = Node(
        VectorMean,
        bindings={"value": source.value},
        null_policies={"value": "drop"},
    )

    result = Graph(mean).execute()

    assert result["timestamp"].to_list() == [dt(1)]
    assert result["mean"].to_list() == pytest.approx([3.0])


def test_custom_null_handler_function_prepares_batch_input_before_bridge() -> None:
    pytest.importorskip("torch")
    source = vector_source(((1.0, None), (2.0, 4.0)))
    mean = Node(
        VectorMean,
        bindings={"value": source.value},
        null_handlers={"value": fill_vector_nulls_with_ten},
    )

    result = Graph(mean).execute()

    assert result["mean"].to_list() == pytest.approx([5.5, 3.0])


def test_explicit_null_handler_wrapper_uses_custom_function() -> None:
    pytest.importorskip("torch")
    source = vector_source(((1.0, None),))
    mean = Node(
        VectorMean,
        bindings={"value": source.value},
        null_handlers={"value": NullHandler.from_function(fill_vector_nulls_with_ten)},
    )

    result = Graph(mean).execute()

    assert result["mean"].to_list() == pytest.approx([5.5])


def test_custom_null_handler_identity_affects_node_id() -> None:
    source = vector_source(((1.0, None),))
    custom = Node(
        VectorMean,
        bindings={"value": source.value},
        null_handlers={"value": fill_vector_nulls_with_ten},
    )
    builtin = Node(
        VectorMean,
        bindings={"value": source.value},
        null_policies={"value": "fill"},
        null_fill_values={"value": 10.0},
    )

    assert custom.ID != builtin.ID


def test_custom_null_handlers_must_be_named_top_level_functions() -> None:
    source = vector_source(((1.0, None),))

    with pytest.raises(ValueError, match="top-level"):
        Node(
            VectorMean,
            bindings={"value": source.value},
            null_handlers={"value": lambda series: series},
        )


def test_fill_null_policy_requires_a_fill_value() -> None:
    source = vector_source(((1.0, None),))
    mean = Node(
        VectorMean,
        bindings={"value": source.value},
        null_policies={"value": "fill"},
    )

    with pytest.raises(ValueError, match="requires null_fill_values"):
        Graph(mean)


def test_numpy_to_series_prefers_pyarrow_for_shaped_outputs() -> None:
    fn = VectorMean({})
    array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    series = fn.numpy_to_series(
        "value",
        array,
        dtype=pl.Float64,
        shape=(2,),
        allow_copy=False,
    )
    array[0, 0] = 99.0

    assert series.dtype == pl.Array(pl.Float64, 2)
    assert series.to_list() == [[99.0, 2.0], [3.0, 4.0]]


def test_numpy_to_series_strict_mode_rejects_non_contiguous_outputs() -> None:
    fn = VectorMean({})
    array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64).T

    with pytest.raises(ValueError, match="not C-contiguous"):
        fn.numpy_to_series(
            "value",
            array,
            dtype=pl.Float64,
            shape=(2,),
            allow_copy=False,
        )
