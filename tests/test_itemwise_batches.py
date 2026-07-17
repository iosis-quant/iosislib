from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import log, prod
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
    TimeAxis,
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

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature(columns=(ColumnSignature("value", pl.Float64, (2,)),)),
            FrameSignature(columns=(("mean", pl.Float64),)),
        )

    def batch(self, frame: pl.DataFrame) -> pl.DataFrame:
        tensor = self.series_to_torch(frame["value"], shape=(2,))
        mean = self.torch_to_series(
            "mean",
            tensor.mean(dim=1),
            dtype=pl.Float64,
        )
        return pl.DataFrame([frame["timestamp"], mean])


class VectorStats(BatchTSFN):
    VERSION = "1.0.0"
    CONFIG_CLS = VectorMeanConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature(columns=(ColumnSignature("value", pl.Float64, (2,)),)),
            FrameSignature(
                columns=(
                    ("mean", pl.Float64),
                    ("difference", pl.Float64),
                )
            ),
        )

    def batch(self, frame: pl.DataFrame) -> pl.DataFrame:
        assert frame.columns == ["timestamp", "value"]
        tensor = self.series_to_torch(frame["value"], shape=(2,))
        mean = self.torch_to_series("mean", tensor.mean(dim=1))
        difference = self.torch_to_series(
            "difference",
            tensor[:, 1] - tensor[:, 0],
        )
        return pl.DataFrame([frame["timestamp"], mean, difference])


class ExplodingBatch(VectorMean):
    VERSION = "1.0.0"

    def batch(self, frame: pl.DataFrame) -> pl.DataFrame:
        raise RuntimeError("batch executed")


class NonFrameBatch(VectorMean):
    VERSION = "1.0.0"

    def batch(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame["value"]  # type: ignore[return-value]


class WrongSchemaBatch(VectorMean):
    VERSION = "1.0.0"

    def batch(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.select("timestamp")


class UtcVectorMean(VectorMean):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        time_axis = TimeAxis(timezone="UTC")
        return (
            FrameSignature(
                time=time_axis,
                columns=(ColumnSignature("value", pl.Float64, (2,)),),
            ),
            FrameSignature(time=time_axis, columns=(("mean", pl.Float64),)),
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


def fill_vector_nulls_with_half(series: pl.Series) -> pl.Series:
    if not isinstance(series.dtype, pl.Array):
        return series.fill_null(0.5)

    fill_array = [0.5] * series.dtype.size
    return (
        series.to_frame()
        .select(
            pl.when(pl.col(series.name).is_null())
            .then(pl.lit(fill_array, dtype=series.dtype))
            .otherwise(pl.col(series.name))
            .arr.eval(pl.element().fill_null(0.5))
            .alias(series.name)
        )
        .to_series()
    )


def drop_last_handler_row(series: pl.Series) -> pl.Series:
    return series.head(len(series) - 1)


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


def test_tsfn_centrally_enforces_error_policy_for_native_itemwise_operations() -> None:
    source = vector_source(((0.2, None),))
    logit = Node(
        Logit,
        bindings={"value": source.value},
        null_policies={"value": NullPolicy.ERROR},
    )

    with pytest.raises(RuntimeError, match="NullPolicy.ERROR failed.*value"):
        Graph(logit).execute()


def test_tsfn_centrally_fills_nulls_before_native_itemwise_operations() -> None:
    source = vector_source(((0.2, None),))
    logit = Node(
        Logit,
        bindings={"value": source.value},
        null_policies={"value": NullPolicy.FILL},
        null_fill_values={"value": 0.5},
    )

    result = Graph(logit).execute()

    assert result["logit"].to_list()[0][0] == pytest.approx(log(0.2 / 0.8))
    assert result["logit"].to_list()[0][1] == pytest.approx(0.0)


def test_tsfn_centrally_applies_custom_handlers_before_subclass_execution() -> None:
    source = vector_source(((0.2, None),))
    logit = Node(
        Logit,
        bindings={"value": source.value},
        null_handlers={"value": fill_vector_nulls_with_half},
    )

    result = Graph(logit).execute()

    assert result["logit"].to_list()[0][1] == pytest.approx(0.0)


def test_custom_null_handlers_must_preserve_frame_row_count() -> None:
    source = vector_source(((0.2, None), (0.3, 0.4)))
    logit = Node(
        Logit,
        bindings={"value": source.value},
        null_handlers={"value": drop_last_handler_row},
    )

    with pytest.raises(RuntimeError, match="must preserve row count"):
        Graph(logit).execute()


def test_batch_tsfn_can_use_torch_helpers_for_series_chunks() -> None:
    pytest.importorskip("torch")
    source = vector_source(((1.0, 3.0), (2.0, 4.0)))
    mean = Node(VectorMean, bindings={"value": source.value})

    result = Graph(mean).execute()

    assert result["mean"].to_list() == pytest.approx([2.0, 3.0])


def test_batch_tsfn_maps_complete_frames_and_supports_multiple_outputs() -> None:
    pytest.importorskip("torch")
    fn = VectorStats({})
    lf = pl.DataFrame(
        {
            "timestamp": [dt(0), dt(1)],
            "value": [[1.0, 3.0], [2.0, 8.0]],
            "ignored": [10, 20],
        },
        schema={
            "timestamp": pl.Datetime,
            "value": pl.Array(pl.Float64, 2),
            "ignored": pl.Int64,
        },
    ).lazy()

    result = fn(lf).collect()

    assert result.columns == ["timestamp", "mean", "difference"]
    assert result["mean"].to_list() == pytest.approx([2.0, 5.0])
    assert result["difference"].to_list() == pytest.approx([2.0, 6.0])


def test_batch_tsfn_requires_a_node_materialization_boundary() -> None:
    source = vector_source(((1.0, 3.0),))
    default_node = Node(VectorMean, bindings={"value": source.value})
    disabled_node = Node(
        VectorMean,
        bindings={"value": source.value},
        materialize=False,
    )

    assert default_node.materialize is True
    assert default_node.function.requires_materialization is True
    assert VectorMean.DEFAULT_NULL_POLICY is NullPolicy.ERROR
    assert Logit.DEFAULT_NULL_POLICY is NullPolicy.PROPAGATE
    assert Graph(default_node).materialized_node_ids == frozenset({default_node.ID})
    with pytest.raises(ValueError, match="VectorMean requires materialization"):
        Graph(disabled_node)


def test_batch_udf_is_not_run_by_verify_and_runs_only_during_execute() -> None:
    source = vector_source(((1.0, 3.0),))
    exploding = Node(ExplodingBatch, bindings={"value": source.value})
    graph = Graph(exploding)

    verified = graph.verify()

    assert verified is None
    with pytest.raises(RuntimeError, match="batch executed"):
        graph.execute()


def test_batch_udf_must_return_a_dataframe() -> None:
    source = vector_source(((1.0, 3.0),))
    invalid = Node(NonFrameBatch, bindings={"value": source.value})

    with pytest.raises(RuntimeError, match="batch must return a Polars DataFrame"):
        Graph(invalid).execute()


def test_batch_udf_output_must_match_the_declared_schema() -> None:
    source = vector_source(((1.0, 3.0),))
    invalid = Node(WrongSchemaBatch, bindings={"value": source.value})

    with pytest.raises(RuntimeError, match="schema"):
        Graph(invalid).execute()


def test_batch_tsfn_preserves_an_empty_frame_with_its_declared_schema() -> None:
    fn = VectorMean({})
    lf = pl.DataFrame(
        schema={
            "timestamp": pl.Datetime,
            "value": pl.Array(pl.Float64, 2),
        }
    ).lazy()

    result = fn(lf).collect()

    assert result.shape == (0, 2)
    assert result.schema["timestamp"] == pl.Datetime
    assert result.schema["mean"] == pl.Float64


def test_batch_tsfn_declares_timezone_aware_output_schema() -> None:
    fn = UtcVectorMean({})
    lf = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            "value": [[1.0, 3.0]],
        },
        schema={
            "timestamp": pl.Datetime(time_zone="UTC"),
            "value": pl.Array(pl.Float64, 2),
        },
    ).lazy()

    result = fn(lf).collect()

    assert result.schema["timestamp"] == pl.Datetime(time_zone="UTC")
    assert result["mean"].to_list() == [2.0]


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


def test_series_to_torch_uses_the_arrow_buffer_through_dlpack() -> None:
    fn = VectorMean({})
    series = pl.Series(
        "value",
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=pl.Array(pl.Float64, 2),
    )

    tensor = fn.series_to_torch(series, shape=(2,))
    arrow_address = series.to_arrow().values.buffers()[1].address

    assert tensor.shape == (2, 2)
    assert tensor.data_ptr() == arrow_address


def test_scalar_series_to_torch_uses_the_arrow_buffer_through_dlpack() -> None:
    fn = VectorMean({})
    series = pl.Series("value", [1.0, 2.0], dtype=pl.Float64)

    tensor = fn.series_to_torch(series)

    assert tensor.data_ptr() == series.to_arrow().buffers()[1].address
    assert tensor.tolist() == [1.0, 2.0]


def test_torch_to_series_preserves_the_tensor_buffer() -> None:
    torch = pytest.importorskip("torch")
    fn = VectorMean({})
    tensor = torch.tensor([1.0, 2.0], dtype=torch.float64)

    series = fn.torch_to_series("value", tensor)
    tensor[0] = 9.0

    assert series.to_arrow().buffers()[1].address == tensor.data_ptr()
    assert series.to_list() == [9.0, 2.0]


def test_zero_copy_bridges_reject_inputs_that_require_materialization() -> None:
    torch = pytest.importorskip("torch")
    fn = VectorMean({})

    with pytest.raises(TypeError, match="expected a NumPy ndarray"):
        fn.numpy_to_series("value", [1.0, 2.0])

    non_contiguous = torch.arange(4, dtype=torch.float64).reshape(2, 2).T
    with pytest.raises(ValueError, match="non-contiguous tensor"):
        fn.torch_to_series("value", non_contiguous, shape=(2,))

    wrong_dtype = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(TypeError, match="without copying or casting"):
        fn.numpy_to_series("value", wrong_dtype, dtype=pl.Float64)

    chunked = pl.concat(
        [
            pl.Series("value", [1.0], dtype=pl.Float64),
            pl.Series("value", [2.0], dtype=pl.Float64),
        ],
        rechunk=False,
    )
    assert chunked.n_chunks() == 2
    with pytest.raises(ValueError, match="multi-chunk Series"):
        fn.series_to_torch(chunked)


def test_bridge_copying_requires_an_explicit_opt_in() -> None:
    fn = VectorMean({})

    series = fn.numpy_to_series(
        "value",
        [1.0, 2.0],
        dtype=pl.Float64,
        allow_copy=True,
    )

    assert series.dtype == pl.Float64
    assert series.to_list() == [1.0, 2.0]

    chunked = pl.concat(
        [
            pl.Series("value", [1.0], dtype=pl.Float64),
            pl.Series("value", [2.0], dtype=pl.Float64),
        ],
        rechunk=False,
    )
    tensor = fn.series_to_torch(chunked, allow_copy=True)
    assert tensor.tolist() == [1.0, 2.0]


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
