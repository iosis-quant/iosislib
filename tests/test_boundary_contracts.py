from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from iosislib.core.graph import Graph
from iosislib.core.node import Node
from iosislib.core.tsfn import ColumnSignature, FrameSignature, TSFN, TimeAxis


VALUE_FRAME = FrameSignature(columns=(("value", pl.Int64),))


class NonTupleSignature(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return [FrameSignature.empty(), VALUE_FRAME]  # type: ignore[return-value]

    def apply(self) -> pl.LazyFrame:
        raise AssertionError("apply must not run")


class ShortSignature(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (FrameSignature.empty(),)  # type: ignore[return-value]

    def apply(self) -> pl.LazyFrame:
        raise AssertionError("apply must not run")


class InvalidSignatureItem(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), "not-a-frame"  # type: ignore[return-value]

    def apply(self) -> pl.LazyFrame:
        raise AssertionError("apply must not run")


class DataFrameResult(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), VALUE_FRAME

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {"timestamp": [datetime(2024, 1, 1)], "value": [1]}
        )  # type: ignore[return-value]


class ExtraOutput(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), VALUE_FRAME

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "value": [1],
                "undeclared": [2],
            }
        ).lazy()


class ReorderedOutput(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), VALUE_FRAME

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {"value": [1], "timestamp": [datetime(2024, 1, 1)]}
        ).lazy()


class MissingOutput(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), VALUE_FRAME

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame({"timestamp": [datetime(2024, 1, 1)]}).lazy()


class WrongTypeOutput(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), VALUE_FRAME

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {"timestamp": [datetime(2024, 1, 1)], "value": [1.5]}
        ).lazy()


class ProjectDeclaredInput(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return VALUE_FRAME, VALUE_FRAME

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        assert lf is not None
        return lf.select("timestamp", "value")


@pytest.mark.parametrize(
    ("factory", "exception_type", "message"),
    [
        (lambda: TimeAxis(column=1), TypeError, "time column must be a string"),
        (lambda: TimeAxis(column=" "), ValueError, "time column must be non-empty"),
        (
            lambda: TimeAxis(dtype="bad"),
            TypeError,
            "time axis dtype must be a Polars data type",
        ),
        (
            lambda: TimeAxis(timezone=3),
            TypeError,
            "time axis timezone must be a string or None",
        ),
        (
            lambda: TimeAxis(dtype=pl.Date, timezone="UTC"),
            TypeError,
            "timezone can only be declared for a Datetime",
        ),
        (lambda: ColumnSignature(1, pl.Int64), TypeError, "column name must be a string"),
        (lambda: ColumnSignature("", pl.Int64), ValueError, "column name must be non-empty"),
        (
            lambda: ColumnSignature("value", "bad"),
            TypeError,
            "column dtype must be a Polars data type",
        ),
        (
            lambda: FrameSignature(time="not-an-axis"),
            TypeError,
            "time must be a TimeAxis or None",
        ),
    ],
)
def test_value_objects_reject_malformed_construction(
    factory,
    exception_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception_type, match=message):
        factory()


def test_valid_time_and_dtype_declarations_retain_existing_representation() -> None:
    aware_axis = TimeAxis(
        column="observed_at",
        dtype=pl.Datetime(time_unit="ns"),
        timezone="UTC",
    )
    date_axis = TimeAxis(dtype=pl.Date)
    bare_column = ColumnSignature("value", pl.Int64)

    assert FrameSignature(time=aware_axis, columns=(("value", pl.Float64),)).time == aware_axis
    assert date_axis.dtype is pl.Date
    assert bare_column.dtype is pl.Int64
    assert FrameSignature.empty() == FrameSignature(time=None, columns=())


@pytest.mark.parametrize(
    ("tsfn_cls", "exception_type", "message"),
    [
        (
            NonTupleSignature,
            TypeError,
            r"NonTupleSignature\.type_signature\(\) must return a tuple of exactly two",
        ),
        (
            ShortSignature,
            ValueError,
            r"ShortSignature\.type_signature\(\) must return exactly two items, got 1",
        ),
        (
            InvalidSignatureItem,
            TypeError,
            r"InvalidSignatureItem\.type_signature\(\) item 1 must be a FrameSignature",
        ),
    ],
)
def test_type_signature_protocol_is_validated_deliberately(
    tsfn_cls: type[TSFN],
    exception_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception_type, match=message):
        tsfn_cls({})


def test_apply_must_return_lazyframe_before_schema_inspection() -> None:
    with pytest.raises(
        TypeError,
        match=r"DataFrameResult\.apply\(\) must return a Polars LazyFrame, got DataFrame",
    ):
        DataFrameResult({})()


@pytest.mark.parametrize(
    ("tsfn_cls", "message"),
    [
        (ExtraOutput, "Output schema columns must exactly match the declared order"),
        (ReorderedOutput, "Output schema columns must exactly match the declared order"),
        (MissingOutput, "Missing required output column: 'value'"),
        (WrongTypeOutput, "Column 'value' type mismatch"),
    ],
)
def test_output_schema_requires_exact_membership_order_and_types(
    tsfn_cls: type[TSFN],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        tsfn_cls({})()


def test_input_schema_deliberately_allows_superset_columns_and_order() -> None:
    input_lf = pl.DataFrame(
        {
            "extra": [9],
            "value": [1],
            "timestamp": [datetime(2024, 1, 1)],
        }
    ).lazy()

    result = ProjectDeclaredInput({})(input_lf).collect()

    assert result.columns == ["timestamp", "value"]
    assert result["value"].to_list() == [1]


def test_graph_execution_wraps_boundary_errors_with_node_and_version_context() -> None:
    node = Node(ExtraOutput, name="extra-output")

    with pytest.raises(
        RuntimeError,
        match=r"Execution failed at node 'extra-output' \(ExtraOutput@1\.0\.0\)",
    ) as exc_info:
        Graph(node).execute()

    assert isinstance(exc_info.value.__cause__, ValueError)
