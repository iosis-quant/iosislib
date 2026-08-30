from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from iosislib.core.graph import Graph
from iosislib.core.node import Node
from iosislib.core.tsfn import ColumnSignature, FrameSignature, TimeAxis
from iosislib.tsfn.adapters import DataFrameSource, DataFrameSourceConfig
from iosislib.tsfn.transforms import Delta, Logit


TIMESTAMP = [
    datetime(2026, 1, 1, 0, 0),
    datetime(2026, 1, 1, 0, 1),
    datetime(2026, 1, 1, 0, 2),
]
FLOAT_SIGNATURE = FrameSignature(columns=(("value", pl.Float64),))


def make_df(
    timestamps: list[datetime] | None = None,
    values: list[float] | None = None,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": timestamps or TIMESTAMP,
            "value": values or [0.25, 0.5, 0.75],
        },
        schema={"timestamp": pl.Datetime, "value": pl.Float64},
    )


def test_dataframe_source_returns_lazy_projection() -> None:
    df = make_df()
    config = DataFrameSourceConfig.from_frame(df, FLOAT_SIGNATURE)
    source = Node(DataFrameSource, config=config)

    lazy_result = source.function.apply()
    result = Graph(source).execute()

    assert isinstance(lazy_result, pl.LazyFrame)
    assert source.function.signature[0] == FrameSignature.empty()
    assert result.columns == ["timestamp", "value"]
    assert result["timestamp"].to_list() == TIMESTAMP
    assert result["value"].to_list() == [0.25, 0.5, 0.75]


def test_dataframe_source_with_unnamed_time_axis() -> None:
    df = pl.DataFrame(
        {"ts": TIMESTAMP, "close": [100.0, 101.0, 102.0]},
        schema={"ts": pl.Datetime, "close": pl.Float64},
    )
    signature = FrameSignature(
        time=TimeAxis(column="ts"),
        columns=(("close", pl.Float64),),
    )
    config = DataFrameSourceConfig.from_frame(df, signature)
    source = Node(DataFrameSource, config=config)

    result = Graph(source).execute()

    assert result.columns == ["ts", "close"]
    assert result["close"].to_list() == [100.0, 101.0, 102.0]


def test_dataframe_source_projects_only_declared_columns() -> None:
    df = pl.DataFrame(
        {
            "timestamp": TIMESTAMP,
            "value": [0.25, 0.5, 0.75],
            "ignored": [1, 2, 3],
        },
        schema={"timestamp": pl.Datetime, "value": pl.Float64, "ignored": pl.Int64},
    )
    config = DataFrameSourceConfig.from_frame(df, FLOAT_SIGNATURE)
    source = Node(DataFrameSource, config=config)

    result = Graph(source).execute()

    assert result.columns == ["timestamp", "value"]
    assert result["value"].to_list() == [0.25, 0.5, 0.75]


def test_dataframe_source_preserves_declared_shape() -> None:
    df = pl.DataFrame(
        {"timestamp": TIMESTAMP[:2], "vector": [[1.0, 2.0], [3.0, 4.0]]},
        schema={"timestamp": pl.Datetime, "vector": pl.Array(pl.Float64, 2)},
    )
    signature = FrameSignature(
        columns=(ColumnSignature("vector", pl.Float64, (2,)),),
    )
    config = DataFrameSourceConfig.from_frame(df, signature)
    source = Node(DataFrameSource, config=config)

    result = Graph(source).execute()

    assert result.schema["vector"] == pl.Array(pl.Float64, 2)
    assert result["vector"].to_list() == [[1.0, 2.0], [3.0, 4.0]]


def test_dataframe_source_chained_with_transforms() -> None:
    df = make_df(values=[0.3, 0.7, 0.5])
    config = DataFrameSourceConfig.from_frame(df, FLOAT_SIGNATURE)
    source = Node(DataFrameSource, config=config, name="prices")
    log_odds = Node(
        Logit,
        bindings={"value": source.value},
        parameters={"input_column": "value", "output_column": "log_odds"},
    )
    change = Node(
        Delta,
        bindings={"log_odds": log_odds.log_odds},
        parameters={"input_column": "log_odds", "output_column": "change"},
    )

    result = Graph(change).execute()

    assert result.columns == ["timestamp", "change"]


def test_dataframe_source_config_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="frame must be a polars DataFrame"):
        DataFrameSourceConfig.from_frame("not a dataframe", FLOAT_SIGNATURE)


def test_dataframe_source_config_rejects_empty_signature() -> None:
    df = make_df()
    with pytest.raises(ValueError, match="declare a time axis"):
        DataFrameSourceConfig.from_frame(df, FrameSignature.empty())


def test_dataframe_source_config_rejects_schema_mismatch() -> None:
    df = pl.DataFrame(
        {"timestamp": TIMESTAMP, "value": [1, 2, 3]},
        schema={"timestamp": pl.Datetime, "value": pl.Int64},
    )
    with pytest.raises(ValueError, match="Frame columns/dtypes do not match"):
        DataFrameSourceConfig.from_frame(df, FLOAT_SIGNATURE)


def test_dataframe_source_config_rejects_missing_column() -> None:
    df = pl.DataFrame(
        {"timestamp": TIMESTAMP},
        schema={"timestamp": pl.Datetime},
    )
    with pytest.raises(ValueError, match="Frame columns/dtypes do not match"):
        DataFrameSourceConfig.from_frame(df, FLOAT_SIGNATURE)


def test_dataframe_source_config_rejects_wrong_time_column() -> None:
    df = pl.DataFrame(
        {"ts": TIMESTAMP, "value": [0.25, 0.5, 0.75]},
        schema={"ts": pl.Datetime, "value": pl.Float64},
    )
    with pytest.raises(ValueError, match="Frame columns/dtypes do not match"):
        DataFrameSourceConfig.from_frame(df, FLOAT_SIGNATURE)


def test_dataframe_source_deterministic_ids() -> None:
    df = make_df()
    config1 = DataFrameSourceConfig.from_frame(df, FLOAT_SIGNATURE)
    config2 = DataFrameSourceConfig.from_frame(df, FLOAT_SIGNATURE)
    node1 = Node(DataFrameSource, config=config1)
    node2 = Node(DataFrameSource, config=config2)

    assert config1 == config2
    assert config1.to_dict() == config2.to_dict()
    assert node1.ID == node2.ID


def test_dataframe_source_different_frames_different_ids() -> None:
    df1 = make_df(values=[1.0, 2.0, 3.0])
    df2 = make_df(values=[4.0, 5.0, 6.0])
    node1 = Node(DataFrameSource, config=DataFrameSourceConfig.from_frame(
        df1, FLOAT_SIGNATURE,
    ))
    node2 = Node(DataFrameSource, config=DataFrameSourceConfig.from_frame(
        df2, FLOAT_SIGNATURE,
    ))

    assert node1.ID != node2.ID


def test_dataframe_source_rejects_type_mismatch_at_construction() -> None:
    df = pl.DataFrame(
        {"timestamp": TIMESTAMP, "value": ["a", "b", "c"]},
        schema={"timestamp": pl.Datetime, "value": pl.String},
    )
    with pytest.raises(ValueError, match="Frame columns/dtypes do not match"):
        DataFrameSourceConfig.from_frame(df, FLOAT_SIGNATURE)


def test_dataframe_source_version() -> None:
    assert DataFrameSource.VERSION == "0.1.0"


def test_dataframe_source_roundtrip_preserves_data() -> None:
    df = make_df()
    config = DataFrameSourceConfig.from_frame(df, FLOAT_SIGNATURE)

    reconstructed = config.frame
    assert reconstructed["timestamp"].to_list() == TIMESTAMP
    assert reconstructed["value"].to_list() == [0.25, 0.5, 0.75]
