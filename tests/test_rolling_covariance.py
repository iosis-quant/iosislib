from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl
import pytest

from iosislib.core.graph import Graph
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, NullPolicy, TSFN, TSFNConfig
from iosislib.strategy.lowering import builtin_registry
from iosislib.tsfn.transforms import (
    EwmMean,
    RollingCov,
    RollingCovConfig,
    RollingCovMatrix,
    RollingCovMatrixConfig,
    RollingStd,
    RollingVar,
    RollingVarConfig,
)


def dt(minute: int) -> datetime:
    return datetime(2026, 1, 1, 0, minute)


@dataclass(frozen=True)
class FrameSeriesConfig(TSFNConfig):
    columns: tuple[str, ...]
    series: tuple[tuple[float | None, ...], ...]

    def __post_init__(self) -> None:
        if len(self.columns) != len(self.series):
            raise ValueError("columns and series must have the same length")
        lengths = {len(values) for values in self.series}
        if len(lengths) != 1:
            raise ValueError("every series must cover the same timestamps")


class FrameSeriesSource(TSFN):
    VERSION = "1.0.0"
    CONFIG_CLS = FrameSeriesConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature.empty(),
            FrameSignature(
                columns=tuple((name, pl.Float64) for name in self.parameters.columns)
            ),
        )

    def apply(self) -> pl.LazyFrame:
        params = self.parameters
        rows = len(params.series[0])
        data: dict[str, list[datetime] | list[float | None]] = {
            "timestamp": [dt(index) for index in range(rows)]
        }
        for name, values in zip(params.columns, params.series):
            data[name] = list(values)
        return pl.DataFrame(
            data,
            schema={
                "timestamp": pl.Datetime,
                **{name: pl.Float64 for name in params.columns},
            },
        ).lazy()


def source(
    series: tuple[tuple[float | None, ...], ...],
    columns: tuple[str, ...] = ("x", "y"),
) -> Node:
    return Node(
        FrameSeriesSource,
        parameters={"columns": columns, "series": series},
    )


def collect(node: Node) -> pl.DataFrame:
    return Graph(node).execute()


def reference_cov(
    x: tuple[float | None, ...],
    y: tuple[float | None, ...],
    *,
    window_size: int,
    min_samples: int,
) -> list[float | None]:
    return (
        pl.DataFrame(
            {"x": list(x), "y": list(y)},
            schema={"x": pl.Float64, "y": pl.Float64},
        )
        .select(
            pl.rolling_cov(
                "x", "y", window_size=window_size, min_samples=min_samples, ddof=1
            )
            .fill_nan(None)
            .alias("cov")
        )["cov"]
        .to_list()
    )


def test_rolling_var_matches_std_squared_and_manual_window() -> None:
    values = (1.0, 2.0, 3.0, 4.0, 5.0)
    src = source((values, values), columns=("value", "other"))
    var = collect(
        Node(
            RollingVar,
            bindings={"value": src.value},
            parameters={"periods": 5, "min_samples": 5},
        )
    )
    std = collect(
        Node(
            RollingStd,
            bindings={"value": src.value},
            parameters={"periods": 5, "min_samples": 5},
        )
    )

    assert var["rolling_var"].to_list()[:-1] == [None] * 4
    assert var["rolling_var"].to_list()[-1] == pytest.approx(2.5)
    assert var["rolling_var"].to_list()[-1] == pytest.approx(
        std["rolling_std"].to_list()[-1] ** 2
    )


def test_rolling_var_ddof_selects_sample_or_population_divisor() -> None:
    values = (1.0, 2.0, 3.0, 4.0, 5.0)
    src = source((values, values), columns=("value", "other"))

    sample = collect(
        Node(
            RollingVar,
            bindings={"value": src.value},
            parameters={"periods": 3, "min_samples": 3, "ddof": 1},
        )
    )
    population = collect(
        Node(
            RollingVar,
            bindings={"value": src.value},
            parameters={"periods": 3, "min_samples": 3, "ddof": 0},
        )
    )

    assert sample["rolling_var"].to_list() == pytest.approx([None, None, 1.0, 1.0, 1.0])
    assert population["rolling_var"].to_list() == pytest.approx(
        [None, None, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]
    )


def test_rolling_cov_pairwise_tracks_scaled_variance_symmetrically() -> None:
    x = (1.0, 2.0, 3.0, 4.0, 5.0)
    y = (2.0, 4.0, 6.0, 8.0, 10.0)
    src = source((x, y))

    forward = collect(
        Node(
            RollingCov,
            bindings={"left": src.x, "right": src.y},
            parameters={"periods": 3, "min_samples": 3},
        )
    )
    backward = collect(
        Node(
            RollingCov,
            bindings={"left": src.y, "right": src.x},
            parameters={"periods": 3, "min_samples": 3},
        )
    )

    assert forward["rolling_cov"].to_list() == pytest.approx(
        [None, None, 2.0, 2.0, 2.0]
    )
    assert backward["rolling_cov"].to_list() == pytest.approx(
        forward["rolling_cov"].to_list()
    )


def test_rolling_cov_matrix_is_row_major_symmetric_with_variance_diagonal() -> None:
    x = (1.0, 2.0, 3.0, 4.0, 5.0)
    y = (2.0, 4.0, 6.0, 8.0, 10.0)
    src = source((x, y))
    node = Node(
        RollingCovMatrix,
        bindings={"x": src.x, "y": src.y},
        parameters={"input_columns": ["x", "y"], "periods": 3, "min_samples": 3},
    )

    result = collect(node)

    assert node.outputs == {"cov_matrix": pl.Array(pl.Float64, 4)}
    assert result.schema["cov_matrix"] == pl.Array(pl.Float64, 4)
    rows = result["cov_matrix"].to_list()
    assert rows[0] == [None, None, None, None]
    assert rows[1] == [None, None, None, None]
    for row in rows[2:]:
        assert row[0] == pytest.approx(1.0)
        assert row[1] == pytest.approx(2.0)
        assert row[2] == pytest.approx(row[1])
        assert row[3] == pytest.approx(4.0)


def test_rolling_cov_matrix_three_assets_supports_var_portfolio_math() -> None:
    x = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    y = (6.0, 5.0, 4.0, 3.0, 2.0, 1.0)
    z = (1.0, 3.0, 2.0, 5.0, 4.0, 6.0)
    src = source((x, y, z), columns=("x", "y", "z"))
    node = Node(
        RollingCovMatrix,
        bindings={"x": src.x, "y": src.y, "z": src.z},
        parameters={
            "input_columns": ["x", "y", "z"],
            "periods": 4,
            "min_samples": 4,
        },
    )

    result = collect(node)
    rows = result["cov_matrix"].to_list()

    assert result.schema["cov_matrix"] == pl.Array(pl.Float64, 9)
    for row in rows[3:]:
        matrix = [row[i * 3 : (i + 1) * 3] for i in range(3)]
        for i in range(3):
            for j in range(3):
                assert matrix[i][j] == pytest.approx(matrix[j][i])
        weights = (0.5, 0.3, 0.2)
        variance = sum(
            weights[i] * weights[j] * matrix[i][j]  # type: ignore[index]
            for i in range(3)
            for j in range(3)
        )
        assert variance == pytest.approx(
            weights[0] ** 2 * matrix[0][0]
            + weights[1] ** 2 * matrix[1][1]
            + weights[2] ** 2 * matrix[2][2]
            + 2 * weights[0] * weights[1] * matrix[0][1]
            + 2 * weights[0] * weights[2] * matrix[0][2]
            + 2 * weights[1] * weights[2] * matrix[1][2]
        )

    diagonal = collect(
        Node(
            RollingVar,
            bindings={"z": src.z},
            parameters={"input_column": "z", "periods": 4, "min_samples": 4},
        )
    )["rolling_var"].to_list()
    assert [row[8] for row in rows] == pytest.approx(diagonal)


def test_rolling_covariance_matches_polars_reference_with_nulls() -> None:
    x = (1.0, None, 3.0, 4.0, 5.0)
    y = (2.0, 4.0, None, 8.0, 10.0)
    src = source((x, y))

    var = collect(
        Node(RollingVar, bindings={"x": src.x}, parameters={"input_column": "x"})
    )["rolling_var"].to_list()
    cov = collect(Node(RollingCov, bindings={"left": src.x, "right": src.y}))[
        "rolling_cov"
    ].to_list()
    matrix = collect(
        Node(
            RollingCovMatrix,
            bindings={"x": src.x, "y": src.y},
            parameters={"input_columns": ["x", "y"]},
        )
    )["cov_matrix"].to_list()

    reference_var = (
        pl.Series("x", x).rolling_var(window_size=3, min_samples=1, ddof=1).to_list()
    )
    expected_cov = reference_cov(x, y, window_size=3, min_samples=1)
    assert var == pytest.approx(reference_var)
    assert cov == pytest.approx(expected_cov)
    assert [row[0] for row in matrix] == pytest.approx(reference_var)
    assert [row[1] for row in matrix] == pytest.approx(expected_cov)


def test_rolling_covariance_sorts_before_windowing_and_never_looks_ahead() -> None:
    lf = pl.DataFrame(
        {
            "timestamp": [dt(3), dt(0), dt(2), dt(1)],
            "left": [40.0, 1.0, 30.0, 20.0],
            "right": [4.0, 10.0, 3.0, 2.0],
        },
        schema={"timestamp": pl.Datetime, "left": pl.Float64, "right": pl.Float64},
    ).lazy()

    var = RollingVar({"input_column": "left", "periods": 2, "min_samples": 2})(
        lf
    ).collect()
    cov = RollingCov({"periods": 2, "min_samples": 2})(lf).collect()
    matrix = RollingCovMatrix({"input_columns": ["left", "right"]})(
        lf.select("timestamp", "left", "right")
    ).collect()

    assert var["timestamp"].to_list() == [dt(0), dt(1), dt(2), dt(3)]
    assert var["rolling_var"].to_list() == pytest.approx(
        [None, 180.5, 50.0, 50.0]
    )
    assert cov["rolling_cov"].to_list() == pytest.approx(
        [None, -76.0, 5.0, 5.0]
    )
    assert matrix["cov_matrix"].to_list()[1][1] == pytest.approx(-76.0)


def test_rolling_covariance_null_policy_drop_removes_rows_before_windowing() -> None:
    src = source(((1.0, None, 3.0, 4.0), (2.0, 4.0, 6.0, 8.0)))
    node = Node(
        RollingCov,
        bindings={"left": src.x, "right": src.y},
        parameters={"periods": 2, "min_samples": 2},
        null_policies={"left": NullPolicy.DROP},
    )

    result = collect(node)

    assert result["timestamp"].to_list() == [dt(0), dt(2), dt(3)]
    assert result["rolling_cov"].to_list() == pytest.approx([None, 4.0, 1.0])


def test_rolling_covariance_config_rejects_invalid_windows() -> None:
    with pytest.raises(TypeError, match="periods must be an integer"):
        RollingVarConfig(periods=True)
    with pytest.raises(TypeError, match="ddof must be an integer"):
        RollingCovConfig(ddof=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ddof must be between 0 and periods - 1"):
        RollingVarConfig(periods=3, ddof=3)
    with pytest.raises(ValueError, match="ddof must be between 0 and periods - 1"):
        RollingCovConfig(periods=3, ddof=-1)
    with pytest.raises(ValueError, match="Duplicate column names"):
        RollingCovConfig(left_column="right")
    with pytest.raises(ValueError, match="at least two columns"):
        RollingCovMatrixConfig(input_columns=("x",))
    with pytest.raises(ValueError, match="Duplicate input_columns"):
        RollingCovMatrixConfig(input_columns=("x", "x"))
    with pytest.raises(ValueError, match="must not be one of the input columns"):
        RollingCovMatrixConfig(input_columns=("x", "cov_matrix"))
    with pytest.raises(TypeError, match="must be a sequence of strings"):
        RollingCovMatrixConfig(input_columns="xy")  # type: ignore[arg-type]


def test_rolling_covariance_typed_and_mapping_config_share_identity() -> None:
    src = source(((1.0, 2.0, 3.0), (2.0, 1.0, 0.0)))
    typed = Node(
        RollingCovMatrix,
        config=RollingCovMatrixConfig(input_columns=("x", "y"), periods=2),
        bindings={"x": src.x, "y": src.y},
    )
    mapping = Node(
        RollingCovMatrix,
        parameters={"input_columns": ["x", "y"], "periods": 2},
        bindings={"x": src.x, "y": src.y},
    )
    different = Node(
        RollingCovMatrix,
        parameters={"input_columns": ["x", "y"], "periods": 3},
        bindings={"x": src.x, "y": src.y},
    )

    assert typed.ID == mapping.ID
    assert typed.ID != different.ID
    assert typed.definition == mapping.definition


def test_ewm_mean_smooths_scalar_rolling_covariance_output() -> None:
    x = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    y = (2.0, 1.0, 4.0, 3.0, 6.0, 5.0)
    src = source((x, y))
    cov = Node(
        RollingCov,
        bindings={"left": src.x, "right": src.y},
        parameters={"periods": 3, "min_samples": 1},
    )
    smooth = Node(
        EwmMean,
        bindings={"rolling_cov": cov.rolling_cov},
        parameters={"input_column": "rolling_cov", "alpha": 0.5},
    )

    result = collect(smooth)
    reference = (
        pl.Series(
            "cov", reference_cov(x, y, window_size=3, min_samples=1), dtype=pl.Float64
        )
        .ewm_mean(alpha=0.5, adjust=True, min_samples=1)
        .to_list()
    )

    assert result["ewm_mean"].to_list() == pytest.approx(reference)


def test_rolling_covariance_transforms_are_discovered_by_the_builtin_registry() -> None:
    registry = builtin_registry()

    expected = {
        "transform.rolling_var": RollingVar,
        "transform.rolling_cov": RollingCov,
        "transform.rolling_cov_matrix": RollingCovMatrix,
    }
    for operation, cls in expected.items():
        assert registry.resolve(operation, cls.VERSION) is cls
