from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import sqrt

import polars as pl
import pytest

from iosislib.core.graph import Graph
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, TSFN, TSFNConfig
from iosislib.strategy import builtin_registry
from iosislib.tsfn.transforms import (
    Abs,
    AbsConfig,
    Clip,
    ClipConfig,
    Power,
    PowerConfig,
    Sign,
    SignConfig,
    Sqrt,
    SqrtConfig,
)
from _floats import assert_lists_close


@dataclass(frozen=True)
class UnarySeriesConfig(TSFNConfig):
    minutes: tuple[int, ...]
    values: tuple[float | None, ...]
    output_column: str = "value"


class UnarySeriesSource(TSFN):
    VERSION = "1.0.0"
    CONFIG_CLS = UnarySeriesConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature.empty(),
            FrameSignature(columns=((self.parameters.output_column, pl.Float64),)),
        )

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "timestamp": [dt(minute) for minute in self.parameters.minutes],
                self.parameters.output_column: list(self.parameters.values),
            },
            schema={
                "timestamp": pl.Datetime,
                self.parameters.output_column: pl.Float64,
            },
        ).lazy()


def dt(minute: int) -> datetime:
    return datetime(2026, 1, 1, 0, minute)


def source(
    values: tuple[float | None, ...],
    *,
    output_column: str = "value",
) -> Node:
    return Node(
        UnarySeriesSource,
        parameters={
            "minutes": tuple(range(len(values))),
            "values": values,
            "output_column": output_column,
        },
    )


def unary_result(
    cls: type,
    values: tuple[float | None, ...],
    parameters: dict[str, object] | None = None,
) -> pl.DataFrame:
    node = Node(
        cls,
        bindings={"value": source(values).value},
        parameters={} if parameters is None else parameters,
    )
    return Graph(node).execute()


def test_abs_and_sign_transform_values_and_preserve_nulls() -> None:
    values = (-3.0, 4.0, None, 0.0)

    abs_result = unary_result(Abs, values)
    sign_result = unary_result(Sign, values)

    assert abs_result["abs"].to_list() == pytest.approx([3.0, 4.0, None, 0.0])
    assert sign_result["sign"].to_list() == pytest.approx([-1.0, 1.0, None, 0.0])


def test_sqrt_nulls_negative_inputs_instead_of_emitting_nan() -> None:
    result = unary_result(Sqrt, (4.0, -1.0, 0.0, None))

    assert result["sqrt"].to_list() == pytest.approx([2.0, None, 0.0, None])


def test_power_squares_by_default() -> None:
    result = unary_result(Power, (3.0, -2.0, None))

    assert result["power"].to_list() == pytest.approx([9.0, 4.0, None])


def test_power_nulls_negative_bases_under_fractional_exponents() -> None:
    result = unary_result(
        Power,
        (-2.0, 4.0, None),
        {"exponent": 0.5},
    )

    assert result["power"].to_list() == [None, 2.0, None]


def test_power_supports_configured_exponents() -> None:
    result = unary_result(Power, (-3.0, 4.0), {"exponent": 3.0})

    assert result["power"].to_list() == pytest.approx([-27.0, 64.0])
    assert unary_result(Power, (3.0,), {"exponent": 0.5})["power"].to_list() == (
        pytest.approx([sqrt(3.0)])
    )


def test_power_config_rejects_invalid_exponents() -> None:
    with pytest.raises(TypeError, match="exponent must be a number"):
        PowerConfig(exponent=True)
    with pytest.raises(TypeError, match="exponent must be a number"):
        PowerConfig(exponent="two")
    with pytest.raises(ValueError, match="exponent must be finite"):
        PowerConfig(exponent=float("inf"))
    with pytest.raises(ValueError, match="exponent must be finite"):
        PowerConfig(exponent=float("nan"))


def test_clip_clamps_between_two_sided_bounds() -> None:
    result = unary_result(
        Clip,
        (-5.0, 0.5, 7.0, None),
        {"lower": -1.0, "upper": 1.0},
    )

    assert result["clip"].to_list() == pytest.approx([-1.0, 0.5, 1.0, None])


def test_clip_supports_one_sided_bounds() -> None:
    lower_only = unary_result(Clip, (-5.0, 0.5, 7.0), {"lower": -1.0})
    upper_only = unary_result(Clip, (-5.0, 0.5, 7.0), {"upper": 1.0})

    assert lower_only["clip"].to_list() == pytest.approx([-1.0, 0.5, 7.0])
    assert upper_only["clip"].to_list() == pytest.approx([-5.0, 0.5, 1.0])


def test_clip_config_requires_an_ordered_bound() -> None:
    with pytest.raises(ValueError, match="At least one of lower and upper"):
        ClipConfig()
    with pytest.raises(ValueError, match="lower must not exceed upper"):
        ClipConfig(lower=2.0, upper=1.0)
    with pytest.raises(TypeError, match="lower must be a number or None"):
        ClipConfig(lower=True)
    with pytest.raises(ValueError, match="lower must be finite"):
        ClipConfig(lower=float("inf"))


def test_unary_transforms_support_custom_column_names() -> None:
    src = source((0.25, -0.25), output_column="probability")
    node = Node(
        Abs,
        bindings={"probability": src.probability},
        parameters={
            "input_column": "probability",
            "output_column": "probability_abs",
        },
    )

    result = Graph(node).execute()

    assert node.outputs == {"probability_abs": pl.Float64}
    assert result.columns == ["timestamp", "probability_abs"]
    assert result["probability_abs"].to_list() == pytest.approx([0.25, 0.25])


@dataclass(frozen=True)
class UnaryVectorConfig(TSFNConfig):
    values: tuple[tuple[float | None, ...], ...]


class UnaryVectorSource(TSFN):
    VERSION = "1.0.0"
    CONFIG_CLS = UnaryVectorConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature.empty(),
            FrameSignature(columns=(("value", pl.Float64, (2,)),)),
        )

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "timestamp": [dt(index) for index in range(len(self.parameters.values))],
                "value": pl.Series(
                    "value",
                    list(self.parameters.values),
                    dtype=pl.Array(pl.Float64, 2),
                ),
            },
            schema={
                "timestamp": pl.Datetime,
                "value": pl.Array(pl.Float64, 2),
            },
        ).lazy()


def vector_source(values: tuple[tuple[float | None, ...], ...]) -> Node:
    return Node(UnaryVectorSource, parameters={"values": values})


def test_abs_inherits_vector_shape_through_the_graph() -> None:
    src = vector_source(((-4.0, 3.0), (9.0, -1.0)))
    node = Node(Abs, bindings={"value": src.value})

    result = Graph(node).execute()

    assert node.outputs == {"abs": pl.Array(pl.Float64, 2)}
    assert_lists_close(
        result["abs"].to_list(),
        [[4.0, 3.0], [9.0, 1.0]],
    )


def test_sqrt_inherits_vector_shape_and_nulls_negative_cells() -> None:
    src = vector_source(((-4.0, 3.0), (4.0, -1.0)))
    node = Node(Sqrt, bindings={"value": src.value})

    result = Graph(node).execute()

    assert node.outputs == {"sqrt": pl.Array(pl.Float64, 2)}
    assert_lists_close(
        result["sqrt"].to_list(),
        [[None, sqrt(3.0)], [2.0, None]],
    )


def test_clip_inherits_vector_shape_through_the_graph() -> None:
    src = vector_source(((-4.0, 3.0), (9.0, -1.0)))
    node = Node(
        Clip,
        bindings={"value": src.value},
        parameters={"lower": -3.0, "upper": 4.0},
    )

    result = Graph(node).execute()

    assert node.outputs == {"clip": pl.Array(pl.Float64, 2)}
    assert_lists_close(
        result["clip"].to_list(),
        [[-3.0, 3.0], [4.0, -1.0]],
    )


def test_unary_transforms_are_discovered_by_the_builtin_registry() -> None:
    registry = builtin_registry()

    expected = {
        "transform.abs": Abs,
        "transform.power": Power,
        "transform.sign": Sign,
        "transform.sqrt": Sqrt,
        "transform.clip": Clip,
    }
    for operation, cls in expected.items():
        assert registry.resolve(operation, cls.VERSION) is cls
