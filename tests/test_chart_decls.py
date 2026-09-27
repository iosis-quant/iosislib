from __future__ import annotations

from datetime import date, datetime, timedelta

import matplotlib
import polars as pl
import pytest

from iosislib.charting import (
    CHART_KINDS,
    ChartDecl,
    parse_chart_decls,
    placeholder_chart_svg,
    render_chart,
    render_chart_decls,
)
from iosislib.strategy import dumps, loads

matplotlib.use("Agg")


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1) + timedelta(days=i) for i in range(4)],
            "equity": [100.0, 110.0, 95.0, 105.0],
            "prediction": [1.0, 2.0, 3.0, 4.0],
            "target": [1.2, 1.9, 3.3, 3.8],
        }
    )


def test_parse_returns_empty_tuple_without_charts() -> None:
    assert parse_chart_decls({}) == ()
    assert parse_chart_decls(None) == ()
    assert parse_chart_decls({"charts": None}) == ()
    assert parse_chart_decls({"charts": []}) == ()


def test_chart_kinds_constant() -> None:
    assert CHART_KINDS == frozenset({"bars", "equity", "line", "scatter"})


def test_declared_output_defaults_to_the_sole_output() -> None:
    decls = parse_chart_decls(
        {"charts": [{"kind": "equity", "columns": ["equity"]}]},
        outputs=["backtest"],
    )
    assert decls == (
        ChartDecl(
            kind="equity",
            name="backtest_equity",
            output="backtest",
            columns=("equity",),
        ),
    )


def test_name_defaults_and_explicit_values_are_kept() -> None:
    decls = parse_chart_decls(
        {"charts": [{"kind": "line", "output": "a", "columns": ["x"], "name": "My.Chart"}]},
        outputs=["a", "b"],
    )
    assert decls[0].name == "My.Chart"


def test_scalar_y_is_normalized_to_a_tuple() -> None:
    decls = parse_chart_decls(
        {"charts": [{"kind": "scatter", "x": "prediction", "y": "target"}]},
        outputs=["p"],
    )
    assert decls[0].y == ("target",)


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\.kind must be one of"):
        parse_chart_decls({"charts": [{"kind": "nope"}]}, outputs=["o"])


def test_output_is_required_for_multiple_outputs() -> None:
    with pytest.raises(ValueError, match="multiple outputs"):
        parse_chart_decls({"charts": [{"kind": "line"}]}, outputs=["a", "b"])


def test_chart_without_outputs_is_rejected() -> None:
    with pytest.raises(ValueError, match="the strategy has no outputs"):
        parse_chart_decls({"charts": [{"kind": "line"}]}, outputs=[])


def test_unknown_output_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match="references unknown output 'zz'"):
        parse_chart_decls(
            {"charts": [{"kind": "line", "output": "zz"}]}, outputs=["a"]
        )


def test_scatter_requires_x_and_y() -> None:
    with pytest.raises(ValueError, match="scatter charts require x and y"):
        parse_chart_decls(
            {"charts": [{"kind": "scatter", "output": "a"}]}, outputs=["a"]
        )


def test_xy_are_rejected_outside_scatter() -> None:
    with pytest.raises(ValueError, match="x/y are only valid for scatter"):
        parse_chart_decls(
            {"charts": [{"kind": "line", "output": "a", "x": "v"}]}, outputs=["a"]
        )


def test_scatter_rejects_columns() -> None:
    with pytest.raises(ValueError, match="scatter charts declare x/y, not columns"):
        parse_chart_decls(
            {
                "charts": [
                    {"kind": "scatter", "output": "a", "x": "p", "y": "t", "columns": ["p"]}
                ]
            },
            outputs=["a"],
        )


def test_columns_must_be_a_list_of_names() -> None:
    with pytest.raises(ValueError, match="list of column names"):
        parse_chart_decls(
            {"charts": [{"kind": "line", "output": "a", "columns": "x"}]},
            outputs=["a"],
        )
    with pytest.raises(ValueError, match="at least one non-empty column name"):
        parse_chart_decls(
            {"charts": [{"kind": "line", "output": "a", "columns": []}]},
            outputs=["a"],
        )
    with pytest.raises(ValueError, match="duplicate column names"):
        parse_chart_decls(
            {"charts": [{"kind": "line", "output": "a", "columns": ["x", "x"]}]},
            outputs=["a"],
        )


def test_invalid_chart_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="must start with a letter"):
        parse_chart_decls(
            {"charts": [{"kind": "line", "output": "a", "name": "bad name!"}]},
            outputs=["a"],
        )
    with pytest.raises(ValueError, match="duplicates chart name 'a_line'"):
        parse_chart_decls(
            {"charts": [{"kind": "line", "output": "a"}, {"kind": "line", "output": "a"}]},
            outputs=["a"],
        )


def test_unknown_fields_and_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[0\] has unknown field\(s\): typo"):
        parse_chart_decls(
            {"charts": [{"kind": "line", "output": "a", "typo": 1}]}, outputs=["a"]
        )
    with pytest.raises(ValueError, match=r"\[0\] must be a mapping"):
        parse_chart_decls({"charts": ["nope"]}, outputs=["a"])
    with pytest.raises(ValueError, match=r"\.charts must be a list"):
        parse_chart_decls({"charts": "nope"}, outputs=["a"])
    with pytest.raises(TypeError, match=r"\$\.metadata must be a mapping"):
        parse_chart_decls(["nope"], outputs=["a"])


def test_outputs_argument_must_be_a_name_sequence() -> None:
    with pytest.raises(TypeError, match="sequence of output names"):
        parse_chart_decls({}, outputs="ab")
    with pytest.raises(TypeError, match="non-empty strings"):
        parse_chart_decls({}, outputs=[1])


def test_chart_decls_survive_a_strategy_round_trip() -> None:
    yaml = """
format: iosis.strategy
version: 0.1.0
name: charted
nodes:
  change:
    op: transform.delta
    version: 0.2.0
    params:
      output_column: change
outputs:
  backtest: change.change
metadata:
  charts:
    - kind: line
      columns: [change]
      title: Change
    - kind: scatter
      name: fit
      output: backtest
      x: change
      y: [change]
"""
    strategy = loads(dumps(loads(yaml)))
    decls = parse_chart_decls(strategy.metadata, outputs=list(strategy.outputs))
    assert [(decl.kind, decl.name, decl.output) for decl in decls] == [
        ("line", "backtest_line", "backtest"),
        ("scatter", "fit", "backtest"),
    ]
    assert decls[0].title == "Change"
    assert decls[1].x == "change"
    assert decls[1].y == ("change",)


def test_render_chart_draws_each_kind() -> None:
    frame = _frame()
    line = ChartDecl(kind="line", name="signal", output="a", columns=("prediction",))
    equity = ChartDecl(kind="equity", name="curve", output="a", columns=("equity",))
    scatter = ChartDecl(
        kind="scatter", name="fit", output="a", x="prediction", y=("target",)
    )
    bar_frame = pl.DataFrame(
        {
            "day": [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)],
            "value": [1.0, -2.0, 3.0],
        }
    )
    bars = ChartDecl(kind="bars", name="daily", output="a", columns=("value",))

    for decl, source in ((line, frame), (equity, frame), (scatter, frame), (bars, bar_frame)):
        svg = render_chart(source, decl)
        assert "<svg" in svg
        assert "chart unavailable" not in svg


def test_render_chart_returns_placeholder_on_failure() -> None:
    decl = ChartDecl(kind="line", name="broken", output="a", columns=("missing",))
    svg = render_chart(_frame(), decl)
    assert "chart unavailable" in svg
    assert "missing" in svg


def test_render_chart_closes_every_figure_it_creates() -> None:
    import matplotlib.pyplot as plt

    before = set(plt.get_fignums())
    render_chart(
        _frame(),
        ChartDecl(kind="line", name="signal", output="a", columns=("prediction",)),
    )
    render_chart(
        _frame(),
        ChartDecl(kind="line", name="broken", output="a", columns=("missing",)),
    )
    assert set(plt.get_fignums()) == before


def test_render_chart_requires_a_chart_decl() -> None:
    with pytest.raises(TypeError, match="decl must be a ChartDecl"):
        render_chart(_frame(), "signal")  # type: ignore[arg-type]


def test_render_chart_decls_maps_names_to_svgs() -> None:
    decls = (
        ChartDecl(kind="line", name="signal", output="backtest", columns=("prediction",)),
        ChartDecl(kind="equity", name="gone", output="other", columns=("equity",)),
    )
    rendered = render_chart_decls({"backtest": _frame()}, decls)
    assert set(rendered) == {"signal", "gone"}
    assert "chart unavailable" not in rendered["signal"]
    assert "has no frame" in rendered["gone"]


def test_render_chart_decls_uses_the_only_frame_for_outputless_decls() -> None:
    decl = ChartDecl(kind="line", name="signal", output=None, columns=("prediction",))
    rendered = render_chart_decls({"only": _frame()}, [decl])
    assert "chart unavailable" not in rendered["signal"]

    missing = ChartDecl(kind="line", name="other", output=None, columns=("prediction",))
    rendered = render_chart_decls({"a": _frame(), "b": _frame()}, [missing])
    assert "does not name an output" in rendered["other"]


def test_render_chart_decls_validates_arguments() -> None:
    with pytest.raises(TypeError, match="mapping of output name to frame"):
        render_chart_decls([], [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sequence of ChartDecl"):
        render_chart_decls({}, "decl")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="only ChartDecl"):
        render_chart_decls({}, ["signal"])  # type: ignore[list-item]


def test_placeholder_svg_is_deterministic_and_escaped() -> None:
    first = placeholder_chart_svg("boom <script>")
    second = placeholder_chart_svg("boom <script>")
    assert first == second
    assert "chart unavailable" in first
    assert "<script>" not in first
    assert "&lt;script&gt;" in first
