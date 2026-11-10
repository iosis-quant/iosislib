from __future__ import annotations

import pytest

from iosislib.strategy import (
    Input,
    Node,
    Reference,
    Strategy,
    expand_variants,
)


def make_strategy() -> Strategy:
    return Strategy(
        name="demo",
        nodes={
            "data": Node(op="source.csv_source", version="0.2.0"),
            "x_lag": Node(
                op="transform.lag",
                version="0.2.0",
                inputs={"x": Input(source=Reference("data", "x"))},
                params={"input_column": "x", "output_column": "x_lag", "periods": 1},
            ),
        },
        outputs={
            "target": Reference("data", "y"),
            "temporal": Reference("x_lag", "x_lag"),
        },
    )


def build_variant(label: str) -> Node:
    return Node(
        op="perturb.block_permute",
        version="0.1.0",
        inputs={
            "x": Input(source=Reference("data", "x"), nulls="propagate"),
            "y": Input(source=Reference("data", "y"), nulls="propagate"),
        },
        params={"block_length": int(label[1:]), "seed": 42},
    )


def test_expand_variants_clones_downstream_and_rewires() -> None:
    strategy = make_strategy()

    expanded = expand_variants(
        strategy, at="data", build_variant=build_variant, labels=["b1", "b4"]
    )

    assert "data__b1" in expanded.strategy.nodes
    assert "x_lag__b1" in expanded.strategy.nodes
    assert "data__b4" in expanded.strategy.nodes
    assert "x_lag__b4" in expanded.strategy.nodes

    clone = expanded.strategy.nodes["x_lag__b1"]
    assert clone.inputs["x"].source.node == "data__b1"
    assert clone.inputs["x"].source.output == "x"

    original_clone = expanded.strategy.nodes["x_lag"]
    assert original_clone.inputs["x"].source.node == "data"


def test_expand_variants_preserves_baseline_outputs() -> None:
    strategy = make_strategy()

    expanded = expand_variants(
        strategy, at="data", build_variant=build_variant, labels=["b1"]
    )

    assert expanded.strategy.outputs["target"] == Reference("data", "y")
    assert expanded.strategy.outputs["temporal"] == Reference("x_lag", "x_lag")
    assert expanded.variant_outputs["target"] is None
    assert expanded.variant_outputs["temporal"] is None


def test_expand_variants_adds_variant_suffixed_outputs() -> None:
    strategy = make_strategy()

    expanded = expand_variants(
        strategy, at="data", build_variant=build_variant, labels=["b1", "b4"]
    )

    assert expanded.strategy.outputs["b1_target"] == Reference("data__b1", "y")
    assert expanded.strategy.outputs["b1_temporal"] == Reference("x_lag__b1", "x_lag")
    assert expanded.strategy.outputs["b4_target"] == Reference("data__b4", "y")
    assert expanded.strategy.outputs["b4_temporal"] == Reference("x_lag__b4", "x_lag")
    assert expanded.variant_outputs["b1_target"] == "b1"
    assert expanded.variant_outputs["b4_temporal"] == "b4"


def test_expand_variants_preserves_input_policies() -> None:
    strategy = make_strategy()

    expanded = expand_variants(
        strategy, at="data", build_variant=build_variant, labels=["b1"]
    )

    clone = expanded.strategy.nodes["x_lag__b1"]
    assert clone.inputs["x"].nulls is None
    assert clone.inputs["x"].tolerance is None


def test_expand_variants_preserves_node_params_and_materialize() -> None:
    strategy = Strategy(
        name="demo",
        nodes={
            "data": Node(op="source.csv_source", version="0.2.0"),
            "scale": Node(
                op="transform.negate",
                version="0.1.0",
                inputs={"value": Input(source=Reference("data", "x"))},
                params={"input_column": "x", "output_column": "neg_x"},
                materialize=True,
            ),
        },
        outputs={"out": Reference("scale", "neg_x")},
    )

    expanded = expand_variants(
        strategy,
        at="data",
        build_variant=lambda label: Node(
            op="perturb.block_permute",
            version="0.1.0",
            inputs={"x": Input(source=Reference("data", "x"))},
            params={"seed": 0},
        ),
        labels=["b1"],
    )

    clone = expanded.strategy.nodes["scale__b1"]
    assert clone.params == {"input_column": "x", "output_column": "neg_x"}
    assert clone.materialize is True


def test_expand_variants_preserves_upstream_nodes_unchanged() -> None:
    strategy = make_strategy()
    original_data = strategy.nodes["data"]

    expanded = expand_variants(
        strategy, at="data", build_variant=build_variant, labels=["b1"]
    )

    assert expanded.strategy.nodes["data"] is original_data


def test_expand_variants_rejects_unknown_injection_node() -> None:
    strategy = make_strategy()

    with pytest.raises(ValueError, match="node 'missing' not found"):
        expand_variants(
            strategy, at="missing", build_variant=build_variant, labels=["b1"]
        )


def test_expand_variants_rejects_empty_labels() -> None:
    strategy = make_strategy()

    with pytest.raises(ValueError, match="at least one variant label"):
        expand_variants(strategy, at="data", build_variant=build_variant, labels=[])


def test_expand_variants_rejects_duplicate_labels() -> None:
    strategy = make_strategy()

    with pytest.raises(ValueError, match="duplicate label"):
        expand_variants(
            strategy, at="data", build_variant=build_variant, labels=["b1", "b1"]
        )


def test_expand_variants_rejects_node_name_collisions() -> None:
    strategy = Strategy(
        name="demo",
        nodes={
            "data": Node(op="source.csv_source", version="0.2.0"),
            "data__b1": Node(op="source.csv_source", version="0.2.0"),
            "scale": Node(
                op="transform.negate",
                version="0.1.0",
                inputs={"value": Input(source=Reference("data", "x"))},
                params={"input_column": "x", "output_column": "neg_x"},
            ),
        },
        outputs={
            "out": Reference("scale", "neg_x"),
            "collision": Reference("data__b1", "x"),
        },
    )

    with pytest.raises(ValueError, match="already exists"):
        expand_variants(
            strategy, at="data", build_variant=build_variant, labels=["b1"]
        )


def test_expand_variants_rejects_non_node_return_from_builder() -> None:
    strategy = make_strategy()

    with pytest.raises(TypeError, match="build_variant must return a Node"):
        expand_variants(
            strategy, at="data", build_variant=lambda label: "not a node", labels=["b1"]
        )


def test_expand_variants_expanded_strategy_is_valid() -> None:
    strategy = make_strategy()

    expanded = expand_variants(
        strategy, at="data", build_variant=build_variant, labels=["b1", "b4"]
    )

    assert isinstance(expanded.strategy, Strategy)
    assert len(expanded.strategy.nodes) == 6
    assert len(expanded.strategy.outputs) == 6


def test_expand_variants_output_count_matches_expectation() -> None:
    strategy = make_strategy()

    expanded = expand_variants(
        strategy, at="data", build_variant=build_variant, labels=["b1", "b4"]
    )

    baseline = sum(1 for v in expanded.variant_outputs.values() if v is None)
    assert baseline == 2
    assert sum(1 for v in expanded.variant_outputs.values() if v is not None) == 4
