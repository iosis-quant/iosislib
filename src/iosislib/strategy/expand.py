"""Clone the downstream subgraph of a node once per variant label."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from iosislib.strategy.ir import (
    STRATEGY_FORMAT,
    STRATEGY_VERSION,
    Input,
    Node,
    Reference,
    Strategy,
    _check_keys,
    _fail,
    _identifier,
    _mapping,
    _semver,
)

EXPANSION_FORMAT = "iosis.expansion"
EXPANSION_VERSION = "0.1.0"

_EXPANSION_KEYS = {
    "format",
    "version",
    "name",
    "description",
    "metadata",
    "nodes",
    "outputs",
    "expand",
}
_EXPANSION_REQUIRED = {"format", "version", "name", "nodes", "outputs", "expand"}
_EXPAND_KEYS = {"at", "template", "variants"}
_VARIANT_KEYS = {"label", "op", "version", "inputs", "params", "materialize"}
_NODE_OVERRIDE_KEYS = ("op", "version", "inputs", "materialize")


def _downstream_closure(strategy: Strategy, start: str) -> frozenset[str]:
    consumers: dict[str, set[str]] = {}
    for name, node in strategy.nodes.items():
        for input_value in node.inputs.values():
            parent = input_value.source.node
            consumers.setdefault(parent, set()).add(name)
    downstream: set[str] = set()
    stack = list(consumers.get(start, ()))
    while stack:
        current = stack.pop()
        if current in downstream:
            continue
        downstream.add(current)
        stack.extend(consumers.get(current, ()))
    return frozenset(downstream)


def _rewire_input(input_value: Input, parent_map: Mapping[str, str]) -> Input:
    new_parent = parent_map.get(input_value.source.node)
    if new_parent is None:
        return input_value
    return Input(
        source=Reference(node=new_parent, output=input_value.source.output),
        tolerance=input_value.tolerance,
        nulls=input_value.nulls,
        fill=input_value.fill,
    )


@dataclass(frozen=True, slots=True)
class ExpandedStrategy:
    strategy: Strategy
    variant_outputs: Mapping[str, str | None] = field(default_factory=dict)

    @classmethod
    def from_data(cls, value: object) -> ExpandedStrategy:
        """Parse an ``iosis.expansion`` document into an expanded strategy."""
        data = _mapping(value, "$")
        _check_keys(
            data,
            "$",
            allowed=_EXPANSION_KEYS,
            required=_EXPANSION_REQUIRED,
        )
        if data["format"] != EXPANSION_FORMAT:
            _fail(
                "$",
                f"expected format {EXPANSION_FORMAT!r}, got {data['format']!r}",
            )
        version = _semver(data["version"], "$.version")
        if version != EXPANSION_VERSION:
            _fail(
                "$",
                f"expected version {EXPANSION_VERSION!r}, got {version!r}",
            )
        name = data["name"]
        if not isinstance(name, str) or not name.strip():
            _fail("$.name", "expected a non-empty string")

        strategy_doc: dict[str, object] = {
            "format": STRATEGY_FORMAT,
            "version": STRATEGY_VERSION,
            "name": name,
            "nodes": data["nodes"],
            "outputs": data["outputs"],
        }
        description = data.get("description")
        if description is not None:
            strategy_doc["description"] = description
        if "metadata" in data:
            strategy_doc["metadata"] = data["metadata"]
        strategy = Strategy.from_data(strategy_doc)
        return _expand_from(data["expand"], strategy)


def _merge_variant(
    template: Mapping[str, object],
    variant: Mapping[str, object],
    path: str,
) -> dict[str, object]:
    merged = dict(template)
    for key in _NODE_OVERRIDE_KEYS:
        if key in variant:
            merged[key] = variant[key]
    if "params" in variant:
        overrides = _mapping(variant["params"], f"{path}.params")
        base_params = template.get("params")
        merged_params = dict(base_params) if isinstance(base_params, Mapping) else {}
        merged_params.update(overrides)
        merged["params"] = merged_params
    return merged


def _expand_from(value: object, strategy: Strategy) -> ExpandedStrategy:
    expand = _mapping(value, "$.expand")
    _check_keys(expand, "$.expand", allowed=_EXPAND_KEYS, required=_EXPAND_KEYS)
    at = _identifier(expand["at"], "$.expand.at")
    if at not in strategy.nodes:
        _fail("$.expand.at", f"references unknown node {at!r}")

    template_data = _mapping(expand["template"], "$.expand.template")
    Node.from_data(template_data, "$.expand.template")

    variants = expand["variants"]
    if isinstance(variants, (str, bytes)) or not isinstance(variants, Sequence):
        _fail("$.expand.variants", "expected a list of variants")
    if not variants:
        _fail("$.expand.variants", "must contain at least one variant")

    labels: list[str] = []
    nodes_by_label: dict[str, Node] = {}
    for index, raw in enumerate(variants):
        path = f"$.expand.variants[{index}]"
        variant = _mapping(raw, path)
        _check_keys(variant, path, allowed=_VARIANT_KEYS, required={"label"})
        label = _identifier(variant["label"], f"{path}.label")
        if label in nodes_by_label:
            _fail(f"{path}.label", f"duplicate variant label {label!r}")
        merged = _merge_variant(template_data, variant, path)
        nodes_by_label[label] = Node.from_data(merged, path)
        labels.append(label)

    def build_variant(label: str) -> Node:
        return nodes_by_label[label]

    return expand_variants(strategy, at=at, build_variant=build_variant, labels=labels)


def expand_variants(
    strategy: Strategy,
    *,
    at: str,
    build_variant: Callable[[str], Node],
    labels: Sequence[str],
) -> ExpandedStrategy:
    if at not in strategy.nodes:
        raise ValueError(f"node {at!r} not found in strategy nodes")
    if not labels:
        raise ValueError("labels must contain at least one variant label")
    seen_labels: set[str] = set()
    for label in labels:
        if not isinstance(label, str) or not label:
            raise ValueError("each label must be a non-empty string")
        if label in seen_labels:
            raise ValueError(f"duplicate label {label!r}")
        seen_labels.add(label)

    downstream = _downstream_closure(strategy, at)
    baseline_outputs = dict(strategy.outputs)

    new_nodes: dict[str, Node] = dict(strategy.nodes)
    new_outputs: dict[str, Reference] = dict(baseline_outputs)
    variant_map: dict[str, str | None] = {name: None for name in baseline_outputs}

    for label in labels:
        variant_name = f"{at}__{label}"
        if variant_name in new_nodes:
            raise ValueError(f"generated node name {variant_name!r} already exists")
        variant_node = build_variant(label)
        if not isinstance(variant_node, Node):
            raise TypeError("build_variant must return a Node")
        new_nodes[variant_name] = variant_node

        parent_map: dict[str, str] = {at: variant_name}
        parent_map.update({name: f"{name}__{label}" for name in downstream})

        for name in sorted(downstream):
            clone_name = parent_map[name]
            if clone_name in new_nodes:
                raise ValueError(
                    f"generated node name {clone_name!r} already exists"
                )
            original = strategy.nodes[name]
            rewired_inputs = {
                input_name: _rewire_input(input_value, parent_map)
                for input_name, input_value in original.inputs.items()
            }
            new_nodes[clone_name] = Node(
                op=original.op,
                version=original.version,
                inputs=rewired_inputs,
                params=original.params,
                materialize=original.materialize,
            )

        for output_name, reference in baseline_outputs.items():
            new_parent = parent_map.get(reference.node, reference.node)
            variant_output = f"{label}_{output_name}"
            if variant_output in new_outputs:
                raise ValueError(
                    f"generated output name {variant_output!r} already exists"
                )
            new_outputs[variant_output] = Reference(
                node=new_parent, output=reference.output
            )
            variant_map[variant_output] = label

    expanded = Strategy(
        name=strategy.name,
        nodes=new_nodes,
        outputs=new_outputs,
        description=strategy.description,
        metadata=strategy.metadata,
    )
    return ExpandedStrategy(strategy=expanded, variant_outputs=variant_map)


__all__ = [
    "EXPANSION_FORMAT",
    "EXPANSION_VERSION",
    "ExpandedStrategy",
    "expand_variants",
]
