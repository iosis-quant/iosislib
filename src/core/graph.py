from __future__ import annotations

import abc
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping

import polars as pl

from src.core.node import Node
from src.core.tsfn import (
    NullPolicy,
    _column_signature_map,
    _column_signature_matches,
    _format_column_signature,
)
from src.core.utils import AsofTolerance, _dtype_matches


class Executor(abc.ABC):
    """Lower and execute a verified graph with time-aware input alignment."""

    def execute(self, graph: Graph) -> pl.DataFrame:
        graph.verify()
        root_lf = self._evaluate_to_root(graph)
        return self.materialize(graph.root_node, root_lf)

    def _evaluate_to_root(self, graph: Graph) -> pl.LazyFrame:
        """Evaluate required boundaries and return the root lazy value."""
        graph.verify()
        results: dict[str, pl.LazyFrame] = {}

        for node in graph.node_list:
            try:
                node_input_lf = (
                    None
                    if not node.bindings
                    else self.align_inputs(node, results)
                )
                results[node.ID] = self.lower_node(node, node_input_lf)
            except Exception as exc:
                raise RuntimeError(
                    f"Execution failed at node '{node.name or node.ID[:8]}' "
                    f"({node.function_cls.__name__}@{node.function.version}): {exc}"
                ) from exc

            if (
                node.ID in graph.materialized_node_ids
                and node.ID != graph.root_node.ID
            ):
                results[node.ID] = self.materialize(node, results[node.ID]).lazy()

        return results[graph.root_node.ID]

    def lower_node(
        self,
        node: Node,
        input_lf: pl.LazyFrame | None,
    ) -> pl.LazyFrame:
        if input_lf is None:
            return node.function()
        return node.function(input_lf)

    def align_inputs(
        self,
        node: Node,
        results: Mapping[str, pl.LazyFrame],
    ) -> pl.LazyFrame:
        """Build a union timeline and backward-asof align a node's inputs."""
        parent_to_bindings: dict[
            tuple[Node, AsofTolerance],
            list[tuple[str, str]],
        ] = defaultdict(list)
        for input_name, (parent_node, parent_column) in node.bindings.items():
            tolerance = node.tolerances.get(input_name)
            parent_to_bindings[(parent_node, tolerance)].append(
                (parent_column, input_name)
            )

        input_time = node.function.signature[0].time
        if input_time is None:
            raise ValueError(
                f"Bound node '{node.name or node.ID}' must declare an input time axis"
            )
        time_column = input_time.column

        parent_frames: list[tuple[pl.LazyFrame, AsofTolerance]] = []
        for (parent_node, tolerance), bindings in parent_to_bindings.items():
            parent_lf = results[parent_node.ID]
            parent_time = parent_node.function.signature[1].time
            if parent_time is None:
                raise ValueError(
                    f"Parent node '{parent_node.name or parent_node.ID}' "
                    "must declare an output time axis"
                )
            select_expressions = [pl.col(parent_time.column).alias(time_column)]
            select_expressions.extend(
                pl.col(parent_column).alias(input_name)
                for parent_column, input_name in bindings
            )
            parent_frames.append(
                (parent_lf.select(select_expressions).sort(time_column), tolerance)
            )

        node_input_lf = (
            pl.concat(
                [parent_lf.select(time_column) for parent_lf, _ in parent_frames],
                how="vertical",
            )
            .unique()
            .sort(time_column)
        )
        for parent_lf, tolerance in parent_frames:
            node_input_lf = node_input_lf.join_asof(
                parent_lf,
                on=time_column,
                strategy="backward",
                tolerance=tolerance,
            )
        return node_input_lf

    @abc.abstractmethod
    def materialize(self, node: Node, lf: pl.LazyFrame) -> pl.DataFrame:
        """Materialize one graph boundary into the executor's local table type."""
        pass


class LocalExecutor(Executor):
    """Execute a graph on one machine using Polars' local query engine."""

    def materialize(self, node: Node, lf: pl.LazyFrame) -> pl.DataFrame:
        try:
            return lf.collect()
        except Exception as exc:
            raise RuntimeError(
                f"Execution failed while materializing node "
                f"'{node.name or node.ID[:8]}' "
                f"({node.function_cls.__name__}@{node.function.version}): {exc}"
            ) from exc


class Graph:
    def __init__(
        self,
        root_node: Node,
        *,
        executor: Executor | None = None,
    ):
        if not isinstance(root_node, Node):
            raise TypeError("Graph root_node must be a Node")
        if executor is not None and not isinstance(executor, Executor):
            raise TypeError("Graph executor must be an Executor")

        self.root_node = root_node
        self._declared_nodes = self.get_declared_nodes(self.root_node)
        self.node_list = self.get_subgraph_execution_order(self.root_node)
        self.executor = LocalExecutor() if executor is None else executor
        self.materialized_node_ids = frozenset(
            node.ID for node in self._declared_nodes if node.materialize
        )

        self.verify()
        self.ID = self._generate_persistent_id()

    def __repr__(self) -> str:
        return (
            f"Graph(id={self.ID[:8]!r}, root={self.root_node.ID[:8]!r}, "
            f"nodes={len(self.node_list)}, "
            f"executor={self.executor.__class__.__name__})"
        )

    def verify(self) -> None:
        """Verify graph structure and type contracts without executing."""
        self._validate_materializations()
        self._validate_graph()

    def _validate_materializations(self) -> None:
        for node in self._declared_nodes:
            if node.function.requires_materialization and not node.materialize:
                raise ValueError(
                    f"Materialization validation failed for node "
                    f"'{node.name or node.ID}': {node.function_cls.__name__} "
                    "requires materialization"
                )

    def get_declared_nodes(self, target_node: Node) -> tuple[Node, ...]:
        """Return every declaration, including semantically duplicate nodes."""
        visited_objects: set[int] = set()
        declared_nodes: list[Node] = []

        def visit(node: Node) -> None:
            object_id = id(node)
            if object_id in visited_objects:
                return
            visited_objects.add(object_id)
            for parent, _ in node.bindings.values():
                visit(parent)
            declared_nodes.append(node)

        visit(target_node)
        return tuple(declared_nodes)

    def get_subgraph_execution_order(self, target_node: Node) -> list[Node]:
        visited: set[str] = set()
        visiting: set[str] = set()
        ordered_nodes: list[Node] = []

        def dfs(node: Node) -> None:
            if node.ID in visiting:
                raise ValueError(
                    f"Cycle detected at node '{node.name or node.ID}'! "
                    "Graphs must be acyclic."
                )
            if node.ID in visited:
                return

            visiting.add(node.ID)
            for parent_node in node.inputs:
                dfs(parent_node)
            visiting.remove(node.ID)
            visited.add(node.ID)
            ordered_nodes.append(node)

        dfs(target_node)
        return ordered_nodes

    def _validate_graph(self) -> None:
        """Validate bindings, outputs, and type compatibility."""
        node_ids = {graph_node.ID for graph_node in self.node_list}

        for node in self.node_list:
            input_signature = node.function.signature[0]
            input_columns = _column_signature_map(input_signature)
            expected_inputs = set(input_columns)
            bound_inputs = set(node.bindings)
            tolerance_inputs = set(node.tolerances)
            null_handler_inputs = set(node.null_handlers)
            null_fill_inputs = set(node.null_fill_values)

            if not node.bindings and not input_signature.is_empty():
                raise ValueError(
                    f"Binding validation failed for node '{node.name or node.ID}'. "
                    "Nodes with no predecessors must declare an empty input signature."
                )

            extra_tolerances = tolerance_inputs - bound_inputs
            if extra_tolerances:
                raise ValueError(
                    f"Binding validation failed for node '{node.name or node.ID}'. "
                    f"Unexpected tolerances for unbound inputs: {sorted(extra_tolerances)}"
                )

            extra_null_handlers = null_handler_inputs - expected_inputs
            if extra_null_handlers:
                raise ValueError(
                    f"Binding validation failed for node '{node.name or node.ID}'. "
                    f"Unexpected null handlers for inputs: {sorted(extra_null_handlers)}"
                )

            extra_null_fill_values = null_fill_inputs - expected_inputs
            if extra_null_fill_values:
                raise ValueError(
                    f"Binding validation failed for node '{node.name or node.ID}'. "
                    f"Unexpected null fill values for inputs: "
                    f"{sorted(extra_null_fill_values)}"
                )

            missing_fill_values = {
                input_name
                for input_name, handler in node.null_handlers.items()
                if handler.policy is NullPolicy.FILL
                and input_name not in node.null_fill_values
            }
            if missing_fill_values:
                raise ValueError(
                    f"Binding validation failed for node '{node.name or node.ID}'. "
                    "NullPolicy.FILL requires null_fill_values for inputs: "
                    f"{sorted(missing_fill_values)}"
                )

            missing = expected_inputs - bound_inputs
            if missing:
                raise ValueError(
                    f"Binding validation failed for node '{node.name or node.ID}'. "
                    f"Missing expected inputs: {sorted(missing)}"
                )

            extra = bound_inputs - expected_inputs
            if extra:
                raise ValueError(
                    f"Binding validation failed for node '{node.name or node.ID}'. "
                    f"Unexpected bound inputs: {sorted(extra)}"
                )

            if node.bindings and input_signature.time is None:
                raise ValueError(
                    f"Binding validation failed for node '{node.name or node.ID}'. "
                    "Bound nodes must declare an input time axis."
                )

            for input_name, (parent_node, parent_column) in node.bindings.items():
                if parent_node.ID not in node_ids:
                    raise ValueError(
                        f"Node '{node.name or node.ID}' binds to parent "
                        f"'{parent_node.name or parent_node.ID}' outside this graph."
                    )

                parent_outputs = _column_signature_map(
                    parent_node.function.signature[1]
                )
                if parent_column not in parent_outputs:
                    raise ValueError(
                        f"Binding validation failed for node '{node.name or node.ID}'. "
                        f"Parent node '{parent_node.name or parent_node.ID}' does not "
                        f"expose output '{parent_column}'. Available outputs: "
                        f"{list(parent_outputs)}"
                    )

                expected_column = input_columns[input_name]
                actual_column = parent_outputs[parent_column]
                if not _column_signature_matches(actual_column, expected_column):
                    raise TypeError(
                        f"Type mismatch at node '{node.name or node.ID}' for input "
                        f"'{input_name}': expected "
                        f"{_format_column_signature(expected_column)}, got "
                        f"{_format_column_signature(actual_column)} from "
                        f"'{parent_node.name or parent_node.ID}.{parent_column}'"
                    )

                self._validate_time_axis_compatibility(node, parent_node)

    def _validate_time_axis_compatibility(
        self,
        node: Node,
        parent_node: Node,
    ) -> None:
        child_time = node.function.signature[0].time
        parent_time = parent_node.function.signature[1].time

        if child_time is None:
            raise ValueError(
                f"Bound node '{node.name or node.ID}' must declare an input time axis"
            )
        if parent_time is None:
            raise ValueError(
                f"Parent node '{parent_node.name or parent_node.ID}' must declare "
                "an output time axis"
            )

        if parent_time.column != child_time.column:
            raise ValueError(
                f"Time axis mismatch at node '{node.name or node.ID}': "
                f"expected parent time column '{child_time.column}', got "
                f"'{parent_time.column}' from "
                f"'{parent_node.name or parent_node.ID}'"
            )

        if not _dtype_matches(parent_time.dtype, child_time.dtype):
            raise TypeError(
                f"Time axis dtype mismatch at node '{node.name or node.ID}': "
                f"expected {child_time.dtype}, got {parent_time.dtype} from "
                f"'{parent_node.name or parent_node.ID}'"
            )

        if parent_time.timezone != child_time.timezone:
            raise TypeError(
                f"Time axis timezone mismatch at node '{node.name or node.ID}': "
                f"expected {child_time.timezone}, got {parent_time.timezone} from "
                f"'{parent_node.name or parent_node.ID}'"
            )

    def _generate_persistent_id(self) -> str:
        graph_definition = {
            "root_id": self.root_node.ID,
            "nodes": sorted(node.ID for node in self.node_list),
        }
        serialized_data = json.dumps(graph_definition, sort_keys=True)
        return hashlib.sha256(serialized_data.encode("utf-8")).hexdigest()

    def execute(self, executor: Executor | None = None) -> pl.DataFrame:
        selected_executor = self.executor if executor is None else executor
        if not isinstance(selected_executor, Executor):
            raise TypeError("Graph executor must be an Executor")
        return selected_executor.execute(self)


__all__ = ["Executor", "Graph", "LocalExecutor"]
