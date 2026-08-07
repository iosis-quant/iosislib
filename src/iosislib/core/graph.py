from __future__ import annotations

import abc
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import polars as pl

from iosislib.core.node import Node
from iosislib.core.tsfn import (
    NullHandler,
    NullPolicy,
    _column_signature_map,
    _column_signature_matches,
    _format_column_signature,
    _format_frame_signature,
)
from iosislib.core.utils import (
    AsofTolerance,
    S3CredentialsProvider,
    _dtype_matches,
    _format_tolerance,
    _s3_credentials_scope,
    _serialize_value,
)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One graph invariant violation with stable declaration context."""

    code: str
    category: str
    message: str
    node_id: str
    node_name: str | None
    tsfn_class: str
    tsfn_version: str
    input_name: str | None = None
    output_name: str | None = None
    _node_position: int = field(default=-1, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "category": self.category,
            "message": self.message,
            "node_id": self.node_id,
            "node_name": self.node_name,
            "tsfn_class": self.tsfn_class,
            "tsfn_version": self.tsfn_version,
            "input_name": self.input_name,
            "output_name": self.output_name,
        }

    def _sort_key(self) -> tuple[Any, ...]:
        return (
            self._node_position,
            self.code,
            self.input_name or "",
            self.output_name or "",
            self.node_name or "",
            self.message,
        )


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Deterministically ordered graph validation results."""

    issues: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        unique: dict[ValidationIssue, ValidationIssue] = {}
        for issue in self.issues:
            current = unique.get(issue)
            if current is None or issue._sort_key() < current._sort_key():
                unique[issue] = issue
        normalized = tuple(
            sorted(unique.values(), key=ValidationIssue._sort_key)
        )
        object.__setattr__(self, "issues", normalized)

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "issues": [issue.to_dict() for issue in self.issues],
        }


class GraphValidationError(ValueError):
    """Raised when graph construction or verification finds invalid declarations."""

    def __init__(self, report: ValidationReport):
        self.report = report
        issue_lines = "\n".join(
            f"  {index}. [{issue.code}] {issue.message}"
            for index, issue in enumerate(report.issues, start=1)
        )
        super().__init__(
            f"Graph validation failed with {len(report.issues)} issue(s):\n"
            f"{issue_lines}"
        )


class Executor(abc.ABC):
    """Lower and execute a verified graph with time-aware input alignment."""

    def __init__(
        self,
        s3_credentials: S3CredentialsProvider | None = None,
    ) -> None:
        """Optionally scope explicit S3 credentials for the whole execution.

        ``s3_credentials`` may be an ``S3Credentials`` value or a provider
        callable that returns fresh credentials (or ``None``) each execution,
        so a long-running process can keep temporary task-role credentials
        valid. Credentials are execution state only and never contribute to
        node or graph identity.
        """
        self._s3_credentials: S3CredentialsProvider | None = s3_credentials

    def execute(self, graph: Graph) -> pl.DataFrame:
        with _s3_credentials_scope(self._s3_credentials):
            root_lf = self._evaluate_to_root(graph)
            return self.materialize(graph.root_node, root_lf)

    def _evaluate_to_root(self, graph: Graph) -> pl.LazyFrame:
        """Evaluate required boundaries and return the root lazy value."""
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
        for input_name, (parent_node, parent_column) in sorted(
            node.bindings.items()
        ):
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
    __slots__ = (
        "ID",
        "_frozen",
        "_validation_report",
        "materialized_node_ids",
        "node_list",
        "root_node",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("Graph instances are immutable")
        object.__setattr__(self, name, value)

    def __init__(self, root_node: Node):
        if not isinstance(root_node, Node):
            raise TypeError("Graph root_node must be a Node")

        node_list, report = self._validated_declaration(root_node)
        if not report.is_valid:
            raise GraphValidationError(report)

        object.__setattr__(self, "root_node", root_node)
        object.__setattr__(self, "node_list", node_list)
        object.__setattr__(self, "_validation_report", report)
        object.__setattr__(
            self,
            "materialized_node_ids",
            frozenset(node.ID for node in node_list if node.materialize),
        )
        object.__setattr__(self, "ID", self._generate_persistent_id())
        object.__setattr__(self, "_frozen", True)

    def __repr__(self) -> str:
        return (
            f"Graph(id={self.ID[:8]!r}, root={self.root_node.ID[:8]!r}, "
            f"nodes={len(self.node_list)})"
        )

    def verify(self) -> None:
        """Consult the immutable validation result established at construction."""
        if not self._validation_report.is_valid:
            raise GraphValidationError(self._validation_report)

    @classmethod
    def validate(cls, root_node: Node) -> ValidationReport:
        """Return all safely diagnosable declaration issues without executing TSFNs."""
        if not isinstance(root_node, Node):
            raise TypeError("Graph root_node must be a Node")

        _, report = cls._validated_declaration(root_node)
        return report

    @classmethod
    def _validated_declaration(
        cls,
        root_node: Node,
    ) -> tuple[tuple[Node, ...], ValidationReport]:
        ordered_nodes, cycle = cls._dependency_order(root_node)
        if cycle is not None:
            cycle_node = cycle[-1]
            cycle_path = " -> ".join(node.name or node.ID for node in cycle)
            issue = cls._issue(
                cycle_node,
                code="CYCLE",
                category="structure",
                message=f"Cycle detected: {cycle_path}. Graphs must be acyclic.",
            )
            return ordered_nodes, ValidationReport(
                (replace(issue, _node_position=0),)
            )

        issues: list[ValidationIssue] = []
        for node in ordered_nodes:
            if node.function.requires_materialization and not node.materialize:
                issues.append(
                    cls._issue(
                        node,
                        code="REQUIRED_MATERIALIZATION_DISABLED",
                        category="materialization",
                        message=(
                            f"Materialization validation failed for node "
                            f"'{node.name or node.ID}': {node.function_cls.__name__} "
                            "requires materialization"
                        ),
                    )
                )

        node_ids = {node.ID for node in ordered_nodes}
        for node in ordered_nodes:
            cls._collect_node_issues(node, node_ids, issues)

        positions = {node.ID: index for index, node in enumerate(ordered_nodes)}
        positioned_issues = tuple(
            replace(
                issue,
                _node_position=positions.get(issue.node_id, len(ordered_nodes)),
            )
            for issue in issues
        )
        return ordered_nodes, ValidationReport(positioned_issues)

    @classmethod
    def _dependency_order(
        cls,
        target_node: Node,
    ) -> tuple[tuple[Node, ...], tuple[Node, ...] | None]:
        """Return one canonical, ID-deduplicated dependency traversal."""
        visited: set[str] = set()
        visiting: dict[str, int] = {}
        stack: list[Node] = []
        ordered_nodes: list[Node] = []
        cycle: tuple[Node, ...] | None = None

        def dfs(node: Node) -> None:
            nonlocal cycle
            if cycle is not None:
                return
            if node.ID in visiting:
                cycle = tuple(stack[visiting[node.ID] :] + [node])
                return
            if node.ID in visited:
                return

            visiting[node.ID] = len(stack)
            stack.append(node)
            for parent_node in cls._canonical_parents(node):
                dfs(parent_node)
            stack.pop()
            visiting.pop(node.ID)
            if cycle is not None:
                return
            visited.add(node.ID)
            ordered_nodes.append(node)

        dfs(target_node)
        return tuple(ordered_nodes), cycle

    @staticmethod
    def _canonical_parents(node: Node) -> tuple[Node, ...]:
        """Order declared parents by consuming input, deduplicating by Node ID."""
        declared_parent_ids = {parent.ID for parent in node.inputs}
        parent_entries: dict[str, tuple[str, Node]] = {}

        for input_name, (parent, _) in sorted(node.bindings.items()):
            if parent.ID in declared_parent_ids:
                parent_entries.setdefault(parent.ID, (input_name, parent))

        for parent in node.inputs:
            parent_entries.setdefault(parent.ID, ("\uffff", parent))

        return tuple(
            parent
            for _, parent in sorted(
                parent_entries.values(),
                key=lambda entry: (entry[0], entry[1].ID),
            )
        )

    @classmethod
    def _collect_node_issues(
        cls,
        node: Node,
        node_ids: set[str],
        issues: list[ValidationIssue],
    ) -> None:
        input_signature = node.function.signature[0]
        input_columns = _column_signature_map(input_signature)
        expected_inputs = set(input_columns)
        bound_inputs = set(node.bindings)

        if not node.bindings and not input_signature.is_empty():
            issues.append(
                cls._issue(
                    node,
                    code="NON_EMPTY_INPUT_WITHOUT_BINDINGS",
                    category="binding",
                    message=(
                        f"Binding validation failed for node '{node.name or node.ID}'. "
                        "Nodes with no predecessors must declare an empty input "
                        "signature."
                    ),
                )
            )

        cls._collect_unexpected_input_metadata(
            node,
            set(node.tolerances) - bound_inputs,
            code="UNEXPECTED_TOLERANCE",
            label="Unexpected tolerances for unbound inputs",
            issues=issues,
        )
        cls._collect_unexpected_input_metadata(
            node,
            set(node.null_handlers) - expected_inputs,
            code="UNEXPECTED_NULL_HANDLER",
            label="Unexpected null handlers for inputs",
            issues=issues,
        )
        cls._collect_unexpected_input_metadata(
            node,
            set(node.null_fill_values) - expected_inputs,
            code="UNEXPECTED_NULL_FILL_VALUE",
            label="Unexpected null fill values for inputs",
            issues=issues,
        )

        for input_name in sorted(node.null_handlers):
            handler = node.null_handlers[input_name]
            if (
                handler.policy is NullPolicy.FILL
                and input_name not in node.null_fill_values
            ):
                issues.append(
                    cls._issue(
                        node,
                        code="MISSING_NULL_FILL_VALUE",
                        category="null_policy",
                        message=(
                            f"Binding validation failed for node "
                            f"'{node.name or node.ID}'. NullPolicy.FILL requires "
                            f"null_fill_values for inputs: ['{input_name}']"
                        ),
                        input_name=input_name,
                    )
                )

        if node.bindings:
            for input_name in sorted(expected_inputs - bound_inputs):
                issues.append(
                    cls._issue(
                        node,
                        code="MISSING_BINDING",
                        category="binding",
                        message=(
                            f"Binding validation failed for node "
                            f"'{node.name or node.ID}'. Missing expected inputs: "
                            f"['{input_name}']"
                        ),
                        input_name=input_name,
                    )
                )

        for input_name in sorted(bound_inputs - expected_inputs):
            issues.append(
                cls._issue(
                    node,
                    code="UNEXPECTED_BINDING",
                    category="binding",
                    message=(
                        f"Binding validation failed for node '{node.name or node.ID}'. "
                        f"Unexpected bound inputs: ['{input_name}']"
                    ),
                    input_name=input_name,
                )
            )

        child_has_time = input_signature.time is not None
        if node.bindings and not child_has_time:
            issues.append(
                cls._issue(
                    node,
                    code="BOUND_INPUT_TIME_AXIS_MISSING",
                    category="time_axis",
                    message=(
                        f"Binding validation failed for node '{node.name or node.ID}'. "
                        "Bound nodes must declare an input time axis."
                    ),
                )
            )

        for input_name, (parent_node, parent_column) in sorted(node.bindings.items()):
            if parent_node.ID not in node_ids:
                issues.append(
                    cls._issue(
                        node,
                        code="PARENT_OUTSIDE_GRAPH",
                        category="structure",
                        message=(
                            f"Node '{node.name or node.ID}' binds to parent "
                            f"'{parent_node.name or parent_node.ID}' outside this graph."
                        ),
                        input_name=input_name,
                        output_name=parent_column,
                    )
                )
                continue

            parent_outputs = _column_signature_map(parent_node.function.signature[1])
            if parent_column not in parent_outputs:
                issues.append(
                    cls._issue(
                        node,
                        code="PARENT_OUTPUT_MISSING",
                        category="binding",
                        message=(
                            f"Binding validation failed for node "
                            f"'{node.name or node.ID}'. Parent node "
                            f"'{parent_node.name or parent_node.ID}' does not expose "
                            f"output '{parent_column}'. Available outputs: "
                            f"{list(parent_outputs)}"
                        ),
                        input_name=input_name,
                        output_name=parent_column,
                    )
                )
                continue

            if input_name not in input_columns:
                continue

            expected_column = input_columns[input_name]
            actual_column = parent_outputs[parent_column]
            if not _column_signature_matches(actual_column, expected_column):
                issues.append(
                    cls._issue(
                        node,
                        code="INPUT_TYPE_MISMATCH",
                        category="type",
                        message=(
                            f"Type mismatch at node '{node.name or node.ID}' for input "
                            f"'{input_name}': expected "
                            f"{_format_column_signature(expected_column)}, got "
                            f"{_format_column_signature(actual_column)} from "
                            f"'{parent_node.name or parent_node.ID}.{parent_column}'"
                        ),
                        input_name=input_name,
                        output_name=parent_column,
                    )
                )

            if child_has_time:
                cls._collect_time_axis_issues(
                    node,
                    parent_node,
                    input_name,
                    parent_column,
                    issues,
                )

    @classmethod
    def _collect_unexpected_input_metadata(
        cls,
        node: Node,
        input_names: set[str],
        *,
        code: str,
        label: str,
        issues: list[ValidationIssue],
    ) -> None:
        for input_name in sorted(input_names):
            issues.append(
                cls._issue(
                    node,
                    code=code,
                    category="binding",
                    message=(
                        f"Binding validation failed for node '{node.name or node.ID}'. "
                        f"{label}: ['{input_name}']"
                    ),
                    input_name=input_name,
                )
            )

    @classmethod
    def _collect_time_axis_issues(
        cls,
        node: Node,
        parent_node: Node,
        input_name: str,
        parent_column: str,
        issues: list[ValidationIssue],
    ) -> None:
        child_time = node.function.signature[0].time
        parent_time = parent_node.function.signature[1].time

        assert child_time is not None
        if parent_time is None:
            issues.append(
                cls._issue(
                    node,
                    code="PARENT_OUTPUT_TIME_AXIS_MISSING",
                    category="time_axis",
                    message=(
                        f"Parent node '{parent_node.name or parent_node.ID}' must "
                        "declare an output time axis"
                    ),
                    input_name=input_name,
                    output_name=parent_column,
                )
            )
            return

        if parent_time.column != child_time.column:
            issues.append(
                cls._issue(
                    node,
                    code="TIME_COLUMN_MISMATCH",
                    category="time_axis",
                    message=(
                        f"Time axis mismatch at node '{node.name or node.ID}': "
                        f"expected parent time column '{child_time.column}', got "
                        f"'{parent_time.column}' from "
                        f"'{parent_node.name or parent_node.ID}'"
                    ),
                    input_name=input_name,
                    output_name=parent_column,
                )
            )

        if not _dtype_matches(parent_time.dtype, child_time.dtype):
            issues.append(
                cls._issue(
                    node,
                    code="TIME_DTYPE_MISMATCH",
                    category="type",
                    message=(
                        f"Time axis dtype mismatch at node '{node.name or node.ID}': "
                        f"expected {child_time.dtype}, got {parent_time.dtype} from "
                        f"'{parent_node.name or parent_node.ID}'"
                    ),
                    input_name=input_name,
                    output_name=parent_column,
                )
            )

        if parent_time.timezone != child_time.timezone:
            issues.append(
                cls._issue(
                    node,
                    code="TIMEZONE_MISMATCH",
                    category="type",
                    message=(
                        f"Time axis timezone mismatch at node "
                        f"'{node.name or node.ID}': expected {child_time.timezone}, "
                        f"got {parent_time.timezone} from "
                        f"'{parent_node.name or parent_node.ID}'"
                    ),
                    input_name=input_name,
                    output_name=parent_column,
                )
            )

    @staticmethod
    def _issue(
        node: Node,
        *,
        code: str,
        category: str,
        message: str,
        input_name: str | None = None,
        output_name: str | None = None,
    ) -> ValidationIssue:
        return ValidationIssue(
            code=code,
            category=category,
            message=message,
            node_id=node.ID,
            node_name=node.name,
            tsfn_class=(
                f"{node.function_cls.__module__}.{node.function_cls.__qualname__}"
            ),
            tsfn_version=node.function.version,
            input_name=input_name,
            output_name=output_name,
        )

    def describe(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible graph metadata without executing."""
        return {
            "id": self.ID,
            "root_id": self.root_node.ID,
            "nodes": [self._describe_node(node) for node in self.node_list],
        }

    def _describe_node(self, node: Node) -> dict[str, Any]:
        input_signature, output_signature = node.function.signature
        input_columns = _column_signature_map(input_signature)
        reasons: list[str] = []
        declared_boundary = node.ID in self.materialized_node_ids
        if node.function.requires_materialization:
            reasons.append("tsfn_required")
        elif declared_boundary:
            reasons.append("node_requested")
        if node.ID == self.root_node.ID:
            reasons.append("root_result")

        return {
            "id": node.ID,
            "name": node.name,
            "function": {
                "module": node.function_cls.__module__,
                "qualname": node.function_cls.__qualname__,
                "version": node.function.version,
            },
            "parameters": _serialize_value(node.parameters.to_dict()),
            "input_signature": _format_frame_signature(input_signature),
            "output_signature": _format_frame_signature(output_signature),
            "bindings": {
                input_name: {
                    "parent_id": parent_node.ID,
                    "parent_name": parent_node.name,
                    "output": parent_column,
                }
                for input_name, (parent_node, parent_column) in sorted(
                    node.bindings.items()
                )
            },
            "tolerances": {
                input_name: _format_tolerance(node.tolerances.get(input_name))
                for input_name in sorted(node.bindings)
            },
            "null_handlers": {
                input_name: self._describe_null_handler(
                    node.null_handlers[input_name],
                    node.null_handler_versions.get(input_name),
                )
                for input_name in sorted(input_columns)
            },
            "null_fill_values": {
                input_name: _serialize_value(fill_value)
                for input_name, fill_value in sorted(node.null_fill_values.items())
            },
            "materialization": {
                "boundary": bool(reasons),
                "effective": node.materialize,
                "required_by_tsfn": node.function.requires_materialization,
                "declared_by_node": declared_boundary,
                "reasons": reasons,
            },
        }

    @staticmethod
    def _describe_null_handler(
        handler: NullHandler,
        version: str | None,
    ) -> dict[str, str]:
        if handler.policy is not None:
            return {"kind": "policy", "value": handler.policy.value}

        assert handler.function is not None
        assert version is not None
        return {
            "kind": "function",
            "module": handler.function.__module__,
            "qualname": handler.function.__qualname__,
            "version": version,
        }

    def _generate_persistent_id(self) -> str:
        graph_definition = {
            "root_id": self.root_node.ID,
            "nodes": tuple(node.ID for node in self.node_list),
        }
        serialized_data = json.dumps(graph_definition, sort_keys=True)
        return hashlib.sha256(serialized_data.encode("utf-8")).hexdigest()

    def execute(self, executor: Executor | None = None) -> pl.DataFrame:
        selected_executor = LocalExecutor() if executor is None else executor
        if not isinstance(selected_executor, Executor):
            raise TypeError("Graph executor must be an Executor")
        return selected_executor.execute(self)


__all__ = [
    "Executor",
    "Graph",
    "GraphValidationError",
    "LocalExecutor",
    "ValidationIssue",
    "ValidationReport",
]
