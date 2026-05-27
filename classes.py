from __future__ import annotations
import abc
import hashlib
import json
import pyarrow as pa
import numpy as np
from typing import Any, Type
from dataclasses import dataclass, fields, MISSING


class Schema:
    def __init__(self, columns: list[tuple[str, pa.DataType]]):
        self.columns = sorted(columns, key=lambda c: c[0])
        self._type_map: dict[str, pa.DataType] = dict(self.columns)

    def validate(self, table: pa.Table) -> bool:
        table_schema = table.schema
        for name, expected_type in self.columns:
            idx = table_schema.get_field_index(name)
            if idx == -1:
                raise ValueError(f"Schema violation! Missing required column: '{name}'")
            actual_type = table_schema.field(idx).type
            if actual_type != expected_type:
                raise TypeError(
                    f"Type mismatch on '{name}': expected {expected_type}, got {actual_type}"
                )
        return True

    def serialize(self) -> str:
        return json.dumps([[name, str(dtype)] for name, dtype in self.columns])


@dataclass(frozen=True)
class TSFNConfig(abc.ABC):
    pass

    def __str__(self):
        # 1. Sort the metadata fields first by name
        sorted_fields = sorted(fields(self), key=lambda f: f.name)

        # 2. Build a standard dict, ensuring inner dicts are also sorted by key
        canonical_map = {
            f.name: (
                {k: v for k, v in sorted(getattr(self, f.name).items())}
                if isinstance(getattr(self, f.name), dict)
                else getattr(self, f.name)
            )
            for f in sorted_fields
        }

        return str(canonical_map)


class TSFN(abc.ABC):
    CONFIG_CLS: Type[TSFNConfig] = TSFNConfig

    def __init__(
        self,
        input_cols: list[tuple[str, pa.DataType]],
        output_cols: list[tuple[str, pa.DataType]],
        parameters: dict[str, Any],
    ):
        self.input_schema = Schema(input_cols)
        self.output_schema = Schema(output_cols)

        self.parameters = self._bind_and_validate_config(parameters)

    def _bind_and_validate_config(self, params: dict[str, Any]) -> TSFNConfig:
        """Filters environment variables and extracts missing required parameters."""
        # 1. Map fields and find which ones are strictly required
        config_fields = fields(self.CONFIG_CLS)
        allowed_fields = {f.name for f in config_fields}

        # A field is required if it has no default value and no default factory
        required_fields = {
            f.name
            for f in config_fields
            if f.default is MISSING and f.default_factory is MISSING
        }

        # 2. Extract the intersection of provided keys and expected configuration fields
        filtered_params = {k: v for k, v in params.items() if k in allowed_fields}

        # 3. Check for any missing required parameters before attempting initialization
        missing_keys = required_fields - filtered_params.keys()
        if missing_keys:
            raise ValueError(
                f"Parameter validation failed for {self.__class__.__name__}. "
                f"Missing required parameters with no default values: {sorted(missing_keys)}. "
                f"Expected schema: {self.CONFIG_CLS.__annotations__}"
            )

        try:
            return self.CONFIG_CLS(**filtered_params)
        except Exception as e:
            raise e

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(in:{self.input_schema.serialize()}, out:{self.output_schema.serialize()})"

    def __call__(self, table: pa.Table) -> pa.Table:
        """Pipeline boundary: Validates native PyArrow tables without overhead."""
        self.input_schema.validate(table)
        # apply method is where we execute our lags, windows, functions etc
        result = self.apply(table)
        self.output_schema.validate(result)
        return result

    @abc.abstractmethod
    def apply(self, table: pa.Table) -> pa.Table:
        """Override this method for raw PyArrow transformations if needed."""
        pass


class Node:
    def __init__(
        self,
        inputs: list[Node],
        function_cls: type[TSFN],
        input_cols: list[tuple[str, pa.DataType]],
        output_cols: list[tuple[str, pa.DataType]],
        parameters: dict[str, Any] | None,
        name: str | None,
    ):
        self.inputs = tuple(inputs)
        self.name = name

        self.function_cls = function_cls
        self.function = function_cls(input_cols, output_cols, parameters or {})
        self.parameters = self.function.parameters

        self.outputs: tuple[tuple[str, pa.DataType], ...] = tuple(
            sorted(self.function.output_schema.columns, key=lambda c: c[0])
        )

        self._verify_schema_compatibility()
        self.ID = self._generate_persistent_id()


    def _verify_schema_compatibility(self):
        parent_output_map: dict[str, pa.DataType] = {}
        for parent in self.inputs:
            for name, dtype in parent.outputs:
                parent_output_map[name] = dtype

        required_cols = {name for name, _ in self.function.input_schema.columns}
        available_cols = set(parent_output_map.keys())

        missing = required_cols - available_cols
        if missing:
            raise ValueError(
                f"Schema violation for Node '{self.name or self.function_cls.__name__}'. "
                f"Missing required columns: {sorted(missing)}"
            )

        type_mismatches = {
            name: (expected, parent_output_map[name])
            for name, expected in self.function.input_schema.columns
            if parent_output_map[name] != expected
        }
        if type_mismatches:
            mismatch_details = ", ".join(
                f"'{col}': expected {exp}, got {act}"
                for col, (exp, act) in sorted(type_mismatches.items())
            )
            raise TypeError(
                f"Type mismatch for Node '{self.name or self.function_cls.__name__}'. "
                f"Mismatched columns: {mismatch_details}"
            )

    def _generate_persistent_id(self) -> str:
        parent_ids = [parent.ID for parent in self.inputs]

        # Using the stringified version of the typed parameters object guarantees
        # that default arguments are captured and serialization won't randomise.
        node_definition = {
            "parent_ids": parent_ids,
            "outputs": list(self.outputs),
            "function": str(self.function),
            "parameters": str(self.parameters),
        }

        serialized_data = json.dumps(node_definition, sort_keys=True)
        return hashlib.sha256(serialized_data.encode("utf-8")).hexdigest()


class AbstractGraph:
    def __init__(self, root_node: Node):
        self.root_node = root_node
        self.node_list = self.get_subgraph_execution_order(self.root_node)

        self.ID = self._generate_persistent_id()

    def get_subgraph_execution_order(self, target_node: Node) -> list[Node]:
        """
        Traces the dependencies of an arbitrary target node backwards on the fly.
        Returns a perfectly topologically sorted list of Node objects ready for execution.
        """
        visited: set[str] = set()
        visiting: set[str] = set()  # Crucial for detecting infinite loops/cycles
        ordered_execution_nodes: list[Node] = []

        def dfs(node: Node):
            if node.ID in visiting:
                raise ValueError(
                    f"Cycle detected at {node.ID}, also known as {node.name}! Graphs must be acyclic."
                )
            if node.ID in visited:
                return

            visiting.add(node.ID)

            for parent_node in node.inputs:
                dfs(parent_node)

            visiting.remove(node.ID)
            visited.add(node.ID)
            ordered_execution_nodes.append(node)

        dfs(target_node)
        return ordered_execution_nodes

    def _generate_persistent_id(self) -> str:
        """Generates a stable, content-based SHA-256 hash for the graph."""
        # 1. Use the root node id as this is effectively a merkle tree root
        graph_definition = {
            "root_id": self.root_node.ID,
        }

        # 2. Serialize to a strict JSON string (sorted keys prevent randomness)
        serialized_data = json.dumps(graph_definition, sort_keys=True)

        # 3. Hash the string using SHA-256
        return hashlib.sha256(serialized_data.encode("utf-8")).hexdigest()
