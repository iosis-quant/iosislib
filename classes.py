from __future__ import annotations

import abc
import hashlib
import json
from dataclasses import MISSING, dataclass, fields
from typing import Any, Type

import polars as pl
import json

@dataclass(frozen=True)
class TSFNConfig(abc.ABC):
    pass

    def __str__(self) -> str:
        def serializer(obj):
            if isinstance(obj, pl.DataType):
                return str(obj)
            raise TypeError(f"Type {type(obj)} not serializable")
            
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        return json.dumps(data, sort_keys=True, default=serializer)

class TSFN(abc.ABC):
    CONFIG_CLS: Type[TSFNConfig] = TSFNConfig

    def __init__(
        self,
        parameters: dict[str, Any],
    ):
        self.signature = self.type_signature()
        self.parameters = self._bind_and_validate_config(parameters)

    def _bind_and_validate_config(self, params: dict[str, Any]) -> TSFNConfig:
        config_fields = fields(self.CONFIG_CLS)
        allowed_fields = {f.name for f in config_fields}

        # 1. Check for unexpected parameters (prevents silent typos)
        extra_keys = params.keys() - allowed_fields
        if extra_keys:
            raise ValueError(
                f"Unexpected parameters for {self.__class__.__name__}: {sorted(extra_keys)}"
            )

        # 2. Check for missing required parameters
        required_fields = {
            f.name for f in config_fields
            if f.default is MISSING and f.default_factory is MISSING
        }
        missing_keys = required_fields - params.keys()

        if missing_keys:
            raise ValueError(
                f"Parameter validation failed for {self.__class__.__name__}. "
                f"Missing required parameters: {sorted(missing_keys)}. "
                f"Expected schema: {self.CONFIG_CLS.__annotations__}"
            )

        return self.CONFIG_CLS(**params)

    @classmethod
    @abc.abstractmethod
    def type_signature(cls) -> tuple[
        tuple[tuple[str, pl.DataType], ...],
        tuple[tuple[str, pl.DataType], ...],
    ]:
        """Return ((input_col_specs, ...), (output_col_specs, ...))."""
        pass

    def __str__(self) -> str:
        input_sig, output_sig = self.signature
        return (
            f"{self.__class__.__name__}"
            f"(in:{json.dumps([(n, str(d)) for n, d in input_sig])}, "
            f"out:{json.dumps([(n, str(d)) for n, d in output_sig])})"
        )

    def validate_input_schema(self, lf: pl.LazyFrame) -> None:
        """Validates that the incoming LazyFrame matches the expected input signature."""
        # Uses cheap metadata schema collection!
        current_schema = lf.collect_schema()
        input_specs = self.signature[0]

        for col_name, expected_type in input_specs:
            if col_name not in current_schema:
                raise ValueError(f"Missing required input column: '{col_name}'")
            
            actual_type = current_schema[col_name]
            if actual_type != expected_type:
                raise TypeError(
                    f"Column '{col_name}' type mismatch. "
                    f"Expected {expected_type}, got {actual_type}"
                )

    def __call__(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        # Validate schema instantly using Polars metadata before processing data
        self.validate_input_schema(lf)
        return self.apply(lf)

    @abc.abstractmethod
    def apply(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """All transformations should be done lazily here."""
        pass

class Node:
    def __init__(
        self,
        function_cls: type[TSFN],
        bindings: dict[
            str,
            tuple[Node, str],
        ],
        parameters: dict[str, Any] | None,
        name: str | None,
    ):
        self.inputs = tuple(inputs)
        self.name = name

        self.function_cls = function_cls
        self.function = function_cls(
            input_cols,
            output_cols,
            parameters or {},
        )

        self.parameters = self.function.parameters

        self.outputs: tuple[tuple[str, pl.DataType], ...] = tuple(
            sorted(
                self.function.output_schema.columns,
                key=lambda c: c[0],
            )
        )

        self._verify_schema_compatibility()

        self.ID = self._generate_persistent_id()

    def _verify_schema_compatibility(self):
        parent_schema = pl.Schema()

        for parent in self.inputs:
            for name, dtype in parent.outputs.items():
                parent_schema[name] = dtype

        required_schema = self.function.input_schema.schema

        missing = set(required_schema) - set(parent_schema)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        mismatches = {}

        for col, expected_dtype in required_schema.items():
            actual_dtype = parent_schema[col]

            if actual_dtype != expected_dtype:
                mismatches[col] = (
                    expected_dtype,
                    actual_dtype,
                )

        if mismatches:
            raise TypeError(
                f"Type mismatches: {mismatches}"
            )

    def _generate_persistent_id(self) -> str:
        parent_ids = [parent.ID for parent in self.inputs]

        node_definition = {
            "parent_ids": parent_ids,
            "outputs": [
                (name, str(dtype))
                for name, dtype in self.outputs
            ],
            "function": str(self.function),
            "parameters": str(self.parameters),
        }

        serialized_data = json.dumps(
            node_definition,
            sort_keys=True,
        )

        return hashlib.sha256(
            serialized_data.encode("utf-8")
        ).hexdigest()

class Graph:
    def __init__(self, root_node: Node):
        self.root_node = root_node
        self.node_list = self.get_subgraph_execution_order(
            self.root_node
        )

        self.ID = self._generate_persistent_id()

    def get_subgraph_execution_order(
        self,
        target_node: Node,
    ) -> list[Node]:
        visited: set[str] = set()
        visiting: set[str] = set()

        ordered_execution_nodes: list[Node] = []

        def dfs(node: Node):
            if node.ID in visiting:
                raise ValueError(
                    f"Cycle detected at {node.ID}, "
                    f"also known as {node.name}! "
                    f"Graphs must be acyclic."
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
        graph_definition = {
            "root_id": self.root_node.ID,
        }

        serialized_data = json.dumps(
            graph_definition,
            sort_keys=True,
        )

        return hashlib.sha256(
            serialized_data.encode("utf-8")
        ).hexdigest()

    def execute(
        self,
        sources: dict[str, pl.LazyFrame],
    ) -> pl.DataFrame:

        results: dict[str, pl.LazyFrame] = {}

        for node in self.node_list:

            if not node.inputs:
                lf = sources[node.ID]
            else:
                parent_frames = [
                    results[parent.ID]
                    for parent in node.inputs
                ]

                lf = pl.concat(
                    parent_frames,
                    how="horizontal",
                )

            results[node.ID] = node.function(lf)

        return results[self.root_node.ID].collect()