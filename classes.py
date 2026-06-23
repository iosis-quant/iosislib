from __future__ import annotations

import abc
import hashlib
import json
from dataclasses import MISSING, dataclass, fields
from typing import Any, Type

import polars as pl
import json


class Schema:
    def __init__(self, schema: pl.Schema):
        self.schema = schema

    def validate(self, lf: pl.LazyFrame) -> bool:
        actual = lf.collect_schema()

        for name, expected_dtype in self.schema.items():
            if name not in actual:
                raise ValueError(
                    f"Schema violation! Missing required column '{name}'"
                )

            actual_dtype = actual[name]

            if actual_dtype != expected_dtype:
                raise TypeError(
                    f"Type mismatch on '{name}': "
                    f"expected {expected_dtype}, got {actual_dtype}"
                )

        return True

    def serialize(self) -> str:
        return json.dumps(
            [(name, str(dtype)) for name, dtype in self.schema.items()]
        )


@dataclass(frozen=True)
class TSFNConfig(abc.ABC):
    pass

    def __str__(self):
        sorted_fields = sorted(fields(self), key=lambda f: f.name)

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
        input_cols: list[tuple[str, pl.DataType]],
        output_cols: list[tuple[str, pl.DataType]],
        parameters: dict[str, Any],
    ):
        self.input_schema = Schema(input_cols)
        self.output_schema = Schema(output_cols)

        self.parameters = self._bind_and_validate_config(parameters)

    def _bind_and_validate_config(
        self, params: dict[str, Any]
    ) -> TSFNConfig:
        config_fields = fields(self.CONFIG_CLS)
        allowed_fields = {f.name for f in config_fields}

        required_fields = {
            f.name
            for f in config_fields
            if f.default is MISSING and f.default_factory is MISSING
        }

        filtered_params = {
            k: v for k, v in params.items() if k in allowed_fields
        }

        missing_keys = required_fields - filtered_params.keys()

        if missing_keys:
            raise ValueError(
                f"Parameter validation failed for {self.__class__.__name__}. "
                f"Missing required parameters with no default values: "
                f"{sorted(missing_keys)}. "
                f"Expected schema: {self.CONFIG_CLS.__annotations__}"
            )

        return self.CONFIG_CLS(**filtered_params)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(in:{self.input_schema.serialize()}, "
            f"out:{self.output_schema.serialize()})"
        )

    def __call__(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        self.input_schema.validate(lf)

        result = self.apply(lf)

        self.output_schema.validate(result)

        return result

    @abc.abstractmethod
    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        pass


class Node:
    def __init__(
        self,
        inputs: list["Node"],
        function_cls: type[TSFN],
        input_cols: list[tuple[str, pl.DataType]],
        output_cols: list[tuple[str, pl.DataType]],
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