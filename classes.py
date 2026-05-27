from __future__ import annotations
import abc
import hashlib
import json
from typing import *
import pyarrow as pa
import numpy as np


class Schema:
    def __init__(self, column_names: List[str]):
        self.column_names: List[str] = sorted(column_names)
        self._set_cache: Set[str] = set(self.column_names)

    def validate(self, table: pa.Table) -> bool:
        """Validates columns instantly using PyArrow table metadata."""
        table_columns = set(table.column_names)
        if not self._set_cache.issubset(table_columns):
            missing = self._set_cache - table_columns
            raise ValueError(f"Schema violation! Missing required columns: {missing}")
        return True

    def serialize(self) -> str:
        return json.dumps(self.column_names)


class TSFN(abc.ABC):
    def __init__(
        self,
        input_cols: List[str],
        output_cols: List[str],
        parameters: Optional[Dict[str, Any]],
    ):
        self.input_schema = Schema(input_cols)
        self.output_schema = Schema(output_cols)
        self.parameters = parameters or {}

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(in:{self.input_schema.serialize()}, out:{self.output_schema.serialize()})"

    def __call__(self, table: pa.Table) -> pa.Table:
        """Pipeline boundary: Validates native PyArrow tables without overhead."""
        self.input_schema.validate(table)
        # apply method is where we execute our lags, windows, functions etc
        result = self.apply(table, self.parameters)
        self.output_schema.validate(result)
        return result

    def apply(self, table: pa.Table, parameters: Optional[Dict[str, Any]]) -> pa.Table:
        """Override this method for raw PyArrow transformations if needed."""
        raise NotImplementedError


class Node:
    def __init__(
        self,
        inputs: List[Node],
        function_cls: type[TSFN],
        input_cols: List[str],
        output_cols: List[str],
        parameters: Optional[Dict[str, Any]],
        name: Optional[str],
    ):
        self.inputs = inputs
        self.name = name
        self.parameters = parameters or {}

        # 1. Instantiate the function first so schemas exist
        self.function = function_cls(input_cols, output_cols, self.parameters)
        self.outputs = sorted(self.function.output_schema.column_names)

        self._verify_schema_compatibility()
        self.ID = self._generate_persistent_id()

    def _verify_schema_compatibility(self):
        parent_output_cols = set()

        for parent in self.inputs:
            parent_output_cols.update(parent.outputs)

        required_inputs = self.function.input_schema._set_cache

        missing = required_inputs - parent_output_cols

        if missing:
            raise ValueError(f"Schema violation! Missing required columns: {missing}")

    def _generate_persistent_id(self) -> str:
        """Generates a stable, content-based SHA-256 hash for the node."""
        # 1. Collect IDs of parent nodes to embed graph structure
        parent_ids = [parent.ID for parent in self.inputs]

        # 2. Create a unique, reproducible blueprint of this node
        node_definition = {
            "parent_ids": parent_ids,
            "outputs": self.outputs,
            "function": str(self.function),
            "parameters": json.dumps(self.parameters, sort_keys=True),
        }

        # 3. Serialize to a strict JSON string (sorted keys prevent randomness)
        serialized_data = json.dumps(node_definition, sort_keys=True)

        # 4. Hash the string using SHA-256
        return hashlib.sha256(serialized_data.encode("utf-8")).hexdigest()


class AbstractGraph:
    def __init__(self, root_node: Node):
        self.root_node = root_node
        self.node_list = self.get_subgraph_execution_order(self.root_node)

        self.ID = self._generate_persistent_id()

    def get_subgraph_execution_order(self, target_node: Node) -> List[Node]:
        """
        Traces the dependencies of an arbitrary target node backwards on the fly.
        Returns a perfectly topologically sorted list of Node objects ready for execution.
        """
        visited: Set[str] = set()
        visiting: Set[str] = set()  # Crucial for detecting infinite loops/cycles
        ordered_execution_nodes: List[Node] = []

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
