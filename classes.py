from __future__ import annotations

import abc
import hashlib
import json
from dataclasses import MISSING, dataclass, fields
from typing import Any, Type
from collections import defaultdict

import polars as pl


@dataclass(frozen=True)
class TSFNConfig(abc.ABC):
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
        self.parameters = self._bind_and_validate_config(parameters)
        self.signature = self.type_signature()

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

    @abc.abstractmethod
    def type_signature(self) -> tuple[
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
        bindings: dict[str, tuple[Node, str]] | None = None,
        parameters: dict[str, Any] | None = None,
        name: str | None = None,
    ):
        self.name = name
        self.function_cls = function_cls
        self.function = function_cls(parameters or {})
        self.parameters = self.function.parameters
        
        # bindings map TSFN semantic input names to (parent_node, parent_output_column)
        self.bindings = bindings or {}
        
        # Deduplicate and extract unique parent Nodes preserving order
        self.inputs = tuple(dict.fromkeys(parent for parent, _ in self.bindings.values()))
        
        # Map exposed output names to their corresponding data types
        self.outputs = {name: dtype for name, dtype in self.function.signature[1]}
        
        self.ID = self._generate_persistent_id()

    def __getattr__(self, name: str) -> tuple[Node, str]:
        """Provides syntactical sugar to reference outputs (e.g., node.lagged)."""
        if name in self.outputs:
            return (self, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' or its configured TSFN '{self.function_cls.__name__}' "
            f"does not expose output: '{name}'"
        )

    def _generate_persistent_id(self) -> str:
        # Serialize bindings deterministically by sorting semantic input keys
        serialized_bindings = {
            input_name: [parent.ID, parent_col]
            for input_name, (parent, parent_col) in sorted(self.bindings.items())
        }
        
        node_definition = {
            "name": self.name,
            "bindings": serialized_bindings,
            "function": str(self.function),
            "parameters": str(self.parameters),
            "outputs": sorted([(name, str(dtype)) for name, dtype in self.outputs.items()]),
        }

        serialized_data = json.dumps(node_definition, sort_keys=True)
        return hashlib.sha256(serialized_data.encode("utf-8")).hexdigest()


class Graph:
    def __init__(self, root_node: Node):
        self.root_node = root_node
        self.node_list = self.get_subgraph_execution_order(self.root_node)
        
        # Perform global static validation
        self._validate_graph()
        
        self.ID = self._generate_persistent_id()

    def get_subgraph_execution_order(self, target_node: Node) -> list[Node]:
        visited: set[str] = set()
        visiting: set[str] = set()
        ordered_execution_nodes: list[Node] = []

        def dfs(node: Node):
            if node.ID in visiting:
                raise ValueError(
                    f"Cycle detected at node '{node.name or node.ID}'! "
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

    def _validate_graph(self) -> None:
        """Validates bindings, outputs, and type compatibility at construction."""
        for node in self.node_list:
            expected_inputs = {name for name, _ in node.function.signature[0]}
            bound_inputs = set(node.bindings.keys())

            # 1. Binding completeness
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

            # 2. Output existence & 3. Type compatibility
            input_types = dict(node.function.signature[0])
            for input_name, (parent_node, parent_col) in node.bindings.items():
                if parent_node not in self.node_list:
                    raise ValueError(
                        f"Node '{node.name or node.ID}' binds to parent "
                        f"'{parent_node.name or parent_node.ID}' which is not in this graph."
                    )
                
                # Check output existence
                if parent_col not in parent_node.outputs:
                    raise ValueError(
                        f"Binding validation failed for node '{node.name or node.ID}'. "
                        f"Parent node '{parent_node.name or parent_node.ID}' does not expose "
                        f"output '{parent_col}'. Available outputs: {list(parent_node.outputs.keys())}"
                    )

                # Check type compatibility
                expected_dtype = input_types[input_name]
                actual_dtype = parent_node.outputs[parent_col]
                if actual_dtype != expected_dtype:
                    raise TypeError(
                        f"Type mismatch at node '{node.name or node.ID}' for input '{input_name}': "
                        f"expected {expected_dtype}, got {actual_dtype} "
                        f"from '{parent_node.name or parent_node.ID}.{parent_col}'"
                    )

    def _generate_persistent_id(self) -> str:
        graph_definition = {
            "root_id": self.root_node.ID,
            "nodes": [node.ID for node in self.node_list],
        }

        serialized_data = json.dumps(graph_definition, sort_keys=True)
        return hashlib.sha256(serialized_data.encode("utf-8")).hexdigest()

    def execute(self, sources: dict[str, pl.LazyFrame]) -> pl.DataFrame:
        results: dict[str, pl.LazyFrame] = {}

        for node in self.node_list:
            if not node.bindings:
                # Leaf / Source node getting input straight from sources
                if node.name in sources:
                    lf = sources[node.name]
                elif node.ID in sources:
                    lf = sources[node.ID]
                else:
                    raise KeyError(
                        f"Source dataframe not found for node '{node.name or node.ID}'"
                    )
                results[node.ID] = node.function(lf)
            else:
                # Construct TSFN input dataframe by grouping projections per parent
                parent_to_bindings = defaultdict(list)
                for input_name, (parent_node, parent_col) in node.bindings.items():
                    parent_to_bindings[parent_node].append((parent_col, input_name))

                parent_frames = []
                for parent_node, binds in parent_to_bindings.items():
                    parent_lf = results[parent_node.ID]
                    # Select expected columns and alias them directly to the TSFN's semantic input name
                    select_exprs = [pl.col(p_col).alias(i_name) for p_col, i_name in binds]
                    parent_frames.append(parent_lf.select(select_exprs))

                # Assemble node's input dataframe
                node_input_lf = pl.concat(parent_frames, how="horizontal")
                results[node.ID] = node.function(node_input_lf)

        return results[self.root_node.ID].collect()