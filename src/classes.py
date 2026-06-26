from __future__ import annotations

import abc
import hashlib
import json
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Type
from collections import defaultdict

import polars as pl


@dataclass(frozen=True)
class TimeAxis:
    column: str = "timestamp"
    dtype: pl.DataType = pl.Datetime
    timezone: str | None = None


@dataclass(frozen=True)
class FrameSignature:
    time: TimeAxis | None = field(default_factory=TimeAxis)
    columns: tuple[tuple[str, pl.DataType], ...] = ()

    @classmethod
    def empty(cls) -> FrameSignature:
        return cls(time=None, columns=())

    def __post_init__(self) -> None:
        if not isinstance(self.columns, tuple):
            raise TypeError("columns must be a tuple, not a list")
        columns = self.columns
        if self.time is None and columns:
            raise ValueError("Inputless frame signatures cannot declare value columns")
        if self.time is not None and any(name == self.time.column for name, _ in columns):
            raise ValueError(
                f"Time column '{self.time.column}' must not be listed as a value column"
            )

    def is_empty(self) -> bool:
        return self.time is None and not self.columns


def _dtype_matches(actual: pl.DataType, expected: pl.DataType) -> bool:
    actual_is_class = _is_dtype_class(actual)
    expected_is_class = _is_dtype_class(expected)

    if actual_is_class or expected_is_class:
        if actual_is_class and expected_is_class:
            return actual is expected

        if expected_is_class:
            return _matches_default_dtype_instance(actual, expected)

        return _matches_default_dtype_instance(expected, actual)

    if _is_list_instance(actual) and _is_list_instance(expected):
        return _dtype_matches(_list_inner_dtype(actual), _list_inner_dtype(expected))

    return actual == expected


def _is_dtype_class(dtype: pl.DataType) -> bool:
    return isinstance(dtype, type) and issubclass(dtype, pl.DataType)


def _is_list_instance(dtype: pl.DataType) -> bool:
    return not _is_dtype_class(dtype) and isinstance(dtype, pl.List)


def _list_inner_dtype(dtype: pl.DataType) -> pl.DataType:
    return dtype.inner


def _matches_default_dtype_instance(
    dtype_instance: pl.DataType,
    dtype_cls: type[pl.DataType],
) -> bool:
    if dtype_cls is pl.List or _is_list_instance(dtype_instance):
        return False

    try:
        default_instance = dtype_cls()
    except TypeError:
        return False

    return (
        type(dtype_instance) is type(default_instance)
        and repr(dtype_instance) == repr(default_instance)
    )


def _format_frame_signature(signature: FrameSignature) -> dict[str, Any]:
    if signature.is_empty():
        return {"time": None, "columns": []}

    assert signature.time is not None
    return {
        "time": {
            "column": signature.time.column,
            "dtype": str(signature.time.dtype),
            "timezone": signature.time.timezone,
        },
        "columns": [(name, str(dtype)) for name, dtype in signature.columns],
    }


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
        FrameSignature,
        FrameSignature,
    ]:
        """Return (input_frame_signature, output_frame_signature)."""
        pass

    def __str__(self) -> str:
        input_sig, output_sig = self.signature
        return (
            f"{self.__class__.__name__}"
            f"(in:{json.dumps(_format_frame_signature(input_sig), sort_keys=True)}, "
            f"out:{json.dumps(_format_frame_signature(output_sig), sort_keys=True)})"
        )

    def validate_input_schema(self, lf: pl.LazyFrame | None) -> None:
        """Validates that the incoming LazyFrame matches the expected input signature."""
        self._validate_schema(lf, self.signature[0], "input")

    def validate_output_schema(self, lf: pl.LazyFrame) -> None:
        """Validates that the returned LazyFrame matches the expected output signature."""
        self._validate_schema(lf, self.signature[1], "output")

    def _validate_schema(
        self,
        lf: pl.LazyFrame | None,
        signature: FrameSignature,
        schema_name: str,
    ) -> None:
        if signature.is_empty():
            if lf is not None:
                raise ValueError(f"Expected no {schema_name} frame for inputless signature")
            return

        if lf is None:
            raise ValueError(f"Missing required {schema_name} frame")

        if signature.time is None:
            raise ValueError(f"{schema_name.capitalize()} signature must declare a time axis")

        current_schema = lf.collect_schema()
        time_axis = signature.time

        if time_axis.column not in current_schema:
            raise ValueError(f"Missing required {schema_name} time column: '{time_axis.column}'")

        actual_time_type = current_schema[time_axis.column]
        if not _dtype_matches(actual_time_type, time_axis.dtype):
            raise TypeError(
                f"Time column '{time_axis.column}' type mismatch. "
                f"Expected {time_axis.dtype}, got {actual_time_type}"
            )

        actual_timezone = getattr(actual_time_type, "time_zone", None)
        if actual_timezone != time_axis.timezone:
            raise TypeError(
                f"Time column '{time_axis.column}' timezone mismatch. "
                f"Expected {time_axis.timezone}, got {actual_timezone}"
            )

        for col_name, expected_type in signature.columns:
            if col_name not in current_schema:
                raise ValueError(f"Missing required {schema_name} column: '{col_name}'")
            
            actual_type = current_schema[col_name]
            if not _dtype_matches(actual_type, expected_type):
                raise TypeError(
                    f"Column '{col_name}' type mismatch. "
                    f"Expected {expected_type}, got {actual_type}"
                )

    def __call__(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        input_signature = self.signature[0]
        self.validate_input_schema(lf)

        if input_signature.is_empty():
            output_lf = self.apply()
        else:
            output_lf = self.apply(lf)

        self.validate_output_schema(output_lf)
        return output_lf

    @abc.abstractmethod
    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
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
        self.outputs = {name: dtype for name, dtype in self.function.signature[1].columns}
        
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
            input_signature = node.function.signature[0]
            expected_inputs = {name for name, _ in input_signature.columns}
            bound_inputs = set(node.bindings.keys())

            if not node.bindings and not input_signature.is_empty():
                raise ValueError(
                    f"Binding validation failed for node '{node.name or node.ID}'. "
                    "Nodes with no predecessors must declare an empty input signature."
                )

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

            if node.bindings and input_signature.time is None:
                raise ValueError(
                    f"Binding validation failed for node '{node.name or node.ID}'. "
                    "Bound nodes must declare an input time axis."
                )

            # 2. Output existence & 3. Type compatibility
            input_types = dict(input_signature.columns)
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
                if not _dtype_matches(actual_dtype, expected_dtype):
                    raise TypeError(
                        f"Type mismatch at node '{node.name or node.ID}' for input '{input_name}': "
                        f"expected {expected_dtype}, got {actual_dtype} "
                        f"from '{parent_node.name or parent_node.ID}.{parent_col}'"
                    )

                self._validate_time_axis_compatibility(node, parent_node)

    def _validate_time_axis_compatibility(self, node: Node, parent_node: Node) -> None:
        child_time = node.function.signature[0].time
        parent_time = parent_node.function.signature[1].time

        if child_time is None:
            raise ValueError(
                f"Bound node '{node.name or node.ID}' must declare an input time axis"
            )
        if parent_time is None:
            raise ValueError(
                f"Parent node '{parent_node.name or parent_node.ID}' must declare an output time axis"
            )

        if parent_time.column != child_time.column:
            raise ValueError(
                f"Time axis mismatch at node '{node.name or node.ID}': "
                f"expected parent time column '{child_time.column}', "
                f"got '{parent_time.column}' from '{parent_node.name or parent_node.ID}'"
            )

        if not _dtype_matches(parent_time.dtype, child_time.dtype):
            raise TypeError(
                f"Time axis dtype mismatch at node '{node.name or node.ID}': "
                f"expected {child_time.dtype}, got {parent_time.dtype} "
                f"from '{parent_node.name or parent_node.ID}'"
            )

        if parent_time.timezone != child_time.timezone:
            raise TypeError(
                f"Time axis timezone mismatch at node '{node.name or node.ID}': "
                f"expected {child_time.timezone}, got {parent_time.timezone} "
                f"from '{parent_node.name or parent_node.ID}'"
            )

    def _generate_persistent_id(self) -> str:
        graph_definition = {
            "root_id": self.root_node.ID,
            "nodes": [node.ID for node in self.node_list],
        }

        serialized_data = json.dumps(graph_definition, sort_keys=True)
        return hashlib.sha256(serialized_data.encode("utf-8")).hexdigest()

    def execute(self) -> pl.DataFrame:
        results: dict[str, pl.LazyFrame] = {}

        for node in self.node_list:
            if not node.bindings:
                results[node.ID] = node.function()
            else:
                # Construct TSFN input dataframe by grouping projections per parent
                parent_to_bindings = defaultdict(list)
                for input_name, (parent_node, parent_col) in node.bindings.items():
                    parent_to_bindings[parent_node].append((parent_col, input_name))

                input_time = node.function.signature[0].time
                if input_time is None:
                    raise ValueError(
                        f"Bound node '{node.name or node.ID}' must declare an input time axis"
                    )
                time_col = input_time.column
                parent_frames = []
                for parent_node, binds in parent_to_bindings.items():
                    parent_lf = results[parent_node.ID]
                    parent_time = parent_node.function.signature[1].time
                    if parent_time is None:
                        raise ValueError(
                            f"Parent node '{parent_node.name or parent_node.ID}' "
                            "must declare an output time axis"
                        )
                    # Select expected columns and alias them directly to the TSFN's semantic input name.
                    select_exprs = [pl.col(parent_time.column).alias(time_col)]
                    select_exprs.extend(pl.col(p_col).alias(i_name) for p_col, i_name in binds)
                    parent_frames.append(parent_lf.select(select_exprs).sort(time_col))

                # Preserve every parent timestamp, then align each parent without lookahead.
                node_input_lf = (
                    pl.concat(
                        [parent_lf.select(time_col) for parent_lf in parent_frames],
                        how="vertical",
                    )
                    .unique()
                    .sort(time_col)
                )
                for parent_lf in parent_frames:
                    node_input_lf = node_input_lf.join_asof(
                        parent_lf,
                        on=time_col,
                        strategy="backward",
                    )
                results[node.ID] = node.function(node_input_lf)

        return results[self.root_node.ID].collect()
