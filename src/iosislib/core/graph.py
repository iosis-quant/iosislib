from __future__ import annotations

import abc
import hashlib
import json
import logging
import math
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
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
    _model_dir_scope,
    _s3_credentials_scope,
    _serialize_value,
    current_s3_credentials,
)

_LOG = logging.getLogger(__name__)


def _trainer_groups(nodes: Iterable[Node]) -> set[str]:
    """Collect model groups published by trainer nodes, ignoring misconfigurations.

    Misconfigured trainers are reported by graph validation; the executor only
    needs the well-formed groups to order live inference after training.
    """
    groups: set[str] = set()
    for node in nodes:
        if getattr(node.function_cls, "IS_TRAINER", False) is not True:
            continue
        model_group = getattr(node.function, "model_group", None)
        if not callable(model_group):
            continue
        try:
            group = model_group()
        except Exception:
            continue
        if isinstance(group, str) and group:
            groups.add(group)
    return groups


def _is_trainer_node(node: Node) -> bool:
    """Whether a node publishes finished models to the executor-owned store."""
    return getattr(node.function_cls, "IS_TRAINER", False) is True


def _is_source_node(node: Node) -> bool:
    """Whether a node is a primary source (inputless type signature).

    Sources are never cached: they are always re-pulled unless every
    dependent is already cached (in which case the frontier prunes them
    without evaluation). Detection is via the input frame signature,
    falling back to bindings for robustness on unvalidated nodes.
    """
    try:
        return bool(node.function.signature[0].is_empty())
    except (AttributeError, IndexError, TypeError):
        return not node.bindings


def _is_live_inference(node: Node, trainer_groups: set[str]) -> bool:
    """Whether an inference node must run after its group's trainer.

    Live means unpinned, unfrozen, and sharing a group with a trainer in the
    same graph. Frozen or pinned inference resolves purely from the store and
    needs no ordering beyond its data parents.
    """
    if getattr(node.function_cls, "IS_INFERENCE", False) is not True:
        return False
    function: Any = node.function
    is_frozen = getattr(function, "is_frozen", None)
    pinned_model_id = getattr(function, "pinned_model_id", None)
    model_group = getattr(function, "model_group", None)
    if not callable(is_frozen) or not callable(pinned_model_id):
        return False
    if not callable(model_group):
        return False
    try:
        frozen = is_frozen()
        pinned = pinned_model_id()
        group = model_group()
    except Exception:
        return False
    return (not frozen) and pinned is None and group in trainer_groups


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
        *,
        model_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self._s3_credentials: S3CredentialsProvider | None = s3_credentials
        self._model_dir_raw: str | os.PathLike[str] | None = model_dir

    def execute(self, graph: Graph) -> pl.DataFrame | dict[str, pl.DataFrame]:
        """Execute every terminal branch; single-terminal graphs return a frame."""
        with _s3_credentials_scope(self._s3_credentials):
            with _model_dir_scope(self._model_dir_raw):
                terminal_frames = self._evaluate_to_terminals(graph)
                if len(graph.terminal_nodes) == 1:
                    sole = graph.terminal_nodes[0]
                    return self.materialize(sole, terminal_frames[sole.ID])
                return {
                    node.ID: self.materialize(node, terminal_frames[node.ID])
                    for node in graph.terminal_nodes
                }

    def _evaluate_to_terminals(
        self, graph: Graph
    ) -> dict[str, pl.LazyFrame]:
        """Evaluate required boundaries and return each terminal lazy value.

        Live inference (unpinned, unfrozen, sharing a group with a trainer in
        the same graph) evaluates after every other node so the trainer's
        manifest entry is visible in the executor-owned model store.
        """
        results: dict[str, pl.LazyFrame] = {}
        terminal_ids = {node.ID for node in graph.terminal_nodes}
        trainer_groups = _trainer_groups(graph.node_list)
        deferred_ids = {
            node.ID
            for node in graph.node_list
            if _is_live_inference(node, trainer_groups)
        }
        ordered = [node for node in graph.node_list if node.ID not in deferred_ids]
        ordered.extend(node for node in graph.node_list if node.ID in deferred_ids)
        for node in ordered:
            self._evaluate_node(node, graph, results, terminal_ids)
        return {node.ID: results[node.ID] for node in graph.terminal_nodes}

    def _evaluate_node(
        self,
        node: Node,
        graph: Graph,
        results: dict[str, pl.LazyFrame],
        terminal_ids: set[str],
    ) -> None:
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

        if _is_trainer_node(node):
            # Force the manifest side effect now so deferred live inference
            # resolves against a populated store. The kept lazy frame is
            # already in memory, so the later terminal collect is free.
            results[node.ID] = self.materialize(node, results[node.ID]).lazy()
        elif node.ID in graph.materialized_node_ids and node.ID not in terminal_ids:
            results[node.ID] = self.materialize(node, results[node.ID]).lazy()

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
    """Execute a graph on one machine using Polars' local query engine.

    Materialized node results are persisted as Parquet files and reused
    on subsequent executions. Backward reachability search determines the
    minimal execution frontier, skipping computation and cache retrieval
    for superseded upstream nodes.
    """

    _DEFAULT_SHARD_BYTES: int = 512 * 1024 * 1024  # 512 MB

    def __init__(
        self,
        cache_dir: str | os.PathLike[str] | None = None,
        no_cache: bool = False,
        s3_credentials: S3CredentialsProvider | None = None,
        cache_shard_bytes: int | None = None,
        model_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        super().__init__(s3_credentials=s3_credentials, model_dir=model_dir)
        raw: str | None = None
        if cache_dir is not None:
            raw = str(cache_dir)
        elif not no_cache:
            raw = os.environ.get("IOSIS_CACHE_DIR")
        self._cache_raw: str | None = raw
        self._cache_s3: bool = raw is not None and raw.startswith("s3://")
        self._cache_dir: Path | None = (
            None if raw is None or self._cache_s3 else Path(raw)
        )
        self._no_cache = no_cache

        # Resolve shard size: explicit param > env var > default
        if cache_shard_bytes is not None:
            self._cache_shard_bytes: int = cache_shard_bytes
        else:
            env_shard = os.environ.get("IOSIS_CACHE_SHARD_BYTES")
            if env_shard is not None:
                try:
                    self._cache_shard_bytes = int(env_shard)
                except ValueError:
                    _LOG.warning(
                        "Invalid IOSIS_CACHE_SHARD_BYTES=%r, using default",
                        env_shard,
                    )
                    self._cache_shard_bytes = self._DEFAULT_SHARD_BYTES
            else:
                self._cache_shard_bytes = self._DEFAULT_SHARD_BYTES

        # Ensure local directory exists so caching is enabled on new paths
        if self._cache_dir is not None and not self._no_cache and not self._cache_s3:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _cache_enabled(self) -> bool:
        if self._no_cache or self._cache_raw is None:
            return False
        if self._cache_s3:
            return True
        return self._cache_dir is not None

    def _cache_entry_key(self, node_id: str) -> str:
        assert self._cache_raw is not None
        prefix = self._cache_raw.rstrip("/")
        return f"{prefix}/{node_id[:2]}/{node_id[2:4]}/{node_id[4:6]}/{node_id[6:]}"

    def _cache_entry_dir(self, node_id: str) -> Path:
        assert self._cache_dir is not None
        return (
            self._cache_dir
            / node_id[:2]
            / node_id[2:4]
            / node_id[4:6]
            / node_id[6:]
        )

    def _split_into_shards(
        self, df: pl.DataFrame
    ) -> list[pl.DataFrame]:
        """Split a DataFrame into byte-budgeted shards."""
        if df.is_empty() or self._cache_shard_bytes <= 0:
            return [df]
        estimated = df.estimated_size()
        if estimated <= self._cache_shard_bytes:
            return [df]
        total_rows = df.shape[0]
        n_shards = math.ceil(estimated / self._cache_shard_bytes)
        n_shards = min(n_shards, total_rows)
        if n_shards <= 1:
            return [df]
        rows_per = math.ceil(total_rows / n_shards)
        return [
            df.slice(i * rows_per, min(rows_per, total_rows - i * rows_per))
            for i in range(n_shards)
        ]

    @staticmethod
    def _delete_legacy_shards(
        filesystem: Any,
        bare: str,
        manifest_path: str,
    ) -> None:
        """Remove a legacy ``data.parquet`` and any stale ``part-*`` files."""
        from pyarrow.fs import FileType

        for name in ("data.parquet",):
            key = f"{bare}/{name}"
            info = filesystem.get_file_info(key)
            if info.type == FileType.File:
                filesystem.delete_file(key)
        manifest_info = filesystem.get_file_info(manifest_path)
        if manifest_info.type == FileType.File:
            filesystem.delete_file(manifest_path)

    def _s3_storage_options(self) -> dict[str, str]:
        credentials = current_s3_credentials()
        if credentials is None:
            return {}
        options: dict[str, str] = {
            "aws_access_key_id": credentials.access_key,
            "aws_secret_access_key": credentials.secret_key,
        }
        if credentials.session_token is not None:
            options["aws_session_token"] = credentials.session_token
        if credentials.region is not None:
            options["aws_region"] = credentials.region
        return options

    def _open_s3_filesystem(self):  # type: ignore[no-untyped-def]
        from pyarrow.fs import S3FileSystem

        credentials = current_s3_credentials()
        if credentials is None:
            return S3FileSystem()
        return S3FileSystem(
            access_key=credentials.access_key,
            secret_key=credentials.secret_key,
            session_token=credentials.session_token,
            region=credentials.region,
        )

    def _read_cache(self, node_id: str) -> pl.LazyFrame | None:
        if self._cache_s3:
            return self._read_cache_s3(node_id)
        return self._read_cache_local(node_id)

    def _read_cache_local(self, node_id: str) -> pl.LazyFrame | None:
        try:
            entry = self._cache_entry_dir(node_id)
            manifest_path = entry / "manifest.json"
            if not manifest_path.exists():
                return None
            manifest = json.loads(manifest_path.read_text())
            if not manifest.get("success", False):
                return None
            parquet_files = sorted(entry.glob("*.parquet"))
            if not parquet_files:
                return None
            return pl.scan_parquet(
                [str(p) for p in parquet_files],
                hive_partitioning=False,
            )
        except Exception:
            return None

    def _read_cache_s3(self, node_id: str) -> pl.LazyFrame | None:
        try:
            key_prefix = self._cache_entry_key(node_id)
            bare = key_prefix.removeprefix("s3://")
            filesystem = self._open_s3_filesystem()

            from pyarrow.fs import FileType

            manifest_key = f"{bare}/manifest.json"
            info = filesystem.get_file_info(manifest_key)
            if info.type != FileType.File:
                return None
            with filesystem.open_input_stream(manifest_key) as stream:
                manifest = json.loads(stream.read().decode("utf-8"))
            if not manifest.get("success", False):
                return None
            storage_options = self._s3_storage_options()
            return pl.scan_parquet(
                f"{key_prefix}/*.parquet",
                hive_partitioning=False,
                storage_options=storage_options,
            )
        except Exception:
            return None

    def _write_cache(self, node_id: str, df: pl.DataFrame) -> None:
        if self._cache_s3:
            self._write_cache_s3(node_id, df)
        else:
            self._write_cache_local(node_id, df)

    def _write_cache_local(self, node_id: str, df: pl.DataFrame) -> None:
        try:
            entry = self._cache_entry_dir(node_id)
            entry.mkdir(parents=True, exist_ok=True)

            # Remove legacy single-file cache if present
            legacy = entry / "data.parquet"
            if legacy.exists():
                legacy.unlink()

            # Remove stale shards from a previous (different) write
            for stale in sorted(entry.glob("part-*.parquet")):
                stale.unlink()

            shards = self._split_into_shards(df)
            for idx, shard in enumerate(shards):
                shard_path = entry / f"part-{idx:05d}.parquet"
                shard.write_parquet(shard_path)

            total_bytes = sum(
                (entry / f"part-{i:05d}.parquet").stat().st_size
                for i in range(len(shards))
            )

            manifest = {
                "node_id": node_id,
                "success": True,
                "row_count": df.shape[0],
                "column_count": df.shape[1],
                "columns": df.columns,
                "schema": {col: str(df.schema[col]) for col in df.columns},
                "byte_size": total_bytes,
                "shard_count": len(shards),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            (entry / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True)
            )
        except Exception:
            _LOG.warning("cache write failed for node %s", node_id[:12], exc_info=True)

    def _write_cache_s3(self, node_id: str, df: pl.DataFrame) -> None:
        try:
            from pyarrow.fs import FileSelector, FileType

            key_prefix = self._cache_entry_key(node_id)
            bare = key_prefix.removeprefix("s3://")
            filesystem = self._open_s3_filesystem()
            filesystem.create_dir(bare, recursive=True)

            # Remove legacy single-file cache and stale shards
            manifest_key = f"{bare}/manifest.json"
            self._delete_legacy_shards(filesystem, bare, manifest_key)

            # Remove stale part-*.parquet from previous writes. Note:
            # get_file_info() with a plain string returns a single FileInfo
            # (it does not glob), so list via a selector instead.
            selector = FileSelector(bare, recursive=False)
            for shard_info in filesystem.get_file_info(selector):
                if shard_info.type != FileType.File:
                    continue
                shard_name = shard_info.path.rsplit("/", 1)[-1]
                if shard_name.startswith("part-") and shard_name.endswith(
                    ".parquet"
                ):
                    filesystem.delete_file(shard_info.path)

            import io

            shards = self._split_into_shards(df)
            total_bytes = 0
            for idx, shard in enumerate(shards):
                buffer = io.BytesIO()
                shard.write_parquet(buffer)
                shard_bytes = buffer.getvalue()
                total_bytes += len(shard_bytes)
                shard_key = f"{bare}/part-{idx:05d}.parquet"
                with filesystem.open_output_stream(shard_key) as stream:
                    stream.write(shard_bytes)

            manifest = {
                "node_id": node_id,
                "success": True,
                "row_count": df.shape[0],
                "column_count": df.shape[1],
                "columns": df.columns,
                "schema": {col: str(df.schema[col]) for col in df.columns},
                "byte_size": total_bytes,
                "shard_count": len(shards),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            with filesystem.open_output_stream(manifest_key) as stream:
                stream.write(
                    json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
                )
        except Exception:
            _LOG.warning("S3 cache write failed for node %s", node_id[:12], exc_info=True)

    def _resolve_cache_frontier(
        self, graph: Graph
    ) -> tuple[set[str], dict[str, pl.LazyFrame]]:
        """Walk backward from every terminal to identify required nodes and hits.

        Stops searching upstream along any branch as soon as a cached node is found.
        Source nodes (inputless type signature) are never read from cache: they
        are always re-pulled unless every dependent already hit (pruned above).
        Returns:
            required_ids: set of node IDs that must be evaluated or read from cache.
            cached_frames: dict mapping node_id -> pl.LazyFrame for frontier cache hits.
        """
        if not self._cache_enabled:
            return {node.ID for node in graph.node_list}, {}

        required_ids: set[str] = set()
        cached_frames: dict[str, pl.LazyFrame] = {}
        queue: list[Node] = list(graph.terminal_nodes)

        while queue:
            node = queue.pop()
            if node.ID in required_ids:
                continue

            required_ids.add(node.ID)

            cached = None if _is_source_node(node) else self._read_cache(node.ID)
            if cached is not None:
                cached_frames[node.ID] = cached
                # Frontier cache hit: stop traversing upstream along this branch!
                continue

            # Cache miss: traverse upstream to parent input nodes
            parents = set(node.inputs) | {
                parent for parent, _ in node.bindings.values()
            }
            for parent_node in parents:
                queue.append(parent_node)

        return required_ids, cached_frames

    def execute(
        self, graph: Graph
    ) -> pl.DataFrame | dict[str, pl.DataFrame]:
        with _s3_credentials_scope(self._s3_credentials):
            with _model_dir_scope(self._model_dir_raw):
                terminal_frames, cached_ids = self._evaluate_to_terminals_with_hits(
                    graph
                )
                outputs: dict[str, pl.DataFrame] = {}
                for node in graph.terminal_nodes:
                    result = self.materialize(node, terminal_frames[node.ID])
                    if (
                        self._cache_enabled
                        and node.ID in graph.materialized_node_ids
                        and node.ID not in cached_ids
                        and not _is_source_node(node)
                    ):
                        self._write_cache(node.ID, result)
                    outputs[node.ID] = result
                if len(graph.terminal_nodes) == 1:
                    return outputs[graph.terminal_nodes[0].ID]
                return outputs

    def _evaluate_to_terminals(
        self, graph: Graph
    ) -> dict[str, pl.LazyFrame]:
        frames, _ = self._evaluate_to_terminals_with_hits(graph)
        return frames

    def _evaluate_to_terminals_with_hits(
        self, graph: Graph
    ) -> tuple[dict[str, pl.LazyFrame], set[str]]:
        required_ids, cached_frames = self._resolve_cache_frontier(graph)
        results: dict[str, pl.LazyFrame] = {}
        terminal_ids = {node.ID for node in graph.terminal_nodes}
        trainer_groups = _trainer_groups(graph.node_list)
        deferred_ids = {
            node.ID
            for node in graph.node_list
            if _is_live_inference(node, trainer_groups)
        }
        ordered = [node for node in graph.node_list if node.ID not in deferred_ids]
        ordered.extend(node for node in graph.node_list if node.ID in deferred_ids)

        for node in ordered:
            if node.ID not in required_ids:
                continue  # Skip unneeded upstream nodes entirely

            if node.ID in cached_frames:
                results[node.ID] = cached_frames[node.ID]
                continue

            try:
                node_input_lf = (
                    None
                    if not node.bindings
                    else self.align_inputs(node, results)
                )
                lf = self.lower_node(node, node_input_lf)
            except Exception as exc:
                raise RuntimeError(
                    f"Execution failed at node "
                    f"'{node.name or node.ID[:8]}' "
                    f"({node.function_cls.__name__}@"
                    f"{node.function.version}): {exc}"
                ) from exc

            is_terminal = node.ID in terminal_ids
            is_materialized = node.ID in graph.materialized_node_ids

            if _is_trainer_node(node):
                # Force the manifest side effect now; the terminal cache
                # write still happens once in execute().
                results[node.ID] = self.materialize(node, lf).lazy()
            elif is_materialized and not is_terminal and not _is_source_node(node):
                df = self.materialize(node, lf)
                if self._cache_enabled and node.ID not in cached_frames:
                    self._write_cache(node.ID, df)
                results[node.ID] = df.lazy()
            elif is_materialized and not is_terminal:
                # Source with materialize:true: collect for run-local reuse
                # but never persist. `materialize` is ignored for caching.
                results[node.ID] = self.materialize(node, lf).lazy()
            else:
                results[node.ID] = lf

        frames = {node.ID: results[node.ID] for node in graph.terminal_nodes}
        return frames, set(cached_frames)

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
        "terminal_nodes",
    )

    ID: str
    _frozen: bool
    _validation_report: ValidationReport
    materialized_node_ids: frozenset[str]
    node_list: tuple[Node, ...]
    root_node: Node
    terminal_nodes: tuple[Node, ...]

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("Graph instances are immutable")
        object.__setattr__(self, name, value)

    def __init__(self, root: Node | Iterable[Node]):
        """Build a validated graph from one root or a full node list.

        A single ``Node`` preserves the legacy single-terminal graph. An
        iterable is treated as the full graph membership: terminals (nodes no
        other member consumes) are derived automatically and every branch
        executes. Trainers are terminal by construction and run alongside the
        inference path over the same data.
        """
        terminals, membership = self._coerce_terminals(root)

        node_list, report = self._validated_declaration(terminals, membership)
        if not report.is_valid:
            raise GraphValidationError(report)

        primary = min(terminals, key=lambda node: node.ID)
        object.__setattr__(self, "root_node", primary)
        object.__setattr__(
            self, "terminal_nodes", tuple(sorted(terminals, key=lambda n: n.ID))
        )
        object.__setattr__(self, "node_list", node_list)
        object.__setattr__(self, "_validation_report", report)
        object.__setattr__(
            self,
            "materialized_node_ids",
            frozenset(node.ID for node in node_list if node.materialize),
        )
        object.__setattr__(self, "ID", self._generate_persistent_id())
        object.__setattr__(self, "_frozen", True)

    @staticmethod
    def _coerce_terminals(
        root: Node | Iterable[Node],
    ) -> tuple[tuple[Node, ...], tuple[Node, ...] | None]:
        """Normalize constructor input to (terminals, membership-or-None)."""
        if isinstance(root, Node):
            return (root,), None
        if isinstance(root, Iterable):
            members = tuple(dict.fromkeys(root))
        else:
            raise TypeError("Graph root must be a Node or an iterable of Nodes")
        if not members:
            raise ValueError("Graph requires at least one node")
        for member in members:
            if not isinstance(member, Node):
                raise TypeError("Graph membership must contain only Nodes")
        parent_ids = {
            parent.ID
            for member in members
            for parent, _ in member.bindings.values()
        }
        terminals = tuple(member for member in members if member.ID not in parent_ids)
        if not terminals:
            raise GraphValidationError(
                ValidationReport(
                    (
                        ValidationIssue(
                            code="NO_TERMINALS",
                            category="structure",
                            message=(
                                "Graph membership has no terminal nodes: every "
                                "node is consumed by another member, so the "
                                "declaration contains a cycle."
                            ),
                            node_id=members[0].ID,
                            node_name=members[0].name,
                            tsfn_class=(
                                f"{members[0].function_cls.__module__}."
                                f"{members[0].function_cls.__qualname__}"
                            ),
                            tsfn_version=members[0].function.version,
                        ),
                    )
                )
            )
        return terminals, members

    def __repr__(self) -> str:
        return (
            f"Graph(id={self.ID[:8]!r}, root={self.root_node.ID[:8]!r}, "
            f"nodes={len(self.node_list)}, terminals={len(self.terminal_nodes)})"
        )

    def verify(self) -> None:
        """Consult the immutable validation result established at construction."""
        if not self._validation_report.is_valid:
            raise GraphValidationError(self._validation_report)

    @classmethod
    def validate(cls, root: Node | Iterable[Node]) -> ValidationReport:
        """Return all safely diagnosable declaration issues without executing TSFNs."""
        terminals, membership = cls._coerce_terminals(root)

        _, report = cls._validated_declaration(terminals, membership)
        return report

    @classmethod
    def _validated_declaration(
        cls,
        terminals: tuple[Node, ...],
        membership: tuple[Node, ...] | None = None,
    ) -> tuple[tuple[Node, ...], ValidationReport]:
        ordered_nodes, cycle = cls._multi_dependency_order(terminals)
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
        lookahead_tainted = cls._lookahead_taint(ordered_nodes)
        for node in ordered_nodes:
            cls._collect_node_issues(node, node_ids, issues, lookahead_tainted)
        cls._collect_training_issues(ordered_nodes, membership, issues)

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
    def _collect_training_issues(
        cls,
        ordered_nodes: tuple[Node, ...],
        membership: tuple[Node, ...] | None,
        issues: list[ValidationIssue],
    ) -> None:
        """Validate trainer/inference pairing and nodelist reachability."""
        from iosislib.core.model import _validate_model_group

        trainers_by_group: dict[str, list[Node]] = defaultdict(list)
        for node in ordered_nodes:
            if getattr(node.function_cls, "IS_TRAINER", False) is not True:
                continue
            function: Any = node.function
            model_group = getattr(function, "model_group", None)
            if not callable(model_group):
                issues.append(
                    cls._issue(
                        node,
                        code="INVALID_MODEL_GROUP",
                        category="training",
                        message=(
                            f"Training validation failed for node "
                            f"'{node.name or node.ID}': "
                            f"{node.function_cls.__name__} does not expose "
                            "model_group()"
                        ),
                    )
                )
                continue
            try:
                group = _validate_model_group(model_group())
            except (TypeError, ValueError) as exc:
                issues.append(
                    cls._issue(
                        node,
                        code="INVALID_MODEL_GROUP",
                        category="training",
                        message=(
                            f"Training validation failed for node "
                            f"'{node.name or node.ID}': {exc}"
                        ),
                    )
                )
                continue
            trainers_by_group[group].append(node)

        for group in sorted(trainers_by_group):
            trainers = trainers_by_group[group]
            if len(trainers) < 2:
                continue
            for extra in trainers[1:]:
                issues.append(
                    cls._issue(
                        extra,
                        code="MULTIPLE_TRAINERS",
                        category="training",
                        message=(
                            f"Training validation failed for node "
                            f"'{extra.name or extra.ID}': group {group!r} "
                            "already has a trainer. Retraining appends a new "
                            "finished model run; it never adds an endpoint."
                        ),
                    )
                )

        for node in ordered_nodes:
            if getattr(node.function_cls, "IS_INFERENCE", False) is not True:
                continue
            function = node.function
            model_group = getattr(function, "model_group", None)
            pinned_model_id = getattr(function, "pinned_model_id", None)
            is_frozen = getattr(function, "is_frozen", None)
            if not callable(model_group):
                issues.append(
                    cls._issue(
                        node,
                        code="INVALID_MODEL_GROUP",
                        category="training",
                        message=(
                            f"Training validation failed for node "
                            f"'{node.name or node.ID}': "
                            f"{node.function_cls.__name__} does not expose "
                            "model_group()"
                        ),
                    )
                )
                continue
            try:
                _validate_model_group(model_group())
                if callable(pinned_model_id):
                    pinned = pinned_model_id()
                    if pinned is not None and (
                        not isinstance(pinned, str) or not pinned
                    ):
                        raise ValueError(
                            "pinned model_id must be a non-empty string or None"
                        )
                if callable(is_frozen):
                    is_frozen()
            except (TypeError, ValueError, LookupError) as exc:
                issues.append(
                    cls._issue(
                        node,
                        code="INVALID_MODEL_REFERENCE",
                        category="training",
                        message=(
                            f"Training validation failed for node "
                            f"'{node.name or node.ID}': {exc}"
                        ),
                    )
                )

        if membership is not None:
            reachable = {node.ID for node in ordered_nodes}
            for member in membership:
                if member.ID not in reachable:
                    issues.append(
                        cls._issue(
                            member,
                            code="UNREACHABLE_NODE",
                            category="structure",
                            message=(
                                f"Membership validation failed for node "
                                f"'{member.name or member.ID}': the node is not "
                                "reachable from any terminal and would never "
                                "execute."
                            ),
                        )
                    )

    @classmethod
    def _dependency_order(
        cls,
        target_node: Node,
    ) -> tuple[tuple[Node, ...], tuple[Node, ...] | None]:
        """Return one canonical, ID-deduplicated dependency traversal."""
        return cls._multi_dependency_order((target_node,))

    @classmethod
    def _multi_dependency_order(
        cls,
        terminals: tuple[Node, ...],
    ) -> tuple[tuple[Node, ...], tuple[Node, ...] | None]:
        """Return one canonical traversal over every terminal's ancestors.

        Terminals are visited in ID order with shared visited state, so shared
        ancestors are evaluated once however many branches consume them.
        """
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

        for terminal in sorted(terminals, key=lambda node: node.ID):
            dfs(terminal)
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
    def _lookahead_taint(
        cls,
        ordered_nodes: tuple[Node, ...],
    ) -> dict[str, bool]:
        """Mark nodes whose output frame contains future-derived values.

        A node is tainted when it is itself a look-ahead transform, or when any
        of its parents is tainted, unless the node declares look-ahead inputs
        (a supervised boundary) and therefore converts labels into predictions.
        """
        tainted: dict[str, bool] = {}
        for node in ordered_nodes:
            if node.function_cls.LOOKAHEAD:
                tainted[node.ID] = True
                continue
            if node.function_cls.ALLOW_LOOKAHEAD_INPUTS:
                tainted[node.ID] = False
                continue
            tainted[node.ID] = any(
                tainted[parent.ID] for parent in node.inputs
            )
        return tainted

    @classmethod
    def _collect_node_issues(
        cls,
        node: Node,
        node_ids: set[str],
        issues: list[ValidationIssue],
        lookahead_tainted: Mapping[str, bool],
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

        allowed_lookahead = node.function_cls.ALLOW_LOOKAHEAD_INPUTS
        if allowed_lookahead and node.bindings:
            for input_name, (parent_node, parent_column) in sorted(
                node.bindings.items()
            ):
                if parent_node.ID not in node_ids:
                    continue
                if not lookahead_tainted[parent_node.ID]:
                    continue
                if input_name in allowed_lookahead:
                    continue
                issues.append(
                    cls._issue(
                        node,
                        code="LOOKAHEAD_INTO_FEATURES",
                        category="lookahead",
                        message=(
                            f"Look-ahead validation failed for node "
                            f"'{node.name or node.ID}'. Input '{input_name}' receives "
                            f"future-derived values from "
                            f"'{parent_node.name or parent_node.ID}', which may only "
                            f"feed a declared target/label input: "
                            f"{sorted(allowed_lookahead)}."
                        ),
                        input_name=input_name,
                        output_name=parent_column,
                    )
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
            "terminal_ids": [node.ID for node in self.terminal_nodes],
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
        if len(self.terminal_nodes) == 1:
            graph_definition = {
                "root_id": self.root_node.ID,
                "nodes": tuple(node.ID for node in self.node_list),
            }
        else:
            graph_definition = {
                "terminal_ids": sorted(node.ID for node in self.terminal_nodes),
                "nodes": tuple(node.ID for node in self.node_list),
            }
        serialized_data = json.dumps(graph_definition, sort_keys=True)
        return hashlib.sha256(serialized_data.encode("utf-8")).hexdigest()

    def execute(
        self, executor: Executor | None = None
    ) -> pl.DataFrame | dict[str, pl.DataFrame]:
        """Execute every terminal branch.

        Single-terminal graphs return one frame (legacy behavior);
        multi-terminal graphs return a mapping of terminal node ID to frame.
        """
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
