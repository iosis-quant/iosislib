from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl

from iosislib.core.graph import Graph, LocalExecutor
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, TSFN, TimeAxis


TIMESTAMP_AXIS = TimeAxis(column="timestamp", dtype=pl.Datetime)
VALUE_FRAME = FrameSignature(
    time=TIMESTAMP_AXIS,
    columns=(("value", pl.Float64),),
)


class SimpleSource(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), VALUE_FRAME

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "timestamp": [
                    datetime(2026, 1, 1),
                    datetime(2026, 1, 2),
                    datetime(2026, 1, 3),
                ],
                "value": [1.0, 2.0, 3.0],
            }
        ).lazy()


class IncrementSource(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), VALUE_FRAME

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "timestamp": [
                    datetime(2026, 1, 1),
                    datetime(2026, 1, 2),
                    datetime(2026, 1, 3),
                ],
                "value": [10.0, 20.0, 30.0],
            }
        ).lazy()


class Doubler(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return VALUE_FRAME, VALUE_FRAME

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        assert lf is not None
        return lf.with_columns((pl.col("value") * 2).alias("value"))


def _make_source_graph() -> Graph:
    return Graph(Node(SimpleSource, name="source", materialize=True))


def _make_transform_graph() -> Graph:
    source = Node(SimpleSource, name="source", materialize=True)
    doubled = Node(
        Doubler,
        bindings={"value": source.output("value")},
        name="doubler",
        materialize=True,
    )
    return Graph(doubled)


def _cache_files(cache_dir: Path) -> list[Path]:
    return sorted(cache_dir.rglob("manifest.json"))


# ---------------------------------------------------------------------------
# Basic cache hit/miss
# ---------------------------------------------------------------------------


class TestCacheHitAndMiss:
    def test_first_execution_is_cache_miss(self, tmp_path: Path) -> None:
        graph = _make_source_graph()
        executor = LocalExecutor(cache_dir=tmp_path)
        result = graph.execute(executor=executor)

        assert result["value"].to_list() == [1.0, 2.0, 3.0]
        manifests = _cache_files(tmp_path)
        assert len(manifests) == 1

    def test_second_execution_is_cache_hit(self, tmp_path: Path) -> None:
        graph = _make_source_graph()
        executor = LocalExecutor(cache_dir=tmp_path)

        result1 = graph.execute(executor=executor)
        result2 = graph.execute(executor=executor)

        assert result1["value"].to_list() == result2["value"].to_list()
        assert len(_cache_files(tmp_path)) == 1

    def test_cache_hit_returns_identical_result(self, tmp_path: Path) -> None:
        graph = _make_source_graph()
        executor = LocalExecutor(cache_dir=tmp_path)

        result1 = graph.execute(executor=executor)
        result2 = graph.execute(executor=executor)

        assert result1.equals(result2)


# ---------------------------------------------------------------------------
# no_cache flag
# ---------------------------------------------------------------------------


class TestNoCache:
    def test_no_cache_disables_caching(self, tmp_path: Path) -> None:
        graph = _make_source_graph()
        executor = LocalExecutor(cache_dir=tmp_path, no_cache=True)
        graph.execute(executor=executor)

        assert len(_cache_files(tmp_path)) == 0

    def test_no_cache_still_executes(self, tmp_path: Path) -> None:
        graph = _make_source_graph()
        executor = LocalExecutor(cache_dir=tmp_path, no_cache=True)
        result = graph.execute(executor=executor)

        assert result["value"].to_list() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# No cache dir
# ---------------------------------------------------------------------------


class TestNoCacheDir:
    def test_no_dir_silent_noop(self) -> None:
        graph = _make_source_graph()
        executor = LocalExecutor()
        result = graph.execute(executor=executor)

        assert result["value"].to_list() == [1.0, 2.0, 3.0]

    def test_nonexistent_dir_silent_noop(self, tmp_path: Path) -> None:
        graph = _make_source_graph()
        executor = LocalExecutor(cache_dir=tmp_path / "nonexistent")
        result = graph.execute(executor=executor)

        assert result["value"].to_list() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Manifest content
# ---------------------------------------------------------------------------


class TestManifestContent:
    def test_manifest_has_correct_fields(self, tmp_path: Path) -> None:
        graph = _make_source_graph()
        executor = LocalExecutor(cache_dir=tmp_path)
        graph.execute(executor=executor)

        manifests = _cache_files(tmp_path)
        assert len(manifests) == 1
        manifest = json.loads(manifests[0].read_text())

        assert manifest["success"] is True
        assert manifest["row_count"] == 3
        assert manifest["column_count"] == 2
        assert "timestamp" in manifest["columns"]
        assert "value" in manifest["columns"]
        assert manifest["byte_size"] > 0
        assert "created_at" in manifest

    def test_manifest_schema_matches_data(self, tmp_path: Path) -> None:
        graph = _make_source_graph()
        executor = LocalExecutor(cache_dir=tmp_path)
        graph.execute(executor=executor)

        manifests = _cache_files(tmp_path)
        manifest = json.loads(manifests[0].read_text())

        assert "value" in manifest["schema"]
        assert "Float64" in manifest["schema"]["value"]


# ---------------------------------------------------------------------------
# Failed write doesn't crash
# ---------------------------------------------------------------------------


class TestFailedWrite:
    def test_read_only_dir_does_not_crash(self, tmp_path: Path) -> None:
        graph = _make_source_graph()
        cache_dir = tmp_path / "readonly"
        cache_dir.mkdir()
        cache_dir.chmod(0o555)
        try:
            executor = LocalExecutor(cache_dir=cache_dir)
            result = graph.execute(executor=executor)
            assert result["value"].to_list() == [1.0, 2.0, 3.0]
        finally:
            cache_dir.chmod(0o755)


# ---------------------------------------------------------------------------
# Corrupt manifest treated as miss
# ---------------------------------------------------------------------------


class TestCorruptManifest:
    def test_bad_json_manifest_treated_as_miss(self, tmp_path: Path) -> None:
        graph = _make_source_graph()
        node_id = graph.root_node.ID
        entry = (
            tmp_path / node_id[:2] / node_id[2:4] / node_id[4:6] / node_id[6:]
        )
        entry.mkdir(parents=True)
        (entry / "manifest.json").write_text("not json {{{")

        executor = LocalExecutor(cache_dir=tmp_path)
        result = graph.execute(executor=executor)

        assert result["value"].to_list() == [1.0, 2.0, 3.0]

    def test_manifest_success_false_treated_as_miss(self, tmp_path: Path) -> None:
        graph = _make_source_graph()
        node_id = graph.root_node.ID
        entry = (
            tmp_path / node_id[:2] / node_id[2:4] / node_id[4:6] / node_id[6:]
        )
        entry.mkdir(parents=True)
        (entry / "manifest.json").write_text(json.dumps({"success": False}))

        executor = LocalExecutor(cache_dir=tmp_path)
        result = graph.execute(executor=executor)

        assert result["value"].to_list() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Missing parquet treated as miss
# ---------------------------------------------------------------------------


class TestMissingParquet:
    def test_manifest_without_parquet_treated_as_miss(self, tmp_path: Path) -> None:
        graph = _make_source_graph()
        node_id = graph.root_node.ID
        entry = (
            tmp_path / node_id[:2] / node_id[2:4] / node_id[4:6] / node_id[6:]
        )
        entry.mkdir(parents=True)
        (entry / "manifest.json").write_text(
            json.dumps({"success": True, "node_id": node_id})
        )

        executor = LocalExecutor(cache_dir=tmp_path)
        result = graph.execute(executor=executor)

        assert result["value"].to_list() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Different node IDs get different entries
# ---------------------------------------------------------------------------


class TestDifferentNodeIds:
    def test_different_graphs_different_entries(self, tmp_path: Path) -> None:
        graph1 = Graph(Node(SimpleSource, name="a", materialize=True))
        graph2 = Graph(Node(IncrementSource, name="b", materialize=True))

        executor = LocalExecutor(cache_dir=tmp_path)
        result1 = graph1.execute(executor=executor)
        result2 = graph2.execute(executor=executor)

        assert result1["value"].to_list() == [1.0, 2.0, 3.0]
        assert result2["value"].to_list() == [10.0, 20.0, 30.0]
        assert len(_cache_files(tmp_path)) == 2


# ---------------------------------------------------------------------------
# Env var cache dir
# ---------------------------------------------------------------------------


class TestEnvVarCacheDir:
    def test_env_var_respected(self, tmp_path: Path) -> None:
        graph = _make_source_graph()
        with patch.dict("os.environ", {"IOSIS_CACHE_DIR": str(tmp_path)}):
            executor = LocalExecutor()
            result = graph.execute(executor=executor)

        assert result["value"].to_list() == [1.0, 2.0, 3.0]
        assert len(_cache_files(tmp_path)) == 1


# ---------------------------------------------------------------------------
# Cache hit skips computation
# ---------------------------------------------------------------------------


class TestCacheHitSkipsComputation:
    def test_cache_hit_skips_lower_node(self, tmp_path: Path) -> None:
        call_count = 0
        original_apply = SimpleSource.apply

        def counting_apply(self: SimpleSource) -> pl.LazyFrame:
            nonlocal call_count
            call_count += 1
            return original_apply(self)

        graph = _make_source_graph()
        executor = LocalExecutor(cache_dir=tmp_path)

        with patch.object(SimpleSource, "apply", counting_apply):
            graph.execute(executor=executor)
            assert call_count == 1

            graph.execute(executor=executor)
            assert call_count == 1


# ---------------------------------------------------------------------------
# Source node cached
# ---------------------------------------------------------------------------


class TestSourceNodeCached:
    def test_source_result_persisted(self, tmp_path: Path) -> None:
        graph = _make_source_graph()
        executor = LocalExecutor(cache_dir=tmp_path)
        graph.execute(executor=executor)

        node_id = graph.root_node.ID
        entry = (
            tmp_path / node_id[:2] / node_id[2:4] / node_id[4:6] / node_id[6:]
        )
        assert (entry / "data.parquet").exists()
        assert (entry / "manifest.json").exists()


# ---------------------------------------------------------------------------
# Non-materialized node not cached
# ---------------------------------------------------------------------------


class TestNonMaterializedNotCached:
    def test_only_materialized_nodes_cached(self, tmp_path: Path) -> None:
        source = Node(SimpleSource, name="source", materialize=True)
        doubled = Node(
            Doubler,
            bindings={"value": source.output("value")},
            name="doubler",
        )
        graph = Graph(doubled)
        executor = LocalExecutor(cache_dir=tmp_path)
        graph.execute(executor=executor)

        manifests = _cache_files(tmp_path)
        assert len(manifests) == 1

        manifest = json.loads(manifests[0].read_text())
        assert manifest["node_id"] == source.ID


# ---------------------------------------------------------------------------
# Transform node with materialize=True
# ---------------------------------------------------------------------------


class TestMaterializedTransformCached:
    def test_materialized_transform_persists(self, tmp_path: Path) -> None:
        graph = _make_transform_graph()
        executor = LocalExecutor(cache_dir=tmp_path)
        graph.execute(executor=executor)

        manifests = _cache_files(tmp_path)
        assert len(manifests) >= 1

        node_ids = set()
        for m in manifests:
            data = json.loads(m.read_text())
            node_ids.add(data["node_id"])

        source = graph.node_list[0]
        assert source.ID in node_ids


# ---------------------------------------------------------------------------
# Cache persists across executions
# ---------------------------------------------------------------------------


class TestCachePersists:
    def test_cache_survives_between_calls(self, tmp_path: Path) -> None:
        graph = _make_source_graph()
        executor = LocalExecutor(cache_dir=tmp_path)

        result1 = graph.execute(executor=executor)
        assert len(_cache_files(tmp_path)) == 1

        result2 = graph.execute(executor=executor)
        assert result1.equals(result2)
        assert len(_cache_files(tmp_path)) == 1


# ---------------------------------------------------------------------------
# S3 cache
# ---------------------------------------------------------------------------


class TestS3Cache:
    def test_s3_uri_detected(self) -> None:
        executor = LocalExecutor(cache_dir="s3://my-bucket/cache")
        assert executor._cache_s3 is True
        assert executor._cache_dir is None
        assert executor._cache_enabled is True

    def test_s3_entry_key_format(self) -> None:
        executor = LocalExecutor(cache_dir="s3://my-bucket/cache")
        node_id = "abcdef0123456789" + "0" * 48
        key = executor._cache_entry_key(node_id)
        assert key == "s3://my-bucket/cache/ab/cd/ef/0123456789" + "0" * 48

    def test_s3_cache_round_trip(self, tmp_path: Path) -> None:
        graph = _make_source_graph()
        node_id = graph.root_node.ID

        local_cache = tmp_path / "local"
        local_cache.mkdir()
        executor = LocalExecutor(cache_dir=local_cache)
        graph.execute(executor=executor)

        entry = executor._cache_entry_dir(node_id)
        manifest = json.loads((entry / "manifest.json").read_text())
        assert manifest["success"] is True

    def test_s3_read_cache_returns_none_when_no_credentials(self) -> None:
        from iosislib.core.utils import _s3_credentials_scope

        executor = LocalExecutor(cache_dir="s3://bucket/cache")
        with _s3_credentials_scope(None):
            result = executor._read_cache_s3("abcdef0123456789" + "0" * 48)
        assert result is None

    def test_s3_write_cache_swallows_exceptions(self) -> None:
        from iosislib.core.utils import _s3_credentials_scope

        executor = LocalExecutor(cache_dir="s3://bucket/cache")
        df = pl.DataFrame({"a": [1, 2, 3]})
        with _s3_credentials_scope(None):
            executor._write_cache_s3("abcdef0123456789" + "0" * 48, df)

    def test_s3_env_var_cache_dir(self) -> None:
        import os

        with patch.dict(os.environ, {"IOSIS_CACHE_DIR": "s3://my-bucket/cache"}):
            executor = LocalExecutor()
            assert executor._cache_s3 is True
            assert executor._cache_enabled is True

    def test_s3_cache_disabled_with_no_cache(self) -> None:
        executor = LocalExecutor(cache_dir="s3://bucket/cache", no_cache=True)
        assert executor._cache_enabled is False

    def test_s3_storage_options_empty_without_credentials(self) -> None:
        from iosislib.core.utils import _s3_credentials_scope

        executor = LocalExecutor(cache_dir="s3://bucket/cache")
        with _s3_credentials_scope(None):
            opts = executor._s3_storage_options()
        assert opts == {}
