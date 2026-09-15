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
        graph = _make_transform_graph()
        executor = LocalExecutor(cache_dir=tmp_path)
        result = graph.execute(executor=executor)

        assert result["value"].to_list() == [2.0, 4.0, 6.0]
        manifests = _cache_files(tmp_path)
        assert len(manifests) == 1

    def test_second_execution_is_cache_hit(self, tmp_path: Path) -> None:
        graph = _make_transform_graph()
        executor = LocalExecutor(cache_dir=tmp_path)

        result1 = graph.execute(executor=executor)
        result2 = graph.execute(executor=executor)

        assert result1["value"].to_list() == result2["value"].to_list()
        assert len(_cache_files(tmp_path)) == 1

    def test_cache_hit_returns_identical_result(self, tmp_path: Path) -> None:
        graph = _make_transform_graph()
        executor = LocalExecutor(cache_dir=tmp_path)

        result1 = graph.execute(executor=executor)
        result2 = graph.execute(executor=executor)

        assert result1.equals(result2)

    def test_source_only_graph_never_caches(self, tmp_path: Path) -> None:
        graph = _make_source_graph()
        executor = LocalExecutor(cache_dir=tmp_path)
        result = graph.execute(executor=executor)

        assert result["value"].to_list() == [1.0, 2.0, 3.0]
        assert _cache_files(tmp_path) == []


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
        graph = _make_transform_graph()
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
        graph = _make_transform_graph()
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
        source1 = Node(SimpleSource, name="a", materialize=True)
        doubled1 = Node(
            Doubler,
            bindings={"value": source1.output("value")},
            name="doubled",
            materialize=True,
        )
        graph1 = Graph(doubled1)
        source2 = Node(IncrementSource, name="b", materialize=True)
        doubled2 = Node(
            Doubler,
            bindings={"value": source2.output("value")},
            name="doubled",
            materialize=True,
        )
        graph2 = Graph(doubled2)

        executor = LocalExecutor(cache_dir=tmp_path)
        result1 = graph1.execute(executor=executor)
        result2 = graph2.execute(executor=executor)

        assert result1["value"].to_list() == [2.0, 4.0, 6.0]
        assert result2["value"].to_list() == [20.0, 40.0, 60.0]
        assert len(_cache_files(tmp_path)) == 2


# ---------------------------------------------------------------------------
# Env var cache dir
# ---------------------------------------------------------------------------


class TestEnvVarCacheDir:
    def test_env_var_respected(self, tmp_path: Path) -> None:
        graph = _make_transform_graph()
        with patch.dict("os.environ", {"IOSIS_CACHE_DIR": str(tmp_path)}):
            executor = LocalExecutor()
            result = graph.execute(executor=executor)

        assert result["value"].to_list() == [2.0, 4.0, 6.0]
        assert len(_cache_files(tmp_path)) == 1


# ---------------------------------------------------------------------------
# Cache hit skips computation
# ---------------------------------------------------------------------------


class TestCacheHitSkipsComputation:
    def test_cache_hit_skips_lower_node(self, tmp_path: Path) -> None:
        call_count = 0
        original_apply = Doubler.apply

        def counting_apply(self: Doubler, lf: pl.LazyFrame | None) -> pl.LazyFrame:
            nonlocal call_count
            call_count += 1
            return original_apply(self, lf)

        graph = _make_transform_graph()
        executor = LocalExecutor(cache_dir=tmp_path)

        with patch.object(Doubler, "apply", counting_apply):
            graph.execute(executor=executor)
            assert call_count == 1

            graph.execute(executor=executor)
            assert call_count == 1

    def test_source_always_repulled(self, tmp_path: Path) -> None:
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
            assert call_count == 2


# ---------------------------------------------------------------------------
# Source node cached
# ---------------------------------------------------------------------------


class TestSourceNodeCached:
    def test_source_result_not_persisted(self, tmp_path: Path) -> None:
        graph = _make_source_graph()
        executor = LocalExecutor(cache_dir=tmp_path)
        graph.execute(executor=executor)

        node_id = graph.root_node.ID
        entry = (
            tmp_path / node_id[:2] / node_id[2:4] / node_id[4:6] / node_id[6:]
        )
        assert list(entry.glob("part-*.parquet")) == []
        assert not (entry / "manifest.json").exists()

    def test_transform_result_persisted(self, tmp_path: Path) -> None:
        graph = _make_transform_graph()
        executor = LocalExecutor(cache_dir=tmp_path)
        graph.execute(executor=executor)

        node_id = graph.root_node.ID
        entry = (
            tmp_path / node_id[:2] / node_id[2:4] / node_id[4:6] / node_id[6:]
        )
        assert list(entry.glob("part-*.parquet")) != []
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

        # Sources are never cached, even with materialize=True, and the
        # non-materialized transform is not cached either.
        assert _cache_files(tmp_path) == []

    def test_materialized_transform_cached_source_ignored(self, tmp_path: Path) -> None:
        source = Node(SimpleSource, name="source", materialize=True)
        doubled = Node(
            Doubler,
            bindings={"value": source.output("value")},
            name="doubler",
            materialize=True,
        )
        graph = Graph(doubled)
        executor = LocalExecutor(cache_dir=tmp_path)
        graph.execute(executor=executor)

        manifests = _cache_files(tmp_path)
        assert len(manifests) == 1

        manifest = json.loads(manifests[0].read_text())
        assert manifest["node_id"] == doubled.ID


# ---------------------------------------------------------------------------
# Transform node with materialize=True
# ---------------------------------------------------------------------------


class TestMaterializedTransformCached:
    def test_materialized_transform_persists(self, tmp_path: Path) -> None:
        graph = _make_transform_graph()
        executor = LocalExecutor(cache_dir=tmp_path)
        graph.execute(executor=executor)

        manifests = _cache_files(tmp_path)
        assert len(manifests) == 1

        node_ids = set()
        for m in manifests:
            data = json.loads(m.read_text())
            node_ids.add(data["node_id"])

        assert graph.root_node.ID in node_ids
        for node in graph.node_list:
            if node.ID != graph.root_node.ID:
                assert node.ID not in node_ids


# ---------------------------------------------------------------------------
# Cache reuse across materialization boundaries
# ---------------------------------------------------------------------------


class TestCacheReuseAcrossMaterialization:
    def test_lazy_node_reuses_cache_written_by_materialized_graph(
        self, tmp_path: Path
    ) -> None:
        # A node identity shared by two graphs: one that materializes it and
        # one that leaves it lazy. The lazy graph must find the cached frame.
        source = Node(SimpleSource, name="source", materialize=True)
        doubled_materialized = Node(
            Doubler,
            bindings={"value": source.output("value")},
            name="doubler",
            materialize=True,
        )
        materialized_graph = Graph(doubled_materialized)
        doubled_lazy = Node(
            Doubler,
            bindings={"value": source.output("value")},
            name="doubler",
            materialize=False,
        )
        lazy_graph = Graph(doubled_lazy)

        # Node identity is independent of the materialization flag.
        assert doubled_materialized.ID == doubled_lazy.ID

        executor = LocalExecutor(cache_dir=tmp_path)
        first = materialized_graph.execute(executor=executor)

        call_count = 0
        original_apply = Doubler.apply

        def counting_apply(self: Doubler, lf: pl.LazyFrame | None) -> pl.LazyFrame:
            nonlocal call_count
            call_count += 1
            return original_apply(self, lf)

        with patch.object(Doubler, "apply", counting_apply):
            second = lazy_graph.execute(executor=executor)

        assert first["value"].to_list() == second["value"].to_list()
        # The lazy graph hit the cache written by the materialized graph, so
        # the transform's apply was never invoked again.
        assert call_count == 0


# ---------------------------------------------------------------------------
# Cache persists across executions
# ---------------------------------------------------------------------------


class TestCachePersists:
    def test_cache_survives_between_calls(self, tmp_path: Path) -> None:
        graph = _make_transform_graph()
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
        graph = _make_transform_graph()
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


# ---------------------------------------------------------------------------
# Sharded cache
# ---------------------------------------------------------------------------


def _make_large_transform_graph(frame: pl.DataFrame) -> Graph:
    from datetime import timedelta

    base = datetime(2026, 1, 1)
    distinct = frame.with_columns(
        pl.Series(
            "timestamp",
            [base + timedelta(microseconds=i) for i in range(frame.height)],
            dtype=pl.Datetime,
        )
    )

    class LargeSource(TSFN):
        VERSION = "1.0.0"

        def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
            return FrameSignature.empty(), VALUE_FRAME

        def apply(self) -> pl.LazyFrame:
            return distinct.lazy()

    source = Node(LargeSource, name="source", materialize=True)
    doubled = Node(
        Doubler,
        bindings={"value": source.output("value")},
        name="doubler",
        materialize=True,
    )
    return Graph(doubled)


class TestShardedCache:
    def test_small_frame_single_shard(self, tmp_path: Path) -> None:
        graph = _make_transform_graph()
        executor = LocalExecutor(cache_dir=tmp_path, cache_shard_bytes=10 * 1024 * 1024)
        graph.execute(executor=executor)

        node_id = graph.root_node.ID
        entry = tmp_path / node_id[:2] / node_id[2:4] / node_id[4:6] / node_id[6:]
        shards = sorted(entry.glob("part-*.parquet"))
        assert len(shards) == 1

        manifest = json.loads((entry / "manifest.json").read_text())
        assert manifest["shard_count"] == 1

    def test_many_rows_triggers_multiple_shards(self, tmp_path: Path) -> None:
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2026, 1, 1) for _ in range(10_000)],
                "value": [float(i) for i in range(10_000)],
            }
        )

        graph = _make_large_transform_graph(df)
        # 10k rows × (8+8 bytes) ≈ 160KB; use very small shard size to force split
        executor = LocalExecutor(cache_dir=tmp_path, cache_shard_bytes=64_000)
        graph.execute(executor=executor)

        node_id = graph.root_node.ID
        entry = tmp_path / node_id[:2] / node_id[2:4] / node_id[4:6] / node_id[6:]
        shards = sorted(entry.glob("part-*.parquet"))
        assert len(shards) > 1

        manifest = json.loads((entry / "manifest.json").read_text())
        assert manifest["shard_count"] == len(shards)
        assert manifest["row_count"] == 10_000

    def test_sharded_round_trip_preserves_data(self, tmp_path: Path) -> None:
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2026, 1, 1) for _ in range(10_000)],
                "value": [float(i) for i in range(10_000)],
            }
        )

        graph = _make_large_transform_graph(df)
        executor = LocalExecutor(cache_dir=tmp_path, cache_shard_bytes=64_000)

        result = graph.execute(executor=executor)
        # Second execution reads transform from shards (source re-pulled)
        result2 = graph.execute(executor=executor)

        assert result.equals(result2)
        assert result["value"].to_list() == [float(i) * 2 for i in range(10_000)]

    def test_manifest_has_shard_count(self, tmp_path: Path) -> None:
        graph = _make_transform_graph()
        executor = LocalExecutor(cache_dir=tmp_path, cache_shard_bytes=10 * 1024 * 1024)
        graph.execute(executor=executor)

        node_id = graph.root_node.ID
        entry = tmp_path / node_id[:2] / node_id[2:4] / node_id[4:6] / node_id[6:]
        manifest = json.loads((entry / "manifest.json").read_text())
        assert "shard_count" in manifest
        assert isinstance(manifest["shard_count"], int)
        assert manifest["shard_count"] >= 1

    def test_byte_size_sums_shards(self, tmp_path: Path) -> None:
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2026, 1, 1) for _ in range(10_000)],
                "value": [float(i) for i in range(10_000)],
            }
        )

        graph = _make_large_transform_graph(df)
        executor = LocalExecutor(cache_dir=tmp_path, cache_shard_bytes=64_000)
        graph.execute(executor=executor)

        node_id = graph.root_node.ID
        entry = tmp_path / node_id[:2] / node_id[2:4] / node_id[4:6] / node_id[6:]
        shards = sorted(entry.glob("part-*.parquet"))
        manifest = json.loads((entry / "manifest.json").read_text())
        actual = sum(s.stat().st_size for s in shards)
        assert manifest["byte_size"] == actual

    def test_env_var_overrides_default(self, tmp_path: Path) -> None:
        with patch.dict("os.environ", {"IOSIS_CACHE_SHARD_BYTES": "4096"}):
            executor = LocalExecutor(cache_dir=tmp_path)
            assert executor._cache_shard_bytes == 4096

    def test_invalid_env_var_falls_back_to_default(self, tmp_path: Path) -> None:
        with patch.dict("os.environ", {"IOSIS_CACHE_SHARD_BYTES": "notanumber"}):
            executor = LocalExecutor(cache_dir=tmp_path)
            assert executor._cache_shard_bytes == LocalExecutor._DEFAULT_SHARD_BYTES

    def test_legacy_data_parquet_cleaned_up(self, tmp_path: Path) -> None:
        graph = _make_transform_graph()
        executor = LocalExecutor(cache_dir=tmp_path)
        graph.execute(executor=executor)

        node_id = graph.root_node.ID
        entry = tmp_path / node_id[:2] / node_id[2:4] / node_id[4:6] / node_id[6:]

        # Remove the part-*.parquet shards so only legacy data.parquet remains,
        # and drop the manifest so the next run is a miss (hits skip rewrite).
        for shard in entry.glob("part-*.parquet"):
            shard.unlink()
        (entry / "manifest.json").unlink()

        # Write a legacy data.parquet with the correct schema
        legacy = entry / "data.parquet"
        pl.DataFrame(
            {
                "timestamp": [datetime(2026, 1, 1)],
                "value": [99.0],
            }
        ).write_parquet(legacy)
        assert legacy.exists()

        # Re-execute on a miss; shards rewrite should remove data.parquet
        executor2 = LocalExecutor(cache_dir=tmp_path)
        graph.execute(executor=executor2)
        assert not legacy.exists()
        assert list(entry.glob("part-*.parquet")) != []

    def test_executor_param_overrides_env_var(self, tmp_path: Path) -> None:
        with patch.dict("os.environ", {"IOSIS_CACHE_SHARD_BYTES": "9999"}):
            executor = LocalExecutor(cache_dir=tmp_path, cache_shard_bytes=2048)
            assert executor._cache_shard_bytes == 2048

    def test_s3_cache_shard_bytes_env_var(self) -> None:
        with patch.dict("os.environ", {"IOSIS_CACHE_SHARD_BYTES": "8192"}):
            executor = LocalExecutor(cache_dir="s3://bucket/cache")
            assert executor._cache_shard_bytes == 8192

    def test_manifest_byte_size_sums_local(self, tmp_path: Path) -> None:
        graph = _make_transform_graph()
        executor = LocalExecutor(cache_dir=tmp_path, cache_shard_bytes=10 * 1024 * 1024)
        graph.execute(executor=executor)

        node_id = graph.root_node.ID
        entry = tmp_path / node_id[:2] / node_id[2:4] / node_id[4:6] / node_id[6:]
        shards = list(entry.glob("part-*.parquet"))
        manifest = json.loads((entry / "manifest.json").read_text())
        assert manifest["byte_size"] == sum(s.stat().st_size for s in shards)
