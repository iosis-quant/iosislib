from __future__ import annotations

import io
import json
import re
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

import iosislib.tsfn.adapters.parquet_stream as parquet_stream
from iosislib.core.graph import Graph
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, TimeAxis
from iosislib.tsfn.adapters import (
    StreamingParquetSource,
    StreamingParquetSourceConfig,
    build_parquet_chunk_manifest,
    chunk_merkle_root,
    merkle_sha256_parquet_source,
)


TIMESTAMP = [
    datetime(2026, 1, 1, 0, 0),
    datetime(2026, 1, 1, 0, 1),
    datetime(2026, 1, 1, 0, 2),
]
FLOAT_SIGNATURE = FrameSignature(columns=(("value", pl.Float64),))
DIGEST_64 = "0" * 64


def _parquet_bytes(timestamp: list[datetime], values: list[float]) -> bytes:
    buffer = io.BytesIO()
    pl.DataFrame(
        {"timestamp": timestamp, "value": values},
        schema={"timestamp": pl.Datetime, "value": pl.Float64},
    ).write_parquet(buffer)
    return buffer.getvalue()


def write_chunks(directory: Path) -> None:
    (directory / "part-000.parquet").write_bytes(
        _parquet_bytes(TIMESTAMP[:1], [1.0])
    )
    (directory / "part-001.parquet").write_bytes(
        _parquet_bytes(TIMESTAMP[1:2], [2.0])
    )


def source_parameters(
    path: Path | str,
    content_sha256: str = "",
    **extra: object,
) -> dict[str, object]:
    return {
        "path": path,
        "output_signature": FLOAT_SIGNATURE,
        "content_sha256": content_sha256,
        **extra,
    }


def test_streaming_source_streams_local_directory_in_manifest_order(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "prices"
    directory.mkdir()
    write_chunks(directory)
    root = build_parquet_chunk_manifest(directory)

    result = Graph(
        Node(
            StreamingParquetSource,
            parameters=source_parameters(directory, root),
            name="prices",
        )
    ).execute()

    assert result.columns == ["timestamp", "value"]
    assert result["timestamp"].to_list() == TIMESTAMP[:2]
    assert result["value"].to_list() == [1.0, 2.0]


def test_streaming_source_construction_does_not_touch_filesystem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def never(location: str):
        calls.append(location)
        raise AssertionError("construction must not touch the filesystem")

    monkeypatch.setattr(parquet_stream, "_open_parquet_filesystem", never)
    node = Node(
        StreamingParquetSource,
        parameters={
            "path": "s3://market-data/prices",
            "output_signature": FLOAT_SIGNATURE,
            "content_sha256": DIGEST_64,
        },
    )

    assert node.parameters.path == "s3://market-data/prices"
    assert calls == []


def test_apply_loads_only_the_manifest_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyarrow.fs import LocalFileSystem

    directory = tmp_path / "prices"
    directory.mkdir()
    write_chunks(directory)
    root = build_parquet_chunk_manifest(directory)
    manifest_path = str(directory / "manifest.json")
    opened: list[str] = []
    real_fs = LocalFileSystem()

    class CountingLocalFS:
        def get_file_info(self, target):
            return real_fs.get_file_info(target)

        def open_input_file(self, path):
            opened.append(path)
            return real_fs.open_input_file(path)

    monkeypatch.setattr(
        parquet_stream,
        "_open_parquet_filesystem",
        lambda location: (CountingLocalFS(), manifest_path),
    )
    node = Node(
        StreamingParquetSource,
        parameters=source_parameters(directory, root),
    )

    lazy = node.function.apply()

    assert isinstance(lazy, pl.LazyFrame)
    assert opened == [manifest_path]


def test_merkle_root_is_deterministic_and_order_sensitive(tmp_path: Path) -> None:
    directory = tmp_path / "prices"
    directory.mkdir()
    write_chunks(directory)
    root = build_parquet_chunk_manifest(directory)

    assert re.fullmatch(r"[0-9a-f]{64}", root)
    assert root == merkle_sha256_parquet_source(directory)
    manifest = parquet_stream.ChunkManifest.from_bytes(
        (directory / "manifest.json").read_bytes()
    )
    assert chunk_merkle_root(manifest.chunks) == root
    swapped = parquet_stream.ChunkManifest(chunks=tuple(reversed(manifest.chunks)))
    assert swapped.merkle_root != root


def test_manifest_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    directory = tmp_path / "prices"
    directory.mkdir()
    write_chunks(directory)
    build_parquet_chunk_manifest(directory)
    manifest_path = directory / "manifest.json"
    manifest = parquet_stream.ChunkManifest.from_bytes(manifest_path.read_bytes())

    assert manifest.format == "iosis.parquet-chunk-manifest"
    assert manifest.version == 1
    assert len(manifest.chunks) == 2
    assert parquet_stream.ChunkManifest.from_bytes(
        manifest.to_json_bytes()
    ) == manifest

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["merkle"]["root"] = DIGEST_64
    with pytest.raises(ValueError, match="does not match its chunks"):
        parquet_stream.ChunkManifest.from_bytes(
            json.dumps(data, sort_keys=True).encode("utf-8")
        )


def test_pinned_root_mismatch_fails_with_execution_context(tmp_path: Path) -> None:
    directory = tmp_path / "prices"
    directory.mkdir()
    write_chunks(directory)
    build_parquet_chunk_manifest(directory)
    node = Node(
        StreamingParquetSource,
        parameters=source_parameters(directory, DIGEST_64),
        name="snapshot",
    )

    with pytest.raises(
        RuntimeError,
        match=(
            rf"Execution failed at node 'snapshot' "
            rf"\(StreamingParquetSource@{re.escape(StreamingParquetSource.VERSION)}\).*"
            r"digest mismatch"
        ),
    ):
        Graph(node).execute()


def test_empty_digest_streams_without_verification(tmp_path: Path) -> None:
    directory = tmp_path / "prices"
    directory.mkdir()
    write_chunks(directory)
    build_parquet_chunk_manifest(directory)

    result = Graph(
        Node(StreamingParquetSource, parameters=source_parameters(directory, ""))
    ).execute()

    assert result["value"].to_list() == [1.0, 2.0]


def test_verify_layout_detects_missing_object(tmp_path: Path) -> None:
    directory = tmp_path / "prices"
    directory.mkdir()
    write_chunks(directory)
    root = build_parquet_chunk_manifest(directory)
    (directory / "part-001.parquet").unlink()
    node = Node(
        StreamingParquetSource,
        parameters=source_parameters(directory, root, verify_layout=True),
        name="layout",
    )

    with pytest.raises(RuntimeError, match="chunk is missing"):
        Graph(node).execute()


def test_verify_layout_rejects_changed_size(tmp_path: Path) -> None:
    directory = tmp_path / "prices"
    directory.mkdir()
    write_chunks(directory)
    root = build_parquet_chunk_manifest(directory)
    with (directory / "part-000.parquet").open("ab") as file:
        file.write(b"trailing-bytes")
    node = Node(
        StreamingParquetSource,
        parameters=source_parameters(directory, root, verify_layout=True),
        name="layout",
    )

    with pytest.raises(RuntimeError, match="size mismatch"):
        Graph(node).execute()


def test_s3_apply_reads_only_manifest_and_layout_stats_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyarrow.fs import FileInfo, FileType

    directory = tmp_path / "prices"
    directory.mkdir()
    write_chunks(directory)
    root = build_parquet_chunk_manifest(directory)
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = parquet_stream.ChunkManifest.from_bytes(manifest_bytes)
    chunk_sizes = {chunk.key: chunk.size for chunk in manifest.chunks}
    opened: list[str] = []
    statted: set[str] = set()

    class FakeS3FileSystem:
        def get_file_info(self, target: str) -> FileInfo:
            if target == "market-data/prices/manifest.json":
                return FileInfo(target, FileType.File)
            if target.startswith("market-data/prices/"):
                key = target.removeprefix("market-data/prices/")
                statted.add(key)
                size = chunk_sizes.get(key)
                if size is not None:
                    return FileInfo(target, FileType.File, size=size)
            return FileInfo(target, FileType.NotFound)

        def open_input_file(self, path: str) -> io.BytesIO:
            opened.append(path)
            if path != "market-data/prices/manifest.json":
                raise AssertionError("chunk bytes must not be opened at apply time")
            return io.BytesIO(manifest_bytes)

    def fake_open(location: str):
        if location.endswith("/manifest.json"):
            return FakeS3FileSystem(), "market-data/prices/manifest.json"
        return FakeS3FileSystem(), "market-data/prices"

    monkeypatch.setattr(parquet_stream, "_open_parquet_filesystem", fake_open)
    node = Node(
        StreamingParquetSource,
        parameters=source_parameters("s3://market-data/prices", root),
    )

    lazy = node.function.apply()
    assert isinstance(lazy, pl.LazyFrame)
    assert opened == ["market-data/prices/manifest.json"]
    assert not statted

    statted.clear()
    node = Node(
        StreamingParquetSource,
        parameters=source_parameters(
            "s3://market-data/prices", root, verify_layout=True
        ),
    )
    node.function.apply()
    assert statted == {chunk.key for chunk in manifest.chunks}


def test_schema_dsl_form(tmp_path: Path) -> None:
    directory = tmp_path / "prices"
    directory.mkdir()
    write_chunks(directory)
    build_parquet_chunk_manifest(directory)

    result = Graph(
        Node(
            StreamingParquetSource,
            parameters={
                "path": directory,
                "schema": {"time": "timestamp", "columns": {"value": "float64"}},
                "content_sha256": "",
            },
        )
    ).execute()

    assert result.schema["value"] == pl.Float64
    assert result["value"].to_list() == [1.0, 2.0]


def test_config_validation_is_explicit(tmp_path: Path) -> None:
    directory = tmp_path / "prices"
    directory.mkdir()

    with pytest.raises(ValueError, match="declare a time axis"):
        StreamingParquetSourceConfig(directory, FrameSignature.empty(), DIGEST_64)

    with pytest.raises(ValueError, match="hexadecimal"):
        StreamingParquetSourceConfig(directory, FLOAT_SIGNATURE, "not-a-digest")

    with pytest.raises(ValueError, match="exactly one"):
        StreamingParquetSourceConfig(
            directory,
            FLOAT_SIGNATURE,
            DIGEST_64,
            schema={"time": "timestamp", "columns": {"value": "float64"}},
        )

    with pytest.raises(TypeError, match="verify_layout"):
        StreamingParquetSourceConfig(
            directory, FLOAT_SIGNATURE, "", verify_layout=1
        )


def test_s3_location_normalization() -> None:
    config = StreamingParquetSourceConfig(
        "s3://market-data/prices/", FLOAT_SIGNATURE, ""
    )
    assert config.path == "s3://market-data/prices"
    assert config.manifest is None
    assert config.content_sha256 == ""

    explicit = StreamingParquetSourceConfig(
        "s3://market-data/prices", FLOAT_SIGNATURE, "", manifest="s3://meta/1.json"
    )
    assert explicit.manifest == "s3://meta/1.json"

    with pytest.raises(ValueError):
        StreamingParquetSourceConfig(
            "https://bucket/prices", FLOAT_SIGNATURE, DIGEST_64
        )


def test_node_ids_are_deterministic(tmp_path: Path) -> None:
    directory = tmp_path / "prices"
    directory.mkdir()
    write_chunks(directory)
    root = build_parquet_chunk_manifest(directory)
    parameters = source_parameters(directory, root)

    first = Node(StreamingParquetSource, parameters=parameters)
    second = Node(StreamingParquetSource, parameters=parameters)

    assert first.ID == second.ID
    assert first.parameters == second.parameters


def test_streaming_source_version() -> None:
    assert StreamingParquetSource.VERSION == "0.1.0"
    assert StreamingParquetSource.type_signature  # concrete class


def test_missing_manifest_fails_with_execution_context(tmp_path: Path) -> None:
    directory = tmp_path / "prices"
    directory.mkdir()
    write_chunks(directory)
    node = Node(
        StreamingParquetSource,
        parameters=source_parameters(directory, ""),
        name="manifestless",
    )

    with pytest.raises(RuntimeError, match="manifest does not exist"):
        Graph(node).execute()


def test_streamed_frame_preserves_declared_time_axis(tmp_path: Path) -> None:
    directory = tmp_path / "prices"
    directory.mkdir()
    write_chunks(directory)
    build_parquet_chunk_manifest(directory)
    signature = FrameSignature(
        time=TimeAxis(dtype=pl.Datetime),
        columns=(("value", pl.Float64),),
    )

    result = Graph(
        Node(
            StreamingParquetSource,
            parameters={
                "path": directory,
                "output_signature": signature,
                "content_sha256": "",
            },
        )
    ).execute()

    assert result.schema == pl.Schema(
        {"timestamp": pl.Datetime("us"), "value": pl.Float64}
    )