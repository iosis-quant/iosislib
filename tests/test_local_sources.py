from __future__ import annotations

import io
import re
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

import iosislib.tsfn.adapters.local_sources as local_sources
from iosislib.core.graph import Graph
from iosislib.core.node import Node
from iosislib.core.tsfn import ColumnSignature, FrameSignature, TimeAxis
from iosislib.tsfn.adapters import (
    CSVSource,
    CSVSourceConfig,
    ParquetSource,
    ParquetSourceConfig,
    sha256_file,
    sha256_parquet_source,
)
from iosislib.tsfn.transforms import Delta, Logit


TIMESTAMP = [
    datetime(2026, 1, 1, 0, 0),
    datetime(2026, 1, 1, 0, 1),
    datetime(2026, 1, 1, 0, 2),
]
FLOAT_SIGNATURE = FrameSignature(columns=(("value", pl.Float64),))


def source_parameters(
    path: Path,
    signature: FrameSignature = FLOAT_SIGNATURE,
) -> dict[str, object]:
    return {
        "path": path,
        "output_signature": signature,
        "content_sha256": sha256_file(path),
    }


def write_csv(path: Path) -> None:
    pl.DataFrame(
        {
            "timestamp": TIMESTAMP,
            "value": [0.25, 0.5, 0.75],
            "ignored": [1, 2, 3],
        },
        schema={
            "timestamp": pl.Datetime,
            "value": pl.Float64,
            "ignored": pl.Int64,
        },
    ).write_csv(path)


def write_parquet(path: Path) -> None:
    pl.DataFrame(
        {
            "timestamp": TIMESTAMP,
            "value": [1.0, 2.0, 3.0],
            "ignored": [1, 2, 3],
        },
        schema={
            "timestamp": pl.Datetime,
            "value": pl.Float64,
            "ignored": pl.Int64,
        },
    ).write_parquet(path)


@pytest.mark.parametrize(
    ("source_cls", "writer", "suffix", "expected"),
    [
        (CSVSource, write_csv, ".csv", [0.25, 0.5, 0.75]),
        (ParquetSource, write_parquet, ".parquet", [1.0, 2.0, 3.0]),
    ],
)
def test_local_sources_return_lazy_projection_from_verified_snapshot(
    tmp_path: Path,
    source_cls: type[CSVSource] | type[ParquetSource],
    writer,
    suffix: str,
    expected: list[float],
) -> None:
    path = tmp_path / f"prices{suffix}"
    writer(path)
    source = Node(source_cls, parameters=source_parameters(path))

    lazy_result = source.function.apply()
    result = Graph(source).execute()

    assert isinstance(lazy_result, pl.LazyFrame)
    assert source.function.signature[0] == FrameSignature.empty()
    assert result.columns == ["timestamp", "value"]
    assert result["timestamp"].to_list() == TIMESTAMP
    assert result["value"].to_list() == expected


def test_parquet_source_preserves_declared_shape(tmp_path: Path) -> None:
    path = tmp_path / "vectors.parquet"
    pl.DataFrame(
        {
            "timestamp": TIMESTAMP[:2],
            "vector": [[1.0, 2.0], [3.0, 4.0]],
        },
        schema={
            "timestamp": pl.Datetime,
            "vector": pl.Array(pl.Float64, 2),
        },
    ).write_parquet(path)
    signature = FrameSignature(
        columns=(ColumnSignature("vector", pl.Float64, (2,)),),
    )

    result = Graph(
        Node(ParquetSource, parameters=source_parameters(path, signature))
    ).execute()

    assert result.schema["vector"] == pl.Array(pl.Float64, 2)
    assert result["vector"].to_list() == [[1.0, 2.0], [3.0, 4.0]]


def test_local_source_construction_does_not_inspect_file(tmp_path: Path) -> None:
    missing = tmp_path / "not-created.csv"
    node = Node(
        CSVSource,
        parameters={
            "path": missing,
            "output_signature": FLOAT_SIGNATURE,
            "content_sha256": "0" * 64,
        },
    )

    assert node.parameters.path == str(missing)
    assert node.outputs == {"value": pl.Float64}


def test_local_source_configs_and_node_ids_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "prices.csv"
    write_csv(path)
    digest = sha256_file(path)
    first_config = CSVSourceConfig(path, FLOAT_SIGNATURE, digest.upper())
    second_config = CSVSourceConfig(str(path), FLOAT_SIGNATURE, digest)
    first_node = Node(CSVSource, parameters=source_parameters(path))
    second_node = Node(CSVSource, parameters=source_parameters(path))

    assert first_config == second_config
    assert first_config.to_dict() == second_config.to_dict()
    assert first_node.ID == second_node.ID


def test_csv_source_preserves_separator_date_parsing_and_schema(tmp_path: Path) -> None:
    path = tmp_path / "prices.csv"
    path.write_text(
        "timestamp;value;ignored\n"
        "2026-01-01T00:00:00;0.25;discarded\n"
        "2026-01-01T00:01:00;0.50;discarded\n",
        encoding="utf-8",
    )
    node = Node(
        CSVSource,
        parameters={**source_parameters(path), "separator": ";"},
    )

    result = Graph(node).execute()

    assert result.schema == pl.Schema(
        {"timestamp": pl.Datetime("us"), "value": pl.Float64}
    )
    assert result["timestamp"].to_list() == TIMESTAMP[:2]
    assert result["value"].to_list() == [0.25, 0.5]


def test_local_source_config_validation_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="declare a time axis"):
        ParquetSourceConfig(tmp_path / "data.parquet", FrameSignature.empty(), "0" * 64)

    with pytest.raises(ValueError, match="64-character hexadecimal"):
        ParquetSourceConfig(tmp_path / "data.parquet", FLOAT_SIGNATURE, "not-a-digest")

    shaped = FrameSignature(
        columns=(ColumnSignature("vector", pl.Float64, (2,)),),
    )
    with pytest.raises(ValueError, match="does not support shaped columns"):
        CSVSourceConfig(tmp_path / "data.csv", shaped, "0" * 64)

    with pytest.raises(ValueError, match="single-byte character"):
        CSVSourceConfig(
            tmp_path / "data.csv",
            FLOAT_SIGNATURE,
            "0" * 64,
            separator="::",
        )


@pytest.mark.parametrize("source_cls", [CSVSource, ParquetSource])
def test_missing_local_file_retains_execution_context(
    tmp_path: Path,
    source_cls: type[CSVSource] | type[ParquetSource],
) -> None:
    node = Node(
        source_cls,
        parameters={
            "path": tmp_path / "missing.data",
            "output_signature": FLOAT_SIGNATURE,
            "content_sha256": "0" * 64,
        },
        name="local_prices",
    )

    with pytest.raises(
        RuntimeError,
        match=rf"Execution failed at node 'local_prices' \({source_cls.__name__}@{re.escape(source_cls.VERSION)}\)",
    ):
        Graph(node).execute()


@pytest.mark.parametrize(
    ("source_cls", "writer", "suffix"),
    [
        (CSVSource, write_csv, ".csv"),
        (ParquetSource, write_parquet, ".parquet"),
    ],
)
def test_initial_digest_mismatch_fails_with_execution_context(
    tmp_path: Path,
    source_cls: type[CSVSource] | type[ParquetSource],
    writer,
    suffix: str,
) -> None:
    path = tmp_path / f"prices{suffix}"
    writer(path)
    parameters = source_parameters(path)
    with path.open("ab") as file:
        file.write(b"changed")
    node = Node(source_cls, parameters=parameters, name="snapshot")

    with pytest.raises(
        RuntimeError,
        match=(
            rf"Execution failed at node 'snapshot' "
            rf"\({source_cls.__name__}@{re.escape(source_cls.VERSION)}\).*digest mismatch"
        ),
    ):
        Graph(node).execute()


@pytest.mark.parametrize(
    ("source_cls", "writer", "suffix", "original", "replacement"),
    [
        (CSVSource, write_csv, ".csv", [0.25, 0.5, 0.75], [0.1, 0.2, 0.3]),
        (ParquetSource, write_parquet, ".parquet", [1.0, 2.0, 3.0], [7.0, 8.0, 9.0]),
    ],
)
def test_source_parses_verified_bytes_when_path_mutates_after_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_cls: type[CSVSource] | type[ParquetSource],
    writer,
    suffix: str,
    original: list[float],
    replacement: list[float],
) -> None:
    path = tmp_path / f"prices{suffix}"
    writer(path)
    node = Node(source_cls, parameters=source_parameters(path), name="snapshot")

    def replace_source_file() -> None:
        replacement_frame = pl.DataFrame(
            {
                "timestamp": TIMESTAMP,
                "value": replacement,
                "ignored": [4, 5, 6],
            },
            schema={
                "timestamp": pl.Datetime,
                "value": pl.Float64,
                "ignored": pl.Int64,
            },
        )
        if suffix == ".csv":
            replacement_frame.write_csv(path)
        else:
            replacement_frame.write_parquet(path)

    if source_cls is CSVSource:
        read_verified_snapshot = local_sources._read_verified_snapshot

        def read_then_replace(snapshot_path: str, expected_sha256: str) -> bytes:
            snapshot = read_verified_snapshot(snapshot_path, expected_sha256)
            replace_source_file()
            return snapshot

        monkeypatch.setattr(
            local_sources,
            "_read_verified_snapshot",
            read_then_replace,
        )
    else:
        read_verified_parquet = local_sources._read_verified_parquet_snapshot

        def read_parquet_then_replace(
            snapshot_path: str,
            expected_sha256: str,
        ) -> local_sources.ParquetSnapshot:
            snapshot = read_verified_parquet(snapshot_path, expected_sha256)
            replace_source_file()
            return snapshot

        monkeypatch.setattr(
            local_sources,
            "_read_verified_parquet_snapshot",
            read_parquet_then_replace,
        )

    result = Graph(node).execute()

    assert result["value"].to_list() == original
    if suffix == ".csv":
        assert pl.read_csv(path)["value"].to_list() == replacement
    else:
        assert pl.read_parquet(path)["value"].to_list() == replacement


def test_parquet_schema_mismatch_is_clear_and_contextual(tmp_path: Path) -> None:
    path = tmp_path / "wrong-value.parquet"
    pl.DataFrame(
        {"timestamp": TIMESTAMP[:1], "value": [1]},
        schema={"timestamp": pl.Datetime, "value": pl.Int64},
    ).write_parquet(path)
    node = Node(
        ParquetSource,
        parameters=source_parameters(path),
        name="wrong_value",
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"Execution failed at node 'wrong_value' "
            r"\(ParquetSource@0\.3\.0\).*Column 'value' type mismatch"
        ),
    ):
        Graph(node).execute()


@pytest.mark.parametrize(
    ("actual_time", "declared_time", "message"),
    [
        (pl.Date, TimeAxis(dtype=pl.Datetime), "type mismatch"),
        (
            pl.Datetime(time_zone="UTC"),
            TimeAxis(dtype=pl.Datetime, timezone=None),
            "timezone mismatch",
        ),
    ],
)
def test_parquet_timestamp_contract_mismatch_is_clear(
    tmp_path: Path,
    actual_time: pl.DataType,
    declared_time: TimeAxis,
    message: str,
) -> None:
    path = tmp_path / f"wrong-time-{message.replace(' ', '-')}.parquet"
    timestamp = (
        [datetime(2026, 1, 1).date()]
        if actual_time == pl.Date
        else [datetime(2026, 1, 1)]
    )
    pl.DataFrame(
        {"timestamp": timestamp, "value": [1.0]},
        schema={"timestamp": actual_time, "value": pl.Float64},
    ).write_parquet(path)
    signature = FrameSignature(time=declared_time, columns=(("value", pl.Float64),))
    node = Node(ParquetSource, parameters=source_parameters(path, signature))

    with pytest.raises(RuntimeError, match=message):
        Graph(node).execute()


@pytest.mark.parametrize(
    ("source_cls", "suffix", "content"),
    [
        (
            CSVSource,
            ".csv",
            b"timestamp,value\n2026-01-01T00:00:00,not-a-float\n",
        ),
        (ParquetSource, ".parquet", b"not a parquet file"),
    ],
)
def test_malformed_source_fails_with_execution_context(
    tmp_path: Path,
    source_cls: type[CSVSource] | type[ParquetSource],
    suffix: str,
    content: bytes,
) -> None:
    path = tmp_path / f"invalid{suffix}"
    path.write_bytes(content)
    node = Node(source_cls, parameters=source_parameters(path), name="malformed")

    with pytest.raises(
        RuntimeError,
        match=(
            rf"Execution failed at node 'malformed' "
            rf"\({source_cls.__name__}@{re.escape(source_cls.VERSION)}\)"
        ),
    ):
        Graph(node).execute()


def _parquet_bytes(timestamp: list[datetime], values: list[float]) -> bytes:
    buffer = io.BytesIO()
    pl.DataFrame(
        {"timestamp": timestamp, "value": values},
        schema={"timestamp": pl.Datetime, "value": pl.Float64},
    ).write_parquet(buffer)
    return buffer.getvalue()


def test_parquet_source_loads_local_directory_snapshot(tmp_path: Path) -> None:
    directory = tmp_path / "prices"
    (directory / "nested").mkdir(parents=True)
    first = directory / "nested" / "part-000.parquet"
    second = directory / "part-001.parquet"
    first.write_bytes(_parquet_bytes(TIMESTAMP[:1], [1.0]))
    second.write_bytes(_parquet_bytes(TIMESTAMP[1:2], [2.0]))
    (directory / "README.txt").write_text("ignored", encoding="utf-8")

    node = Node(
        ParquetSource,
        parameters={
            "path": directory,
            "output_signature": FLOAT_SIGNATURE,
            "content_sha256": sha256_parquet_source(directory),
        },
    )

    result = Graph(node).execute()

    assert result["timestamp"].to_list() == TIMESTAMP[:2]
    assert result["value"].to_list() == [1.0, 2.0]
    assert sha256_parquet_source(first) == sha256_file(first)

    single_directory = tmp_path / "single"
    single_directory.mkdir()
    only_part = single_directory / "part.parquet"
    only_part.write_bytes(_parquet_bytes(TIMESTAMP[:1], [3.0]))
    assert sha256_parquet_source(single_directory) == sha256_file(only_part)


def test_parquet_source_loads_s3_prefix_without_network_in_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyarrow.fs import FileInfo, FileType

    objects = {
        "market-data/prices/part-001.parquet": _parquet_bytes(TIMESTAMP[1:2], [2.0]),
        "market-data/prices/nested/part-000.parquet": _parquet_bytes(
            TIMESTAMP[:1], [1.0]
        ),
        "market-data/prices/notes.txt": b"ignored",
    }

    class FakeS3FileSystem:
        def __init__(self) -> None:
            self.opened: list[str] = []
            self.lookups = 0

        def get_file_info(self, target):
            self.lookups += 1
            if isinstance(target, str):
                return FileInfo(target, FileType.Directory)
            return [FileInfo(path, FileType.File) for path in reversed(tuple(objects))]

        def open_input_file(self, path: str) -> io.BytesIO:
            self.opened.append(path)
            return io.BytesIO(objects[path])

    filesystem = FakeS3FileSystem()
    monkeypatch.setattr(
        local_sources,
        "_open_parquet_filesystem",
        lambda location: (filesystem, "market-data/prices"),
    )
    location = "s3://market-data/prices/"
    unexecuted = Node(
        ParquetSource,
        parameters={
            "path": location,
            "output_signature": FLOAT_SIGNATURE,
            "content_sha256": "0" * 64,
        },
    )

    assert unexecuted.parameters.path == "s3://market-data/prices"
    assert filesystem.lookups == 0

    digest = sha256_parquet_source(location)
    node = Node(
        ParquetSource,
        parameters={
            "path": location,
            "output_signature": FLOAT_SIGNATURE,
            "content_sha256": digest,
        },
    )
    result = Graph(node).execute()

    assert result["timestamp"].to_list() == TIMESTAMP[:2]
    assert result["value"].to_list() == [1.0, 2.0]
    assert (
        filesystem.opened
        == [
            "market-data/prices/nested/part-000.parquet",
            "market-data/prices/part-001.parquet",
        ]
        * 2
    )


def test_parquet_source_loads_explicit_s3_object_without_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyarrow.fs import FileInfo, FileType

    content = _parquet_bytes(TIMESTAMP[:1], [4.0])

    class FakeS3FileSystem:
        def __init__(self) -> None:
            self.opened: list[str] = []

        def get_file_info(self, target: str) -> FileInfo:
            return FileInfo(target, FileType.File)

        def open_input_file(self, path: str) -> io.BytesIO:
            self.opened.append(path)
            return io.BytesIO(content)

    filesystem = FakeS3FileSystem()
    monkeypatch.setattr(
        local_sources,
        "_open_parquet_filesystem",
        lambda location: (filesystem, "market-data/snapshot"),
    )
    location = "s3://market-data/snapshot"
    node = Node(
        ParquetSource,
        parameters={
            "path": location,
            "output_signature": FLOAT_SIGNATURE,
            "content_sha256": sha256_parquet_source(location),
        },
    )

    result = Graph(node).execute()

    assert result["value"].to_list() == [4.0]
    assert filesystem.opened == ["market-data/snapshot"] * 2


@pytest.mark.parametrize("location", ["s3:///prices", "https://bucket/prices"])
def test_parquet_source_rejects_invalid_remote_location(location: str) -> None:
    with pytest.raises(ValueError):
        ParquetSourceConfig(location, FLOAT_SIGNATURE, "0" * 64)


def test_source_versions_describe_verified_snapshot_behavior() -> None:
    assert CSVSource.VERSION == "0.2.0"
    assert ParquetSource.VERSION == "0.3.0"


def test_offline_quickstart_graph(tmp_path: Path) -> None:
    csv_path = tmp_path / "prices.csv"
    pl.DataFrame(
        {
            "timestamp": TIMESTAMP,
            "probability": [0.25, 0.5, 0.75],
        }
    ).write_csv(csv_path)

    source = Node(
        CSVSource,
        parameters={
            "path": csv_path,
            "output_signature": FrameSignature(
                columns=(("probability", pl.Float64),),
            ),
            "content_sha256": sha256_file(csv_path),
        },
        name="prices",
    )
    log_odds = Node(
        Logit,
        bindings={"probability": source.probability},
        parameters={
            "input_column": "probability",
            "output_column": "log_odds",
        },
    )
    change = Node(
        Delta,
        bindings={"log_odds": log_odds.log_odds},
        parameters={
            "input_column": "log_odds",
            "output_column": "change",
        },
    )

    graph = Graph(change)
    graph.verify()
    result = graph.execute()

    assert result.columns == ["timestamp", "change"]
    assert result["change"].round(12).to_list() == [
        None,
        1.098612288668,
        1.098612288668,
    ]
