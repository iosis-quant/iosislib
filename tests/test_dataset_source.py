from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from iosislib.core.graph import Graph
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature
from iosislib.tsfn.adapters import (
    DatasetManifest,
    DatasetSource,
    DatasetSourceConfig,
)


TIMESTAMP = [
    datetime(2026, 1, 1, 0, 0),
    datetime(2026, 1, 1, 0, 1),
    datetime(2026, 1, 1, 0, 2),
    datetime(2026, 1, 2, 0, 0),
    datetime(2026, 1, 2, 0, 1),
]
FLOAT_SIGNATURE = FrameSignature(columns=(("value", pl.Float64),))


def _parquet_bytes(timestamps: list[datetime], values: list[float]) -> bytes:
    buffer = io.BytesIO()
    pl.DataFrame(
        {"timestamp": timestamps, "value": values},
        schema={"timestamp": pl.Datetime, "value": pl.Float64},
    ).write_parquet(buffer)
    return buffer.getvalue()


def write_hive_dataset(base: Path) -> None:
    day1 = base / "year=2026" / "month=01" / "day=01"
    day1.mkdir(parents=True)
    (day1 / "data.parquet").write_bytes(_parquet_bytes(TIMESTAMP[:3], [1.0, 2.0, 3.0]))

    day2 = base / "year=2026" / "month=01" / "day=02"
    day2.mkdir(parents=True)
    (day2 / "data.parquet").write_bytes(_parquet_bytes(TIMESTAMP[3:], [4.0, 5.0]))


# ---------------------------------------------------------------------------
# DatasetManifest tests
# ---------------------------------------------------------------------------


class TestDatasetManifest:
    def test_round_trip(self) -> None:
        manifest = DatasetManifest(
            format="iosis.cloud-dataset-v1",
            path="/data/**/*.parquet",
            schema={"time": "timestamp", "columns": {"value": "float64"}},
        )
        raw = manifest.to_json_bytes()
        restored = DatasetManifest.from_bytes(raw)
        assert restored.path == "/data/**/*.parquet"
        assert restored.schema == {"time": "timestamp", "columns": {"value": "float64"}}

    def test_to_dict_includes_time_range(self) -> None:
        manifest = DatasetManifest(
            format="iosis.cloud-dataset-v1",
            path="/data/**/*.parquet",
            schema={},
            time_range=("2020-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        )
        d = manifest.to_dict()
        assert d["time_range"] == {
            "start": "2020-01-01T00:00:00Z",
            "end": "2026-01-01T00:00:00Z",
        }

    def test_to_dict_omits_none_time_range(self) -> None:
        manifest = DatasetManifest(
            format="iosis.cloud-dataset-v1",
            path="/data/**/*.parquet",
            schema={},
        )
        assert "time_range" not in manifest.to_dict()

    def test_rejects_wrong_format(self) -> None:
        with pytest.raises(ValueError, match="manifest format"):
            DatasetManifest(format="wrong", path="/x", schema={})

    def test_rejects_empty_path(self) -> None:
        with pytest.raises(ValueError, match="path"):
            DatasetManifest(format="iosis.cloud-dataset-v1", path="", schema={})

    def test_from_dict_missing_required_field(self) -> None:
        with pytest.raises(ValueError, match="missing"):
            DatasetManifest.from_dict({"format": "iosis.cloud-dataset-v1"})


# ---------------------------------------------------------------------------
# DatasetSourceConfig tests
# ---------------------------------------------------------------------------


class TestDatasetSourceConfig:
    def test_local_path(self, tmp_path: Path) -> None:
        config = DatasetSourceConfig(
            path=str(tmp_path / "**" / "*.parquet"),
            output_signature=FLOAT_SIGNATURE,
        )
        assert config.path == str(tmp_path / "**" / "*.parquet")

    def test_s3_path(self) -> None:
        config = DatasetSourceConfig(
            path="s3://my-bucket/data/**/*.parquet",
            output_signature=FLOAT_SIGNATURE,
        )
        assert config.path == "s3://my-bucket/data/**/*.parquet"

    def test_validates_time_range(self, tmp_path: Path) -> None:
        config = DatasetSourceConfig(
            path=str(tmp_path / "**" / "*.parquet"),
            output_signature=FLOAT_SIGNATURE,
            time_range=("2026-01-01", "2026-12-31"),
        )
        assert config.time_range == ("2026-01-01", "2026-12-31")

    def test_rejects_invalid_time_range(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="time_range"):
            DatasetSourceConfig(
                path=str(tmp_path / "**" / "*.parquet"),
                output_signature=FLOAT_SIGNATURE,
                time_range="2026-01-01",
            )


# ---------------------------------------------------------------------------
# DatasetSource tests
# ---------------------------------------------------------------------------


class TestDatasetSource:
    def test_scans_hive_partitioned_dataset(self, tmp_path: Path) -> None:
        base = tmp_path / "data"
        write_hive_dataset(base)

        result = Graph(
            Node(
                DatasetSource,
                parameters={
                    "path": str(base / "**" / "*.parquet"),
                    "output_signature": FLOAT_SIGNATURE,
                },
                name="prices",
            )
        ).execute()

        assert result.columns == ["timestamp", "value"]
        assert result.height == 5
        assert result["value"].to_list() == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_hive_partition_columns_not_in_result(self, tmp_path: Path) -> None:
        base = tmp_path / "data"
        write_hive_dataset(base)

        result = Graph(
            Node(
                DatasetSource,
                parameters={
                    "path": str(base / "**" / "*.parquet"),
                    "output_signature": FrameSignature(
                        columns=(("value", pl.Float64),)
                    ),
                },
                name="prices",
            )
        ).execute()

        assert result.columns == ["timestamp", "value"]
        assert "year" not in result.columns

    def test_time_range_filter_prunes_data(self, tmp_path: Path) -> None:
        base = tmp_path / "data"
        write_hive_dataset(base)

        result = Graph(
            Node(
                DatasetSource,
                parameters={
                    "path": str(base / "**" / "*.parquet"),
                    "output_signature": FLOAT_SIGNATURE,
                    "time_range": ("2026-01-02T00:00:00", "2026-01-03T00:00:00"),
                },
                name="prices",
            )
        ).execute()

        assert result.height == 2
        assert result["value"].to_list() == [4.0, 5.0]

    def test_schema_output_matches_declaration(self, tmp_path: Path) -> None:
        base = tmp_path / "data"
        write_hive_dataset(base)

        result = Graph(
            Node(
                DatasetSource,
                parameters={
                    "path": str(base / "**" / "*.parquet"),
                    "output_signature": FLOAT_SIGNATURE,
                },
                name="prices",
            )
        ).execute()

        assert result.schema["timestamp"] == pl.Datetime
        assert result.schema["value"] == pl.Float64

    def test_flat_files_no_partitioning(self, tmp_path: Path) -> None:
        flat = tmp_path / "flat"
        flat.mkdir()
        (flat / "a.parquet").write_bytes(_parquet_bytes([datetime(2026, 1, 1)], [1.0]))
        (flat / "b.parquet").write_bytes(_parquet_bytes([datetime(2026, 1, 2)], [2.0]))

        result = Graph(
            Node(
                DatasetSource,
                parameters={
                    "path": str(flat / "*.parquet"),
                    "output_signature": FLOAT_SIGNATURE,
                },
                name="flat",
            )
        ).execute()

        assert result.height == 2
        assert "year" not in result.columns

    def test_custom_partition_layout(self, tmp_path: Path) -> None:
        base = tmp_path / "weather"
        d1 = base / "region=us-east" / "date=2026-01-01"
        d1.mkdir(parents=True)
        (d1 / "readings.parquet").write_bytes(_parquet_bytes([datetime(2026, 1, 1, 12, 0)], [72.5]))

        d2 = base / "region=eu-west" / "date=2026-01-02"
        d2.mkdir(parents=True)
        (d2 / "readings.parquet").write_bytes(_parquet_bytes([datetime(2026, 1, 2, 12, 0)], [55.1]))

        result = Graph(
            Node(
                DatasetSource,
                parameters={
                    "path": str(base / "**" / "*.parquet"),
                    "output_signature": FLOAT_SIGNATURE,
                },
                name="weather",
            )
        ).execute()

        assert result.height == 2
        assert sorted(result["value"].to_list()) == [55.1, 72.5]
