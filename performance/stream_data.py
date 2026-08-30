"""Streaming data generation for performance tests.

Generates synthetic datasets and streams them to disk in CSV or Parquet format.
Supports configurable row counts, column widths, and data domains.

Usage:
    python performance/stream_data.py                           # generate all defaults
    python performance/stream_data.py --rows 500000 --width 16  # large backtest data
    python performance/stream_data.py --format parquet          # parquet output
    python performance/stream_data.py --domain prices --stream   # streaming write
"""

from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl

from iosislib.core.utils import numpy_to_series


# ---------------------------------------------------------------------------
# Data domain generators
# ---------------------------------------------------------------------------

def _timestamp_column(rows: int, freq_ms: int = 60_000) -> pl.Series:
    epoch_us = 1_735_689_600_000_000
    return pl.Series(
        "timestamp",
        [epoch_us + i * freq_ms * 1000 for i in range(rows)],
        dtype=pl.Datetime("us"),
    )


def generate_prices(rows: int, rng: np.random.Generator) -> pl.DataFrame:
    dt = 1.0 / 252 / 390
    mu, sigma = 0.05, 0.2
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(rows)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    spread = rng.uniform(0.001, 0.01, rows) * close
    open_ = close * (1 + rng.normal(0, 0.001, rows))
    high = np.maximum(open_, close) + spread * rng.uniform(0, 1, rows)
    low = np.minimum(open_, close) - spread * rng.uniform(0, 1, rows)
    volume = rng.lognormal(10, 1, rows).astype(int)
    return pl.DataFrame({
        "timestamp": _timestamp_column(rows),
        "open": np.round(open_, 4),
        "high": np.round(high, 4),
        "low": np.round(low, 4),
        "close": np.round(close, 4),
        "volume": volume,
    })


def generate_weather(rows: int, rng: np.random.Generator) -> pl.DataFrame:
    t = np.arange(rows, dtype=float)
    temp_base = 15.0 + 10.0 * np.sin(2 * np.pi * t / 1440 - np.pi / 2)
    temperature = temp_base + rng.normal(0, 2, rows)
    humidity = np.clip(60 + 20 * np.sin(2 * np.pi * t / 1440) + rng.normal(0, 5, rows), 0, 100)
    wind_speed = np.clip(rng.exponential(5, rows), 0, 80)
    precipitation = np.where(rng.random(rows) < 0.05, rng.exponential(2, rows), 0.0)
    return pl.DataFrame({
        "timestamp": _timestamp_column(rows),
        "temperature": np.round(temperature, 2),
        "humidity": np.round(humidity, 2),
        "wind_speed": np.round(wind_speed, 2),
        "precipitation": np.round(precipitation, 3),
    })


def generate_sensors(rows: int, rng: np.random.Generator) -> pl.DataFrame:
    t = np.arange(rows, dtype=float)
    temp = 40.0 + 5.0 * np.sin(2 * np.pi * t / 720) + rng.normal(0, 0.5, rows)
    pressure = 1013.0 + rng.normal(0, 2, rows) + 0.001 * np.cumsum(rng.standard_normal(rows))
    humidity = np.clip(50 + 10 * np.sin(2 * np.pi * t / 1440) + rng.normal(0, 3, rows), 0, 100)
    vibration = np.abs(rng.normal(0.5, 0.2, rows)) + 0.1 * np.sin(2 * np.pi * t / 360)
    return pl.DataFrame({
        "timestamp": _timestamp_column(rows, freq_ms=10_000),
        "temperature": np.round(temp, 2),
        "pressure": np.round(pressure, 2),
        "humidity": np.round(humidity, 2),
        "vibration": np.round(vibration, 4),
    })


def generate_shipping(rows: int, rng: np.random.Generator) -> pl.DataFrame:
    oil_base = rng.lognormal(np.log(500_000), 0.3, rows)
    grain_base = rng.lognormal(np.log(2_000), 0.4, rows)
    containers = rng.poisson(50, rows).astype(float)
    return pl.DataFrame({
        "timestamp": _timestamp_column(rows, freq_ms=3_600_000),
        "oil_litres": np.round(oil_base, 0),
        "grain_tonnes": np.round(grain_base, 1),
        "container_count": containers,
    })


def generate_market_data(
    rows: int, width: int, rng: np.random.Generator
) -> pl.DataFrame:
    dt = 1.0 / 252 / 390
    mu, sigma = 0.05, 0.2
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal((rows, width))
    close = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    spread = rng.uniform(0.001, 0.01, (rows, width)) * close
    bid = close - spread * 0.5
    ask = close + spread * 0.5
    signal = rng.standard_normal((rows, width)) * 0.1

    return pl.DataFrame({
        "timestamp": _timestamp_column(rows),
        "bid": numpy_to_series("bid", bid, allow_copy=True, shape=(width,)),
        "ask": numpy_to_series("ask", ask, allow_copy=True, shape=(width,)),
        "signal": numpy_to_series("signal", signal, allow_copy=True, shape=(width,)),
    })


def generate_wide_prices(
    rows: int, width: int, rng: np.random.Generator
) -> pl.DataFrame:
    """Narrow price data replicated across `width` assets for transform benchmarks."""
    dt = 1.0 / 252 / 390
    mu, sigma = 0.05, 0.2
    base_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(rows)
    close_base = 100.0 * np.exp(np.cumsum(base_returns))
    assets: dict[str, Any] = {"timestamp": _timestamp_column(rows)}
    for i in range(width):
        noise = rng.normal(1.0, 0.01, rows)
        assets[f"close_{i}"] = np.round(close_base * noise, 4)
    return pl.DataFrame(assets)


DOMAINS: dict[str, Callable[[int, np.random.Generator], pl.DataFrame]] = {
    "prices": generate_prices,
    "weather": generate_weather,
    "sensors": generate_sensors,
    "shipping": generate_shipping,
}


# ---------------------------------------------------------------------------
# Streaming writers
# ---------------------------------------------------------------------------

@dataclass
class WriteResult:
    path: Path
    rows: int
    cols: int
    format: str
    bytes_written: int
    elapsed: float
    sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stream_csv(df: pl.DataFrame, path: Path) -> WriteResult:
    t0 = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)
    elapsed = time.perf_counter() - t0
    size = path.stat().st_size
    return WriteResult(
        path=path,
        rows=df.height,
        cols=df.width,
        format="csv",
        bytes_written=size,
        elapsed=elapsed,
        sha256=_sha256_file(path),
    )


def stream_parquet(df: pl.DataFrame, path: Path) -> WriteResult:
    t0 = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    elapsed = time.perf_counter() - t0
    size = path.stat().st_size
    return WriteResult(
        path=path,
        rows=df.height,
        cols=df.width,
        format="parquet",
        bytes_written=size,
        elapsed=elapsed,
        sha256=_sha256_file(path),
    )


def stream_chunked_csv(
    df: pl.DataFrame, path: Path, chunk_rows: int = 50_000
) -> WriteResult:
    """Write a DataFrame to CSV in chunks to simulate streaming from a source."""
    t0 = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    total = df.height
    written = 0
    with path.open("w", encoding="utf-8") as f:
        header_written = False
        for start in range(0, total, chunk_rows):
            chunk = df.slice(start, min(chunk_rows, total - start))
            if not header_written:
                f.write(chunk.write_csv(include_header=True))
                header_written = True
            else:
                f.write(chunk.write_csv(include_header=False))
            written += chunk.height
    elapsed = time.perf_counter() - t0
    size = path.stat().st_size
    return WriteResult(
        path=path,
        rows=written,
        cols=df.width,
        format="csv_chunked",
        bytes_written=size,
        elapsed=elapsed,
        sha256=_sha256_file(path),
    )


def stream_chunked_parquet(
    df: pl.DataFrame, path: Path, chunk_rows: int = 50_000
) -> WriteResult:
    """Write a DataFrame to Parquet in chunks (separate files per chunk)."""
    t0 = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    total = df.height
    total_bytes = 0
    written = 0
    for start in range(0, total, chunk_rows):
        chunk = df.slice(start, min(chunk_rows, total - start))
        chunk_path = path.parent / f"{path.stem}_part{start // chunk_rows:04d}.parquet"
        chunk.write_parquet(chunk_path)
        total_bytes += chunk_path.stat().st_size
        written += chunk.height
    elapsed = time.perf_counter() - t0
    return WriteResult(
        path=path,
        rows=written,
        cols=df.width,
        format="parquet_chunked",
        bytes_written=total_bytes,
        elapsed=elapsed,
        sha256="",
    )


# ---------------------------------------------------------------------------
# Batch generation helper
# ---------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    name: str
    rows: int
    width: int = 1
    domain: str = "prices"
    format: str = "csv"
    stream: bool = False
    chunk_rows: int = 50_000


@dataclass
class GeneratedDataset:
    spec: DatasetSpec
    result: WriteResult
    schema: pl.Schema


def generate_dataset(
    spec: DatasetSpec, out_dir: Path, rng: np.random.Generator
) -> GeneratedDataset:
    if spec.domain == "market_data":
        df = generate_market_data(spec.rows, spec.width, rng)
    elif spec.domain == "wide_prices":
        df = generate_wide_prices(spec.rows, spec.width, rng)
    elif spec.domain in DOMAINS:
        df = DOMAINS[spec.domain](spec.rows, rng)
    else:
        raise ValueError(f"Unknown domain: {spec.domain!r}")

    ext = ".parquet" if "parquet" in spec.format else ".csv"
    path = out_dir / f"{spec.name}{ext}"

    if spec.stream and "csv" in spec.format:
        result = stream_chunked_csv(df, path, spec.chunk_rows)
    elif spec.stream and "parquet" in spec.format:
        result = stream_chunked_parquet(df, path, spec.chunk_rows)
    elif "parquet" in spec.format:
        result = stream_parquet(df, path)
    else:
        result = stream_csv(df, path)

    return GeneratedDataset(spec=spec, result=result, schema=df.schema)


def generate_all(
    specs: list[DatasetSpec], out_dir: Path, seed: int = 42
) -> list[GeneratedDataset]:
    rng = np.random.default_rng(seed)
    results = []
    for spec in specs:
        print(f"  Generating {spec.name} ({spec.rows:,} rows, {spec.domain}, {spec.format})...")
        gen = generate_dataset(spec, out_dir, rng)
        r = gen.result
        print(
            f"    -> {r.path.name}: {r.rows:,} rows, {r.cols} cols, "
            f"{r.bytes_written / 1024 / 1024:.1f} MB, {r.elapsed:.2f}s"
        )
        results.append(gen)
    return results
