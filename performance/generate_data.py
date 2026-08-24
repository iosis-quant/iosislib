"""Generate synthetic CSV datasets for performance benchmarking.

Produces realistic time-series data across multiple domains:
  - prices:        Financial OHLCV data
  - weather:       Meteorological observations
  - accounting:    Company financial statements
  - shipping:      Port cargo manifests
  - sensors:       Industrial IoT readings

Usage:
    python performance/generate_data.py                     # generate all into data/
    python performance/generate_data.py --rows 50000        # custom row count
    python performance/generate_data.py --out /tmp          # custom output dir
    python performance/generate_data.py --update-strategies # also update strategy YAML hashes
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import polars as pl


def _timestamp_column(rows: int, freq_ms: int = 60_000) -> pl.Series:
    """Monotonic microsecond timestamps starting at 2025-01-01T00:00:00Z."""
    epoch_us = 1_735_689_600_000_000  # 2025-01-01T00:00:00Z in microseconds
    return pl.Series(
        "timestamp",
        [epoch_us + i * freq_ms * 1000 for i in range(rows)],
        dtype=pl.Datetime("us"),
    )


def generate_prices(rows: int, rng: np.random.Generator) -> pl.DataFrame:
    """Geometric Brownian Motion OHLCV data."""
    dt = 1.0 / 252 / 390  # ~minute bars in trading-year fractions
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
    """Synthetic meteorological readings with diurnal cycle."""
    t = np.arange(rows, dtype=float)
    temp_base = 15.0 + 10.0 * np.sin(2 * np.pi * t / 1440 - np.pi / 2)  # 24h cycle at 1-min
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


def generate_accounting(rows: int, rng: np.random.Generator) -> pl.DataFrame:
    """Simulated quarterly company financials with seasonal patterns."""
    quarters = rows // 90 + 1
    base_revenue = np.linspace(1_000_000, 1_500_000, quarters)
    seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * np.arange(quarters) / 4)
    revenue = (base_revenue * seasonal + rng.normal(0, 50_000, quarters)).repeat(90)[:rows]
    cost = revenue * rng.uniform(0.55, 0.70, rows)
    profit = revenue - cost
    assets = np.cumsum(profit) + 5_000_000 + rng.normal(0, 10_000, rows)
    liabilities = assets * rng.uniform(0.3, 0.6, rows)
    return pl.DataFrame({
        "timestamp": _timestamp_column(rows, freq_ms=86_400_000),  # daily
        "revenue": np.round(revenue, 2),
        "cost": np.round(cost, 2),
        "profit": np.round(profit, 2),
        "assets": np.round(assets, 2),
        "liabilities": np.round(liabilities, 2),
    })


def generate_shipping(rows: int, rng: np.random.Generator) -> pl.DataFrame:
    """Port cargo manifest: oil, grain, containers."""
    oil_base = rng.lognormal(np.log(500_000), 0.3, rows)  # litres
    grain_base = rng.lognormal(np.log(2_000), 0.4, rows)   # tonnes
    containers = rng.poisson(50, rows).astype(float)
    return pl.DataFrame({
        "timestamp": _timestamp_column(rows, freq_ms=3_600_000),  # hourly
        "oil_litres": np.round(oil_base, 0),
        "grain_tonnes": np.round(grain_base, 1),
        "container_count": containers,
    })


def generate_sensors(rows: int, rng: np.random.Generator) -> pl.DataFrame:
    """Industrial IoT sensor readings with drift and noise."""
    t = np.arange(rows, dtype=float)
    temp = 40.0 + 5.0 * np.sin(2 * np.pi * t / 720) + rng.normal(0, 0.5, rows)
    pressure = 1013.0 + rng.normal(0, 2, rows) + 0.001 * np.cumsum(rng.standard_normal(rows))
    humidity = np.clip(50 + 10 * np.sin(2 * np.pi * t / 1440) + rng.normal(0, 3, rows), 0, 100)
    vibration = np.abs(rng.normal(0.5, 0.2, rows)) + 0.1 * np.sin(2 * np.pi * t / 360)
    return pl.DataFrame({
        "timestamp": _timestamp_column(rows, freq_ms=10_000),  # 10-second intervals
        "temperature": np.round(temp, 2),
        "pressure": np.round(pressure, 2),
        "humidity": np.round(humidity, 2),
        "vibration": np.round(vibration, 4),
    })


BACKTEST_WIDTH = 8


def generate_market_data(rows: int, rng: np.random.Generator) -> None:
    """Generate shaped OHLCV+signal parquet for backtest benchmarks.

    Writes directly to Parquet (CSVSource doesn't support shaped columns).
    Returns nothing; the data is written to data/market_data.parquet.
    """
    w = BACKTEST_WIDTH
    dt = 1.0 / 252 / 390
    mu, sigma = 0.05, 0.2
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal((rows, w))
    close = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    spread = rng.uniform(0.001, 0.01, (rows, w)) * close
    bid = close - spread * 0.5
    ask = close + spread * 0.5
    signal = rng.standard_normal((rows, w)) * 0.1

    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    from iosislib.core.utils import numpy_to_series

    df = pl.DataFrame({
        "timestamp": _timestamp_column(rows),
        "bid": numpy_to_series("bid", bid, allow_copy=True, shape=(w,)),
        "ask": numpy_to_series("ask", ask, allow_copy=True, shape=(w,)),
        "signal": numpy_to_series("signal", signal, allow_copy=True, shape=(w,)),
    })
    path = out_dir / "market_data.parquet"
    df.write_parquet(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(
        f"  {path.name:<20s}  {df.height:>10,} rows  {df.width} cols  sha256={digest[:16]}\u2026"
    )


CSV_GENERATORS = {
    "prices": generate_prices,
    "weather": generate_weather,
    "accounting": generate_accounting,
    "shipping": generate_shipping,
    "sensors": generate_sensors,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_strategy_hashes(strategies_dir: Path, hashes: dict[str, str]) -> None:
    """Replace placeholder SHA-256 hashes in strategy YAML files with real ones."""
    placeholder = "FILL_HASH_AFTER_RUNNING_generate_data_py"
    updated = 0
    for path in strategies_dir.glob("*.strategy.yaml"):
        text = path.read_text(encoding="utf-8")
        new_text = text
        for name, h in hashes.items():
            # Replace placeholder in any source node that references this dataset
            if name in new_text and placeholder in new_text:
                new_text = new_text.replace(placeholder, h)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            updated += 1
            print(f"  Updated {path.name}")
    if updated:
        print(f"  Updated {updated} strategy file(s) with SHA-256 hashes")
    else:
        print("  No strategy files needed updating")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark CSV datasets")
    parser.add_argument("--rows", type=int, default=100_000, help="Rows per dataset")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument("--update-strategies", action="store_true",
                        help="Update strategy YAML files with correct SHA-256 hashes")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Specific datasets to generate (default: all)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    names = args.datasets if args.datasets else list(CSV_GENERATORS.keys())
    rng = np.random.default_rng(42)

    hashes: dict[str, str] = {}
    for name in names:
        if name not in CSV_GENERATORS:
            print(f"Unknown dataset: {name!r}. Available: {sorted(CSV_GENERATORS)}", file=sys.stderr)
            sys.exit(1)
        df = CSV_GENERATORS[name](args.rows, rng)
        path = out_dir / f"{name}.csv"
        df.write_csv(path)
        h = sha256_file(path)
        hashes[name] = h
        print(f"  {path.name:<20s}  {df.height:>10,} rows  {df.width} cols  sha256={h[:16]}\u2026")

    if "market_data" in (args.datasets or ["market_data"]):
        generate_market_data(args.rows, rng)
        hashes["market_data"] = sha256_file(out_dir / "market_data.parquet")

    print(f"\nGenerated {len(names)} dataset(s) in {out_dir}")

    # Print SHA-256 hashes for embedding in strategy YAML files
    print("\n# SHA-256 content hashes for strategy YAML files:")
    for name, h in sorted(hashes.items()):
        ext = ".parquet" if name == "market_data" else ".csv"
        print(f"#   {name}{ext}: {h}")

    if args.update_strategies:
        print("\nUpdating strategy YAML files...")
        strategies_dir = Path(__file__).parent / "strategies"
        _update_strategy_hashes(strategies_dir, hashes)


if __name__ == "__main__":
    main()
