"""CSV-domain data generators for performance tests.

Thin registry over stream_data generators, keyed by domain name.
Imported by test_perf.py as ``CSV_GENERATORS[domain](rows, rng)``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from stream_data import (
    generate_prices,
    generate_sensors,
    generate_shipping,
    generate_weather,
)

CSV_GENERATORS: dict[str, Callable[[int, np.random.Generator], pl.DataFrame]] = {
    "prices": generate_prices,
    "weather": generate_weather,
    "sensors": generate_sensors,
    "shipping": generate_shipping,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
