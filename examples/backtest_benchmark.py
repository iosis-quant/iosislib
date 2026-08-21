"""Benchmark the immediate-execution backtest loop on 1M x 64 Float64 rows.

Run with ``python examples/backtest_benchmark.py``. Prints rows/s and per-row
microseconds for the pooled buffer loop. ``run_example()`` returns a small
validated result frame without the full benchmark load.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import polars as pl

from iosislib.backtest import (
    BacktestConfig,
    BacktestTSFN,
    MarketState,
    Policy,
    Venue,
)
from iosislib.backtest.feeds import L1Feed
from iosislib.backtest.policy import Array
from iosislib.core.utils import numpy_to_series

WIDTH = 64
ROWS = 1_000_000


@dataclass(frozen=True)
class SignalPolicy(Policy):
    VERSION = "1.0.0"

    def decide(
        self, policy_state, state: MarketState, cash: float, balances: Array, orders: Array, row: int
    ) -> None:
        del policy_state, cash, balances
        orders[row] = state.information


def _frame(rows: int, width: int) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    bid = 100.0 + rng.random((rows, width))
    ask = bid + 0.01
    signal = rng.standard_normal((rows, width)) * 0.1
    return pl.DataFrame(
        [
            pl.Series("timestamp", np.arange(rows), dtype=pl.Datetime),
            numpy_to_series("bid", bid, allow_copy=True, shape=(width,)),
            numpy_to_series("ask", ask, allow_copy=True, shape=(width,)),
            numpy_to_series("signal", signal, allow_copy=True, shape=(width,)),
        ]
    )


def _backtest(width: int) -> BacktestTSFN:
    return BacktestTSFN(
        BacktestConfig(
            feed=L1Feed(Venue("bench", tuple(f"A{i}" for i in range(width)))),
            policy=SignalPolicy(),
            initial_cash=1_000_000.0,
        )
    )


def run_example() -> pl.DataFrame:
    return _backtest(WIDTH).batch(_frame(20_000, WIDTH))


def run_benchmark() -> None:
    frame = _frame(ROWS, WIDTH)
    started = perf_counter()
    result = _backtest(WIDTH).batch(frame)
    elapsed = perf_counter() - started
    print(f"rows: {ROWS}  width: {WIDTH}  height: {result.height}")
    print(f"elapsed: {elapsed:.3f}s  rows/s: {ROWS / elapsed:.0f}")
    print(f"per-row: {elapsed / ROWS * 1e6:.2f} us")


def main() -> None:
    run_benchmark()


if __name__ == "__main__":
    main()
