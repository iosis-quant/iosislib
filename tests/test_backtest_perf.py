"""Backtest performance profiling and benchmarking harness.

Run benchmarks: pytest tests/test_backtest_perf.py -v -s
Run profiles:   pytest tests/test_backtest_perf.py -v -s -k profile
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time
from dataclasses import dataclass

import numpy as np
import polars as pl
import pytest

from iosislib.backtest import (
    BacktestConfig,
    BacktestTSFN,
    FractionalLimitPolicy,
    MarketState,
    Policy,
    PolicyState,
    StatefulPolicy,
    Venue,
)
from iosislib.backtest.feeds import L1Feed
from iosislib.backtest.policy import Array
from iosislib.core.utils import numpy_to_series

ROWS_MED = 100_000
ROWS_FULL = 1_000_000
WIDTHS = [1, 8, 64]


@dataclass(frozen=True)
class SignalPolicy(Policy):
    VERSION = "1.0.0"

    def decide(self, policy_state, state, cash, balances, orders, row):
        del policy_state, cash, balances
        orders[row] = state.information


@dataclass(frozen=True)
class HeavyPolicy(Policy):
    VERSION = "1.0.0"

    def decide(self, policy_state, state, cash, balances, orders, row):
        del policy_state, cash
        orders[row] = state.information * np.dot(balances, state.bid)


@dataclass(frozen=True)
class CounterState(PolicyState):
    tick: int = 0


@dataclass(frozen=True)
class CounterPolicy(StatefulPolicy):
    VERSION = "1.0.0"

    def initial_state(self):
        return CounterState()

    def decide(self, policy_state, state, cash, balances, orders, row):
        del state, cash, balances
        assert isinstance(policy_state, CounterState)
        orders[row] = float(policy_state.tick + 1)
        return CounterState(policy_state.tick + 1)


def _build_frame(rows: int, width: int) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    bid = 100.0 + rng.random((rows, width))
    ask = bid + 0.01
    signal = rng.standard_normal((rows, width)) * 0.1
    return pl.DataFrame([
        pl.Series("timestamp", np.arange(rows), dtype=pl.Datetime),
        numpy_to_series("bid", bid, allow_copy=True, shape=(width,)),
        numpy_to_series("ask", ask, allow_copy=True, shape=(width,)),
        numpy_to_series("signal", signal, allow_copy=True, shape=(width,)),
    ])


def _make_bt(policy, risk=None, width=1):
    return BacktestTSFN(BacktestConfig(
        feed=L1Feed(Venue("t", tuple(f"A{i}" for i in range(width)))),
        policy=policy, initial_cash=1_000_000.0, risk_policy=risk,
    ))


def _time_batch(bt, frame):
    t0 = time.perf_counter()
    result = bt.batch(frame)
    dt = time.perf_counter() - t0
    return result, dt


def _profile_batch(bt, frame, top_n=25):
    pr = cProfile.Profile()
    pr.enable()
    bt.batch(frame)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(top_n)
    return s.getvalue()


class TestBenchmarks:

    @pytest.mark.parametrize("width", WIDTHS)
    def test_signal_no_risk(self, width):
        frame = _build_frame(ROWS_MED, width)
        bt = _make_bt(SignalPolicy(), width=width)
        _, dt = _time_batch(bt, frame)
        us = dt / ROWS_MED * 1e6
        print(f"\n  [Signal w={width:2d}] {dt:.3f}s  {us:.2f} us/row  {ROWS_MED/dt:.0f} rows/s")

    @pytest.mark.parametrize("width", WIDTHS)
    def test_signal_with_limit(self, width):
        frame = _build_frame(ROWS_MED, width)
        bt = _make_bt(SignalPolicy(), risk=FractionalLimitPolicy(0.5), width=width)
        _, dt = _time_batch(bt, frame)
        us = dt / ROWS_MED * 1e6
        print(f"\n  [Signal+Limit w={width:2d}] {dt:.3f}s  {us:.2f} us/row  {ROWS_MED/dt:.0f} rows/s")

    @pytest.mark.parametrize("width", WIDTHS)
    def test_stateful_no_risk(self, width):
        frame = _build_frame(ROWS_MED, width)
        bt = _make_bt(CounterPolicy(), width=width)
        _, dt = _time_batch(bt, frame)
        us = dt / ROWS_MED * 1e6
        print(f"\n  [Counter w={width:2d}] {dt:.3f}s  {us:.2f} us/row  {ROWS_MED/dt:.0f} rows/s")

    def test_1m_w64_signal(self):
        frame = _build_frame(ROWS_FULL, 64)
        bt = _make_bt(SignalPolicy(), width=64)
        _, dt = _time_batch(bt, frame)
        us = dt / ROWS_FULL * 1e6
        print(f"\n  [1M x 64 Signal] {dt:.3f}s  {us:.2f} us/row  {ROWS_FULL/dt:.0f} rows/s")

    def test_1m_w64_signal_limit(self):
        frame = _build_frame(ROWS_FULL, 64)
        bt = _make_bt(SignalPolicy(), risk=FractionalLimitPolicy(0.5), width=64)
        _, dt = _time_batch(bt, frame)
        us = dt / ROWS_FULL * 1e6
        print(f"\n  [1M x 64 Signal+Limit] {dt:.3f}s  {us:.2f} us/row  {ROWS_FULL/dt:.0f} rows/s")

    def test_1m_w64_stateful(self):
        frame = _build_frame(ROWS_FULL, 64)
        bt = _make_bt(CounterPolicy(), width=64)
        _, dt = _time_batch(bt, frame)
        us = dt / ROWS_FULL * 1e6
        print(f"\n  [1M x 64 Counter] {dt:.3f}s  {us:.2f} us/row  {ROWS_FULL/dt:.0f} rows/s")

    def test_1m_w64_stateful_limit(self):
        frame = _build_frame(ROWS_FULL, 64)
        bt = _make_bt(CounterPolicy(), risk=FractionalLimitPolicy(0.5), width=64)
        _, dt = _time_batch(bt, frame)
        us = dt / ROWS_FULL * 1e6
        print(f"\n  [1M x 64 Counter+Limit] {dt:.3f}s  {us:.2f} us/row  {ROWS_FULL/dt:.0f} rows/s")

    def test_1m_w64_heavy(self):
        frame = _build_frame(ROWS_FULL, 64)
        bt = _make_bt(HeavyPolicy(), width=64)
        _, dt = _time_batch(bt, frame)
        us = dt / ROWS_FULL * 1e6
        print(f"\n  [1M x 64 Heavy] {dt:.3f}s  {us:.2f} us/row  {ROWS_FULL/dt:.0f} rows/s")


class TestProfiles:

    def test_profile_signal_w64(self):
        frame = _build_frame(ROWS_MED, 64)
        bt = _make_bt(SignalPolicy(), width=64)
        print("\n" + _profile_batch(bt, frame))

    def test_profile_counter_w64(self):
        frame = _build_frame(ROWS_MED, 64)
        bt = _make_bt(CounterPolicy(), width=64)
        print("\n" + _profile_batch(bt, frame))

    def test_profile_signal_limit_w64(self):
        frame = _build_frame(ROWS_MED, 64)
        bt = _make_bt(SignalPolicy(), risk=FractionalLimitPolicy(0.5), width=64)
        print("\n" + _profile_batch(bt, frame))

    def test_profile_counter_limit_w64(self):
        frame = _build_frame(ROWS_MED, 64)
        bt = _make_bt(CounterPolicy(), risk=FractionalLimitPolicy(0.5), width=64)
        print("\n" + _profile_batch(bt, frame))

    def test_profile_heavy_w64(self):
        frame = _build_frame(ROWS_MED, 64)
        bt = _make_bt(HeavyPolicy(), width=64)
        print("\n" + _profile_batch(bt, frame))
