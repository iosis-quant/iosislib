"""Backtest performance profiling and benchmarking harness.

Run benchmarks:  pytest tests/test_backtest_perf.py -v -s
Run profiles:    pytest tests/test_backtest_perf.py -v -s -k profile
Run summary:     pytest tests/test_backtest_perf.py -v -s -k summary
Run validate:    pytest tests/test_backtest_perf.py -v -s -k validate
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
    FractionalKellyPolicy,
    MarketState,
    Policy,
    PolicyState,
    StatefulPolicy,
    Venue,
)
from iosislib.backtest.feeds import L1Feed
from iosislib.backtest.policy import Array
from iosislib.core.utils import numpy_to_series

ROWS_SMALL = 10_000
ROWS_MED = 100_000
ROWS_FULL = 1_000_000
WARMUP = 2
REPEATS = 5


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


def _make_bt(policy, risk=None, width=1, validate=True):
    return BacktestTSFN(BacktestConfig(
        feed=L1Feed(Venue("t", tuple(f"A{i}" for i in range(width)))),
        policy=policy, initial_cash=1_000_000.0,
        risk_policy=risk, validate=validate,
    ))


def _bench(fn):
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def _profile(bt, frame, top_n=25):
    pr = cProfile.Profile()
    pr.enable()
    bt.batch(frame)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(top_n)
    return s.getvalue()


def _fmt(dt, rows):
    us = dt / rows * 1e6
    return f"{dt:.3f}s  {us:.1f} us/row  {rows/dt:,.0f} rows/s"


# ---------------------------------------------------------------------------
# Kernel benchmarks: stateless policies
# ---------------------------------------------------------------------------

class TestKernelStateless:

    @pytest.mark.parametrize("width", [1, 8, 64])
    @pytest.mark.parametrize("rows", [ROWS_SMALL, ROWS_MED, ROWS_FULL])
    def test_signal_no_risk(self, width, rows):
        frame = _build_frame(rows, width)
        bt = _make_bt(SignalPolicy(), width=width)
        best = _bench(lambda: bt.batch(frame))
        print(f"\n  Signal w={width:2d} rows={rows:>9,}: {_fmt(best, rows)}")

    @pytest.mark.parametrize("width", [1, 8, 64])
    @pytest.mark.parametrize("rows", [ROWS_SMALL, ROWS_MED, ROWS_FULL])
    def test_signal_with_limit(self, width, rows):
        frame = _build_frame(rows, width)
        bt = _make_bt(SignalPolicy(), risk=FractionalLimitPolicy(0.5), width=width)
        best = _bench(lambda: bt.batch(frame))
        print(f"\n  Signal+Limit w={width:2d} rows={rows:>9,}: {_fmt(best, rows)}")

    @pytest.mark.parametrize("width", [1, 8, 64])
    @pytest.mark.parametrize("rows", [ROWS_SMALL, ROWS_MED, ROWS_FULL])
    def test_signal_with_kelly(self, width, rows):
        frame = _build_frame(rows, width)
        rng = np.random.default_rng(42)
        sig = np.clip(rng.random((rows, width)), 0.01, 0.99)
        frame = _build_frame(rows, width).with_columns(
            numpy_to_series("signal", sig, allow_copy=True, shape=(width,))
        )
        bt = _make_bt(SignalPolicy(), risk=FractionalKellyPolicy(1.0), width=width)
        best = _bench(lambda: bt.batch(frame))
        print(f"\n  Signal+Kelly w={width:2d} rows={rows:>9,}: {_fmt(best, rows)}")


# ---------------------------------------------------------------------------
# Kernel benchmarks: stateful policies
# ---------------------------------------------------------------------------

class TestKernelStateful:

    @pytest.mark.parametrize("width", [1, 8, 64])
    def test_stateful_no_risk(self, width):
        frame = _build_frame(ROWS_MED, width)
        bt = _make_bt(CounterPolicy(), width=width)
        best = _bench(lambda: bt.batch(frame))
        print(f"\n  Counter w={width:2d}: {_fmt(best, ROWS_MED)}")

    @pytest.mark.parametrize("width", [1, 8, 64])
    def test_stateful_with_limit(self, width):
        frame = _build_frame(ROWS_MED, width)
        bt = _make_bt(CounterPolicy(), risk=FractionalLimitPolicy(0.5), width=width)
        best = _bench(lambda: bt.batch(frame))
        print(f"\n  Counter+Limit w={width:2d}: {_fmt(best, ROWS_MED)}")

    def test_heavy_policy_w64(self):
        frame = _build_frame(ROWS_MED, 64)
        bt = _make_bt(HeavyPolicy(), width=64)
        best = _bench(lambda: bt.batch(frame))
        print(f"\n  Heavy w=64: {_fmt(best, ROWS_MED)}")


# ---------------------------------------------------------------------------
# Validate flag comparison
# ---------------------------------------------------------------------------

class TestValidateFlag:

    @pytest.mark.parametrize("validate", [True, False])
    def test_validate_impact(self, validate):
        frame = _build_frame(ROWS_FULL, 64)
        bt = _make_bt(SignalPolicy(), width=64, validate=validate)
        best = _bench(lambda: bt.batch(frame))
        label = "validate=True " if validate else "validate=False"
        print(f"\n  [{label}] 1M x 64 Signal: {_fmt(best, ROWS_FULL)}")

    @pytest.mark.parametrize("validate", [True, False])
    def test_validate_impact_with_risk(self, validate):
        frame = _build_frame(ROWS_FULL, 64)
        bt = _make_bt(SignalPolicy(), risk=FractionalLimitPolicy(0.5), width=64, validate=validate)
        best = _bench(lambda: bt.batch(frame))
        label = "validate=True " if validate else "validate=False"
        print(f"\n  [{label}] 1M x 64 Signal+Limit: {_fmt(best, ROWS_FULL)}")


# ---------------------------------------------------------------------------
# Frame build overhead
# ---------------------------------------------------------------------------

class TestFrameBuild:

    @pytest.mark.parametrize("width", [1, 8, 64])
    def test_frame_build_vs_kernel(self, width):
        rows = ROWS_MED
        best_build = _bench(lambda: _build_frame(rows, width))
        bt = _make_bt(SignalPolicy(), width=width)
        frame = _build_frame(rows, width)
        best_kernel = _bench(lambda: bt.batch(frame))
        total = best_build + best_kernel
        print(f"\n  w={width:2d} frame build: {_fmt(best_build, rows)}")
        print(f"  w={width:2d} kernel:     {_fmt(best_kernel, rows)}")
        print(f"  w={width:2d} split:      {best_build/total*100:.0f}% build / {best_kernel/total*100:.0f}% kernel")


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

class TestProfiles:

    def test_profile_signal_w64(self):
        frame = _build_frame(ROWS_MED, 64)
        bt = _make_bt(SignalPolicy(), width=64)
        print("\n--- cProfile: SignalPolicy w=64, no risk ---")
        print(_profile(bt, frame))

    def test_profile_signal_limit_w64(self):
        frame = _build_frame(ROWS_MED, 64)
        bt = _make_bt(SignalPolicy(), risk=FractionalLimitPolicy(0.5), width=64)
        print("\n--- cProfile: SignalPolicy + FractionalLimit w=64 ---")
        print(_profile(bt, frame))

    def test_profile_stateful_w64(self):
        frame = _build_frame(ROWS_MED, 64)
        bt = _make_bt(CounterPolicy(), width=64)
        print("\n--- cProfile: CounterPolicy w=64, no risk ---")
        print(_profile(bt, frame))

    def test_profile_heavy_w64(self):
        frame = _build_frame(ROWS_MED, 64)
        bt = _make_bt(HeavyPolicy(), width=64)
        print("\n--- cProfile: HeavyPolicy w=64, no risk ---")
        print(_profile(bt, frame))


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

class TestSummary:

    def test_summary_table(self):
        configs = [
            ("Signal w=1", SignalPolicy(), None, 1),
            ("Signal w=8", SignalPolicy(), None, 8),
            ("Signal w=64", SignalPolicy(), None, 64),
            ("Signal+Limit w=1", SignalPolicy(), FractionalLimitPolicy(0.5), 1),
            ("Signal+Limit w=8", SignalPolicy(), FractionalLimitPolicy(0.5), 8),
            ("Signal+Limit w=64", SignalPolicy(), FractionalLimitPolicy(0.5), 64),
            ("Counter w=1", CounterPolicy(), None, 1),
            ("Counter w=8", CounterPolicy(), None, 8),
            ("Counter w=64", CounterPolicy(), None, 64),
            ("Counter+Limit w=64", CounterPolicy(), FractionalLimitPolicy(0.5), 64),
            ("Heavy w=64", HeavyPolicy(), None, 64),
        ]
        rows = ROWS_MED
        print(f"\n{'Config':<25s} {'Time':>8s} {'us/row':>8s} {'rows/s':>12s}")
        print("-" * 55)
        for label, policy, risk, width in configs:
            frame = _build_frame(rows, width)
            bt = _make_bt(policy, risk=risk, width=width)
            best = _bench(lambda: bt.batch(frame))
            us = best / rows * 1e6
            rps = rows / best
            print(f"{label:<25s} {best:>7.3f}s {us:>7.1f} {rps:>11,.0f}")

    def test_scale_comparison(self):
        """Show how each config scales from 10k to 1M rows at w=64."""
        configs = [
            ("Signal", SignalPolicy(), None),
            ("Signal+Limit", SignalPolicy(), FractionalLimitPolicy(0.5)),
            ("Counter", CounterPolicy(), None),
            ("Counter+Limit", CounterPolicy(), FractionalLimitPolicy(0.5)),
        ]
        print(f"\n{'Config':<20s}", end="")
        for r in [ROWS_SMALL, ROWS_MED, ROWS_FULL]:
            print(f" {r:>10,}", end="")
        print()
        print("-" * 62)
        for label, policy, risk in configs:
            print(f"{label:<20s}", end="")
            for rows in [ROWS_SMALL, ROWS_MED, ROWS_FULL]:
                frame = _build_frame(rows, 64)
                bt = _make_bt(policy, risk=risk, width=64)
                best = _bench(lambda: bt.batch(frame))
                us = best / rows * 1e6
                print(f" {us:>7.1f} us", end="")
            print()
