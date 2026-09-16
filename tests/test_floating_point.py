"""Floating-point edge cases: diagnose platform drift early.

Each subtest pins one way last-ulp noise enters the engine (decimal
inexactness, non-associative reduction, catastrophic cancellation, dust
quantities) and asserts the hardened helper absorbs it on every BLAS.
"""

from __future__ import annotations

import numpy as np
import pytest

from iosislib.backtest._tolerances import (
    cost_eps,
    equity_eps,
    has_fill,
    has_liquidity,
    is_ruined,
    is_zero_qty,
    notional_exceeds,
    qty_side,
)


class TestDecimalLiteralsAreInexact:
    def test_point_one_plus_point_two_is_not_point_three(self) -> None:
        assert 0.1 + 0.2 != 0.3

    def test_cash_after_round_trip_is_dust_not_zero(self) -> None:
        cash = 1.0 - (10.0 * 1.0 + (-10.0) * 0.9)
        assert abs(cash) < 1e-12
        assert is_ruined(cash, 1.0)

    def test_dot_of_offsetting_legs_is_dust(self) -> None:
        balances = np.array([10.0, -10.0] + [0.0] * 6)
        bid = np.array([0.01] * 8)
        assert abs(float(np.dot(balances, bid))) < 1e-12


class TestReductionOrderDrift:
    @pytest.mark.parametrize("width", [2, 8, 64])
    def test_dot_sign_is_unstable_at_zero_but_band_is_stable(self, width: int) -> None:
        balances = np.zeros(width)
        balances[0] = 10.0
        balances[1] = -10.0
        bid = np.full(width, 0.01)
        equity = float(np.dot(balances, bid))
        assert abs(equity) < 1e-12
        assert is_ruined(equity, 1.0)

    def test_forward_vs_reverse_accumulation_agree_inside_band(self) -> None:
        values = [0.1, 0.2, -0.3, 1e-16, -1e-16]
        forward = sum(values)
        reverse = sum(reversed(values))
        assert abs(forward - reverse) < 1e-15


class TestCatastrophicCancellation:
    def test_need_minus_cum_plus_vol_yields_dust_take(self) -> None:
        need, cum, vol = 10.0, 10.0 - 1e-14, 1.0
        take = min(vol, max(need - cum + vol, 0.0))
        assert take > 0.0
        assert not has_fill(take - vol)  # dust remainder is not a fill

    def test_cost_over_dust_fill_must_be_rejected(self) -> None:
        assert not has_fill(1e-17)


class TestQuantityDeadZone:
    @pytest.mark.parametrize("qty", [0.0, 1e-13, -1e-13, -0.0])
    def test_dust_quantities_are_flat(self, qty: float) -> None:
        assert is_zero_qty(qty)
        assert qty_side(qty) == 0

    @pytest.mark.parametrize(("qty", "side"), [(1e-9, 1), (-1e-9, -1), (10.0, 1)])
    def test_real_quantities_keep_sign(self, qty: float, side: int) -> None:
        assert qty_side(qty) == side


class TestRuinBand:
    @pytest.mark.parametrize("equity", [-1e-10, 0.0, 1e-10, 1e-12, -1e-12])
    def test_dust_equity_counts_as_ruined(self, equity: float) -> None:
        assert is_ruined(equity, 1.0)

    def test_healthy_equity_is_not_ruined(self) -> None:
        assert not is_ruined(1.0, 1.0)
        assert not is_ruined(1e-6, 1.0)

    def test_band_scales_with_notional(self) -> None:
        assert equity_eps(1e6) > equity_eps(1.0)
        assert is_ruined(5e-4, 1e6)  # dust at 1M scale
        assert not is_ruined(0.5, 1e6)


class TestSolvencyVetoBand:
    def test_spread_cost_at_boundary_is_accepted(self) -> None:
        equity, qty, spread = 1.0, 10.0, 0.1
        assert equity - qty * spread <= 1e-12  # exactly at the edge
        assert not (equity - qty * spread < -cost_eps(equity))

    def test_clearly_insolvent_is_rejected(self) -> None:
        assert 1.0 - 100.0 * 0.1 < -cost_eps(1.0)


class TestNotionalAndLiquidity:
    def test_one_ulp_over_cap_does_not_clamp(self) -> None:
        assert not notional_exceeds(100.0 + 1e-13, 100.0)

    def test_material_breach_clamps(self) -> None:
        assert notional_exceeds(101.0, 100.0)

    @pytest.mark.parametrize("vol", [0.0, 1e-300, 1e-13])
    def test_dust_volume_is_not_liquidity(self, vol: float) -> None:
        assert not has_liquidity(vol)

    def test_real_volume_is_liquidity(self) -> None:
        assert has_liquidity(1e-9)


class TestPlatformCanaries:
    def test_dot_matches_manual_sum_within_band_not_bitwise(self) -> None:
        rng = np.random.default_rng(0)
        values = rng.normal(size=1024)
        assert abs(float(np.dot(values, np.ones_like(values))) - float(np.sum(values))) < 1e-9

    def test_float32_vs_float64_contraction_agree_inside_band(self) -> None:
        a, b, c = 0.1, 0.2, 0.3
        f32 = float(np.float32(a) * np.float32(b) + np.float32(c))
        f64 = a * b + c
        assert abs(f32 - f64) < 1e-6
