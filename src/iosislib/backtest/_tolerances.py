"""Shared floating-point tolerances for backtest control flow.

BLAS reduction order, SIMD width, and FMA contraction differ across
platforms, so a computed equity of ``~0 +/- 1e-16`` must not flip
liquidation, fills, or retraining. These helpers put an explicit dead-band
around every razor-edge branch so all platforms agree.
"""

from __future__ import annotations


EQUITY_ABS_EPS = 1e-9
EQUITY_REL_EPS = 1e-9
QTY_ABS_EPS = 1e-12
FILL_ABS_EPS = 1e-12
COST_ABS_EPS = 1e-9
COST_REL_EPS = 1e-9
PRICE_ABS_EPS = 1e-12
NOTIONAL_REL_EPS = 1e-9
SIGNAL_ABS_EPS = 1e-12


def equity_eps(scale: float = 1.0) -> float:
    return max(EQUITY_ABS_EPS, EQUITY_REL_EPS * max(1.0, abs(scale)))


def is_ruined(equity: float, scale: float = 1.0) -> bool:
    """Dust equity counts as ruined so every BLAS agrees on liquidation."""
    return equity <= equity_eps(scale)


def cost_eps(scale: float = 1.0) -> float:
    return max(COST_ABS_EPS, COST_REL_EPS * max(1.0, abs(scale)))


def is_zero_qty(qty: float) -> bool:
    return abs(qty) <= QTY_ABS_EPS


def qty_side(qty: float) -> int:
    if qty > QTY_ABS_EPS:
        return 1
    if qty < -QTY_ABS_EPS:
        return -1
    return 0


def has_fill(fill: float) -> bool:
    return fill > FILL_ABS_EPS


def is_zero_fill(fill: float) -> bool:
    return fill <= FILL_ABS_EPS


def has_liquidity(volume: float) -> bool:
    return volume > PRICE_ABS_EPS


def notional_exceeds(notional: float, cap: float) -> bool:
    return notional > cap + max(COST_ABS_EPS, NOTIONAL_REL_EPS * max(1.0, abs(cap)))
