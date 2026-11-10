from __future__ import annotations

from iosislib.models.cnn1d import CNN1D
from iosislib.perturb import BlockPermute
from iosislib.tsfn.transforms import (
    Delta,
    EwmMean,
    EwmStd,
    Lag,
    Lead,
    LogReturn,
    Negate,
    PctChange,
    RollingCov,
    RollingCovMatrix,
    RollingMax,
    RollingMean,
    RollingMedian,
    RollingMin,
    RollingPercentileRank,
    RollingQuantile,
    RollingStd,
    RollingSum,
    RollingVar,
    RollingZScore,
)

ALL_ROLLING = (
    RollingMean,
    RollingStd,
    RollingSum,
    RollingMax,
    RollingMin,
    RollingMedian,
    RollingZScore,
    RollingVar,
    RollingQuantile,
    RollingPercentileRank,
)


def test_default_trailing_rows_is_zero() -> None:
    assert Negate({}).trailing_rows() == 0


def test_every_rolling_transform_reports_periods() -> None:
    for transform in ALL_ROLLING:
        assert transform({"periods": 5}).trailing_rows() == 5
        assert transform({}).trailing_rows() == 3


def test_shift_family_reports_periods() -> None:
    assert Delta({}).trailing_rows() == 1
    assert Delta({"periods": 3}).trailing_rows() == 3
    assert Lag({"periods": 2}).trailing_rows() == 2
    assert PctChange({"periods": 4}).trailing_rows() == 4
    assert LogReturn({"periods": 5}).trailing_rows() == 5


def test_unbounded_ops_report_none() -> None:
    assert EwmMean({}).trailing_rows() is None
    assert EwmStd({}).trailing_rows() is None
    assert BlockPermute({}).trailing_rows() is None


def test_lead_reports_zero() -> None:
    assert Lead({"periods": 1}).trailing_rows() == 0


def test_multivariate_rolling_reports_periods() -> None:
    assert RollingCov({"periods": 7}).trailing_rows() == 7
    assert (
        RollingCovMatrix({"input_columns": ("left", "right"), "periods": 6})
        .trailing_rows()
        == 6
    )


def test_cnn1d_reports_sequence_lookback() -> None:
    assert CNN1D({"sequence_length": 4}).trailing_rows() == 3
    assert CNN1D({}).trailing_rows() == 31
