from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import pytest

FLOAT_REL_TOL = 1e-6
FLOAT_ABS_TOL = 1e-9


def approx(expected, *, rel: float = FLOAT_REL_TOL, abs: float = FLOAT_ABS_TOL):  # noqa: A002
    return pytest.approx(expected, rel=rel, abs=abs)


def assert_float_close(actual: object, expected: object, *, rel: float = FLOAT_REL_TOL, abs: float = FLOAT_ABS_TOL) -> None:  # noqa: A002
    if expected is None or actual is None:
        assert actual is expected, f"{actual!r} is not {expected!r}"
        return
    if isinstance(expected, (float, int)) and not isinstance(expected, bool) and isinstance(actual, (float, int)) and not isinstance(actual, bool):
        assert math.isclose(float(actual), float(expected), rel_tol=rel, abs_tol=abs), f"{actual!r} != {expected!r}"
        return
    assert actual == expected, f"{actual!r} != {expected!r}"


def assert_nested_close(actual: object, expected: object, *, rel: float = FLOAT_REL_TOL, abs: float = FLOAT_ABS_TOL) -> None:  # noqa: A002
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping), f"{actual!r} is not a mapping"
        assert set(actual) == set(expected), f"keys {set(actual)!r} != {set(expected)!r}"
        for key in expected:
            assert_nested_close(actual[key], expected[key], rel=rel, abs=abs)
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)), f"{actual!r} is not a sequence"
        assert len(actual) == len(expected), f"length {len(actual)} != {len(expected)}: {actual!r} != {expected!r}"
        for a_item, e_item in zip(actual, expected):
            assert_nested_close(a_item, e_item, rel=rel, abs=abs)
        return
    assert_float_close(actual, expected, rel=rel, abs=abs)


def assert_lists_close(actual: Sequence[object], expected: Sequence[object], *, rel: float = FLOAT_REL_TOL, abs: float = FLOAT_ABS_TOL) -> None:  # noqa: A002
    assert_nested_close(actual, expected, rel=rel, abs=abs)


def assert_dicts_close(actual: Sequence[Mapping[str, object]], expected: Sequence[Mapping[str, object]], *, rel: float = FLOAT_REL_TOL, abs: float = FLOAT_ABS_TOL) -> None:  # noqa: A002
    assert_nested_close(actual, expected, rel=rel, abs=abs)


def assert_dict_close(actual: Mapping[str, object], expected: Mapping[str, object], *, rel: float = FLOAT_REL_TOL, abs: float = FLOAT_ABS_TOL) -> None:  # noqa: A002
    assert_nested_close(actual, expected, rel=rel, abs=abs)


def assert_frames_close(actual, expected, *, rel: float = FLOAT_REL_TOL, abs: float = FLOAT_ABS_TOL) -> None:  # noqa: A002
    assert actual.shape == expected.shape, f"shape {actual.shape} != {expected.shape}"
    assert actual.columns == expected.columns, f"columns {actual.columns} != {expected.columns}"
    assert_nested_close(actual.to_dicts(), expected.to_dicts(), rel=rel, abs=abs)
