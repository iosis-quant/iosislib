"""Decision-policy contracts for the minimal event backtest."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, ClassVar, cast

import numpy as np
import numpy.typing as npt
import polars as pl

from iosislib.core.model import (
    DatasetSplitter,
    MetricItems,
    ScheduleContext,
    Scheduler,
    SupervisedModel,
)
from iosislib.core.utils import (
    _canonical_json,
    _serialize_value,
    numpy_to_series,
    series_to_numpy,
)


Array = npt.NDArray[np.float64]
Shape = tuple[int, ...]


def _shape_width(shape: Shape) -> int:
    width = 1
    for size in shape:
        width *= size
    return width


def _validate_shape(name: str, shape: Shape) -> None:
    if not isinstance(shape, tuple):
        raise TypeError(f"{name} must be a tuple")
    if any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in shape):
        raise ValueError(f"{name} dimensions must be positive integers")


def _row_vector(value: object, shape: Shape, name: str) -> Array:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(
            f"{name} must have shape {shape or '()'} for each backtest row"
        )
    if array.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must expose Float64 values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return np.ascontiguousarray(array.reshape(-1))


class MarketState:
    """A reusable, read-only view over the current row of market inputs."""

    __slots__ = ("timestamp", "row", "_information", "_bid", "_ask", "_target")

    def __init__(
        self,
        information: Array,
        bid: Array,
        ask: Array,
        target: Array | None = None,
    ) -> None:
        for value in (information, bid, ask, target):
            if value is not None:
                value.setflags(write=False)
        self.timestamp: object | None = None
        self.row = 0
        self._information = information
        self._bid = bid
        self._ask = ask
        self._target = target

    def move_to(self, timestamp: object, row: int) -> None:
        self.timestamp = timestamp
        self.row = row

    @property
    def row_count(self) -> int:
        return len(self._information)

    @property
    def information(self) -> Array:
        return cast(Array, self._information[self.row])

    @property
    def bid(self) -> Array:
        return cast(Array, self._bid[self.row])

    @property
    def previous_bid(self) -> Array:
        """Return the previous bid, or the current bid on the first row."""
        return cast(Array, self._bid[max(self.row - 1, 0)])

    @property
    def ask(self) -> Array:
        return cast(Array, self._ask[self.row])

    @property
    def target(self) -> Array:
        if self._target is None:
            raise ValueError("This policy requires a resolved target input")
        return cast(Array, self._target[self.row])


@dataclass(frozen=True)
class PolicyState:
    """Immutable local state threaded through one backtest execution."""


@dataclass(frozen=True, eq=False)
class Order:
    """Immediate signed Float64 quantities in feed-venue universe order.

    A passive value type for analytics and serialization; the backtest loop
    itself never constructs orders and instead writes into pooled buffers.
    """

    quantities: Array

    def __post_init__(self) -> None:
        if not isinstance(self.quantities, np.ndarray):
            raise TypeError("quantities must be a NumPy array")
        if self.quantities.ndim != 1:
            raise ValueError("order quantities must be one-dimensional")
        if self.quantities.dtype != np.dtype(np.float64):
            raise TypeError("order quantities must have dtype float64")
        self.quantities.setflags(write=False)


class Policy(ABC):
    """Answer: given current market and portfolio state, what do we do?"""

    _SERIALIZE_WITH_TO_DICT: ClassVar[bool] = True
    VERSION: ClassVar[str]

    @abstractmethod
    def decide(
        self, state: MarketState, cash: float, balances: Array, orders: Array, row: int
    ) -> None:
        """Write the immediate order quantities into ``orders[row]``.

        ``orders`` is the executor-owned (rows, width) batch buffer. Policies
        must write their target quantities into ``orders[row]`` and must not
        retain ``state``, ``balances``, or ``orders``.
        """

    def to_dict(self) -> dict[str, Any]:
        values: dict[str, Any] = (
            {
                item.name: _serialize_value(getattr(self, item.name))
                for item in fields(cast(Any, self))
            }
            if is_dataclass(self)
            else {}
        )
        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "version": self.VERSION,
            **values,
        }

    def __str__(self) -> str:
        return _canonical_json(self.to_dict())


class StatefulPolicy(Policy, ABC):
    """A policy whose immutable state remains local to one execution."""

    @abstractmethod
    def initial_state(self) -> PolicyState:
        """Return fresh state for one run; never retain it on the policy."""

    @abstractmethod
    def decide_stateful(
        self,
        policy_state: PolicyState,
        state: MarketState,
        cash: float,
        balances: Array,
        orders: Array,
        row: int,
    ) -> PolicyState:
        """Write order quantities into ``orders[row]`` and return successor state."""

    def decide(
        self, state: MarketState, cash: float, balances: Array, orders: Array, row: int
    ) -> None:
        del state, cash, balances, orders, row
        raise TypeError("StatefulPolicy must be driven via decide_stateful")


class FeatureBuffer:
    """Bounded mutable labelled history owned exclusively by one policy state."""

    def __init__(
        self,
        capacity: int,
        feature_shape: Shape,
        target_shape: Shape,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        _validate_shape("feature_shape", feature_shape)
        _validate_shape("target_shape", target_shape)
        self._capacity = capacity
        self._feature_shape = feature_shape
        self._target_shape = target_shape
        self._start = 0
        self._size = 0
        self._features: Array | None = None
        self._target: Array | None = None

    @property
    def row_count(self) -> int:
        return self._size

    def append(self, features: Array, target: Array) -> None:
        if features.ndim != 1 or target.ndim != 1:
            raise ValueError("ModelPolicy features and targets must be vectors")
        if features.dtype != np.dtype(np.float64) or target.dtype != np.dtype(np.float64):
            raise TypeError("ModelPolicy features and targets must be Float64")
        if not np.isfinite(features).all() or not np.isfinite(target).all():
            raise ValueError("ModelPolicy features and targets must be finite")
        if len(features) != _shape_width(self._feature_shape):
            raise ValueError("ModelPolicy features do not match feature_shape")
        if len(target) != _shape_width(self._target_shape):
            raise ValueError("ModelPolicy targets do not match target_shape")
        if self._features is None:
            self._features = np.empty(
                (self._capacity, _shape_width(self._feature_shape)),
                dtype=np.float64,
            )
            self._target = np.empty(
                (self._capacity, _shape_width(self._target_shape)),
                dtype=np.float64,
            )
        assert self._target is not None

        index = (self._start + self._size) % self._capacity
        self._features[index] = features
        self._target[index] = target
        if self._size < self._capacity:
            self._size += 1
        else:
            self._start = (self._start + 1) % self._capacity

    def frame(self) -> pl.DataFrame:
        if self._features is None or self._target is None or not self._size:
            raise ValueError("ModelPolicy cannot fit without historical labelled rows")
        indices = (self._start + np.arange(self._size)) % self._capacity
        features = np.ascontiguousarray(self._features[indices])
        target = np.ascontiguousarray(self._target[indices])
        feature_values = (
            features.reshape((self._size, *self._feature_shape))
            if self._feature_shape
            else features.reshape(self._size)
        )
        target_values = (
            target.reshape((self._size, *self._target_shape))
            if self._target_shape
            else target.reshape(self._size)
        )
        return pl.DataFrame(
            [
                numpy_to_series(
                    "features", feature_values, shape=self._feature_shape or None
                ),
                numpy_to_series(
                    "target", target_values, shape=self._target_shape or None
                ),
            ]
        )


@dataclass(frozen=True)
class ModelPolicyState(PolicyState):
    active_model: SupervisedModel
    buffer: FeatureBuffer
    rows_seen: int = 0
    last_retrain_at: int = 0
    retrain_count: int = 0
    next_schedule_at: int = 0
    metrics: MetricItems = ()


@dataclass(frozen=True)
class ModelPolicy(StatefulPolicy, ABC):
    """Fit a supervised model and interpret each one-row prediction as an order."""

    VERSION = "1.0.0"

    model: SupervisedModel
    scheduler: Scheduler
    splitter: DatasetSplitter
    feature_shape: Shape
    target_shape: Shape
    history_rows: int
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.model, SupervisedModel):
            raise TypeError("model must be a SupervisedModel")
        if not isinstance(self.scheduler, Scheduler):
            raise TypeError("scheduler must be a Scheduler")
        if not isinstance(self.splitter, DatasetSplitter):
            raise TypeError("splitter must be a DatasetSplitter")
        _validate_shape("feature_shape", self.feature_shape)
        _validate_shape("target_shape", self.target_shape)
        if (
            isinstance(self.history_rows, bool)
            or not isinstance(self.history_rows, int)
            or self.history_rows < 1
        ):
            raise ValueError("history_rows must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")

    @abstractmethod
    def interpret(self, prediction: Array, orders: Array, row: int) -> None:
        """Translate one normalized model prediction into ``orders[row]``."""

    def prediction_metrics(self, prediction: Array, target: Array) -> MetricItems:
        """Return prediction metrics; subclasses may define task-specific metrics."""
        if prediction.shape != target.shape:
            return ()
        return (("mse", float(np.mean((prediction - target) ** 2))),)

    def initial_state(self) -> PolicyState:
        return ModelPolicyState(
            active_model=self.model,
            buffer=FeatureBuffer(
                self.history_rows,
                self.feature_shape,
                self.target_shape,
            ),
        )

    def decide_stateful(
        self,
        policy_state: PolicyState,
        state: MarketState,
        cash: float,
        balances: Array,
        orders: Array,
        row: int,
    ) -> PolicyState:
        del cash, balances
        if not isinstance(policy_state, ModelPolicyState):
            raise TypeError("ModelPolicy requires a ModelPolicyState")
        if policy_state.rows_seen != state.row:
            raise RuntimeError("ModelPolicy state is not aligned with market time")

        active_model = cast(SupervisedModel, policy_state.active_model)
        last_retrain_at = policy_state.last_retrain_at
        retrain_count = policy_state.retrain_count
        next_schedule_at = policy_state.next_schedule_at
        if policy_state.rows_seen >= next_schedule_at:
            decision = self.scheduler.decide(
                ScheduleContext(
                    total_rows=state.row_count,
                    rows_seen=policy_state.rows_seen,
                    rows_since_retrain=policy_state.rows_seen - last_retrain_at,
                    retrain_count=retrain_count,
                    metrics=policy_state.metrics,
                )
            )
            next_schedule_at = decision.predict_until
            if decision.retrain:
                seed = self.seed + retrain_count
                active_model = active_model.fit(
                    self.splitter.split(policy_state.buffer.frame(), seed=seed),
                    seed=seed,
                )
                if not isinstance(active_model, SupervisedModel):
                    raise TypeError("fitted model must be a SupervisedModel")
                last_retrain_at = policy_state.rows_seen
                retrain_count += 1

        feature_vector = _row_vector(
            state.information,
            self.feature_shape,
            "ModelPolicy features",
        )
        target = _row_vector(state.target, self.target_shape, "ModelPolicy targets")
        if not active_model.is_trained:
            policy_state.buffer.append(feature_vector, target)
            orders[row].fill(0.0)
            return ModelPolicyState(
                active_model=active_model,
                buffer=policy_state.buffer,
                rows_seen=policy_state.rows_seen + 1,
                last_retrain_at=last_retrain_at,
                retrain_count=retrain_count,
                next_schedule_at=next_schedule_at,
                metrics=policy_state.metrics,
            )
        feature_values = (
            feature_vector.reshape((1, *self.feature_shape))
            if self.feature_shape
            else feature_vector
        )
        feature_series = numpy_to_series(
            "features",
            feature_values,
            shape=self.feature_shape or None,
            allow_copy=True,
        )
        raw_prediction = series_to_numpy(
            active_model.predict(feature_series),
            allow_copy=True,
        )
        if raw_prediction.ndim < 1 or raw_prediction.shape[0] != 1:
            raise ValueError(
                "ModelPolicy model must return exactly one prediction per market row"
            )
        try:
            prediction = np.ascontiguousarray(
                np.asarray(raw_prediction, dtype=np.float64).reshape(-1)
            )
        except (TypeError, ValueError) as error:
            raise TypeError(
                "ModelPolicy predictions must be convertible to Float64"
            ) from error
        if not np.isfinite(prediction).all():
            raise ValueError("ModelPolicy predictions must be finite")
        self.interpret(prediction, orders, row)

        target = _row_vector(state.target, self.target_shape, "ModelPolicy targets")
        metrics = self.prediction_metrics(prediction, target)
        policy_state.buffer.append(feature_vector, target)
        return ModelPolicyState(
            active_model=active_model,
            buffer=policy_state.buffer,
            rows_seen=policy_state.rows_seen + 1,
            last_retrain_at=last_retrain_at,
            retrain_count=retrain_count,
            next_schedule_at=next_schedule_at,
            metrics=metrics,
        )


@dataclass(frozen=True)
class OrderModelPolicy(ModelPolicy):
    """Interpret the model's output vector as immediate signed order quantities."""

    def interpret(self, prediction: Array, orders: Array, row: int) -> None:
        orders[row] = np.ascontiguousarray(prediction)


__all__ = [
    "Array",
    "FeatureBuffer",
    "MarketState",
    "ModelPolicy",
    "ModelPolicyState",
    "Order",
    "OrderModelPolicy",
    "Policy",
    "PolicyState",
    "StatefulPolicy",
]
