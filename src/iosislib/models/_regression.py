from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np
import numpy.typing as npt
import polars as pl

from iosislib.core.model import Dataset, shape_width
from iosislib.core.tsfn import ColumnSignature


FloatMatrix = npt.NDArray[np.float64]


def resolve_shape(
    name: str,
    bound_input_columns: Mapping[str, ColumnSignature],
    configured: int | None,
) -> tuple[int, ...]:
    """Resolve a bound feature/target shape against its configured width."""
    bound = bound_input_columns.get(name)
    if bound is not None:
        shape = bound.shape
        width = shape_width(shape)
        if configured is not None and configured != width:
            raise ValueError(
                f"Configured {name} width {configured} does not match the "
                f"bound width {width}"
            )
        return shape
    if configured is not None:
        return (configured,)
    raise ValueError(
        f"{name} must be connected in the graph or have a configured width"
    )


def feature_matrix(features: pl.Series, *, width: int) -> FloatMatrix:
    values = np.asarray(features.to_list(), dtype=np.float64)
    if values.ndim == 1 and width == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape != (len(features), width):
        raise ValueError(
            f"Expected {width} features per row, got array shape {values.shape}"
        )
    return values


def target_matrix(target: pl.Series, *, width: int) -> FloatMatrix:
    values = np.asarray(target.to_list(), dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape != (len(target), width):
        raise ValueError(
            f"Expected {width} targets per row, got array shape {values.shape}"
        )
    return values


def dataset_arrays(
    dataset: Dataset,
    *,
    feature_width: int,
    target_width: int,
    epoch: int = 0,
    seed: int = 0,
) -> Iterator[tuple[FloatMatrix, FloatMatrix]]:
    for batch in dataset.batches(epoch=epoch, seed=seed):
        yield (
            feature_matrix(batch.get_column("features"), width=feature_width),
            target_matrix(batch.get_column("target"), width=target_width),
        )


def collect_dataset(
    dataset: Dataset,
    *,
    feature_width: int,
    target_width: int,
    seed: int = 0,
) -> tuple[FloatMatrix, FloatMatrix]:
    batches = tuple(
        dataset_arrays(
            dataset,
            feature_width=feature_width,
            target_width=target_width,
            seed=seed,
        )
    )
    if not batches:
        raise ValueError("Cannot fit a model on an empty dataset")
    return (
        np.concatenate([features for features, _ in batches]),
        np.concatenate([target for _, target in batches]),
    )


def mean_squared_error(target: pl.Series, prediction: pl.Series) -> float:
    target_values = np.asarray(target.to_list(), dtype=np.float64)
    prediction_values = np.asarray(prediction.to_list(), dtype=np.float64)
    if target_values.ndim != prediction_values.ndim:
        if (
            target_values.ndim == 1
            and prediction_values.ndim == 2
            and prediction_values.shape[1] == 1
        ):
            target_values = target_values.reshape(-1, 1)
        elif (
            prediction_values.ndim == 1
            and target_values.ndim == 2
            and target_values.shape[1] == 1
        ):
            prediction_values = prediction_values.reshape(-1, 1)
    if target_values.shape != prediction_values.shape:
        raise ValueError("target and prediction must have the same shape")
    if target_values.size == 0:
        raise ValueError("MSE requires at least one target and prediction")
    return float(np.mean((target_values - prediction_values) ** 2))
