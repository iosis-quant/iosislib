from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import polars as pl

from iosislib.core.model import Dataset


FloatMatrix = npt.NDArray[np.float64]
FloatVector = npt.NDArray[np.float64]


def feature_matrix(features: pl.Series, *, width: int) -> FloatMatrix:
    values = np.asarray(features.to_list(), dtype=np.float64)
    if values.ndim != 2 or values.shape != (len(features), width):
        raise ValueError(
            f"Expected {width} features per row, got array shape {values.shape}"
        )
    return values


def target_vector(target: pl.Series) -> FloatVector:
    values = np.asarray(target.to_list(), dtype=np.float64)
    if values.ndim != 1 or values.shape != (len(target),):
        raise ValueError(f"Expected one target per row, got array shape {values.shape}")
    return values


def dataset_arrays(
    dataset: Dataset,
    *,
    width: int,
    epoch: int = 0,
    seed: int = 0,
) -> Iterator[tuple[FloatMatrix, FloatVector]]:
    for batch in dataset.batches(epoch=epoch, seed=seed):
        yield (
            feature_matrix(batch.get_column("features"), width=width),
            target_vector(batch.get_column("target")),
        )


def collect_dataset(
    dataset: Dataset,
    *,
    width: int,
    seed: int = 0,
) -> tuple[FloatMatrix, FloatVector]:
    batches = tuple(dataset_arrays(dataset, width=width, seed=seed))
    if not batches:
        raise ValueError("Cannot fit a model on an empty dataset")
    return (
        np.concatenate([features for features, _ in batches]),
        np.concatenate([target for _, target in batches]),
    )


def mean_squared_error(target: pl.Series, prediction: pl.Series) -> float:
    value = ((target - prediction) ** 2).mean()
    if value is None:
        raise ValueError("MSE requires at least one target and prediction")
    return float(value)
