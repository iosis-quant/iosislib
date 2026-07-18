from typing import Any, assert_type

import numpy as np
import numpy.typing as npt
import polars as pl
import torch

from iosislib import Node
from iosislib.core.tsfn import TSFN
from iosislib.tsfn.transforms import Logit, LogitConfig, SpreadConfig


config = LogitConfig(
    input_column="probability",
    output_column="score",
    timestamp_column="observed_at",
)
node = Node(Logit, config=config)
mapped_node = Node(
    Logit,
    parameters={
        "input_column": "probability",
        "output_column": "score",
        "timestamp_column": "observed_at",
    },
)
function = Logit(config)
binding = node.output("score")
series = pl.Series("probability", [0.25, 0.75])
numpy_values = function.series_to_numpy(series)
torch_values = function.series_to_torch(series, allow_copy=True)

assert_type(node, Node[LogitConfig])
assert_type(mapped_node, Node[LogitConfig])
assert_type(node.function, TSFN[LogitConfig])
assert_type(node.parameters, LogitConfig)
assert_type(function.parameters, LogitConfig)
assert_type(binding, tuple[Node[LogitConfig], str])
assert_type(numpy_values, npt.NDArray[Any])
assert_type(function.numpy_to_series("probability", np.array([0.25])), pl.Series)
assert_type(torch_values, torch.Tensor)
assert_type(function.torch_to_series("probability", torch_values), pl.Series)

wrong_config_node: Node[LogitConfig] = Node(
    Logit,
    config=SpreadConfig(),  # type: ignore[arg-type]
)
