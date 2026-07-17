from __future__ import annotations

from src.core.graph import Executor, Graph, LocalExecutor
from src.core.model import Dataset, Model, Scheduler, SupervisedModelTSFN
from src.core.node import Node
from src.core.tsfn import ColumnSignature, FrameSignature, TSFN, TimeAxis


def test_core_types_live_in_their_domain_modules() -> None:
    assert TimeAxis.__module__ == "src.core.tsfn"
    assert ColumnSignature.__module__ == "src.core.tsfn"
    assert FrameSignature.__module__ == "src.core.tsfn"
    assert TSFN.__module__ == "src.core.tsfn"
    assert Dataset.__module__ == "src.core.model"
    assert Model.__module__ == "src.core.model"
    assert Scheduler.__module__ == "src.core.model"
    assert SupervisedModelTSFN.__module__ == "src.core.model"
    assert Node.__module__ == "src.core.node"
    assert Executor.__module__ == "src.core.graph"
    assert LocalExecutor.__module__ == "src.core.graph"
    assert Graph.__module__ == "src.core.graph"
