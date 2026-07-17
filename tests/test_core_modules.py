from __future__ import annotations

from iosislib.core.graph import Executor, Graph, LocalExecutor
from iosislib.core.model import Dataset, Model, Scheduler, SupervisedModelTSFN
from iosislib.core.node import Node
from iosislib.core.tsfn import ColumnSignature, FrameSignature, TSFN, TimeAxis


def test_core_types_live_in_their_domain_modules() -> None:
    assert TimeAxis.__module__ == "iosislib.core.tsfn"
    assert ColumnSignature.__module__ == "iosislib.core.tsfn"
    assert FrameSignature.__module__ == "iosislib.core.tsfn"
    assert TSFN.__module__ == "iosislib.core.tsfn"
    assert Dataset.__module__ == "iosislib.core.model"
    assert Model.__module__ == "iosislib.core.model"
    assert Scheduler.__module__ == "iosislib.core.model"
    assert SupervisedModelTSFN.__module__ == "iosislib.core.model"
    assert Node.__module__ == "iosislib.core.node"
    assert Executor.__module__ == "iosislib.core.graph"
    assert LocalExecutor.__module__ == "iosislib.core.graph"
    assert Graph.__module__ == "iosislib.core.graph"
