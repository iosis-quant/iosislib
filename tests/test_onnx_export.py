from __future__ import annotations

import pytest
import torch

from iosislib.models._export import OnnxExportError, export_onnx_payload
from iosislib.models.mlp import DenseMLPModel, _network, _parameter_state


def _trained_mlp() -> DenseMLPModel:
    layers = (2, 3, 1)
    torch.manual_seed(0)
    return DenseMLPModel(layers=layers, state=_parameter_state(_network(layers)))


def test_mlp_exports_valid_onnx() -> None:
    onnx = pytest.importorskip("onnx")
    payload, info = export_onnx_payload(_trained_mlp())

    assert isinstance(payload, bytes) and len(payload) > 0
    assert info["framework"] == "pytorch"
    assert info["feature_width"] == 2
    assert info["target_width"] == 1

    model = onnx.load_from_string(payload)
    onnx.checker.check_model(model)
    assert [i.name for i in model.graph.input] == ["input"]
    assert [o.name for o in model.graph.output] == ["output"]


def test_untrained_mlp_export_fails() -> None:
    with pytest.raises(OnnxExportError):
        export_onnx_payload(DenseMLPModel(layers=(2, 3, 1)))
