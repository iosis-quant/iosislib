from __future__ import annotations

import io
from typing import Any

import torch

from iosislib.core.model import Model


class OnnxExportError(RuntimeError):
    """A checkpoint cannot be exported to ONNX in this environment."""


def export_onnx_payload(checkpoint: Model) -> tuple[bytes, dict[str, Any]]:
    """Export a torch-backed checkpoint to ONNX bytes plus digest metadata.

    Returns ``(payload, info)`` where ``info`` carries the framework, input
    width, and output width needed by publishers without loading the payload.
    Raises :class:`OnnxExportError` when the ``onnx`` package is missing or
    the checkpoint type has no exporter.
    """
    try:
        import onnx  # noqa: F401 - required by torch.onnx at export time
    except ImportError as exc:
        raise OnnxExportError(
            "Exporting ONNX requires the 'onnx' package, which is not installed"
        ) from exc

    module, feature_width, target_width = _torch_module_for(checkpoint)
    module.eval()
    example = torch.zeros(1, feature_width, dtype=torch.float64)
    buffer = io.BytesIO()
    torch.onnx.export(
        module,
        example,
        buffer,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    return buffer.getvalue(), {
        "framework": "pytorch",
        "framework_version": torch.__version__,
        "feature_width": feature_width,
        "target_width": target_width,
    }


def _torch_module_for(checkpoint: Model) -> tuple[torch.nn.Module, int, int]:
    """Rebuild the inference module for a supported checkpoint type."""
    from iosislib.models.mlp import _load_parameter_state
    from iosislib.models.mlp import _network
    from iosislib.models.mlp import DenseMLPModel

    if isinstance(checkpoint, DenseMLPModel):
        if checkpoint.state is None:
            raise OnnxExportError("Cannot export an untrained DenseMLPModel")
        module = _network(checkpoint.layers)
        _load_parameter_state(module, checkpoint.state)
        return module, checkpoint.layers[0], checkpoint.layers[-1]
    raise OnnxExportError(
        f"No ONNX exporter for checkpoint type {type(checkpoint).__name__}"
    )


__all__ = ["OnnxExportError", "export_onnx_payload"]
