from __future__ import annotations

import io
from typing import Any

import numpy as np
import onnx
import polars as pl
import torch

from iosislib.core.model import Model


class OnnxExportError(RuntimeError):
    """A checkpoint cannot be exported to ONNX."""


def supports_onnx(checkpoint: Model) -> bool:
    """Whether a checkpoint type has an ONNX exporter."""
    try:
        from iosislib.models.mlp import DenseMLPModel
    except ImportError:
        return False
    if isinstance(checkpoint, DenseMLPModel):
        return checkpoint.state is not None
    return False


def export_onnx_payload(checkpoint: Model) -> tuple[bytes, dict[str, Any]]:
    """Export a torch-backed checkpoint to ONNX bytes plus digest metadata.

    Returns ``(payload, info)`` where ``info`` carries the framework, input
    width, and output width needed by publishers without loading the payload.
    Raises :class:`OnnxExportError` when the checkpoint type has no exporter.
    """
    _ = onnx.__version__  # required by torch.onnx at export time
    try:
        import onnxscript  # noqa: F401  # required by torch.onnx exporter
    except ImportError as exc:
        raise OnnxExportError(
            "ONNX export requires the 'onnxscript' package"
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


def run_onnx_payload(onnx_bytes: bytes, features: np.ndarray) -> np.ndarray:
    """Run an exported ONNX payload over a float64 feature matrix."""
    import onnxruntime as ort

    if not isinstance(onnx_bytes, (bytes, bytearray)) or not onnx_bytes:
        raise ValueError("onnx payload must be non-empty bytes")
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError(f"ONNX inference needs a 2D feature matrix, got {matrix.shape}")
    session = ort.InferenceSession(bytes(onnx_bytes), providers=["CPUExecutionProvider"])
    (output,) = session.run(None, {"input": matrix})
    result = np.asarray(output, dtype=np.float64)
    if result.ndim == 1:
        result = result.reshape(-1, 1)
    return result


def predict_features(
    features: pl.Series, checkpoint: Model, onnx_bytes: bytes | None
) -> pl.Series:
    """Score one feature batch, preferring ONNX when payload bytes exist.

    Falls back to the native ``checkpoint.predict`` for non-NN checkpoints
    (LightGBM, untrained, unsupported) where no ONNX payload is stored.
    """
    if onnx_bytes:
        from iosislib.models._regression import feature_matrix

        width = len(onnx_bytes) and _onnx_feature_width(checkpoint)
        matrix = feature_matrix(features, width=width)
        prediction = run_onnx_payload(onnx_bytes, matrix)
        output_width = prediction.shape[1]
        return pl.Series(
            "prediction",
            prediction,
            dtype=pl.Array(pl.Float64, output_width),
        )
    return checkpoint.predict(features)


def _onnx_feature_width(checkpoint: Model) -> int:
    from iosislib.models.mlp import DenseMLPModel

    if isinstance(checkpoint, DenseMLPModel):
        return checkpoint.layers[0]
    raise OnnxExportError(
        f"No ONNX inference metadata for {type(checkpoint).__name__}"
    )


__all__ = [
    "OnnxExportError",
    "export_onnx_payload",
    "predict_features",
    "run_onnx_payload",
    "supports_onnx",
]
