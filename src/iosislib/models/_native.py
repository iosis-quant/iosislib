"""Native checkpoint payloads: torch state_dict (.pt) and LightGBM text (.txt)."""

from __future__ import annotations

import dataclasses
import io
from collections.abc import Mapping
from typing import Any

import torch

from iosislib.core.model import PT_FILENAME, TXT_FILENAME, Model


def _torch_module_for(checkpoint: Model) -> torch.nn.Module:
    """Rebuild the inference module for a torch-backed checkpoint."""
    from iosislib.models.cnn1d import CNN1DModel
    from iosislib.models.mlp import DenseMLPModel
    from iosislib.models.mlp import _load_parameter_state as _load_mlp_state
    from iosislib.models.mlp import _network as _mlp_network

    if isinstance(checkpoint, DenseMLPModel):
        if checkpoint.state is None:
            raise ValueError("Cannot serialize an untrained DenseMLPModel")
        module = _mlp_network(checkpoint.layers)
        _load_mlp_state(module, checkpoint.state)
        return module
    if isinstance(checkpoint, CNN1DModel):
        from iosislib.models.cnn1d import _load_parameter_state as _load_cnn_state

        if checkpoint.state is None:
            raise ValueError("Cannot serialize an untrained CNN1DModel")
        module = checkpoint._network()
        _load_cnn_state(module, checkpoint.state)
        return module
    raise TypeError(f"No torch module for checkpoint type {type(checkpoint).__name__}")


def _native_widths(checkpoint: Model) -> tuple[int | None, int | None]:
    """Return ``(feature_width, target_width)`` for a native checkpoint."""
    from iosislib.models.cnn1d import CNN1DModel
    from iosislib.models.lightgbm import LightGBMModel
    from iosislib.models.mlp import DenseMLPModel

    if isinstance(checkpoint, DenseMLPModel):
        return checkpoint.layers[0], checkpoint.layers[-1]
    if isinstance(checkpoint, CNN1DModel):
        return checkpoint.feature_width, checkpoint.output_width
    if isinstance(checkpoint, LightGBMModel):
        return checkpoint.feature_width, checkpoint.target_width
    return None, None


def native_payload_for(checkpoint: Model) -> tuple[str, bytes, dict[str, Any]] | None:
    """Serialize a trained checkpoint with its framework-native format.

    Returns ``(filename, payload, info)`` where ``filename`` is ``model.pt``
    (``torch.save`` of the state dict) or ``model.txt`` (LightGBM model
    string), or ``None`` for untrained/unsupported checkpoints which persist
    as JSON only.
    """
    from iosislib.models.cnn1d import CNN1DModel
    from iosislib.models.lightgbm import LightGBMModel
    from iosislib.models.mlp import DenseMLPModel

    feature_width, target_width = _native_widths(checkpoint)
    if isinstance(checkpoint, (DenseMLPModel, CNN1DModel)):
        if checkpoint.state is None:
            return None
        module = _torch_module_for(checkpoint)
        buffer = io.BytesIO()
        torch.save(module.state_dict(), buffer)
        return PT_FILENAME, buffer.getvalue(), {
            "framework": "pytorch",
            "framework_version": torch.__version__,
            "feature_width": feature_width,
            "target_width": target_width,
        }
    if isinstance(checkpoint, LightGBMModel):
        if checkpoint.model_text is None:
            return None
        try:
            import lightgbm

            framework_version = str(lightgbm.__version__)
        except ImportError:
            framework_version = "unknown"
        return TXT_FILENAME, checkpoint.model_text.encode("utf-8"), {
            "framework": "lightgbm",
            "framework_version": framework_version,
            "feature_width": feature_width,
            "target_width": target_width,
        }
    return None


def hydrate_checkpoint(
    checkpoint: Model, payloads: Mapping[str, bytes]
) -> Model:
    """Rebuild a checkpoint's weights from native sidecar payloads.

    Falls back to ``checkpoint`` unchanged when no sidecar applies, so
    JSON-only stores keep working. Raises ``ValueError`` on corrupt payloads.
    """
    from iosislib.models.cnn1d import CNN1DModel
    from iosislib.models.lightgbm import LightGBMModel
    from iosislib.models.mlp import DenseMLPModel

    if isinstance(checkpoint, (DenseMLPModel, CNN1DModel)) and PT_FILENAME in payloads:
        raw = payloads[PT_FILENAME]
        if not isinstance(raw, (bytes, bytearray)) or not raw:
            raise ValueError("model.pt payload must be non-empty bytes")
        try:
            state_dict = torch.load(
                io.BytesIO(bytes(raw)),
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise ValueError(f"model.pt payload is unreadable: {exc}") from exc
        module = _torch_module_for_config(checkpoint)
        try:
            module.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            raise ValueError(
                f"model.pt payload does not match {type(checkpoint).__name__}: {exc}"
            ) from exc
        return dataclasses.replace(checkpoint, state=_parameter_state_for(module))
    if isinstance(checkpoint, LightGBMModel) and TXT_FILENAME in payloads:
        raw = payloads[TXT_FILENAME]
        if not isinstance(raw, (bytes, bytearray)) or not raw:
            raise ValueError("model.txt payload must be non-empty bytes")
        try:
            text = bytes(raw).decode("utf-8")
        except ValueError as exc:
            raise ValueError(f"model.txt payload is not UTF-8: {exc}") from exc
        try:
            from iosislib.models.lightgbm import _import_lightgbm

            _import_lightgbm().Booster(model_str=text)
        except Exception as exc:
            raise ValueError(f"model.txt payload is not a LightGBM model: {exc}") from exc
        return dataclasses.replace(checkpoint, model_text=text)
    return checkpoint


def _torch_module_for_config(checkpoint: Model) -> torch.nn.Module:
    """Build an (uninitialized) inference module from checkpoint config."""
    from iosislib.models.cnn1d import CNN1DModel
    from iosislib.models.mlp import DenseMLPModel
    from iosislib.models.mlp import _network as _mlp_network

    if isinstance(checkpoint, DenseMLPModel):
        return _mlp_network(checkpoint.layers)
    if isinstance(checkpoint, CNN1DModel):
        return checkpoint._network()
    raise TypeError(f"No torch module for checkpoint type {type(checkpoint).__name__}")


def _parameter_state_for(module: torch.nn.Module) -> Any:
    """Read back a ``ParameterState`` tuple from a loaded module."""
    return tuple(
        tuple(float(value) for value in parameter.detach().cpu().reshape(-1).tolist())
        for parameter in module.parameters()
    )


__all__ = ["hydrate_checkpoint", "native_payload_for"]
