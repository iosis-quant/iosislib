"""Framework-native checkpoint payloads with hash-referenced envelopes.

Storage layout per checkpoint::

    model.json  config-only envelope (class identity, version, hyperparams,
                plus a weights digest when a sidecar exists)
    model.pt    torch.save of the state dict (torch checkpoints)
    model.txt   LightGBM model string (LightGBM checkpoints)

``model_id`` is the SHA-256 of the canonical envelope JSON, which commits to
the weight bytes via their digest without containing them. Store paths and
object keys are transport and never enter the hash.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import io
from collections.abc import Mapping
from typing import Any

import torch

from iosislib.core.model import PT_FILENAME, TXT_FILENAME, Model
from iosislib.core.utils import _canonical_json

WEIGHTS_KEY = "weights"


def _torch_module_for_config(checkpoint: Model) -> torch.nn.Module:
    """Build an uninitialized inference module from checkpoint config."""
    from iosislib.models.cnn1d import CNN1DModel
    from iosislib.models.mlp import DenseMLPModel
    from iosislib.models.mlp import _network as _mlp_network

    if isinstance(checkpoint, DenseMLPModel):
        return _mlp_network(checkpoint.layers)
    if isinstance(checkpoint, CNN1DModel):
        return checkpoint._network()
    raise TypeError(f"No torch module for checkpoint type {type(checkpoint).__name__}")


def _torch_weights_bytes(checkpoint: Model) -> bytes:
    """Serialize a trained torch checkpoint's state dict."""
    from iosislib.models.cnn1d import CNN1DModel
    from iosislib.models.mlp import DenseMLPModel
    from iosislib.models.mlp import _load_parameter_state as _load_mlp_state

    try:
        from iosislib.models.cnn1d import _load_parameter_state as _load_cnn_state
    except ImportError:
        _load_cnn_state = None

    module = _torch_module_for_config(checkpoint)
    if isinstance(checkpoint, DenseMLPModel):
        if checkpoint.state is None:
            raise ValueError("Cannot serialize an untrained DenseMLPModel")
        _load_mlp_state(module, checkpoint.state)
    elif isinstance(checkpoint, CNN1DModel):
        if checkpoint.state is None or _load_cnn_state is None:
            raise ValueError("Cannot serialize an untrained CNN1DModel")
        _load_cnn_state(module, checkpoint.state)
    else:
        raise TypeError(
            f"No torch weights for checkpoint type {type(checkpoint).__name__}"
        )
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    return buffer.getvalue()


def _torch_state_from_bytes(checkpoint: Model, raw: bytes) -> Any:
    """Load a state dict payload back into a ``ParameterState`` tuple."""
    try:
        state_dict = torch.load(
            io.BytesIO(raw), map_location="cpu", weights_only=True
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
    return tuple(
        tuple(float(value) for value in parameter.detach().cpu().reshape(-1).tolist())
        for parameter in module.parameters()
    )


def _sidecar_for_checkpoint(checkpoint: Model) -> tuple[str, bytes] | None:
    """Build the native weight sidecar, or ``None`` when JSON-only."""
    from iosislib.models.cnn1d import CNN1DModel
    from iosislib.models.lightgbm import LightGBMModel
    from iosislib.models.mlp import DenseMLPModel

    if isinstance(checkpoint, (DenseMLPModel, CNN1DModel)):
        if checkpoint.state is None:
            return None
        return PT_FILENAME, _torch_weights_bytes(checkpoint)
    if isinstance(checkpoint, LightGBMModel):
        if checkpoint.model_text is None:
            return None
        return TXT_FILENAME, checkpoint.model_text.encode("utf-8")
    return None


def _strip_weights(envelope: dict[str, Any]) -> None:
    """Null weight fields in a config-only envelope."""
    state = envelope.get("state")
    if not isinstance(state, dict):
        return
    for field in ("state", "model_text"):
        if field in state and state[field] is not None:
            state[field] = None


def _framework_for(qualname: str) -> str:
    if "LightGBM" in qualname:
        return "lightgbm"
    if "DenseMLP" in qualname or "CNN1D" in qualname:
        return "pytorch"
    return "iosislib"


def _framework_version_for(framework: str) -> str:
    if framework == "pytorch":
        return str(torch.__version__)
    if framework == "lightgbm":
        try:
            import lightgbm

            return str(getattr(lightgbm, "__version__", "unknown"))
        except ImportError:
            return "unknown"
    return "unknown"


def _widths_for_envelope(envelope: Mapping[str, Any]) -> tuple[Any, Any]:
    """Read ``(feature_width, target_width)`` from envelope config."""
    state = envelope.get("state")
    if not isinstance(state, Mapping):
        return None, None
    layers = state.get("layers")
    if isinstance(layers, (list, tuple)) and layers:
        return layers[0], layers[-1]
    feature_width = state.get("feature_width")
    target_width = state.get("target_width", state.get("output_width"))
    return feature_width, target_width


def envelope_info(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Digest fields (framework, widths) for metadata from an envelope."""
    qualname = str(envelope.get("qualname", ""))
    framework = _framework_for(qualname)
    feature_width, target_width = _widths_for_envelope(envelope)
    return {
        "framework": framework,
        "framework_version": _framework_version_for(framework),
        "feature_width": feature_width,
        "target_width": target_width,
    }


def envelope_for(checkpoint: Model) -> tuple[dict[str, Any], str | None, bytes | None]:
    """Split a checkpoint into ``(envelope, sidecar_filename, sidecar_bytes)``.

    The envelope is config-only: weight fields are nulled and replaced by a
    ``weights`` digest entry. Checkpoints without native weights return
    ``(full envelope, None, None)`` with no ``weights`` entry.
    """
    envelope = copy.deepcopy(checkpoint.to_dict())
    sidecar = _sidecar_for_checkpoint(checkpoint)
    if sidecar is None:
        return envelope, None, None
    filename, payload = sidecar
    _strip_weights(envelope)
    envelope[WEIGHTS_KEY] = {
        "filename": filename,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_size": len(payload),
    }
    return envelope, filename, payload


def model_id_for_envelope(envelope: Mapping[str, Any]) -> str:
    """Content-address an envelope (weights committed by digest, not content)."""
    return hashlib.sha256(_canonical_json(dict(envelope)).encode("utf-8")).hexdigest()


def materialize_envelope(
    envelope: Mapping[str, Any], sidecars: Mapping[str, bytes]
) -> Model:
    """Rebuild a checkpoint from an envelope plus sidecar payloads.

    Envelopes without a ``weights`` entry (untrained, custom, or legacy
    weights-inline stores) materialize directly. Otherwise the sidecar is
    required and must match the recorded digest.
    """
    from iosislib.core.model import model_from_dict

    if not isinstance(envelope, Mapping):
        raise TypeError("materialize_envelope requires a mapping")
    weights = envelope.get(WEIGHTS_KEY)
    if weights is None:
        return model_from_dict(envelope)
    if not isinstance(weights, Mapping):
        raise ValueError("Envelope weights entry must be a mapping")
    filename = weights.get("filename")
    expected_digest = weights.get("sha256")
    expected_size = weights.get("byte_size")
    if not isinstance(filename, str) or not isinstance(expected_digest, str):
        raise ValueError("Envelope weights entry is malformed")
    raw = sidecars.get(filename)
    if raw is None:
        raise FileNotFoundError(
            f"Sidecar {filename!r} for this checkpoint is not in the store"
        )
    raw = bytes(raw)
    if not raw:
        raise ValueError(f"Sidecar {filename!r} must be non-empty bytes")
    if isinstance(expected_size, int) and len(raw) != expected_size:
        raise ValueError(
            f"Sidecar {filename!r} size {len(raw)} does not match "
            f"envelope record {expected_size}"
        )
    actual_digest = hashlib.sha256(raw).hexdigest()
    if actual_digest != expected_digest:
        raise ValueError(
            f"Sidecar {filename!r} failed digest check: "
            "payload does not match the envelope record"
        )
    unweighted = {k: v for k, v in dict(envelope).items() if k != WEIGHTS_KEY}
    if filename == PT_FILENAME:
        base = model_from_dict(unweighted)
        state = _torch_state_from_bytes(base, raw)
        return dataclasses.replace(base, state=state)  # type: ignore[arg-type]
    if filename == TXT_FILENAME:
        try:
            text = raw.decode("utf-8")
        except ValueError as exc:
            raise ValueError(f"Sidecar {filename!r} is not UTF-8: {exc}") from exc
        try:
            from iosislib.models.lightgbm import _import_lightgbm

            _import_lightgbm().Booster(model_str=text)
        except Exception as exc:
            raise ValueError(
                f"Sidecar {filename!r} is not a LightGBM model: {exc}"
            ) from exc
        base = model_from_dict(unweighted)
        return dataclasses.replace(base, model_text=text)  # type: ignore[arg-type]
    raise ValueError(f"Unknown sidecar filename {filename!r}")


def export_from_store_files(
    envelope_json: bytes, sidecars: Mapping[str, bytes]
) -> tuple[str, bytes, dict[str, Any]] | None:
    """Resolve a publishable sidecar from stored envelope + sidecar files.

    Handles new hash-referenced envelopes (sidecar passed through after digest
    check) and legacy weights-inline envelopes (sidecar derived from inline
    weights). Returns ``None`` for JSON-only checkpoints.
    """
    import json

    from iosislib.core.model import model_from_dict

    try:
        envelope = json.loads(bytes(envelope_json).decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Stored envelope is unreadable: {exc}") from exc
    if not isinstance(envelope, dict):
        raise ValueError("Stored envelope must be a mapping")
    weights = envelope.get(WEIGHTS_KEY)
    if isinstance(weights, Mapping):
        filename = weights.get("filename")
        if not isinstance(filename, str):
            raise ValueError("Envelope weights entry is malformed")
        raw = sidecars.get(filename)
        if raw is None:
            return None
        materialize_envelope(envelope, sidecars)
        return filename, bytes(raw), envelope_info(envelope)
    checkpoint = model_from_dict(envelope)
    sidecar = _sidecar_for_checkpoint(checkpoint)
    if sidecar is None:
        return None
    filename, payload = sidecar
    return filename, payload, envelope_info(checkpoint.to_dict())


__all__ = [
    "WEIGHTS_KEY",
    "envelope_for",
    "envelope_info",
    "export_from_store_files",
    "materialize_envelope",
    "model_id_for_envelope",
]
