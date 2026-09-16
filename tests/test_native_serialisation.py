from __future__ import annotations

import hashlib
import json

import pytest
import torch

from iosislib.core.model import (
    ModelStore,
    PT_FILENAME,
    TXT_FILENAME,
    model_from_dict,
)
from iosislib.models._native import (
    WEIGHTS_KEY,
    envelope_for,
    envelope_info,
    export_from_store_files,
    materialize_envelope,
    model_id_for_envelope,
)
from iosislib.models.lightgbm import LightGBMModel
from iosislib.models.mlp import DenseMLPModel, _network, _parameter_state


def _trained_mlp() -> DenseMLPModel:
    layers = (2, 3, 1)
    torch.manual_seed(0)
    return DenseMLPModel(layers=layers, state=_parameter_state(_network(layers)))


def _trained_lightgbm() -> LightGBMModel:
    lightgbm = pytest.importorskip("lightgbm")
    import numpy as np

    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 2))
    target = features[:, 0] * 2.0 + 1.0
    dataset = lightgbm.Dataset(features, label=target, free_raw_data=False)
    booster = lightgbm.train(
        {"objective": "regression", "verbosity": -1, "num_threads": 1},
        dataset,
        num_boost_round=5,
    )
    return LightGBMModel(
        feature_width=2, target_width=1, model_text=booster.model_to_string()
    )


def test_envelope_strips_weights_and_commits_by_digest() -> None:
    checkpoint = _trained_mlp()
    envelope, filename, payload = envelope_for(checkpoint)

    assert filename == PT_FILENAME
    assert isinstance(payload, bytes) and len(payload) > 0
    assert envelope["state"]["state"] is None
    assert envelope["state"]["layers"] == [2, 3, 1]
    weights = envelope[WEIGHTS_KEY]
    assert weights["filename"] == PT_FILENAME
    assert weights["sha256"] == hashlib.sha256(payload).hexdigest()
    assert weights["byte_size"] == len(payload)
    assert model_id_for_envelope(envelope) == ModelStore.model_id(checkpoint)


def test_envelope_json_stays_small_without_weights() -> None:
    checkpoint = _trained_mlp()
    envelope, _, payload = envelope_for(checkpoint)

    raw = json.dumps(envelope)
    assert WEIGHTS_KEY in raw
    assert len(raw.encode()) < len(payload)
    assert len(raw.encode()) < 2048


def test_envelope_info_reports_framework_widths() -> None:
    envelope, _, _ = envelope_for(_trained_mlp())
    info = envelope_info(envelope)

    assert info["framework"] == "pytorch"
    assert info["feature_width"] == 2
    assert info["target_width"] == 1


def test_materialize_round_trip_restores_weights() -> None:
    checkpoint = _trained_mlp()
    envelope, filename, payload = envelope_for(checkpoint)

    restored = materialize_envelope(envelope, {filename: payload})

    assert isinstance(restored, DenseMLPModel)
    assert restored == checkpoint


def test_materialize_rejects_tampered_sidecar() -> None:
    envelope, filename, payload = envelope_for(_trained_mlp())
    tampered = bytearray(payload)
    tampered[len(tampered) // 2] ^= 0xFF

    with pytest.raises(ValueError, match="digest check"):
        materialize_envelope(envelope, {filename: bytes(tampered)})


def test_materialize_rejects_missing_sidecar() -> None:
    envelope, _, _ = envelope_for(_trained_mlp())

    with pytest.raises(FileNotFoundError):
        materialize_envelope(envelope, {})


def test_untrained_envelope_has_no_weights_entry() -> None:
    envelope, filename, payload = envelope_for(DenseMLPModel(layers=(2, 3, 1)))

    assert filename is None
    assert payload is None
    assert WEIGHTS_KEY not in envelope
    assert materialize_envelope(envelope, {}) == DenseMLPModel(layers=(2, 3, 1))


def test_legacy_weights_inline_envelope_still_loads() -> None:
    checkpoint = _trained_mlp()
    legacy = checkpoint.to_dict()
    assert WEIGHTS_KEY not in legacy

    assert materialize_envelope(legacy, {}) == checkpoint


def test_lightgbm_envelope_round_trip() -> None:
    checkpoint = _trained_lightgbm()
    envelope, filename, payload = envelope_for(checkpoint)

    assert filename == TXT_FILENAME
    assert payload == checkpoint.model_text.encode("utf-8")
    assert envelope["state"]["model_text"] is None
    assert envelope_info(envelope)["framework"] == "lightgbm"
    assert materialize_envelope(envelope, {filename: payload}) == checkpoint


def test_export_from_store_files_for_new_and_legacy() -> None:
    checkpoint = _trained_mlp()
    envelope, filename, payload = envelope_for(checkpoint)

    resolved = export_from_store_files(
        json.dumps(envelope).encode(), {filename: payload}
    )
    assert resolved is not None
    assert resolved[0] == PT_FILENAME
    assert resolved[1] == payload

    legacy = checkpoint.to_dict()
    resolved_legacy = export_from_store_files(json.dumps(legacy).encode(), {})
    assert resolved_legacy is not None
    assert resolved_legacy[0] == PT_FILENAME

    untrained = DenseMLPModel(layers=(2, 3, 1)).to_dict()
    assert export_from_store_files(json.dumps(untrained).encode(), {}) is None


def test_store_writes_config_only_json_plus_sidecar(tmp_path) -> None:
    store = ModelStore(tmp_path)
    checkpoint = _trained_mlp()
    model_id = store.save_run(group="mlp", finished=checkpoint)

    json_path, pt_path, txt_path, _ = store._artifact_locations(model_id)
    envelope = json.loads(json_path.read_bytes().decode())
    assert envelope["state"]["state"] is None
    assert envelope[WEIGHTS_KEY]["filename"] == PT_FILENAME
    assert pt_path.read_bytes()
    assert txt_path.exists() is False

    loaded = store.load_for_inference("mlp", model_id)
    assert loaded == checkpoint
    assert store.read_manifest("mlp")["runs"][0]["format"] == "pt"
    assert model_from_dict(envelope) != checkpoint


def test_store_json_only_for_untrained(tmp_path) -> None:
    store = ModelStore(tmp_path)
    model_id = store.save_run(
        group="mlp", finished=DenseMLPModel(layers=(2, 3, 1))
    )

    json_path, pt_path, txt_path, _ = store._artifact_locations(model_id)
    assert json_path.read_bytes()
    assert pt_path.exists() is False
    assert txt_path.exists() is False
    assert store.read_manifest("mlp")["runs"][0]["format"] == "json"
