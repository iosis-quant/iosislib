from __future__ import annotations

import pytest
import torch

from iosislib.core.model import ModelStore, PT_FILENAME, TXT_FILENAME
from iosislib.models._native import hydrate_checkpoint, native_payload_for
from iosislib.models.lightgbm import LightGBMModel
from iosislib.models.mlp import DenseMLPModel, _network, _parameter_state


def _trained_mlp() -> DenseMLPModel:
    layers = (2, 3, 1)
    torch.manual_seed(0)
    return DenseMLPModel(layers=layers, state=_parameter_state(_network(layers)))


def test_mlp_serializes_to_torch_state_dict() -> None:
    filename, payload, info = native_payload_for(_trained_mlp())

    assert filename == PT_FILENAME
    assert isinstance(payload, bytes) and len(payload) > 0
    assert info["framework"] == "pytorch"
    assert info["feature_width"] == 2
    assert info["target_width"] == 1


def test_mlp_pt_round_trip_preserves_predictions() -> None:
    checkpoint = _trained_mlp()
    filename, payload, _ = native_payload_for(checkpoint)
    assert filename == PT_FILENAME

    untrained = DenseMLPModel(layers=checkpoint.layers)
    restored = hydrate_checkpoint(untrained, {PT_FILENAME: payload})

    assert isinstance(restored, DenseMLPModel)
    assert restored.state == checkpoint.state


def test_untrained_mlp_has_no_native_payload() -> None:
    assert native_payload_for(DenseMLPModel(layers=(2, 3, 1))) is None


def test_corrupt_pt_payload_fails() -> None:
    with pytest.raises(ValueError):
        hydrate_checkpoint(
            DenseMLPModel(layers=(2, 3, 1)), {PT_FILENAME: b"not-a-state-dict"}
        )


def test_lightgbm_serializes_to_native_text() -> None:
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
    checkpoint = LightGBMModel(
        feature_width=2, target_width=1, model_text=booster.model_to_string()
    )

    filename, payload, info = native_payload_for(checkpoint)

    assert filename == TXT_FILENAME
    assert payload == checkpoint.model_text.encode("utf-8")
    assert info["framework"] == "lightgbm"

    restored = hydrate_checkpoint(
        LightGBMModel(feature_width=2, target_width=1), {TXT_FILENAME: payload}
    )
    assert restored.model_text == checkpoint.model_text


def test_untrained_lightgbm_has_no_native_payload() -> None:
    assert (
        native_payload_for(LightGBMModel(feature_width=2, target_width=1)) is None
    )


def test_store_writes_pt_sidecar_and_hydrates(tmp_path) -> None:
    store = ModelStore(tmp_path)
    checkpoint = _trained_mlp()
    model_id = store.save_run(group="mlp", finished=checkpoint)

    json_path, pt_path, txt_path, _ = store._artifact_locations(model_id)
    assert json_path.read_bytes()
    assert pt_path.read_bytes()
    assert txt_path.exists() is False

    loaded = store.load_for_inference("mlp", model_id)
    assert isinstance(loaded, DenseMLPModel)
    assert loaded.state == checkpoint.state
    assert store.read_manifest("mlp")["runs"][0]["format"] == "pt"


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
