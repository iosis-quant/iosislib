"""Tests for the iosisweb TSFN catalog generator."""

from __future__ import annotations

import json

from iosislib.catalog import dump_tsfn_catalog
from iosislib.strategy import builtin_registry

_CATEGORIES = {"backtest", "model", "source", "transform"}


def test_catalog_covers_registry_operations() -> None:
    registry = builtin_registry()
    catalog = dump_tsfn_catalog(registry)
    operations = {(entry["op"], entry["version"]) for entry in catalog["tsfns"]}
    assert operations == set(registry.operations)


def test_catalog_is_json_serializable() -> None:
    catalog = dump_tsfn_catalog()
    assert json.loads(json.dumps(catalog)) == catalog


def test_catalog_entries_declare_attributes() -> None:
    catalog = dump_tsfn_catalog()
    assert catalog["tsfns"]
    for entry in catalog["tsfns"]:
        assert entry["op"]
        assert entry["version"]
        assert entry["category"] in _CATEGORIES
        assert entry["class"]
        assert entry["module"]
        assert isinstance(entry["description"], str)
        assert isinstance(entry["requiresMaterialization"], bool)
        assert isinstance(entry["defaultNullPolicy"], str)
        assert isinstance(entry["lookahead"], bool)
        assert isinstance(entry["allowLookaheadInputs"], list)
        assert isinstance(entry["parameters"], list)
        for parameter in entry["parameters"]:
            assert parameter["name"]
            assert parameter["type"]
            assert isinstance(parameter["required"], bool)
        assert entry["signature"] is None or {
            "input",
            "output",
        } <= set(entry["signature"])


def test_catalog_some_signatures_resolvable() -> None:
    catalog = dump_tsfn_catalog()
    resolved = [entry for entry in catalog["tsfns"] if entry["signature"] is not None]
    assert resolved