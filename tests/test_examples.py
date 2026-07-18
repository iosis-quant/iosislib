from __future__ import annotations

import builtins
import importlib.util
import socket
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest

from iosislib.core.graph import Graph


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
EXAMPLE_PATHS = tuple(sorted(EXAMPLES_DIR.glob("*.py")))


def _import_example(path: Path) -> ModuleType:
    module_name = f"_iosislib_example_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def _unexpected_side_effect(*args: object, **kwargs: object) -> None:
    del args, kwargs
    raise AssertionError("example performed a side effect while being imported")


@pytest.mark.parametrize("example_path", EXAMPLE_PATHS, ids=lambda path: path.name)
def test_example_imports_are_side_effect_free(
    example_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert EXAMPLE_PATHS, "no tracked Python examples found"

    real_open = builtins.open
    real_import = builtins.__import__

    def guarded_open(
        file: str | bytes | int,
        mode: str = "r",
        *args: object,
        **kwargs: object,
    ) -> object:
        if any(flag in mode for flag in "wax+"):
            _unexpected_side_effect()
        return real_open(file, mode, *args, **kwargs)

    def guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.partition(".")[0] in {"matplotlib", "plotly"}:
            _unexpected_side_effect()
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(builtins, "print", _unexpected_side_effect)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(socket, "create_connection", _unexpected_side_effect)
    monkeypatch.setattr(socket.socket, "connect", _unexpected_side_effect)
    monkeypatch.setattr(tempfile, "TemporaryDirectory", _unexpected_side_effect)
    monkeypatch.setattr(Path, "mkdir", _unexpected_side_effect)
    monkeypatch.setattr(Path, "touch", _unexpected_side_effect)
    monkeypatch.setattr(Path, "write_bytes", _unexpected_side_effect)
    monkeypatch.setattr(Path, "write_text", _unexpected_side_effect)
    monkeypatch.setattr(pl.DataFrame, "write_csv", _unexpected_side_effect)
    monkeypatch.setattr(pl.DataFrame, "write_parquet", _unexpected_side_effect)
    monkeypatch.setattr(Graph, "execute", _unexpected_side_effect)

    module = _import_example(example_path)

    assert callable(module.run_example)
    assert capsys.readouterr() == ("", "")


def test_offline_example_executes_deterministically() -> None:
    module = _import_example(EXAMPLES_DIR / "offline_graph.py")

    result = module.run_example()

    assert result.columns == ["timestamp", "change"]
    assert result["change"].round(12).to_list() == [
        None,
        1.098612288668,
        1.098612288668,
    ]
