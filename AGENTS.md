# Repository Guidelines

## Project Structure & Module Organization

This repository is intentionally compact, but it has moved into a small package-style layout. `src/classes.py` is the core Python module and defines the transformation function base classes (`TSFN`, `TSFNConfig`), frame contracts (`TimeAxis`, `FrameSignature`), and `Node`/`Graph` orchestration. `tests/` contains the pytest suite for schema validation, identity, binding, and execution behavior. `TODO.md` records short design notes.

Use `./temp/` for temporary work: researching approaches, assembling junk files, stringing together experiments, or sketching functionality before it belongs in tracked source. Treat it as the project's blank canvas. Do not put production code there, and do not rely on files in `temp/` for tests or library behavior.

## Build, Test, and Development Commands

- `python -m py_compile src/classes.py`: checks the library module for syntax errors.
- `python -m pytest`: runs the test suite.

`pyproject.toml` contains the minimal package/test configuration. Install runtime and test dependencies in your local environment as needed.

## Coding Style & Naming Conventions

Use modern Python 3 style with `from __future__ import annotations`, type hints, dataclasses, abstract base classes, and Polars `LazyFrame` transformations. Prefer frozen dataclasses for value objects and configs. Keep validation errors explicit and specific: use `ValueError` for invalid structure or missing fields and `TypeError` for dtype/timezone mismatches.

Follow the existing style in `src/classes.py`: four-space indentation, `PascalCase` classes, `snake_case` functions and variables, uppercase class constants such as `VERSION`, and private helper functions prefixed with `_`. Use small helper functions when they make validation or formatting rules reusable.

Model frame contracts with `FrameSignature`, `TimeAxis`, and `ColumnSignature`. Store signature columns as tuples, not lists: use `(name, dtype)` for scalars and `ColumnSignature(name, dtype, shape)` or `(name, dtype, shape)` for shaped array values. New `TSFN` subclasses must define a non-empty string `VERSION`, use a `TSFNConfig` dataclass through `CONFIG_CLS` when parameters are needed, return `(input_signature, output_signature)` from `type_signature()`, and keep transformations lazy inside `apply()`.

Keep node and graph identifiers deterministic. If data contributes to persistent IDs, serialize it with sorted keys or sorted items, and avoid unordered inputs unless they are normalized first. Human-facing names should remain labels only; identity should come from function identity, version, signatures, parameters, outputs, and bindings.

## Testing Guidelines

Use `pytest` for new tests. Place tests in `tests/` and name files `test_*.py`. Prioritize coverage for schema validation, frame signature invariants, graph binding validation, time-axis compatibility, cycle detection, deterministic node and graph IDs, TSFN version/config behavior, and execution behavior with small Polars `LazyFrame` fixtures. Include both success cases and clear failure expectations with `pytest.raises`.

## Commit & Pull Request Guidelines

Recent Git history uses very terse subjects, but new commits should be clearer and imperative, for example `Add graph validation tests` or `Document contributor workflow`. Pull requests should summarize behavior changes, list tests or checks run, link related issues when available, and include screenshots only when visual example output changes.

## Security & Configuration Tips

Do not commit generated data, plots, virtual environments, caches, or local scratch files. Keep throwaway research and experiments in ignored paths such as `temp/`, and avoid embedding secrets or machine-specific paths in tracked files.
