# Repository Guidelines

## Project Structure & Module Organization

This repository is intentionally compact. `classes.py` is the core Python module and currently defines the transformation function base classes (`TSFN`, `TSFNConfig`) plus `Node` and `Graph` orchestration. `TODO.md` records short design notes. `dev/` is ignored by Git and is for local experiments; `dev/synthetic.py` generates sample time-series data and plots it. Put production code in tracked modules, not under `dev/`. Add future tests under `tests/`.

## Build, Test, and Development Commands

- `python -m py_compile classes.py`: checks the library module for syntax errors.
- `python dev/synthetic.py`: runs the local synthetic-data example; it expects optional packages such as Polars, NumPy, and Matplotlib.
- `python -m pytest`: expected test runner once a `tests/` suite exists.

No package manifest is present yet, so install dependencies in your local environment as needed.

## Coding Style & Naming Conventions

Use Python 3 style with type hints, dataclasses, abstract base classes, and Polars `LazyFrame` transformations. Prefer lazy operations inside `TSFN.apply()` and keep validation errors explicit. Use four-space indentation. Name classes with `PascalCase`, functions and variables with `snake_case`, and constants with uppercase names such as `START_DATE`. Keep node and graph identifiers deterministic by avoiding unordered data in ID inputs unless it is sorted first.

## Testing Guidelines

Use `pytest` for new tests. Place tests in `tests/` and name files `test_*.py`. Prioritize coverage for schema validation, graph binding validation, cycle detection, deterministic node and graph IDs, and execution behavior with small Polars `LazyFrame` fixtures. Include both success cases and clear failure expectations with `pytest.raises`.

## Commit & Pull Request Guidelines

Recent Git history uses very terse subjects, but new commits should be clearer and imperative, for example `Add graph validation tests` or `Document contributor workflow`. Pull requests should summarize behavior changes, list tests or checks run, link related issues when available, and include screenshots only when visual example output changes.

## Security & Configuration Tips

Do not commit generated data, plots, virtual environments, or local scratch files. Keep experimental scripts in ignored paths such as `dev/`, and avoid embedding secrets or machine-specific paths in tracked files.
