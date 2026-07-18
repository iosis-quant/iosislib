# iosislib

iosislib is a pre-alpha Python library for typed, deterministic, time-aware
computation graphs over Polars time-series data. It owns graph validation,
causal parent alignment, null policies, materialization boundaries, and model
lifecycle contracts while Polars remains the columnar execution engine.

The distribution is not published yet. Install it from a checkout with Python
3.11, 3.12, or 3.13:

```console
python -m pip install .
```

The installed package namespace is `iosislib`. Owning modules such as
`iosislib.core.graph` remain canonical, with a small top-level convenience
surface for `Graph`, `Node`, and frame signatures.

## Offline quickstart

The tracked [offline graph example](examples/offline_graph.py) snapshots a
temporary local CSV by content hash, then executes `CSVSource -> Logit -> Delta`
without network access. Run it from the repository root after installation:

```console
python examples/offline_graph.py
```

Nodes are immutable declarations. A child owns the alignment tolerance and
null policy for each input it consumes; graph execution aligns parents with a
backward as-of join on the union of their timestamps.

Concrete frozen configs are accepted through the typed `config=` path, while
`parameters=` mappings remain available for dynamic programs. Both normalize to
the same Node identity. Use `node.output("column")` for an explicit typed binding;
`node.column` remains equivalent shorthand.

Graphs are immutable after successful validation. Execution strategies are
runtime choices, supplied with `graph.execute(executor=...)`; they are not part of
the graph declaration or ID. Local CSV and Parquet sources read, hash, and parse
the same in-memory byte snapshot. This gives an exact content guarantee at the
cost of holding the full source bytes and parsed frame at that boundary.

## Development

Install the complete local toolchain:

```console
python -m pip install -e ".[dev]"
```

Run the same quality gates enforced by CI:

```console
python -m compileall -q src/iosislib examples
python -m ruff check src examples
python -m mypy
python examples/offline_graph.py
python -m pytest
python -m build
python -m twine check --strict dist/*
check-wheel-contents dist
```

Ruff checks production source and tracked examples without forcing churn in
deliberately adversarial tests. Strict mypy checks the TSFN and Node contracts,
the concrete transforms, and a public consumer fixture. This is an enforced
public slice, not a claim that every production module is already strict-clean.

Tracked tests use local fixtures or mock remote API boundaries; they do not
require live market or finance services. CI runs the full suite on Windows and
Ubuntu for every declared Python version, then validates both source and wheel
distributions in a clean environment outside the checkout.

## Portable strategy declarations

`iosislib.strategy` provides the backend-independent `iosis.strategy/v1` YAML
representation for storing strategies and passing them between APIs and
frontends. It uses versioned symbolic operations and readable `node.output`
references rather than Python class paths or backend graph IDs. The parser,
deterministic serializer, fingerprint, and packaged JSON Schema are documented in
[the strategy format specification](docs/strategy-format-v1.md). A backend
operation registry/compiler is intentionally a later layer.

See [TODO.md](TODO.md) for the ordered roadmap and [AGENTS.md](AGENTS.md) for
the architectural and contribution constraints.
