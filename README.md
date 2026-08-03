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
the graph declaration or ID. CSV and Parquet sources read, hash, and parse the
same in-memory byte snapshot. This gives an exact content guarantee at the cost
of holding the full source bytes and parsed frame at that boundary.

## Parquet data sources

`ParquetSource` accepts a local file, a local directory, an S3 object, or an S3
bucket/prefix. Directories and prefixes are searched recursively for `.parquet`
files. This source is deliberately non-streaming: it downloads every selected
object, verifies one deterministic dataset digest, parses the snapshot, and then
returns a projected `LazyFrame` with the declared time-series schema.

```python
import polars as pl

from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature
from iosislib.tsfn.adapters import ParquetSource, sha256_parquet_source

location = "s3://my-bucket/prices/"  # A local pathlib.Path works too.
signature = FrameSignature(columns=(("price", pl.Float64),))
source = Node(
    ParquetSource,
    parameters={
        "path": location,
        "output_signature": signature,
        "content_sha256": sha256_parquet_source(location),
    },
)
```

S3 access uses PyArrow's standard AWS credential chain; credentials are not graph
parameters and therefore are not serialized into node identity. Computing the
digest reads the dataset once, and graph execution reads it again and rejects the
result if the bytes or selected object manifest changed between those operations.
The digest identifies the physical snapshot, not only its logical rows, so changing
a multi-file dataset's partitioning intentionally changes its content address. A
single-file directory or prefix retains the bare file's SHA-256 for compatibility.

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

`iosislib.strategy` provides the backend-independent `iosis.strategy` YAML
representation for storing strategies and passing them between APIs and
frontends. Document and operation versions are explicit SemVer fields. It uses
stable symbolic operation names and readable `node.output`
references rather than Python class paths or backend graph IDs. The parser,
deterministic serializer, fingerprint, and packaged JSON Schema are documented in
[the strategy format specification](docs/strategy-format.md). A backend
operation registry/compiler is intentionally a later layer.

See [TODO.md](TODO.md) for the ordered roadmap and [AGENTS.md](AGENTS.md) for
the architectural and contribution constraints.

Third-party TSFN and model authors should start with the
[extension guide](docs/extending-iosislib.md), which explains the graph,
signature, null, identity, materialization, and walk-forward model contracts.
