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

This example snapshots a local CSV by content hash, then executes two lazy
transforms without any network access.

```python
from datetime import datetime

import polars as pl

from iosislib.core.graph import Graph
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature
from iosislib.tsfn.adapters import CSVSource, sha256_file
from iosislib.tsfn.transforms import Delta, Logit


csv_path = "prices.csv"
pl.DataFrame(
    {
        "timestamp": [
            datetime(2026, 1, 1, 0, 0),
            datetime(2026, 1, 1, 0, 1),
            datetime(2026, 1, 1, 0, 2),
        ],
        "probability": [0.25, 0.5, 0.75],
    }
).write_csv(csv_path)

source = Node(
    CSVSource,
    parameters={
        "path": csv_path,
        "output_signature": FrameSignature(
            columns=(("probability", pl.Float64),),
        ),
        "content_sha256": sha256_file(csv_path),
    },
    name="prices",
)
log_odds = Node(
    Logit,
    bindings={"probability": source.probability},
    parameters={
        "input_column": "probability",
        "output_column": "log_odds",
    },
)
change = Node(
    Delta,
    bindings={"log_odds": log_odds.log_odds},
    parameters={
        "input_column": "log_odds",
        "output_column": "change",
    },
)

graph = Graph(change)
graph.verify()
result = graph.execute()
print(result)
```

Nodes are immutable declarations. A child owns the alignment tolerance and
null policy for each input it consumes; graph execution aligns parents with a
backward as-of join on the union of their timestamps.

## Development

Install the complete local toolchain:

```console
python -m pip install -e ".[dev]"
```

Run the same quality gates enforced by CI:

```console
python -m compileall -q src/iosislib
python -m ruff check src
python -m mypy
python -m pytest
python -m build
python -m twine check --strict dist/*
check-wheel-contents dist
```

Ruff checks production source without forcing churn in legacy examples or
deliberately adversarial tests. `mypy` currently applies strict checking to the
transform validation boundary. That deliberately small baseline can expand as
the public namespace and typed configuration work land; it does not imply that
all production modules already pass strict static analysis. Imported modules
are skipped until they are deliberately added to the checked file set.

Tracked tests use local fixtures or mock remote API boundaries; they do not
require live market or finance services. CI runs the full suite on Windows and
Ubuntu for every declared Python version, then validates both source and wheel
distributions in a clean environment outside the checkout.

See [TODO.md](TODO.md) for the ordered roadmap and [AGENTS.md](AGENTS.md) for
the architectural and contribution constraints.
