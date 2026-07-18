# iosis strategy format v1

`iosis.strategy/v1` is a small, portable declaration of a strategy graph. It is
the storage and transport format shared by files, APIs, and frontends. It is not
a serialization of the current Python `Node` or `Graph` classes.

## Example

```yaml
format: iosis.strategy/v1
name: probability-change

nodes:
  prices:
    op: source.csv/v1
    params:
      path: prices.csv
      content_sha256: 0123456789abcdef
      schema:
        time: timestamp
        columns:
          probability: float64

  log_odds:
    op: transform.logit/v1
    inputs:
      probability: prices.probability
    params:
      output: log_odds

  change:
    op: transform.delta/v1
    inputs:
      value:
        from: log_odds.log_odds
        tolerance: 5m
        nulls: propagate
    params:
      periods: 1
      output: change

outputs:
  signal: change.change
```

The short input form is normally enough:

```yaml
inputs:
  probability: prices.probability
```

Use the expanded form when an input needs consumer-owned behavior:

```yaml
inputs:
  probability:
    from: prices.probability
    tolerance: 5m
    nulls: fill
    fill: 0.0
```

## Fields

- `format` is exactly `iosis.strategy/v1`.
- `name` is a human-readable strategy name.
- `description` and `metadata` are optional and have no execution semantics.
- `nodes` maps stable, local identifiers to node declarations. Declaration
  order has no meaning.
- `op` is a stable operation contract name ending in `/vN`. A later operation
  registry will map these names to whichever backend implementation is current.
- `params` contains operation-specific, JSON-compatible values.
- `inputs` maps the operation's input names to `node.output` references.
- `materialize` is optional. Omission lets the operation contract choose its
  required behavior; an explicit value is a strategy declaration.
- `outputs` gives public names to one or more `node.output` references and
  determines which nodes belong to the strategy.

Identifiers begin with a letter and contain only letters, digits, `_`, or `-`.
Dots are reserved as the separator in a reference.

An expanded input accepts:

- `from`: the required source reference;
- `tolerance`: a non-negative number or a Polars-style duration string such as
  `5m`; omission means an unbounded backward as-of match;
- `nulls`: `error`, `propagate`, `drop`, `fill`, or `pass`;
- `fill`: a scalar, required only when `nulls` is `fill`.

## Stability boundary

The format version controls document structure. Each `op` version controls that
operation's parameter, input, and output contract. Backend class names, backend
versions, resolved schemas, content-addressed node IDs, graph IDs, and executor
choices are deliberately absent. A compiler may expose those details in a
separate diagnostic artifact, but must not write them back as strategy meaning.

The v1 parser checks document shape, references, cycles, unused nodes, portable
value types, duplicate YAML keys, and ambiguous YAML features. Operation-specific
parameter and output validation belongs to the future operation registry/compiler.
YAML anchors and aliases are rejected so every value remains visible where it is
used. YAML booleans follow the unambiguous `true`/`false` spelling; values such as
`on`, `off`, `yes`, and ISO dates remain strings.

## Python API

```python
from iosislib.strategy import dumps, load, loads, schema

strategy = load("strategy.yaml")
same_strategy = loads(dumps(strategy))
json_schema = schema()
print(strategy.fingerprint)
```

`dumps()` emits deterministic YAML in dependency order. `fingerprint` is the
SHA-256 of a canonical JSON form of the IR; it identifies the strategy document,
not a compiled backend graph.

The packaged JSON Schema covers the portable document shape. The Python parser
adds the graph checks that JSON Schema cannot express.
