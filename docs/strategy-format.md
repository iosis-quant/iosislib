# iosis strategy format

`iosis.strategy` is a small, portable declaration of a strategy graph. It is
the storage and transport format shared by files, APIs, and frontends. It is not
a serialization of the current Python `Node` or `Graph` classes.

## Example

```yaml
format: iosis.strategy
version: 0.1.0
name: probability-change

nodes:
  prices:
    op: source.csv
    version: 0.2.0
    params:
      path: prices.csv
      content_sha256: 0123456789abcdef
      schema:
        time: timestamp
        columns:
          probability: float64

  log_odds:
    op: transform.logit
    version: 0.1.0
    inputs:
      probability: prices.probability
    params:
      output: log_odds

  change:
    op: transform.delta
    version: 0.1.0
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

- `format` is exactly `iosis.strategy`.
- `version` is the SemVer version of the strategy document contract. The
  current supported version is `0.1.0`.
- `name` is a human-readable strategy name.
- `description` and `metadata` are optional and have no execution semantics.
- `nodes` maps stable, local identifiers to node declarations. Declaration
  order has no meaning.
- `op` is a stable operation contract name such as `transform.logit`.
- Each node's `version` is a SemVer operation-contract version. It is separate
  from `op` and matches the corresponding Python TSFN version.
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

## Model operations

`model.light_gbm@0.3.0` and `model.dense_mlp@0.3.0` consume exactly two
inputs and produce one output:

- `features`: a `Vector[Float64]` column, normally the output of a
  `transform.feature_packer`;
- `target`: a `Vector[Float64]` column (a scalar series is treated as a
  width-1 vector);
- `prediction`: a `Vector[Float64]` column whose width matches `target`.

Feature and target widths are derived from the bound columns; they can be
declared explicitly, in which case they must match the bindings.

A model regresses the target on the features in walk-forward segments. The
`params` for `model.light_gbm` are:

- `num_boost_round`, `learning_rate`, `num_leaves`, `max_depth`,
  `min_data_in_leaf`, `early_stopping_rounds`: LightGBM hyperparameters;
- `scheduler`: when to retrain (see below);
- `splitter`: how the historical prefix is split for training (see below).

The `params` for `model.dense_mlp` are:

- `hidden_layers`: a list of interior layer widths;
- `epochs`, `learning_rate`, `weight_decay`: training hyperparameters;
- `scheduler` and `splitter`: as above.

Both models emit an `mse` segment metric that later scheduler decisions can
observe.

### Scheduler declarations

`scheduler` is omitted, an instance, or one of the following mappings:

```yaml
scheduler: { every: 100 }              # retrain every 100 rows
scheduler: { frozen: true }            # train once on the initial prefix
scheduler:                              # retrain when mse exceeds 0.5
  metric: { name: mse, threshold: 0.5, check_every: 50 }
scheduler:                              # retrain when any sub-scheduler does
  any:
    - { every: 250 }
    - { metric: { name: mse, threshold: 0.5, check_every: 50 } }
```

A `metric` scheduler accepts `name` or `metric_name`, plus `threshold` and
`check_every`. An omitted `scheduler` uses the operation's default
(`{ every: 100 }`).

### Splitter declarations

`splitter` is omitted, an instance, or a mapping of `ChronologicalSplitter`
fields:

```yaml
splitter:
  validation_size: 0.2    # float fraction or integer count
  test_size: 0.0
  gap: 0
  batch_size: null
  shuffle_train: false
  drop_last: false
  purge_window: 0         # rows excluded from the split: their targets are
                          # not yet observable at the retraining boundary
```

`purge_window` drops the final rows of the historical prefix before the split.
It must equal the number of rows the `target` column looks ahead (for example
the horizon of a `transform.lead` producing a forward return); those rows'
labels are only realized after the retraining boundary, so the model training
on them would leak the future. A value of `0` disables purging.

An omitted `splitter` uses the operation's default
(`{ validation_size: 0.2 }`).

## Stability boundary

The top-level version controls document structure. Each node version controls that
operation's parameter, input, and output contract. Backend class names, backend
versions, resolved schemas, content-addressed node IDs, graph IDs, and executor
choices are deliberately absent. A compiler may expose those details in a
separate diagnostic artifact, but must not write them back as strategy meaning.

The parser checks document shape, references, cycles, unused nodes, portable
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
