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
      output_column: log_odds

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
      output_column: change

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
- `description` is optional. `metadata` is optional free-form data; the one
  recognized key is `charts` (see Chart declarations), and the removed key
  `metrics` is rejected with a migration error.
- `nodes` maps stable, local identifiers to node declarations. Declaration
  order has no meaning.
- `op` is a stable operation contract name such as `transform.logit`.
- Each node's `version` is a SemVer operation-contract version. It is separate
  from `op` and matches the corresponding Python TSFN version.
- `params` contains operation-specific, JSON-compatible values.
- `inputs` maps the operation's input names to `node.output` references.
- `materialize` is optional. Omission lets the operation contract choose its
  required behavior; an explicit value is a strategy declaration. It is execution
  state only and never contributes to node or graph identity. Declared nodes are
  collected and their frames persisted to the executor's cache; every node's
  cached frame is read back regardless of this flag.
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

## Source operations

### `source.parquet_source` / `source.csv_source`

Read a local or S3 Parquet/CSV file. Parameters:

- `path`: file path or `s3://` URI (required).
- `content_sha256`: 64-character hex digest for content verification. For
  published datasets, the manifest `id` field is the SHA-256. Use
  `list_dataset_manifests()` or `lookup_dataset()` to obtain it.
- `schema`: column schema declaration (see below).
- `separator`: CSV only, single-byte delimiter (default `,`).

```yaml
nodes:
  data:
    op: source.parquet_source
    version: 0.3.0
    params:
      path: s3://bucket/datasets/ticker_a.parquet
      content_sha256: abc123...
      schema:
        time: timestamp
        columns:
          bid: float64
          ask: float64
          volume: float64
```

### Loading published datasets

Use `source.parquet` (not `source.parquet_source`) with a `data_version`
parameter to reference published datasets by name. The cloud worker's
`DatasetResolver` rewrites these into `source.parquet_source` nodes with the
resolved S3 path and content hash at compile time.

```yaml
nodes:
  data:
    op: source.parquet
    version: 0.3.0
    params:
      path: ticker_a            # dataset name (not an S3 path)
      data_version: latest      # "latest" or ISO-8601 date
      schema:
        time: timestamp
        columns:
          bid: float64
          ask: float64
```

The `path` param here is the dataset name (as returned by `list_datasets()`),
not a file path. The `data_version` selects which snapshot to use. The resolver
looks up the dataset in the published catalog and replaces `path` with the
actual S3 location and `content_sha256` with the manifest `id`.

### `schema` parameter format

The `schema` parameter declares the output frame structure. It is a mapping
with two keys:

```yaml
schema:
  time: <time column name>       # required: string
  columns:                       # required: mapping of name to dtype
    close: float64
    volume: float64
    signals:                     # shaped (array) columns
      dtype: float64
      shape: [3]
```

Supported dtypes: `bool`, `float32`, `float64`, `int32`, `int64`, `string`.
Array columns use a mapping with `dtype` and `shape` keys.

## Transform operations

### `transform.negate`

Negate a numeric column element-wise. Useful for pairs trading where you need
opposite signals (long A, short B) from a single z-score.

```yaml
nodes:
  zscore:
    op: transform.rolling_z_score
    version: 0.1.0
    inputs:
      value: spread.spread
    params:
      window: 20
      output_column: zscore

  neg_zscore:
    op: transform.negate
    version: 0.1.0
    inputs:
      value: zscore.zscore
    params:
      output_column: neg_zscore
```

Parameters: `input_column` (default `"value"`), `output_column` (default
`"negated"`), `timestamp_column` (default `"timestamp"`).

### `transform.feature_packer`

Pack multiple scalar columns into a single fixed-width array column. The
output column name is set by `output_column` (default `"features"`).

**Important:** The YAML input mapping keys must match the `input_columns`
entries. The input mapping key becomes the column name that FeaturePacker
reads. For example:

```yaml
nodes:
  packer:
    op: transform.feature_packer
    version: 0.2.0
    inputs:
      rolling_mean: momentum.rolling_mean   # key = column name
      spread_val: spread.spread             # key = column name
    params:
      input_columns: [rolling_mean, spread_val]  # must match keys above
      output_column: features
```

The output is exposed as the `output_column` name, not as `"features"` unless
that is what `output_column` says.

## Backtest operations

`backtest.backtest` simulates policy orders against a feed's executable quotes,
row by row. It is a materialized operation (requires computation at execution
time).

### Parameters

- `feed`: feed declaration (required)
- `policy`: policy declaration (required)
- `risk_policy`: risk policy declaration (optional)
- `initial_cash`: starting cash (required, float)
- `validate`: whether to validate input frame (default `true`)
- `limit_price_column`: input column with per-asset limit prices; `NaN`
  means market order (optional). Unfilled limit quantity rests as a
  good-til-cancelled working order (cancel-replace per side).
- `cancel_column`: input column; any nonzero, non-`NaN` value cancels all
  working orders for that asset before matching (optional).
- `fee_schedule`: fee declaration, `{kind: fixed, taker_rate, maker_rate}`
  with signed rates (negative = rebate). Market orders and crossing limits
  pay taker; rested working-order fills pay maker (optional, default no fees).
- `slippage_spread_fraction`: L1 slippage as a fraction of the spread added
  against the taker; limit fills are capped at the limit price
  (optional, default `0.0`).

### Feed declaration

The `feed` parameter accepts a declarative mapping:

```yaml
feed:
  kind: l1                          # required: "l1" or "l2"
  venue:
    name: my-venue                  # venue identifier
    universe: [AAPL, GOOGL]         # list of asset names
  bid_column: bid                   # L1 only, optional, default "bid"
  ask_column: ask                   # L1 only, optional, default "ask"
  # L2 only: bid_price_column, bid_volume_column, ask_price_column,
  # ask_volume_column (fixed-size best-first ladders) and depth_levels
  # (levels per side).
```

### Policy declaration

The `policy` parameter accepts a declarative mapping with a `kind` field:

```yaml
policy:
  kind: signal                      # or "threshold"
```

**Policy kinds:**

| `kind` | Class | Description | Parameters |
|--------|-------|-------------|------------|
| `signal` | `SignalPolicy` | Copies signal column directly as order quantities | (none) |
| `threshold` | `ThresholdPolicy` | Long/short based on signal vs threshold | `threshold` (float, default 0.0) |

### Risk policy declaration

The `risk_policy` parameter is optional and accepts a declarative mapping:

```yaml
risk_policy:
  kind: fractional_limit
  fraction: 0.25
```

**Risk policy kinds:**

| `kind` | Class | Description | Parameters |
|--------|-------|-------------|------------|
| `fractional_limit` | `FractionalLimitPolicy` | Caps each position's notional at a fraction of equity | `fraction` (float) |
| `fractional_kelly` | `FractionalKellyPolicy` | Sizes long exposure using Kelly criterion | `fraction` (float) |

### Full backtest example

```yaml
nodes:
  data:
    op: source.parquet
    version: 0.3.0
    params:
      path: market_data
      data_version: latest
      schema:
        time: timestamp
        columns:
          bid: float64
          ask: float64

  zscore:
    op: transform.rolling_z_score
    version: 0.1.0
    inputs:
      value: data.bid
    params:
      window: 20
      output_column: zscore

  bt:
    op: backtest.backtest
    version: 1.2.0
    inputs:
      signal: zscore.zscore
    params:
      feed:
        kind: l1
        venue:
          name: us-equities
          universe: [AAPL, GOOGL]
      policy:
        kind: threshold
        threshold: 0.5
      risk_policy:
        kind: fractional_limit
        fraction: 0.25
      initial_cash: 100000.0

outputs:
  equity: bt.equity
  cash: bt.cash
  orders: bt.order
```

### Backtest outputs

The backtest node always produces these output columns:

| Column | Type | Description |
|--------|------|-------------|
| `cash` | `Float64` | Running cash balance (net of fees) |
| `equity` | `Float64` | Total equity (cash + position value) |
| `balance` | `Array[Float64, N]` | Position quantities per asset |
| `order` | `Array[Float64, N]` | Executed order quantities (working + new fills) |
| `proposed_order` | `Array[Float64, N]` | Order quantities before risk policy |
| `fill_price` | `Array[Float64, N]` | Volume-weighted fill price (`0.0` when nothing filled) |
| `unfilled` | `Array[Float64, N]` | Signal quantity not filled this row |
| `fees` | `Float64` | Cumulative signed fees (negative = net rebates) |
| `open_orders` | `Array[Float64, N×2]` | Resting buy/sell quantities per asset |

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

> **Null policy warning.** Model inputs default to loud null failure: a single
> null in `features` or `target` aborts execution instead of training. Upstream
> transforms almost always produce warm-up nulls (e.g. the first `pct_change`
> row, which then propagates through `rolling_*`), so always declare per-input
> handling on model inputs — `nulls: drop`, or `nulls: fill` with an explicit
> `fill` value. Omitting it is the most common reason a first model run fails.

> **Warmup predictions are NaN.** Walk-forward models only predict rows after a
> retraining boundary. With the default `{ every: 100 }` scheduler the first 100
> predictions are NaN; metric nodes reject non-finite inputs by default, so set
> `drop_nonfinite: true` on a metric node bound to a prediction to drop warm-up
> rows instead of failing.

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

## Metric operations

Metrics are ordinary graph nodes. A metric node consumes its declared inputs,
sorts them by time (keeping input order among equal timestamps), reduces them
to a single value, and emits exactly one row stamped with the last input
timestamp (rows the reduction filters out do not move the stamp).
Declare a metric node like any other node and expose its value through
`outputs`:

```yaml
nodes:
  change:
    op: transform.delta
    version: 0.2.0
    inputs:
      value: prices.probability
    params:
      output_column: change

  sharpe:
    op: metrics.sharpe
    version: 1.0.0
    inputs:
      returns:
        from: change.change
        nulls: drop

outputs:
  signal: change.change
  sharpe: sharpe.sharpe
```

| `op` | Inputs | Value |
|------|--------|-------|
| `metrics.mse` | `prediction`, `target` | mean squared error of `prediction - target` |
| `metrics.mae` | `prediction`, `target` | mean absolute error of `prediction - target` |
| `metrics.max_drawdown` | `equity` | largest peak-relative drawdown of the equity curve |
| `metrics.sharpe` | `returns` | `mean / stdev` of returns (`ddof=1`, not annualized) |
| `metrics.total_return` | `equity` | `last / first - 1` of the equity curve |

Every metric op is version `1.0.0`, emits one `Float64` column named after the
metric (`mse`, `mae`, `max_drawdown`, `sharpe`, `total_return`), and accepts:

- `timestamp_column` (default `timestamp`): the node's time column name; it
  must match the parents' time column like any other node;
- `drop_nonfinite` (default `false`): drop rows where any input is non-finite
  (NaN, infinity, or a row that survived a permissive null policy) before the
  reduction.

Behavior notes:

- Inputs are strict. By default any NaN/inf raises with per-column counts;
  nulls are governed by the per-input null policy and default to a loud
  failure as well, and `drop_nonfinite` never rescues them: it only drops
  NaN/inf rows (plus nulls a permissive policy already let through). For
  warm-up *null* rows (the first `delta` value, predictions before a
  walk-forward model exists) declare `nulls: drop` on the binding; use
  `drop_nonfinite: true` for warm-up rows that arrive as NaN.
- `metrics.sharpe` requires a non-zero standard deviation of returns,
  `metrics.total_return` requires a positive first value, and
  `metrics.max_drawdown` requires a positive running peak (a curve that never
  exceeds zero has no defined relative drawdown). Nodes require at
  least their minimum row count after filtering: `mse`/`mae` need 1 row, the
  other metrics need 2.
- Metric values are not folded into any run summary; they are ordinary named
  outputs like every other node output.

## Chart declarations

`metadata.charts` declares which charts a runner should render for a
strategy's outputs. Nothing is inferred: only declared charts are rendered,
and each declaration must reference a declared output.

```yaml
metadata:
  charts:
    - kind: equity
      columns: [equity]
      title: Equity curve
    - kind: scatter
      name: fit
      output: prediction
      x: prediction
      y: [target]
```

Each entry is a mapping with:

- `kind` (required): one of `line`, `scatter`, `bars`, `equity`.
- `output` (required when the strategy declares more than one output): the
  strategy output name whose frame feeds the chart; defaults to the sole
  output.
- `name`: artifact name, defaulting to `<output>_<kind>`; must start with a
  letter and contain only letters, digits, `.`, `_`, or `-`, and must be
  unique across declarations.
- `columns`: column names to plot; required for `line`, `bars`, and `equity`.
  `bars` requires a `Date`/`Datetime` time column.
- `x`, `y`: required for `scatter`, rejected otherwise. `x` is one column;
  `y` is a column name or list of column names. Both must be scalar numeric.
- `title`: optional chart title.

Validation runs at strategy compile time (`parse_chart_decls`); unknown
fields, unknown outputs, and kind/field mismatches are errors. At render time
(`render_chart` / `render_chart_decls`) a declaration always produces an SVG:
if drawing fails (for example a missing column at render time), a
deterministic placeholder SVG carrying the reason is emitted instead.

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
