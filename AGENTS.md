# Repository Guidelines

## Project Purpose

iosisLib is a typed, deterministic, time-aware computation graph for feature engineering and model workflows over Polars time-series data. A user program constructs the graph IR in ordinary Python. The graph validates schemas and lineage, aligns parent data causally, and executes through an `Executor`. Deterministic identities are intended to support later caching and distributed execution.

Keep the library narrower than a generic workflow orchestrator or backtester. Polars is the columnar execution engine; iosisLib owns graph semantics, temporal alignment, type contracts, governance, and model lifecycle boundaries. Backtesting, caching, Ray execution, a portable TOML/YAML strategy DSL, and richer observability are future layers, not reasons to distort the current core.

## Source Layout And Imports

- `src/iosislib/core/tsfn.py`: `TimeAxis`, `ColumnSignature`, `FrameSignature`, null handling, `TSFN`, `ItemwiseUnaryTSFN`, `ItemwiseStructTSFN`, and `BatchTSFN`.
- `src/iosislib/core/model.py`: datasets, splitters, immutable model checkpoints, schedulers, and `SupervisedModelTSFN`.
- `src/iosislib/core/node.py`: immutable declarative `Node` objects, bindings, materialization choices, and content-addressed IDs.
- `src/iosislib/core/graph.py`: verification, topological ordering, time alignment, `Executor`, and `LocalExecutor`.
- `src/iosislib/core/utils.py`: generic dtype, shape, serialization, tolerance, and zero-copy helpers.
- `src/iosislib/tsfn/transforms/`: concrete transforms such as delta, logit, ratio, and spread.
- `src/iosislib/tsfn/adapters/`: data-producing TSFNs such as Polymarket and yfinance.
- `tests/`: the tracked pytest suite. `examples/` contains runnable examples.
- `temp/`: ignored research and prototypes only. Production code and tracked tests must not depend on it.

The old `src.classes` monolith and temporary `src.*` package namespace were deliberately removed. Do not recreate a compatibility facade. Import from the owning module, for example `from iosislib.core.node import Node` and `from iosislib.core.graph import Graph`. Internally, preserve the dependency direction `utils -> tsfn -> model/node -> graph`; `tsfn.py` must not depend on model or graph state.

## Architectural Invariants

### Nodes Are Declarations

A `Node` says which versioned TSFN runs, with what parameters, which predecessor outputs satisfy its named inputs, and which per-input policies apply. Nodes do not execute, inspect graph state, or own scheduling infrastructure. They are immutable after construction, hash and compare by persistent ID, and expose both typed `node.output("name")` bindings and `node.output_name` sugar.

Do not introduce a `GraphEdge` wrapper. The existing binding `input_name -> (parent_node, parent_output)` already carries the graph relation. Do not infer semantic equality from Python object identity; node identity is content-addressed.

### Graphs Own Execution

`Graph` discovers dependencies, rejects cycles, verifies exact bindings and frame compatibility, and delegates lowering/materialization to an `Executor`. A successfully constructed Graph is an immutable validated declaration with one canonical topological tuple. Executors are runtime strategies passed to `graph.execute(executor=...)`; they are not stored Graph state or identity. Execution errors must retain node name/ID and TSFN version context. Keep execution methods out of `Node`, and do not make TSFNs aware of their consumers or the enclosing graph.

There is no runtime `SourceNode` distinction and no external source-frame dictionary. A node without predecessors is valid only when its TSFN input is `FrameSignature.empty()`. The executor calls that TSFN with no frame, and its parameters generate or load the data. This is a type-contract distinction, not an `isinstance` branch over source subclasses.

`LocalExecutor` currently walks a topological order sequentially. The `Executor` abstraction is the intended extension point for dependency-depth parallelism and a future Ray runtime: independent ready nodes may execute concurrently while Polars remains lazy inside each task. Do not put Ray concerns into TSFNs or make all nodes eager in anticipation. There is currently no `Graph.compile()` or separate compiled/description object; do not add a second graph-shaped type unless it carries a concrete execution invariant that `Graph` and `Executor` cannot express.

### Consumers Own Alignment Policy

A child consumes each bound input, so the child node owns its `tolerances` mapping. Never put tolerance on the parent output or TSFN. Missing tolerance means an unbounded backward as-of match.

For bound nodes, execution:

1. Projects each parent to its time column and requested value columns.
2. Sorts every projection by time.
3. Builds the sorted union of all parent timestamps.
4. Backward `join_asof` aligns each projection using that input's tolerance.
5. Passes the aligned lazy frame to the child.

The union timeline is intentional: no parent timestamp is lost by default. Backward alignment prevents future lookahead. Rows before a parent's first observation remain null. A single parent follows the same path.

## Frame And Type Contracts

`TSFN.type_signature()` returns exactly `(input_frame_signature, output_frame_signature)`. Time is frame metadata, not a bindable value column. `FrameSignature.columns` must be a tuple and contains only value columns. `FrameSignature.empty()` is the only inputless contract.

`TimeAxis` declares column name, Polars dtype, and timezone. Parent and child time axes must match. `ColumnSignature` declares name, element dtype, and optional logical shape. A shaped value is physically stored as a flat fixed-size `pl.Array(element_dtype, product(shape))`; shape carries no domain semantics. Do not add custom tensor containers or graph-wide broadcast machinery without a concrete operation that requires it.

Dtype matching is structural and conservative. Parameterized dtypes must match their parameters. List inner types matter recursively. Bare `pl.List`, `pl.Array`, or `pl.Datetime` classes are not broad wildcards for arbitrary parameterized instances.

Shape-polymorphic unary transforms should resolve their input/output shape from the bound parent and use `arr.eval(pl.element())`. N-ary `ItemwiseStructTSFN` currently requires equal input shapes unless the operation explicitly overrides `resolve_output_shape`. Do not generate large trees of `arr.get` expressions or call Python once per row.

## TSFN Implementation Rules

Every concrete TSFN must:

- Define a non-empty `VERSION`; changing behavior requires a version change.
- Use a frozen `TSFNConfig` dataclass through `CONFIG_CLS` for parameters. Prefer typed `Node(..., config=ExactConfig(...))`; mapping-based `parameters=` remains supported and must normalize identically.
- Declare accurate input and output signatures.
- Return a `pl.LazyFrame` from `apply()`.
- Preserve the declared time contract and emit exactly the declared physical schema.
- Raise explicit `ValueError` for invalid structure and `TypeError` for incompatible types.

Intermediate abstract TSFN subclasses may omit a concrete version; validation runs when a subclass becomes concrete. Ordinary transforms should be deterministic and side-effect free. Source-style adapters and model training may perform I/O or state transitions, but their graph-visible behavior must be governed by parameters, versions, input data, and deterministic seeds.

Prefer native Polars expressions. `ItemwiseUnaryTSFN` is the preferred scalar/fixed-array elementwise path. `ItemwiseStructTSFN` uses a struct batch for n-ary operations. `BatchTSFN` uses `LazyFrame.map_batches` for full-frame operations and is a materialization boundary. Avoid `map_elements`, Python row loops, or arbitrary eager collection inside normal TSFNs.

`TSFN.REQUIRES_MATERIALIZATION` is a class-level minimum imposed by the operation. `Node(materialize=True)` may request an additional persisted boundary for that declaration, but a node cannot disable a boundary required by its TSFN. Effective materialization is Node declaration state and participates in Node/Graph identity. The executor keeps values lazy between boundaries, collects required/intermediate boundaries, and always collects the root result.

## Null And External-Compute Semantics

A Polars null means missing or invalid data, not zero, NaN, or an empty value. Preserve that distinction. Nodes can configure per-input `NullHandler` values using `NullPolicy.ERROR`, `PROPAGATE`, `DROP`, `FILL`, `PASS`, or a named custom function. Every declared input resolves to an effective handler for runtime, description, and identity; omitted and explicit defaults are equivalent. Fill policies require explicit fill values, and fill values for non-`FILL` handlers are rejected. Custom handlers must preserve row count where required.

Custom null-handler functions require an explicit behavior version through `NullHandler`, the Node mapping, or `function.__iosis_version__`; changing handler behavior requires changing that version. Models default to loud failure on null input or prediction output. Do not silently edit market/model data to make it look complete. Diagnose feed behavior, alignment, or plotting before changing values.

When crossing out of Polars, use the existing Arrow/NumPy/Torch bridge methods. They attempt zero-copy transfer and validate aliasing when requested, but zero-copy is a conditional invariant, not a slogan: null bitmaps, non-contiguous buffers, unsupported Arrow DLPack paths, or dtype conversion may require an explicit allowed copy. Keep Python orchestration outside the hot elementwise path.

## Model Lifecycle

Model training is graph computation, not an external side pipeline. It is also a hard materialization boundary, represented by `BatchTSFN.REQUIRES_MATERIALIZATION` and verified by the graph.

A `Model` is an immutable, serializable checkpoint. `SupervisedModel.fit()` consumes a `DatasetSplit` and returns a new checkpoint; it must not mutate and return itself. Implementations may store weights directly or by path without introducing an artifact hierarchy.

`SupervisedModelTSFN` consumes exactly `features` and `target`, emits `prediction`, sorts by graph time, and performs walk-forward segment processing. A `DatasetSplitter` partitions only the historical prefix available at a retraining boundary. A `Scheduler` observes `ScheduleContext` row counts, retrain counts, and prior segment metrics, then returns a `ScheduleDecision`. Metrics are produced during segment operation and can drive later decisions. Seeds must make fitting deterministic for identical graph inputs.

Do not add `available_at` or framework-wide bitemporality casually. The graph enforces causal parent alignment, and the model loop trains only on earlier rows. Domain-specific label availability and sophisticated purging/embargo behavior belong in explicit splitter/feature programs.

## Identity, Serialization, And Caching

TSFN version, qualified class name, resolved signatures, normalized parameters, bindings, tolerances, effective null policies/relevant fill values, effective materialization, and outputs contribute to node identity. Names are human labels only. Graph IDs derive from the canonical topological tuple and root ID.

Serialization must be deterministic: sort mapping keys/items, normalize unordered values, reject unsupported or non-finite values, and never use process-local identity. New value/config/helper classes that enter IDs need stable `to_dict()`/`__str__()` behavior. The `iosisLib` namespace migration intentionally invalidated pre-migration qualified-name-based Node and Graph IDs; future module moves are identity migrations too.

`LocalExecutor` caches materialized node results as Parquet files keyed by node ID. Cache location is set via `cache_dir` parameter or the `IOSIS_CACHE_DIR` environment variable. During topological traversal, materialized nodes check the cache first; on a hit, the result is loaded via `pl.scan_parquet()` and computation is skipped entirely. Cache writes that fail are silently ignored. The root node is cached in `execute()` only when it is a declared materialization boundary. Non-materialized nodes are never cached.

## Adapters And Existing Behavior

Polymarket slug mode resolves Gamma event markets, preserves outcome order from the API, fetches each corresponding CLOB token history, and represents each market row as an outcome-price array. Market column names may be generated or explicitly supplied. Its internal history alignment can use an adapter tolerance; do not assume mismatched CLOB ticks are simultaneous or fabricate complementary prices.

The yfinance adapter imports pandas/yfinance only when used and is covered by the `yfinance` package extra. Network tests should mock API boundaries; tracked tests must not depend on live services.

Local CSV/Parquet sources consume one complete in-memory byte snapshot: they hash and parse the same bytes, then expose the parsed frame lazily downstream. Mutation after capture cannot change that execution. This intentionally uses memory for both the source bytes and parsed frame and creates no staging artifacts.

## Development Workflow

- `python -m pip install -e ".[dev]"`: install package and development extras.
- `python -m compileall -q src/iosislib examples`: syntax-check package modules and tracked examples.
- `python -m ruff check src examples`: lint production source and tracked examples.
- `python -m mypy`: check the enforced strict public typing slice.
- `python examples/offline_graph.py`: execute the offline compatibility example.
- `python -m pytest`: run tracked tests under `tests/`.
- `python -m build`: build wheel and source distributions under `dist/`.

Use Python 3.11+, four-space indentation, type hints, frozen dataclasses for value objects/configs, `PascalCase` classes, `snake_case` functions/variables, and uppercase constants. Keep comments sparse and explanatory.

Tests belong in `tests/test_*.py`. Brutally cover success and failure paths: schema/timezone/dtype/shape validation, parameterized dtype matrices, null policies, missing/extra/wrong bindings, cycles, deterministic IDs, as-of tolerances, union timelines, no-lookahead behavior, materialization boundaries, bridge copy guarantees, model scheduling/splitting, and adapter payload edge cases.

Keep commits imperative and specific, for example `Add scheduler boundary tests`. Pull requests should explain behavior and identity changes, list tests run, and link issues where available. Do not commit secrets, generated distributions, caches, plots, model weights, or machine-specific paths.
