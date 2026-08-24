"""YAML-driven performance benchmarking runner for iosislib strategies.

Accepts standard iosislib strategy YAML files (iosis.strategy 0.1.0 format),
lowers them through the strategy registry, and benchmarks graph execution.

Usage:
    python performance/runner.py                          # run all strategy files
    python performance/runner.py strategies/logit_delta.strategy.yaml
    python performance/runner.py --output change           # benchmark one output
    python performance/runner.py --json
    python performance/runner.py --list-ops                # show available TSFNs
    python performance/runner.py --profile                 # cProfile runs
    python performance/runner.py --no-warmup               # skip warmup (cold-cache)
"""

from __future__ import annotations

import cProfile
import gc
import io
import json
import pstats
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from iosislib.core.graph import Graph, LocalExecutor
from iosislib.strategy.lowering import builtin_registry, lower
from iosislib.strategy.parser import loads


# ---------------------------------------------------------------------------
# Benchmark result
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    strategy_name: str
    output_name: str
    rows: int
    elapsed: float
    us_per_row: float
    rows_per_sec: float
    all_times: list[float] = field(default_factory=list)
    node_times: list[tuple[str, str, float]] = field(default_factory=list)
    median: float = 0.0
    stddev: float = 0.0
    min_time: float = 0.0
    max_time: float = 0.0


# ---------------------------------------------------------------------------
# Profiling executor
# ---------------------------------------------------------------------------

class ProfilingExecutor(LocalExecutor):
    """LocalExecutor that records per-node wall-clock timing.

    Times the full lifecycle of each node: ``lower_node`` (lazy graph
    construction) plus ``materialize`` if the node is a materialization
    boundary.  This gives an accurate picture of where time is spent.
    """

    def __init__(self) -> None:
        super().__init__()
        self.node_times: list[tuple[str, str, float]] = []
        self._pending_boundary: dict[str, float] = {}

    def _evaluate_to_root(self, graph):  # type: ignore[override]
        import polars as pl

        from iosislib.core.graph import _s3_credentials_scope

        results: dict[str, pl.LazyFrame] = {}

        with _s3_credentials_scope(self._s3_credentials):
            for node in graph.node_list:
                t0 = time.perf_counter()
                try:
                    node_input_lf = (
                        None
                        if not node.bindings
                        else self.align_inputs(node, results)
                    )
                    results[node.ID] = self.lower_node(node, node_input_lf)
                except Exception as exc:
                    raise RuntimeError(
                        f"Execution failed at node "
                        f"'{node.name or node.ID[:8]}' "
                        f"({node.function_cls.__name__}@"
                        f"{node.function.version}): {exc}"
                    ) from exc

                if (
                    node.ID in graph.materialized_node_ids
                    and node.ID != graph.root_node.ID
                ):
                    results[node.ID] = self.materialize(
                        node, results[node.ID]
                    ).lazy()

                elapsed = time.perf_counter() - t0
                if node.ID != graph.root_node.ID:
                    self.node_times.append(
                        (node.name or node.ID[:8], node.function_cls.__name__, elapsed)
                    )

            return results[graph.root_node.ID]

    def execute(self, graph):  # type: ignore[override]
        from iosislib.core.graph import _s3_credentials_scope

        with _s3_credentials_scope(self._s3_credentials):
            root_lf = self._evaluate_to_root(graph)
            t0 = time.perf_counter()
            result = self.materialize(graph.root_node, root_lf)
            elapsed = time.perf_counter() - t0
            self.node_times.append(
                (graph.root_node.name or graph.root_node.ID[:8],
                 f"{graph.root_node.function_cls.__name__}+materialize",
                 elapsed)
            )
            return result


# ---------------------------------------------------------------------------
# Core execution helpers
# ---------------------------------------------------------------------------

def _load_strategy(path: Path) -> Any:
    """Parse a strategy YAML file, resolving relative paths to the strategy's directory."""
    import re
    raw = path.read_text(encoding="utf-8")
    strategy_dir = str(path.parent).replace("\\", "/")
    if not strategy_dir.endswith("/"):
        strategy_dir += "/"
    raw = re.sub(
        r'(?<=path: )([a-zA-Z0-9_./-]+\.(?:csv|parquet))',
        lambda m: m.group(1) if Path(m.group(1)).is_absolute() else strategy_dir + m.group(1),
        raw,
    )
    return loads(raw)


def _lower_strategy(strategy: Any) -> Any:
    """Lower a Strategy IR into core Node objects using the builtin registry."""
    registry = builtin_registry()
    return lower(strategy, registry)


def _run_once(
    graph: Graph, executor: LocalExecutor | None = None
) -> tuple[pl.DataFrame, float, list[tuple[str, str, float]]]:
    """Execute a graph once, return (result_df, elapsed_seconds, node_times)."""
    exec_ = executor or LocalExecutor()
    t0 = time.perf_counter()
    result = graph.execute(executor=exec_)
    elapsed = time.perf_counter() - t0
    node_times = getattr(exec_, "node_times", [])
    return result, elapsed, list(node_times)


def _bench_fn(
    graph: Graph,
    warmup: int = 3,
    repeats: int = 5,
    executor_factory: Any = None,
) -> tuple[float, list[float], list[tuple[str, str, float]], pl.DataFrame]:
    """Warm up, then time repeated executions.

    Returns (best_time, all_times, node_times_from_best, result_df).

    Warmup runs prime the OS buffer cache, so this measures warm-cache
    (in-memory) Polars compute performance, not cold-disk I/O.
    """
    last_df: pl.DataFrame | None = None
    for _ in range(warmup):
        last_df, _, _ = _run_once(graph, executor_factory() if executor_factory else None)

    times = []
    best_node_times: list[tuple[str, str, float]] = []
    best_df: pl.DataFrame | None = last_df

    gc.collect()
    gc.disable()
    try:
        for _ in range(repeats):
            exec_ = executor_factory() if executor_factory else None
            df, elapsed, nt = _run_once(graph, exec_)
            times.append(elapsed)
            if elapsed == min(times):
                best_node_times = nt
                best_df = df
    finally:
        gc.enable()

    return min(times), times, best_node_times, best_df  # type: ignore[return-value]


def _profile_fn(graph: Graph, top_n: int = 25) -> str:
    """cProfile one execution, return formatted stats."""
    pr = cProfile.Profile()
    pr.enable()
    _run_once(graph)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(top_n)
    return s.getvalue()


# ---------------------------------------------------------------------------
# Strategy benchmark runner
# ---------------------------------------------------------------------------

def run_strategy_benchmark(
    strategy_path: Path,
    *,
    output_name: str | None = None,
    warmup: int = 3,
    repeats: int = 5,
    run_profile: bool = False,
) -> tuple[list[BenchResult], list[str | None]]:
    """Benchmark all (or one) outputs of a strategy file."""
    strategy = _load_strategy(strategy_path)
    lowered = _lower_strategy(strategy)

    outputs_to_run = (
        [output_name] if output_name else list(lowered.outputs.keys())
    )

    results: list[BenchResult] = []
    profiles: list[str | None] = []

    for out_name in outputs_to_run:
        if out_name not in lowered.outputs:
            raise ValueError(
                f"Output {out_name!r} not found in strategy. "
                f"Available: {sorted(lowered.outputs)}"
            )

        graph = lowered.graph(out_name)
        exec_factory = ProfilingExecutor if run_profile else None
        best, all_times, node_times, result_df = _bench_fn(
            graph, warmup, repeats, exec_factory
        )
        rows = result_df.height

        us = best / rows * 1e6 if rows > 0 else 0.0
        rps = rows / best if best > 0 else 0.0

        if len(all_times) >= 2:
            med = statistics.median(all_times)
            sd = statistics.stdev(all_times)
        else:
            med = best
            sd = 0.0

        results.append(BenchResult(
            strategy_name=strategy.name,
            output_name=out_name,
            rows=rows,
            elapsed=best,
            us_per_row=us,
            rows_per_sec=rps,
            all_times=all_times,
            node_times=node_times,
            median=med,
            stddev=sd,
            min_time=min(all_times),
            max_time=max(all_times),
        ))

        profile_report = None
        if run_profile:
            profile_report = _profile_fn(graph)
        profiles.append(profile_report)

    return results, profiles


# ---------------------------------------------------------------------------
# YAML loader helpers
# ---------------------------------------------------------------------------

def load_all_strategies(strategies_dir: Path) -> list[Path]:
    """Find all .strategy.yaml files in a directory."""
    return sorted(strategies_dir.glob("*.strategy.yaml"))


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def print_table(results: list[BenchResult]) -> None:
    print(f"\n{'Strategy':<20s} {'Output':<20s} {'Rows':>10s} {'Time':>8s} {'us/row':>8s} {'rows/s':>12s} {'stddev':>8s}")
    print("-" * 88)
    for r in results:
        print(
            f"{r.strategy_name:<20s} {r.output_name:<20s} "
            f"{r.rows:>10,} {r.elapsed:>7.3f}s {r.us_per_row:>7.1f} {r.rows_per_sec:>11,.0f} {r.stddev:>7.4f}s"
        )


def export_json(results: list[BenchResult]) -> str:
    return json.dumps(
        [
            {
                "strategy": r.strategy_name,
                "output": r.output_name,
                "rows": r.rows,
                "elapsed_s": round(r.elapsed, 4),
                "us_per_row": round(r.us_per_row, 2),
                "rows_per_sec": round(r.rows_per_sec),
                "median_s": round(r.median, 4),
                "stddev_s": round(r.stddev, 4),
                "min_s": round(r.min_time, 4),
                "max_s": round(r.max_time, 4),
                "all_times": [round(t, 4) for t in r.all_times],
            }
            for r in results
        ],
        indent=2,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_discoverable() -> None:
    """Show all TSFN operations available in the builtin registry."""
    registry = builtin_registry()
    print(f"\n=== Available TSFN operations ({len(registry.operations)}) ===\n")
    for (op_name, version), cls in sorted(registry.operations.items()):
        print(f"  {op_name:<45s}  {cls.__name__} v{version}")
    print()


def main() -> None:
    args = sys.argv[1:]
    do_profile = "--profile" in args
    do_json = "--json" in args
    list_ops = "--list-ops" in args
    no_warmup = "--no-warmup" in args

    # Parse --output <name>
    output_filter = None
    if "--output" in args:
        idx = args.index("--output")
        if idx + 1 < len(args):
            output_filter = args[idx + 1]
        else:
            print("Error: --output requires a name argument", file=sys.stderr)
            sys.exit(1)

    # Parse --warmup and --repeats
    warmup, repeats = (0 if no_warmup else 3), 5
    if "--warmup" in args:
        idx = args.index("--warmup")
        warmup = int(args[idx + 1])
    if "--repeats" in args:
        idx = args.index("--repeats")
        repeats = int(args[idx + 1])

    paths = [a for a in args if not a.startswith("--") and not a.isdigit()]

    if list_ops:
        _print_discoverable()
        return

    strategies_dir = Path(__file__).parent / "strategies"
    if paths:
        strategy_files = [Path(p) for p in paths]
    else:
        strategy_files = load_all_strategies(strategies_dir)

    if not strategy_files:
        print("No strategy files found.")
        print(f"  Looked in: {strategies_dir}")
        print(f"  Run: python performance/generate_data.py  to create benchmark CSVs")
        sys.exit(1)

    all_results: list[BenchResult] = []
    all_profiles: list[str | None] = []

    for path in strategy_files:
        print(f"\n{'#' * 80}")
        print(f"# Strategy: {path.name}")
        print(f"{'#' * 80}")

        try:
            results, profiles = run_strategy_benchmark(
                path,
                output_name=output_filter,
                warmup=warmup,
                repeats=repeats,
                run_profile=do_profile,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            continue

        for r in results:
            print(
                f"  {r.output_name:<20s} {r.rows:>10,} rows  "
                f"{r.elapsed:>7.3f}s  {r.us_per_row:>6.1f} us/row  "
                f"{r.rows_per_sec:>10,.0f} rows/s"
            )
            if r.node_times:
                for name, cls_name, nt in r.node_times:
                    print(f"    {name:<24s} {cls_name:<20s} {nt:>7.4f}s")

        all_results.extend(results)
        all_profiles.extend(profiles)

    print(f"\n{'=' * 80}")
    print(f"SUMMARY ({len(all_results)} benchmarks)")
    print(f"{'=' * 80}")
    print_table(all_results)

    if any(p is not None for p in all_profiles):
        print(f"\n{'=' * 80}")
        print("PROFILES")
        print(f"{'=' * 80}")
        for p in all_profiles:
            if p:
                print(f"\n{p}")

    if do_json:
        print(f"\n{export_json(all_results)}")


if __name__ == "__main__":
    main()
