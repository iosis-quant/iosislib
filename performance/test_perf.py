"""YAML-driven performance benchmark runner for iosislib.

Loads scenario YAML definitions, generates synthetic data, streams to disk,
executes through the graph/backtester, and measures timing, memory, and
per-call profiling hotspots.

This is a standalone CLI tool. Run directly, not via pytest:

    python performance/test_perf.py                                # all scenarios
    python performance/test_perf.py --scenario backtest_scale      # one group
    python performance/test_perf.py --scenario backtest_scale --profile  # with cProfile
    python performance/test_perf.py --list-scenarios               # show available
    python performance/test_perf.py --generate-only                # data only
    python performance/test_perf.py --profile-top 50               # more profile lines
    python performance/test_perf.py --no-warmup                    # cold-cache mode
    python performance/test_perf.py --repeats 10                   # more statistical weight
"""

from __future__ import annotations

import cProfile
import gc
import hashlib
import io
import json
import pstats
import re
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from iosislib.core.graph import Graph, LocalExecutor
from iosislib.strategy.lowering import builtin_registry, lower
from iosislib.strategy.parser import loads as strategy_loads

_PERF_DIR = Path(__file__).parent
sys.path.insert(0, str(_PERF_DIR.parent / "src"))

from generate_data import CSV_GENERATORS, sha256_file  # noqa: E402
from stream_data import (  # noqa: E402
    DatasetSpec,
    generate_all,
    generate_market_data,
    stream_csv,
    stream_parquet,
)


# ---------------------------------------------------------------------------
# Scenario data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Thresholds:
    max_us_per_row: float = 100.0
    max_memory_mb: float = 1000.0

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Thresholds:
        if not d:
            return cls()
        return cls(
            max_us_per_row=d.get("max_us_per_row", 100.0),
            max_memory_mb=d.get("max_memory_mb", 1000.0),
        )


@dataclass(frozen=True)
class ScenarioData:
    name: str
    rows: int
    width: int = 1
    domain: str = "prices"
    format: str = "csv"
    stream: bool = False
    chunk_rows: int = 50_000


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    data: ScenarioData
    strategy: dict[str, Any]
    outputs: list[str]
    thresholds: Thresholds

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Scenario:
        data_raw = d["data"]
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            data=ScenarioData(
                name=data_raw["name"],
                rows=data_raw["rows"],
                width=data_raw.get("width", 1),
                domain=data_raw.get("domain", "prices"),
                format=data_raw.get("format", "csv"),
                stream=data_raw.get("stream", False),
                chunk_rows=data_raw.get("chunk_rows", 50_000),
            ),
            strategy=d["strategy"],
            outputs=d.get("outputs", []),
            thresholds=Thresholds.from_dict(d.get("thresholds")),
        )


# ---------------------------------------------------------------------------
# Benchmark result
# ---------------------------------------------------------------------------

@dataclass
class NodeTiming:
    name: str
    class_name: str
    elapsed: float
    pct_total: float = 0.0


@dataclass
class ProfileHotspot:
    ncalls: str
    tottime: float
    percall: float
    cumtime: float
    percall_cum: float
    filename: str
    lineno: int
    function_name: str


@dataclass
class BenchResult:
    scenario: str
    description: str
    rows: int
    width: int
    elapsed: float
    us_per_row: float
    rows_per_sec: float
    peak_memory_bytes: int
    peak_memory_mb: float
    all_times: list[float] = field(default_factory=list)
    median: float = 0.0
    stddev: float = 0.0
    min_time: float = 0.0
    max_time: float = 0.0
    node_timings: list[NodeTiming] = field(default_factory=list)
    profile_hotspots: list[ProfileHotspot] = field(default_factory=list)
    profile_text: str = ""
    passed: bool = True
    failures: list[str] = field(default_factory=list)
    data_bytes: int = 0


# ---------------------------------------------------------------------------
# Profiling executor
# ---------------------------------------------------------------------------

class ProfilingExecutor(LocalExecutor):
    """LocalExecutor that records per-node wall-clock timing."""

    def __init__(self) -> None:
        super().__init__()
        self.node_times: list[tuple[str, str, float]] = []
        self._s3_credentials: Any = None

    def _evaluate_to_root(self, graph: Graph) -> Any:
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

    def execute(self, graph: Graph) -> pl.DataFrame:
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
# cProfile runner
# ---------------------------------------------------------------------------

def _run_cprofile(
    graph: Graph, top_n: int = 30
) -> tuple[str, list[ProfileHotspot]]:
    """Run one execution under cProfile, return formatted text and parsed hotspots."""
    pr = cProfile.Profile()
    pr.enable()
    graph.execute()
    pr.disable()

    # Parse stats from raw cProfile results
    raw_stats = pr.getstats()
    func_stats: dict[tuple[str, int, str], tuple[int, int, float, float]] = {}
    for entry in raw_stats:
        code = entry.code
        if isinstance(code, str):
            func_key = (code, 0, code)
        else:
            func_key = (code.co_filename, code.co_firstlineno, code.co_name)
        cc = getattr(entry, "callcount", 0)
        nc = getattr(entry, "reccallcount", 0)
        tt = getattr(entry, "totaltime", 0.0)
        ct = getattr(entry, "inlinetime", 0.0)
        if func_key in func_stats:
            prev_cc, prev_nc, prev_tt, prev_ct = func_stats[func_key]
            func_stats[func_key] = (prev_cc + cc, prev_nc + nc, prev_tt + tt, prev_ct + ct)
        else:
            func_stats[func_key] = (cc, nc, tt, ct)

    # Sort by cumtime (tt) descending
    sorted_funcs = sorted(func_stats.items(), key=lambda x: x[1][2], reverse=True)

    hotspots: list[ProfileHotspot] = []
    for func_key, (cc, nc, tt, ct) in sorted_funcs[:top_n]:
        filename, lineno, function_name = func_key
        total_calls = cc + nc
        percall = tt / total_calls if total_calls else 0.0
        percall_cum = ct / total_calls if total_calls else 0.0
        hotspots.append(ProfileHotspot(
            ncalls=f"{cc}/{nc}" if nc else str(cc),
            tottime=round(tt, 6),
            percall=round(percall, 6),
            cumtime=round(ct, 6),
            percall_cum=round(percall_cum, 6),
            filename=filename,
            lineno=lineno,
            function_name=function_name,
        ))

    # Formatted text via pstats
    stream = io.StringIO()
    pstats.Stats(pr, stream=stream).sort_stats("cumtime").print_stats(top_n)
    return stream.getvalue(), hotspots


# ---------------------------------------------------------------------------
# Data generation & resolution
# ---------------------------------------------------------------------------

def _generate_scenario_data(
    scenario: Scenario, out_dir: Path, seed: int = 42
) -> tuple[Path, str]:
    spec = scenario.data
    ext = ".parquet" if "parquet" in spec.format else ".csv"
    path = out_dir / f"{spec.name}{ext}"

    if path.exists():
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        return path, sha

    rng = np.random.default_rng(seed + hash(spec.name) % 10000)

    if spec.domain == "market_data":
        df = generate_market_data(spec.rows, spec.width, rng)
    elif spec.domain in CSV_GENERATORS:
        df = CSV_GENERATORS[spec.domain](spec.rows, rng)
    else:
        raise ValueError(f"Unknown domain: {spec.domain}")

    t0 = time.perf_counter()
    if "parquet" in spec.format:
        stream_parquet(df, path)
    else:
        stream_csv(df, path)
    elapsed = time.perf_counter() - t0

    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"    Generated {path.name}: {df.height:,} rows, {elapsed:.2f}s")
    return path, sha


def _resolve_strategy_paths(
    strategy_def: dict[str, Any], data_path: Path, content_sha256: str
) -> dict[str, Any]:
    raw = yaml.dump(strategy_def, default_flow_style=False)
    raw = raw.replace("FILL_HASH", content_sha256)
    data_dir = str(data_path.parent).replace("\\", "/")
    if not data_dir.endswith("/"):
        data_dir += "/"
    raw = re.sub(
        r'(?<=path: )\.\./data/([a-zA-Z0-9_./-]+)',
        lambda m: data_dir + m.group(1),
        raw,
    )
    return yaml.safe_load(raw)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph(
    strategy_def: dict[str, Any], output_name: str | None = None
) -> Graph:
    raw = yaml.dump(strategy_def, default_flow_style=False)
    strategy = strategy_loads(raw)
    registry = builtin_registry()
    lowered = lower(strategy, registry)
    outputs = [output_name] if output_name else list(lowered.outputs.keys())
    if not outputs:
        raise ValueError("No outputs defined in strategy")
    return lowered.graph(outputs[0])


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------

def _bench_scenario(
    scenario: Scenario,
    strategy_def: dict[str, Any],
    *,
    warmup: int = 3,
    repeats: int = 5,
    profile_top_n: int = 30,
) -> BenchResult:
    """Benchmark a scenario: warmup, timed runs with memory + node timing,
    then a final profiled run for cProfile hotspots."""
    graph = _build_graph(strategy_def, scenario.outputs[0] if scenario.outputs else None)
    rows = scenario.data.rows

    # -- Warmup: prime OS cache, JIT, Polars lazy internals --
    for _ in range(warmup):
        exec_ = ProfilingExecutor()
        graph.execute(executor=exec_)

    # -- Timed runs: GC disabled for stable measurements --
    gc.collect()
    gc.disable()

    times: list[float] = []
    node_timings: list[NodeTiming] = []
    peak_mem = 0

    tracemalloc.start()
    try:
        for i in range(repeats):
            exec_ = ProfilingExecutor()
            t0 = time.perf_counter()
            graph.execute(executor=exec_)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

            _, peak = tracemalloc.get_traced_memory()
            peak_mem = max(peak_mem, peak)

            # Keep node timings from the fastest run
            if elapsed == min(times):
                total = sum(t for _, _, t in exec_.node_times)
                node_timings = [
                    NodeTiming(
                        name=name,
                        class_name=cls_name,
                        elapsed=t,
                        pct_total=(t / total * 100.0) if total > 0 else 0.0,
                    )
                    for name, cls_name, t in exec_.node_times
                ]
    finally:
        gc.enable()
        tracemalloc.stop()

    # -- cProfile run: one execution under the profiler for call-level hotspots --
    profile_text, profile_hotspots = _run_cprofile(graph, top_n=profile_top_n)

    # -- Compute statistics --
    best = min(times)
    us = best / rows * 1e6 if rows > 0 else 0.0
    rps = rows / best if best > 0 else 0.0
    med = statistics.median(times) if len(times) >= 2 else best
    sd = statistics.stdev(times) if len(times) >= 2 else 0.0

    result = BenchResult(
        scenario=scenario.name,
        description=scenario.description,
        rows=rows,
        width=scenario.data.width,
        elapsed=best,
        us_per_row=us,
        rows_per_sec=rps,
        peak_memory_bytes=peak_mem,
        peak_memory_mb=peak_mem / 1024 / 1024,
        all_times=times,
        median=med,
        stddev=sd,
        min_time=min(times),
        max_time=max(times),
        node_timings=node_timings,
        profile_hotspots=profile_hotspots,
        profile_text=profile_text,
    )

    # -- Threshold checks --
    if us > scenario.thresholds.max_us_per_row:
        result.passed = False
        result.failures.append(
            f"us_per_row {us:.1f} > threshold {scenario.thresholds.max_us_per_row}"
        )
    if result.peak_memory_mb > scenario.thresholds.max_memory_mb:
        result.passed = False
        result.failures.append(
            f"peak_memory {result.peak_memory_mb:.1f}MB > "
            f"threshold {scenario.thresholds.max_memory_mb}MB"
        )

    return result


# ---------------------------------------------------------------------------
# Scenario loading
# ---------------------------------------------------------------------------

def load_scenarios(scenarios_dir: Path) -> list[Scenario]:
    all_scenarios: list[Scenario] = []
    for path in sorted(scenarios_dir.glob("*.yaml")):
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for s in data.get("scenarios", []):
            all_scenarios.append(Scenario.from_dict(s))
    return all_scenarios


def load_scenario_group(scenarios_dir: Path, group_name: str) -> list[Scenario]:
    path = scenarios_dir / f"{group_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Scenario group not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [Scenario.from_dict(s) for s in data.get("scenarios", [])]


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _print_node_breakdown(result: BenchResult) -> None:
    if not result.node_timings:
        return
    print(f"      {'Node':<28s} {'TSFN':<22s} {'Time':>8s} {'%Total':>7s}")
    print(f"      {'-'*67}")
    for nt in sorted(result.node_timings, key=lambda x: x.elapsed, reverse=True):
        print(
            f"      {nt.name:<28s} {nt.class_name:<22s} "
            f"{nt.elapsed:>7.4f}s {nt.pct_total:>6.1f}%"
        )


def _print_profile_hotspots(result: BenchResult, top_n: int = 15) -> None:
    if not result.profile_hotspots:
        return
    print(f"\n      {'cProfile top {top_n} by cumulative time:'}")
    print(f"      {'Calls':>8s} {'TotTime':>8s} {'CumTime':>8s} {'Function':<40s} {'Location'}")
    print(f"      {'-'*100}")
    for h in result.profile_hotspots[:top_n]:
        loc = f"{h.filename}:{h.lineno}" if h.filename != "~" else ""
        print(
            f"      {h.ncalls:>8s} {h.tottime:>7.4f}s {h.cumtime:>7.4f}s "
            f"{h.function_name:<40s} {loc}"
        )


def print_summary(results: list[BenchResult]) -> None:
    print(f"\n{'=' * 110}")
    print(f"PERFORMANCE SUMMARY ({len(results)} scenarios)")
    print(f"{'=' * 110}")
    print(
        f"{'Scenario':<35s} {'Rows':>10s} {'W':>3s} "
        f"{'Best':>8s} {'Median':>8s} {'±SD':>7s} "
        f"{'us/row':>8s} {'rows/s':>10s} "
        f"{'PeakMem':>9s} {'Status':>6s}"
    )
    print("-" * 110)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(
            f"{r.scenario:<35s} {r.rows:>10,} {r.width:>3d} "
            f"{r.elapsed:>7.3f}s {r.median:>7.3f}s {r.stddev:>6.4f}s "
            f"{r.us_per_row:>7.1f} {r.rows_per_sec:>9,.0f} "
            f"{r.peak_memory_mb:>7.1f}MB {status:>6s}"
        )
    passed = sum(1 for r in results if r.passed)
    print(f"\n{passed}/{len(results)} passed")


def export_json(results: list[BenchResult]) -> str:
    return json.dumps(
        [
            {
                "scenario": r.scenario,
                "description": r.description,
                "rows": r.rows,
                "width": r.width,
                "elapsed_s": round(r.elapsed, 4),
                "median_s": round(r.median, 4),
                "stddev_s": round(r.stddev, 4),
                "us_per_row": round(r.us_per_row, 2),
                "rows_per_sec": round(r.rows_per_sec),
                "peak_memory_mb": round(r.peak_memory_mb, 2),
                "min_s": round(r.min_time, 4),
                "max_s": round(r.max_time, 4),
                "all_times": [round(t, 4) for t in r.all_times],
                "passed": r.passed,
                "failures": r.failures,
                "node_timings": [
                    {"name": n.name, "class": n.class_name,
                     "elapsed": round(n.elapsed, 4), "pct": round(n.pct_total, 1)}
                    for n in r.node_timings
                ],
                "profile_hotspots": [
                    {"calls": h.ncalls, "tottime": h.tottime, "cumtime": h.cumtime,
                     "function": h.function_name, "file": h.filename, "line": h.lineno}
                    for h in r.profile_hotspots[:20]
                ],
            }
            for r in results
        ],
        indent=2,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]
    do_json = "--json" in args
    list_scenarios = "--list-scenarios" in args
    generate_only = "--generate-only" in args
    no_warmup = "--no-warmup" in args
    do_profile = "--profile" in args

    # Parse --scenario <group|all>
    group_filter = None
    if "--scenario" in args:
        idx = args.index("--scenario")
        if idx + 1 < len(args):
            group_filter = args[idx + 1]
        else:
            print("Error: --scenario requires a group name", file=sys.stderr)
            sys.exit(1)

    # Parse --warmup, --repeats, --profile-top
    warmup = 0 if no_warmup else 3
    repeats = 5
    profile_top_n = 30
    if "--warmup" in args:
        idx = args.index("--warmup")
        warmup = int(args[idx + 1])
    if "--repeats" in args:
        idx = args.index("--repeats")
        repeats = int(args[idx + 1])
    if "--profile-top" in args:
        idx = args.index("--profile-top")
        profile_top_n = int(args[idx + 1])

    scenarios_dir = Path(__file__).parent / "scenarios"
    data_dir = Path(__file__).parent / "data"

    # --list-scenarios
    if list_scenarios:
        all_scenarios = load_scenarios(scenarios_dir)
        print(f"\nAvailable scenarios ({len(all_scenarios)} total):\n")
        for s in all_scenarios:
            print(f"  {s.name:<40s} {s.data.rows:>10,} rows  {s.description}")
        print()
        return

    # --generate-only
    if generate_only:
        all_scenarios = load_scenarios(scenarios_dir)
        seen: set[str] = set()
        specs: list[DatasetSpec] = []
        for s in all_scenarios:
            if s.data.name not in seen:
                seen.add(s.data.name)
                specs.append(DatasetSpec(
                    name=s.data.name, rows=s.data.rows, width=s.data.width,
                    domain=s.data.domain, format=s.data.format,
                    stream=s.data.stream, chunk_rows=s.data.chunk_rows,
                ))
        generate_all(specs, data_dir)
        return

    # Resolve groups
    if group_filter:
        groups = (
            [p.stem for p in sorted(scenarios_dir.glob("*.yaml"))]
            if group_filter == "all"
            else [group_filter]
        )
    else:
        groups = [p.stem for p in sorted(scenarios_dir.glob("*.yaml"))]

    data_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[BenchResult] = []

    for group in groups:
        try:
            scenarios = load_scenario_group(scenarios_dir, group)
        except FileNotFoundError as exc:
            print(f"  {exc}", file=sys.stderr)
            continue

        print(f"\n{'#' * 80}")
        print(f"# Group: {group} ({len(scenarios)} scenarios)")
        print(f"{'#' * 80}")

        for scenario in scenarios:
            print(f"\n  [{scenario.name}] {scenario.description}")
            print(
                f"    Data: {scenario.data.rows:,} rows, "
                f"{scenario.data.width} assets, {scenario.data.domain}"
            )

            try:
                data_path, sha = _generate_scenario_data(scenario, data_dir)
                strategy_def = _resolve_strategy_paths(
                    scenario.strategy, data_path, sha
                )

                result = _bench_scenario(
                    scenario, strategy_def,
                    warmup=warmup, repeats=repeats,
                    profile_top_n=profile_top_n,
                )
                result.data_bytes = data_path.stat().st_size if data_path.exists() else 0

                status = "PASS" if result.passed else "FAIL"
                print(
                    f"    [{status}] {result.elapsed:.3f}s  "
                    f"{result.us_per_row:.1f} us/row  "
                    f"{result.rows_per_sec:,.0f} rows/s  "
                    f"peak={result.peak_memory_mb:.1f}MB"
                )
                if result.failures:
                    for f_msg in result.failures:
                        print(f"      FAIL: {f_msg}")

                # Node breakdown
                if result.node_timings:
                    _print_node_breakdown(result)

                # cProfile hotspots
                if do_profile and result.profile_hotspots:
                    _print_profile_hotspots(result, top_n=profile_top_n)

                all_results.append(result)

            except Exception as exc:
                print(f"    [ERROR] {exc}")

    if all_results:
        print_summary(all_results)

        # Print profiles for all results if --profile
        if do_profile:
            print(f"\n{'=' * 110}")
            print("CPROFILE REPORTS (sorted by cumulative time)")
            print(f"{'=' * 110}")
            for r in all_results:
                if r.profile_text:
                    print(f"\n--- {r.scenario} ({r.elapsed:.3f}s, {r.us_per_row:.1f} us/row) ---")
                    print(r.profile_text)

        if do_json:
            print(f"\n{export_json(all_results)}")


if __name__ == "__main__":
    main()
