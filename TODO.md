# Roadmap

The core graph, causal alignment, null policies, materialization boundaries,
walk-forward model lifecycle, transforms, local CSV/Parquet sources, and
Polymarket/yfinance adapters are implemented. Current work is ordered by the
contracts needed before a release:

1. Rework definition IDs only when cache freshness/dirty semantics have concrete
   requirements; current Node IDs remain authoritative.
2. Add content-addressed result caching, then dependency-depth and Ray executor
   implementations.
3. Expand adapter coverage and domain transforms as concrete workflows require
   them.
4. Build backtesting and richer observability as layers above the graph core.

The numbered design briefs in `temp/todo/` are local coordination material and
are intentionally not part of the published package.
