"""Plot a time-first Polars frame and export raw SVG."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from iosislib.charting import figure_to_svg, plot_frame


def run_example() -> str:
    start = datetime(2026, 1, 1)
    frame = pl.DataFrame(
        {
            "timestamp": [start + timedelta(hours=i) for i in range(120)],
            "price": [100.0 + i * 0.15 for i in range(120)],
            "band": [[95.0 + i * 0.15, 105.0 + i * 0.15] for i in range(120)],
        }
    )
    figure, _ = plot_frame(frame, max_points=120, title="Price and band")
    return figure_to_svg(figure)


def main() -> None:
    print(run_example())


if __name__ == "__main__":
    main()
