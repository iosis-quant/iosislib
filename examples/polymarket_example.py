from __future__ import annotations

import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.classes import Graph, Node
from src.tsfn.adapters import PolymarketPriceHistory


# july 1 2026 shanghai highest temperature
TOKEN_ID = "largest-company-end-of-december-2026"

if not TOKEN_ID:
    raise SystemExit(
        "Set tokenid"
    )


price_history = Node(
    PolymarketPriceHistory,
    parameters={
        "event_slug": TOKEN_ID,
        "interval": "max",
        "fidelity": 1,
        "alignment_tolerance":"1m",
    },
    name="weather_prices",
)


def plot_yes_no_prices(df):
    market_columns = [column for column in df.columns if column != "timestamp"]
    if not market_columns:
        raise ValueError("No market price columns found")

    timestamps = df["timestamp"].to_list()
    fig, axes_grid = plt.subplots(
        1,
        4,
        figsize=(16, 10),
        sharex=True,
        sharey=True,
    )
    axes = list(axes_grid.flatten())

    for ax, column in zip(axes, market_columns):
        prices = df[column].to_list()
        yes_times, yes_prices = _available_outcome_points(timestamps, prices, 0)
        no_times, no_prices = _available_outcome_points(timestamps, prices, 1)

        ax.plot(yes_times, yes_prices, color="green", marker=".", label="Yes")
        ax.plot(no_times, no_prices, color="red", marker=".", label="No")
        ax.set_title(f"Market {column}")
        ax.set_ylabel("price")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(market_columns):]:
        ax.set_visible(False)

    axes[0].legend(loc="upper right")
    axes[0].set_xlabel("time")
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()


def _available_outcome_points(timestamps, price_vectors, outcome_index):
    points = [
        (timestamp, row[outcome_index])
        for timestamp, row in zip(timestamps, price_vectors)
        if row is not None
        and len(row) > outcome_index
        and row[outcome_index] is not None
    ]
    if not points:
        return [], []
    return zip(*points)


result = Graph(price_history).execute()
print(result)
plot_yes_no_prices(result)
