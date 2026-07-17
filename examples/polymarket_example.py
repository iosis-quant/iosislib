from __future__ import annotations

import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

def plot_yes_no_prices(df):
    market_columns = [column for column in df.columns if column != "timestamp"]
    if not market_columns:
        raise ValueError("No market price columns found")

    timestamps = df["timestamp"].to_list()
    fig, axes_grid = plt.subplots(
        1,
        2,
        figsize=(16, 10),
        sharex=True,
        sharey=True,
    )
    axes = list(axes_grid.flatten())

    for ax, column in zip(axes, market_columns):
        prices = df[column].to_list()
        yes_times, yes_prices = _available_outcome_points(timestamps, prices, 0)
        no_times, no_prices = _available_outcome_points(timestamps, prices, 1)

        ax.plot(yes_times, yes_prices, color="green", marker="", label="Yes")
        ax.plot(no_times, no_prices, color="red", marker="", label="No")
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


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.core.graph import Graph
from src.core.node import Node
from src.tsfn.adapters import PolymarketPriceHistory,YFinanceOHLCV
from src.tsfn.transforms import Delta, Spread, Logit, Ratio


price_history1 = Node(
    PolymarketPriceHistory,
    parameters={
        "event_slug": "will-global-art-market-sales-hit-65-billion-for-2026-20260605225003542",
        "interval": "max",
        "fidelity": 1,
        "alignment_tolerance":"1m",
    },
    name="global_art",
)

price_history2 = Node(
    PolymarketPriceHistory,
    parameters={
        "event_slug": "will-china-become-the-2-global-art-market-in-2026-20260626194726882",
        "interval": "max",
        "fidelity": 1,
        "alignment_tolerance":"1m",
    },
    name="china_art",
)

logit1 = Node(
    Logit,
    bindings={"value":(price_history1, "0")},
    name="logit1"
)

logit2 = Node(
    Logit,
    bindings={"value":(price_history2, "0")},
    name="logit2"
)

spread = Node(
    Spread,
    bindings={"left":(logit1, "logit"), "right":(logit2, "logit")},
    name="logit_spread"
)

result = Graph(spread).execute()
print(result)
plot_yes_no_prices(result)
