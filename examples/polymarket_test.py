from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.classes import Graph, Node
from src.tsfn.adapters import PolymarketPriceHistory


# july 1 2026 shanghai highest temperature
TOKEN_ID = "66101062242573835490004104913349045280360775766528514737008188965345903973945"

if not TOKEN_ID:
    raise SystemExit(
        "Set tokenid"
    )


price_history = Node(
    PolymarketPriceHistory,
    parameters={
        "token_id": TOKEN_ID,
        "interval": "1m",
        "fidelity": 1,
        "price_column": "seoul_weather_price",
    },
    name="seoul_weather_price_history",
)

print(Graph(price_history).execute())
