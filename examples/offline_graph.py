from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from iosislib.core.graph import Graph
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature
from iosislib.tsfn.adapters import CSVSource, sha256_file
from iosislib.tsfn.transforms import Delta, Logit


EXPECTED_CHANGES = [None, 1.098612288668, 1.098612288668]


def _write_prices(path: Path) -> None:
    pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 1, 1, 0, 0),
                datetime(2026, 1, 1, 0, 1),
                datetime(2026, 1, 1, 0, 2),
            ],
            "probability": [0.25, 0.5, 0.75],
        }
    ).write_csv(path)


def run_example() -> pl.DataFrame:
    """Execute a deterministic local-source graph and return its result."""
    with TemporaryDirectory() as directory:
        csv_path = Path(directory) / "prices.csv"
        _write_prices(csv_path)

        prices = Node(
            CSVSource,
            parameters={
                "path": csv_path,
                "output_signature": FrameSignature(
                    columns=(("probability", pl.Float64),),
                ),
                "content_sha256": sha256_file(csv_path),
            },
            name="prices",
        )
        log_odds = Node(
            Logit,
            bindings={"probability": prices.probability},
            parameters={
                "input_column": "probability",
                "output_column": "log_odds",
            },
            name="log_odds",
        )
        change = Node(
            Delta,
            bindings={"log_odds": log_odds.log_odds},
            parameters={
                "input_column": "log_odds",
                "output_column": "change",
            },
            name="change",
        )

        result = Graph(change).execute()

    assert result.columns == ["timestamp", "change"]
    assert result["change"].round(12).to_list() == EXPECTED_CHANGES
    return result


def main() -> None:
    run_example()


if __name__ == "__main__":
    main()
