from datetime import datetime, timedelta
from io import BytesIO

import matplotlib
import matplotlib.image as mpimg
import polars as pl
import pytest

from iosislib.charting import to_png, write_png


matplotlib.use("Agg")


def _frame() -> pl.DataFrame:
    start = datetime(2026, 1, 1)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=i) for i in range(4)],
            "value": [1.0, 2.0, 1.5, 3.0],
        }
    )


def test_to_png_returns_high_resolution_png_bytes() -> None:
    image = mpimg.imread(BytesIO(to_png(_frame(), dpi=240)), format="png")

    assert image.shape[1] >= 2000
    assert image.shape[0] >= 1200


def test_to_png_accepts_graph_like_sources() -> None:
    class FakeGraph:
        def execute(self, *, executor: object = None) -> pl.DataFrame:
            del executor
            return _frame()

    assert to_png(FakeGraph()).startswith(b"\x89PNG\r\n\x1a\n")


def test_write_png_writes_the_same_public_format(tmp_path: object) -> None:
    path = tmp_path / "chart.png"
    write_png(_frame(), path)

    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_to_png_rejects_invalid_dpi() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        to_png(_frame(), dpi=0)
