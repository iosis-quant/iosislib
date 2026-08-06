from datetime import datetime, timedelta
from io import BytesIO

import matplotlib
import matplotlib.image as mpimg
import polars as pl
import pytest

from iosislib.charting import figure_to_png_data_uri, to_png


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


def test_to_png_default_dark_background() -> None:
    image = mpimg.imread(BytesIO(to_png(_frame(), dpi=100)), format="png")

    corner = image[0, 0, :3]
    assert corner.max() < 0.2
    assert image.mean(axis=(0, 1))[:3].max() < 0.25


def test_to_png_accepts_graph_like_sources() -> None:
    class FakeGraph:
        def execute(self, *, executor: object = None) -> pl.DataFrame:
            del executor
            return _frame()

    assert to_png(FakeGraph()).startswith(b"\x89PNG\r\n\x1a\n")


def test_figure_to_png_data_uri_is_base64_encoded() -> None:
    figure_bytes = to_png(_frame(), dpi=100)
    import matplotlib.pyplot as plt

    figure = plt.figure()
    figure_to_png_data_uri(figure)
    uri = figure_to_png_data_uri(figure)
    assert uri.startswith("data:image/png;base64,")
    assert uri.split(",", 1)[1]
    assert figure_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    plt.close(figure)


def test_to_png_rejects_invalid_dpi() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        to_png(_frame(), dpi=0)
