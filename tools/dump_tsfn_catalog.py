"""Write the built-in TSFN catalog to a JSON file for the iosisweb API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_OUTPUT = _REPO_ROOT / "iosisweb" / "lib" / "catalog" / "tsfns.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the iosisweb TSFN catalog.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="output JSON path (default: iosisweb/lib/catalog/tsfns.json)",
    )
    args = parser.parse_args(argv)

    from iosislib.catalog import dump_tsfn_catalog

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dump_tsfn_catalog(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
