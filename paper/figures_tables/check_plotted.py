"""Compares the plotted values of every figure and table with those of the paper, byte by byte.

``expected/<stem>.csv`` are the values of the paper's floats; ``outputs/figures/plotted/<stem>.csv`` are written by the
scripts of this folder. A float whose output file is missing (Fig. 7 without predictions) is reported as missing.
Exit status 1 if any file differs.
"""

import sys
from pathlib import Path

from common import PLOTTED_DIR

EXPECTED = Path(__file__).resolve().parent / "expected"


def main() -> None:
    failed = False
    for expected in sorted(EXPECTED.glob("*.csv")):
        produced = PLOTTED_DIR / expected.name
        if not produced.exists():
            status = "missing"
        elif produced.read_bytes() == expected.read_bytes():
            status = "identical"
        else:
            status, failed = "DIFFERS", True
        print(f"{expected.stem:32s} {status}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
