"""Generate printable ArUco sheets from the action registry.

Run from the repository root:

    python tools/generate_markers.py
    python tools/generate_markers.py --side 800 --output reachy_lego/markers

The marker square stays square. Words are drawn below it, outside the
black border. Resize with --side or by uniform scaling when printing.
Do not crop the black border or the white margin.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reachy_lego.block_library import ARUCO_DICTIONARY
from reachy_lego.marker_sheet import PRINT_INSTRUCTIONS, write_marker_pack


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate one ArUco PNG per LEGO action.")
    parser.add_argument(
        "--side",
        type=int,
        default=600,
        help="Square marker size in pixels (default 600). Scale uniformly for the block face; do not crop.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reachy_lego" / "markers",
        help="Directory for the PNG sheets and manifest.json.",
    )
    args = parser.parse_args(argv)
    manifest = write_marker_pack(args.output, args.side)
    print(f"Dictionary: {ARUCO_DICTIONARY}")
    print(f"Wrote {manifest}")
    print(PRINT_INSTRUCTIONS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
