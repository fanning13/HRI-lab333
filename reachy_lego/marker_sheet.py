"""Printable marker sheets. Labels sit outside the black marker square."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from reachy_lego.block_library import ARUCO_DICTIONARY, BLOCK_ACTIONS, BlockAction, filename_for
from reachy_lego.detection import aruco_dictionary


PRINT_INSTRUCTIONS = (
    "Print each sheet at the same scale. The marker itself must stay square: "
    "scale uniformly, and do not crop the black border or the white margin. "
    "To change the printed size, rerun this tool with --side PIXELS or scale "
    "uniformly in the print dialog. Attach the printed marker flat on the "
    "camera-facing side of the block. The camera must see every marker in "
    "the stack; a hidden marker cannot be decoded."
)


def quiet_margin(side: int) -> int:
    """White paper around the square marker, in pixels."""
    return max(48, side // 6)


def render_marker_sheet(action: BlockAction, side: int = 600) -> np.ndarray:
    """Return one grayscale sheet: square marker, then label and syntax.

    ``side`` is the marker square passed to ``generateImageMarker``. The
    sheet is taller than it is wide because the words sit below the marker.
    """
    if side < 80 or side > 2000:
        raise ValueError("marker side must be between 80 and 2000 pixels")
    marker = cv2.aruco.generateImageMarker(aruco_dictionary(), action.marker_id, side)
    if marker.ndim != 2 or marker.shape != (side, side):
        raise RuntimeError("ArUco generator returned a non-square marker")

    margin = quiet_margin(side)
    scale = side / 600.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = (
        (action.label, 1.05, 2),
        (action.syntax, 0.72, 2),
        ("Attach flat, facing the camera.", 0.55, 1),
        ("The camera must see every marker in the stack.", 0.55, 1),
        (f"{ARUCO_DICTIONARY}   id {action.marker_id}", 0.48, 1),
    )
    measured = []
    for text, base_scale, thickness in lines:
        font_scale = base_scale * scale
        stroke = max(1, int(round(thickness * scale)))
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, stroke)
        measured.append((text, font_scale, stroke, text_width, text_height, baseline))

    gap = int(16 * scale) + 8
    text_band = sum(text_height + baseline + gap for _, _, _, _, text_height, baseline in measured)
    width = side + 2 * margin
    height = margin + side + margin + text_band + margin // 2
    sheet = np.full((height, width), 255, dtype=np.uint8)
    sheet[margin : margin + side, margin : margin + side] = marker

    y = margin + side + margin
    for text, font_scale, stroke, text_width, text_height, baseline in measured:
        x = max(8, (width - text_width) // 2)
        y += text_height
        cv2.putText(sheet, text, (x, y), font, font_scale, 0, stroke, cv2.LINE_AA)
        y += baseline + gap
    return sheet


def write_marker_pack(directory: Path, side: int = 600) -> Path:
    """Write one PNG per action plus ``manifest.json``. Returns the manifest path."""
    directory.mkdir(parents=True, exist_ok=True)
    margin = quiet_margin(side)
    records = []
    for action in BLOCK_ACTIONS:
        name = filename_for(action)
        sheet = render_marker_sheet(action, side)
        path = directory / name
        if not cv2.imwrite(str(path), sheet):
            raise RuntimeError(f"Could not write {path}")
        records.append(
            {
                "marker_id": action.marker_id,
                "label": action.label,
                "syntax": action.syntax,
                "action_key": action.action_key,
                "file": name,
            }
        )
    manifest = {
        "dictionary": ARUCO_DICTIONARY,
        "side_px": side,
        "margin_px": margin,
        "print_instructions": PRINT_INSTRUCTIONS,
        "markers": records,
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
