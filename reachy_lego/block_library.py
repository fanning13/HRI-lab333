"""Action-block registry for the Reachy LEGO scanner.

This module is the only source of marker IDs, student-facing labels,
display syntax, and action keys. Detection, marker sheets, and the
Reachy adapter all read from here. The syntax strings are shown to
students; nothing in this project evaluates them.
"""

from __future__ import annotations

from dataclasses import dataclass


# IDs 10–16 are entries in this dictionary, not a separate code.
ARUCO_DICTIONARY = "DICT_6X6_250"


@dataclass(frozen=True)
class BlockAction:
    """One printed action block."""

    marker_id: int
    label: str
    syntax: str
    action_key: str


BLOCK_ACTIONS: tuple[BlockAction, ...] = (
    BlockAction(10, "Look left", "head.turn_left()", "look_left"),
    BlockAction(11, "Look right", "head.turn_right()", "look_right"),
    BlockAction(12, "Nod yes", "head.nod()", "nod"),
    BlockAction(13, "Shake head", "head.shake_no()", "shake_head"),
    BlockAction(14, "Wiggle antennas", "antennas.wiggle_both()", "wiggle_antennas"),
    BlockAction(15, "Twist body left", "body.turn_left()", "body_twist_left"),
    BlockAction(16, "Twist body right", "body.turn_right()", "body_twist_right"),
)


def _index(actions: tuple[BlockAction, ...]) -> tuple[dict[int, BlockAction], dict[str, BlockAction]]:
    by_id: dict[int, BlockAction] = {}
    by_key: dict[str, BlockAction] = {}
    for action in actions:
        if action.marker_id in by_id:
            raise ValueError(f"duplicate marker id {action.marker_id}")
        if action.action_key in by_key:
            raise ValueError(f"duplicate action key {action.action_key}")
        by_id[action.marker_id] = action
        by_key[action.action_key] = action
    return by_id, by_key


BY_ID, BY_KEY = _index(BLOCK_ACTIONS)


def filename_for(action: BlockAction) -> str:
    """Printable PNG name, for example ``10_look_left.png``."""
    return f"{action.marker_id}_{action.action_key}.png"
