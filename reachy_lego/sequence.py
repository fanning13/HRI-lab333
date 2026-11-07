"""Turn ordered marker hits into a display sequence.

The code listing is assembled from registry strings. It is never passed
to eval, exec, or the robot.
"""

from __future__ import annotations

from dataclasses import dataclass

from reachy_lego.block_library import BY_ID, BlockAction
from reachy_lego.detection import MarkerHit


EMPTY_MESSAGE = (
    "No action markers were found. Stack the blocks so each marker faces "
    "the camera, then scan again."
)


@dataclass(frozen=True)
class ResolvedSequence:
    ok: bool
    steps: tuple[BlockAction, ...]
    message: str


def unknown_message(marker_ids: list[int]) -> str:
    listed = ", ".join(str(marker_id) for marker_id in marker_ids)
    return (
        f"A marker in the stack is not one of the seven action blocks (ID {listed}). "
        "Remove it and scan again. Reachy will stay still."
    )


def resolve_hits(hits: list[MarkerHit]) -> ResolvedSequence:
    """Accept the whole sequence, or reject it before any movement.

    Hits must already be in top-to-bottom order. Duplicate IDs stay.
    """
    if not hits:
        return ResolvedSequence(False, (), EMPTY_MESSAGE)
    unknown = [hit.marker_id for hit in hits if hit.marker_id not in BY_ID]
    if unknown:
        return ResolvedSequence(False, (), unknown_message(unknown))
    steps = tuple(BY_ID[hit.marker_id] for hit in hits)
    return ResolvedSequence(True, steps, "")


def natural_language(steps: tuple[BlockAction, ...] | list[BlockAction]) -> str:
    """Student-facing labels in detected order."""
    return "\n".join(f"{index}. {step.label}" for index, step in enumerate(steps, start=1))


def code_listing(steps: tuple[BlockAction, ...] | list[BlockAction]) -> str:
    """Registry syntax for display. Do not evaluate the result."""
    return "\n".join(step.syntax for step in steps)
