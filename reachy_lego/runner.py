"""Scan policy: validate the full sequence, then move, then celebrate."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from reachy_lego.block_library import BlockAction
from reachy_lego.detection import MarkerHit, detect_markers, highlight_markers
from reachy_lego.sequence import resolve_hits


class AppState(enum.Enum):
    READY = "READY"
    SCANNING = "SCANNING"
    SHOWING_SEQUENCE = "SHOWING_SEQUENCE"
    RUNNING = "RUNNING"
    CELEBRATING = "CELEBRATING"
    ERROR = "ERROR"


BUSY_STATES = {
    AppState.SCANNING,
    AppState.SHOWING_SEQUENCE,
    AppState.RUNNING,
    AppState.CELEBRATING,
}


def scan_allowed(state: AppState, camera_ready: bool) -> bool:
    """Scan is available on Ready and Error, and only with a frame source."""
    return camera_ready and state not in BUSY_STATES


@dataclass
class Analysis:
    ok: bool
    message: str
    steps: tuple[BlockAction, ...]
    hits: tuple[MarkerHit, ...]
    annotated: object


@dataclass
class RunUpdate:
    state: AppState
    message: str = ""
    steps: tuple[BlockAction, ...] = ()
    current_index: int | None = None
    annotated: object = None
    performed: tuple[str, ...] = ()
    celebrated: bool = False


def analyze_frame(frame) -> Analysis:
    """Detect and resolve one frame. Does not call a movement handler."""
    if frame is None:
        return Analysis(
            False,
            "The camera did not return a frame. Scan again when a picture is visible.",
            (),
            (),
            None,
        )
    hits = detect_markers(frame)
    annotated = highlight_markers(frame, hits)
    resolved = resolve_hits(hits)
    return Analysis(resolved.ok, resolved.message, resolved.steps, tuple(hits), annotated)


def _emit(on_update, update: RunUpdate) -> RunUpdate:
    if on_update is not None:
        on_update(update)
    return update


def perform_sequence(analysis: Analysis, motion, on_update=None) -> RunUpdate:
    """Play a sequence that has already been accepted.

    An rejected analysis returns before ``perform`` or ``celebrate``.
    A movement error stops the rest of the sequence and skips the celebration.
    """
    if not analysis.ok:
        return _emit(
            on_update,
            RunUpdate(AppState.ERROR, analysis.message, annotated=analysis.annotated),
        )

    steps = analysis.steps
    _emit(
        on_update,
        RunUpdate(
            AppState.SHOWING_SEQUENCE,
            "",
            steps,
            None,
            analysis.annotated,
            (),
        ),
    )
    performed: list[str] = []
    for index, step in enumerate(steps):
        _emit(
            on_update,
            RunUpdate(
                AppState.RUNNING,
                "",
                steps,
                index,
                analysis.annotated,
                tuple(performed),
            ),
        )
        try:
            motion.perform(step.action_key)
        except Exception as exc:
            return _emit(
                on_update,
                RunUpdate(
                    AppState.ERROR,
                    f"Reachy stopped before the sequence finished: {exc}",
                    steps,
                    index,
                    analysis.annotated,
                    tuple(performed),
                    False,
                ),
            )
        performed.append(step.action_key)

    _emit(
        on_update,
        RunUpdate(
            AppState.CELEBRATING,
            "Dance complete!",
            steps,
            None,
            analysis.annotated,
            tuple(performed),
            False,
        ),
    )
    try:
        motion.celebrate()
    except Exception as exc:
        return _emit(
            on_update,
            RunUpdate(
                AppState.ERROR,
                f"The celebration did not finish: {exc}",
                steps,
                None,
                analysis.annotated,
                tuple(performed),
                False,
            ),
        )
    return _emit(
        on_update,
        RunUpdate(
            AppState.READY,
            "Dance complete!",
            steps,
            None,
            analysis.annotated,
            tuple(performed),
            True,
        ),
    )


def run_frame(frame, motion, on_update=None) -> RunUpdate:
    """Analyze one frame and move only when every marker is a known action."""
    _emit(on_update, RunUpdate(AppState.SCANNING, "Scanning the stack."))
    try:
        analysis = analyze_frame(frame)
    except Exception as exc:
        return _emit(
            on_update,
            RunUpdate(AppState.ERROR, f"Could not read the markers: {exc}"),
        )
    if not analysis.ok:
        return _emit(
            on_update,
            RunUpdate(AppState.ERROR, analysis.message, annotated=analysis.annotated),
        )
    return perform_sequence(analysis, motion, on_update)
