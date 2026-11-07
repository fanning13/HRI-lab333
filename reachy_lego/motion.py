"""Movement interface used by the scanner.

The real Reachy adapter and the dry-run recorder both implement
``perform`` and ``celebrate``. Student syntax is not an input.
"""

from __future__ import annotations

import time
from typing import Protocol


class Motion(Protocol):
    def perform(self, action_key: str) -> None:
        """Run one registry action key."""

    def celebrate(self) -> None:
        """Play the fixed end-of-dance movement. Not a student block."""


class RecordingMotion:
    """Dry-run recorder. Stores action keys and does not move hardware."""

    def __init__(self, pause_s: float = 0.0, fail_on: int | None = None):
        self.pause_s = pause_s
        self.fail_on = fail_on
        self.performed: list[str] = []
        self.celebrations = 0

    def perform(self, action_key: str) -> None:
        if self.fail_on is not None and len(self.performed) == self.fail_on:
            raise RuntimeError("movement failed")
        if self.pause_s:
            time.sleep(self.pause_s)
        self.performed.append(action_key)

    def celebrate(self) -> None:
        if self.pause_s:
            time.sleep(self.pause_s)
        self.celebrations += 1


class DisabledMotion:
    """Used when Reachy never connected. Any call is a connection error."""

    def perform(self, action_key: str) -> None:
        raise RuntimeError("Reachy is not connected")

    def celebrate(self) -> None:
        raise RuntimeError("Reachy is not connected")
