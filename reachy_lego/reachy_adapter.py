"""Map registry action keys to short, fixed Reachy Mini moves.

Direction signs are taken from the course helpers already used with this
SDK, not from a guess:

- ``move_head`` documents yaw+ as turn left and pitch+ as look down
  (lab3 ``common.py``).
- The lab 1 greeting uses that same positive-yaw-is-left sign for body yaw.
- ``goto_target`` antennas are ``[right_angle, left_angle]`` in radians.

Body yaw is commanded on its own. Automatic body yaw is turned off at
connect time so a head look does not drag the body, matching the lab 2
choreography note. Angles stay inside the lab 2 conservative caps
(head 18°, antennas 0.50 rad, body yaw 0.20 rad), which are well inside
the SDK limits of pitch/roll ±40°, relative yaw ±65°, and body yaw ±160°.

Each goto uses a duration of at least 0.5 s and blocks until the SDK
reports the move finished. The celebration is a fixed pose sequence,
not a student action key.
"""

from __future__ import annotations

# Commanded poses. Caps match the lab 2 conservative limits.
HEAD_YAW_DEG = 18.0
NOD_PITCH_DEG = 12.0
ANTENNA_RAD = 0.40
BODY_TWIST_RAD = 0.18
MOVE_SECONDS = 0.5

HEAD_YAW_CAP_DEG = 18.0
HEAD_PITCH_CAP_DEG = 18.0
HEAD_ROLL_CAP_DEG = 10.0
ANTENNA_CAP_RAD = 0.50
BODY_YAW_CAP_RAD = 0.20


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _head_pose(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0):
    from reachy_mini.utils import create_head_pose

    return create_head_pose(
        yaw=_clamp(yaw, HEAD_YAW_CAP_DEG),
        pitch=_clamp(pitch, HEAD_PITCH_CAP_DEG),
        roll=_clamp(roll, HEAD_ROLL_CAP_DEG),
        degrees=True,
    )


class ReachyMotion:
    """Registry action keys to predefined ``goto_target`` sequences."""

    def __init__(self, mini):
        self._mini = mini

    def perform(self, action_key: str) -> None:
        try:
            handler = ACTION_HANDLERS[action_key]
        except KeyError as exc:
            raise KeyError(f"unknown action key {action_key}") from exc
        handler(self)

    def _goto(
        self,
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
        antennas: tuple[float, float] = (0.0, 0.0),
        body_yaw: float = 0.0,
        duration: float = MOVE_SECONDS,
    ) -> None:
        # goto_target waits until the move finishes (duration + 1 s timeout).
        self._mini.goto_target(
            head=_head_pose(yaw, pitch, roll),
            antennas=[
                _clamp(antennas[0], ANTENNA_CAP_RAD),
                _clamp(antennas[1], ANTENNA_CAP_RAD),
            ],
            body_yaw=_clamp(body_yaw, BODY_YAW_CAP_RAD),
            duration=max(0.5, float(duration)),
        )

    def look_left(self) -> None:
        self._goto(yaw=HEAD_YAW_DEG)
        self._goto(yaw=0.0)

    def look_right(self) -> None:
        self._goto(yaw=-HEAD_YAW_DEG)
        self._goto(yaw=0.0)

    def nod(self) -> None:
        self._goto(pitch=NOD_PITCH_DEG)
        self._goto(pitch=0.0)
        self._goto(pitch=NOD_PITCH_DEG)
        self._goto(pitch=0.0)

    def shake_head(self) -> None:
        self._goto(yaw=HEAD_YAW_DEG)
        self._goto(yaw=-HEAD_YAW_DEG)
        self._goto(yaw=HEAD_YAW_DEG)
        self._goto(yaw=0.0)

    def wiggle_antennas(self) -> None:
        # SDK order is [right, left].
        self._goto(antennas=(ANTENNA_RAD, -ANTENNA_RAD))
        self._goto(antennas=(-ANTENNA_RAD, ANTENNA_RAD))
        self._goto(antennas=(0.0, 0.0))

    def body_twist_left(self) -> None:
        self._goto(body_yaw=BODY_TWIST_RAD)
        self._goto(body_yaw=0.0)

    def body_twist_right(self) -> None:
        self._goto(body_yaw=-BODY_TWIST_RAD)
        self._goto(body_yaw=0.0)

    def celebrate(self) -> None:
        """Short fixed ending. Not dispatched through an action key."""
        self._goto(antennas=(ANTENNA_RAD, -ANTENNA_RAD))
        self._goto(antennas=(-ANTENNA_RAD, ANTENNA_RAD), pitch=NOD_PITCH_DEG)
        self._goto(antennas=(0.0, 0.0), pitch=0.0, yaw=0.0, body_yaw=0.0)


ACTION_HANDLERS = {
    "look_left": ReachyMotion.look_left,
    "look_right": ReachyMotion.look_right,
    "nod": ReachyMotion.nod,
    "shake_head": ReachyMotion.shake_head,
    "wiggle_antennas": ReachyMotion.wiggle_antennas,
    "body_twist_left": ReachyMotion.body_twist_left,
    "body_twist_right": ReachyMotion.body_twist_right,
}
