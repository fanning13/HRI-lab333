"""Connect to Reachy the same way the guidance study does.

The host comes from ``--robot-ip``, ``REACHY_IP`` in the environment,
or ``guidance_study/.env``. No address is written into this package.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from reachy_lego.reachy_adapter import MOVE_SECONDS, _head_pose


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_robot_host(cli_host: str = "") -> str:
    """Return the daemon host from the CLI or the study's REACHY_IP setting."""
    if cli_host.strip():
        return cli_host.strip()
    load_dotenv(REPO_ROOT / "reachy_lego" / ".env")
    load_dotenv(REPO_ROOT / "guidance_study" / ".env")
    host = os.environ.get("REACHY_IP", "").strip()
    if not host:
        raise RuntimeError(
            "REACHY_IP is not set. Add it to guidance_study/.env, export it, "
            "or pass --robot-ip. This app uses that existing setting and does "
            "not hard-code an address."
        )
    return host


def connect_reachy(host: str):
    """Open the network media connection and hold a neutral pose.

    ``automatic_body_yaw`` is disabled so body twists use the commanded
    yaw instead of following the head. The SDK default is on.
    """
    from reachy_mini import ReachyMini

    mini = ReachyMini(
        host=host,
        connection_mode="network",
        media_backend="default",
        timeout=10,
        automatic_body_yaw=False,
    )
    mini.enable_motors()
    mini.set_automatic_body_yaw(False)
    mini.goto_target(
        head=_head_pose(0, 0, 0),
        antennas=[0.0, 0.0],
        body_yaw=0.0,
        duration=MOVE_SECONDS,
    )
    return mini


def disconnect_reachy(mini) -> None:
    """Return toward neutral, then close the media connection."""
    if mini is None:
        return
    try:
        mini.goto_target(
            head=_head_pose(0, 0, 0),
            antennas=[0.0, 0.0],
            body_yaw=0.0,
            duration=MOVE_SECONDS,
        )
    except Exception:
        pass
    closer = getattr(mini, "__exit__", None)
    if closer is not None:
        try:
            closer(None, None, None)
        except Exception:
            pass
