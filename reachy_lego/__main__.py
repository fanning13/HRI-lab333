"""Reachy LEGO scanner.

    python -m reachy_lego --check
    python -m reachy_lego --dry-run --image reachy_lego/markers/10_look_left.png
    python -m reachy_lego
"""

from __future__ import annotations

import argparse
import traceback

from reachy_lego.block_library import ARUCO_DICTIONARY, BLOCK_ACTIONS
from reachy_lego.detection import opencv_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan a stack of ArUco action blocks and play them on Reachy Mini.",
    )
    parser.add_argument("--check", action="store_true", help="Print the action registry and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Show the decoded steps without moving Reachy.")
    parser.add_argument("--image", default="", help="Scan this still image instead of a live camera.")
    parser.add_argument("--webcam", action="store_true", help="Use this computer's webcam instead of Reachy's camera.")
    parser.add_argument(
        "--robot-ip",
        default="",
        help="Reachy daemon host. Defaults to REACHY_IP from the environment or guidance_study/.env.",
    )
    return parser


def print_registry() -> None:
    print(f"Dictionary: {ARUCO_DICTIONARY}")
    print(opencv_status())
    for action in BLOCK_ACTIONS:
        print(f"{action.marker_id:3}  {action.label:<18}  {action.syntax:<26}  {action.action_key}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        print_registry()
        return 0
    if args.image and args.webcam:
        raise SystemExit("Pass either --image or --webcam, not both.")
    from reachy_lego.runtime import expose_libpython, install_force_stop

    install_force_stop()
    if not args.dry_run:
        expose_libpython()

    from reachy_lego.connection import connect_reachy, disconnect_reachy, resolve_robot_host
    from reachy_lego.frames import StillFrames, UnavailableFrames, WebcamFrames, ReachyFrames, load_image
    from reachy_lego.motion import DisabledMotion, RecordingMotion
    from reachy_lego.reachy_adapter import ReachyMotion

    try:
        import reachy_lego.gui as gui
    except ModuleNotFoundError as exc:
        if exc.name == "_tkinter":
            raise SystemExit(
                "This window uses Tkinter, but this Python was built without Tk. "
                "The project .venv (pyenv 3.11.9) has no _tkinter module. "
                "Use a Python that includes Tk, then run python -m reachy_lego. "
                "Registry and dry-run logic can still be tested with "
                "python -m unittest reachy_lego.test_lego."
            ) from exc
        raise

    source = None
    mini = None
    startup_message = ""
    camera_ready = False
    try:
        if args.image:
            source = StillFrames(load_image(args.image))
            camera_ready = True
        elif args.webcam:
            source = WebcamFrames(0)
            source.start()
            if source.error():
                startup_message = source.error() or "Could not open the webcam."
                camera_ready = False
            else:
                camera_ready = True

        if args.dry_run:
            motion = RecordingMotion(pause_s=0.4)
            if source is None:
                startup_message = (
                    "Dry run is ready, and Reachy will not move. "
                    "Pass --image or --webcam to scan a stack."
                )
                source = UnavailableFrames(startup_message)
                camera_ready = False
        else:
            host = resolve_robot_host(args.robot_ip)
            print(f"Connecting to Reachy at {host}.")
            mini = connect_reachy(host)
            motion = ReachyMotion(mini)
            if source is None:
                source = ReachyFrames(mini)
                source.start()
                camera_ready = True
    except Exception as exc:
        traceback.print_exc()
        startup_message = f"Could not connect to Reachy: {exc}"
        if source is not None:
            source.stop()
        if mini is not None:
            disconnect_reachy(mini)
        source = UnavailableFrames(startup_message)
        motion = DisabledMotion()
        camera_ready = False
        mini = None

    try:
        gui.launch_gui(source, motion, args.dry_run, camera_ready, startup_message)
    finally:
        if source is not None:
            source.stop()
        disconnect_reachy(mini)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
