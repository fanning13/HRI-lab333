"""Registry, detection, ordering, dry-run, and display tests.

Hardware is not required. Reachy pose tests use a fake ``goto_target``.
The Tk window test runs only when this Python has Tk.
"""

from __future__ import annotations

import ast
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from reachy_lego.block_library import (
    ARUCO_DICTIONARY,
    BLOCK_ACTIONS,
    BY_ID,
    BY_KEY,
    BlockAction,
    filename_for,
)
from reachy_lego.detection import MarkerHit, aruco_dictionary, detect_markers, order_hits
from reachy_lego.marker_sheet import quiet_margin, render_marker_sheet, write_marker_pack
from reachy_lego.motion import RecordingMotion
from reachy_lego.runner import AppState, Analysis, perform_sequence, run_frame, scan_allowed
from reachy_lego.sequence import code_listing, natural_language, resolve_hits


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent
EXPECTED = (
    (10, "Look left", "head.turn_left()", "look_left"),
    (11, "Look right", "head.turn_right()", "look_right"),
    (12, "Nod yes", "head.nod()", "nod"),
    (13, "Shake head", "head.shake_no()", "shake_head"),
    (14, "Wiggle antennas", "antennas.wiggle_both()", "wiggle_antennas"),
    (15, "Twist body left", "body.turn_left()", "body_twist_left"),
    (16, "Twist body right", "body.turn_right()", "body_twist_right"),
)


def _hit(marker_id: int, y: float, x: float = 0.0) -> MarkerHit:
    corners = np.array(
        [[x, y], [x + 10, y], [x + 10, y + 10], [x, y + 10]],
        dtype=np.float64,
    )
    return MarkerHit(marker_id, corners)


def _stack(marker_ids: list[int], side: int = 200, gap: int = 90) -> np.ndarray:
    margin = 50
    height = margin + len(marker_ids) * (side + gap)
    canvas = np.full((height, side + 2 * margin), 255, dtype=np.uint8)
    dictionary = aruco_dictionary()
    for index, marker_id in enumerate(marker_ids):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side)
        y = margin + index * (side + gap)
        canvas[y : y + side, margin : margin + side] = marker
    return canvas


class RegistryTests(unittest.TestCase):
    def test_registry_matches_the_seven_actions(self):
        self.assertEqual(ARUCO_DICTIONARY, "DICT_6X6_250")
        found = tuple((item.marker_id, item.label, item.syntax, item.action_key) for item in BLOCK_ACTIONS)
        self.assertEqual(found, EXPECTED)
        self.assertEqual(set(BY_ID), {item[0] for item in EXPECTED})
        self.assertEqual(set(BY_KEY), {item[3] for item in EXPECTED})

    def test_filename_uses_id_and_action_key(self):
        self.assertEqual(filename_for(BY_ID[10]), "10_look_left.png")


class DetectionTests(unittest.TestCase):
    def test_each_registered_marker_generates_and_detects(self):
        for action in BLOCK_ACTIONS:
            with self.subTest(marker_id=action.marker_id):
                sheet = render_marker_sheet(action, side=240)
                hits = detect_markers(sheet)
                self.assertEqual([hit.marker_id for hit in hits], [action.marker_id])

    def test_sheet_keeps_a_square_marker_and_puts_text_outside_it(self):
        action = BY_ID[14]
        side = 240
        margin = quiet_margin(side)
        marker = cv2.aruco.generateImageMarker(aruco_dictionary(), action.marker_id, side)
        sheet = render_marker_sheet(action, side)
        crop = sheet[margin : margin + side, margin : margin + side]
        np.testing.assert_array_equal(crop, marker)
        self.assertEqual(sheet.shape[1], side + 2 * margin)
        self.assertGreater(sheet.shape[0], side + 2 * margin)
        self.assertEqual(crop.shape, (side, side))

    def test_order_is_top_to_bottom_and_keeps_duplicate_ids(self):
        ordered = order_hits([_hit(12, 300), _hit(10, 10), _hit(10, 150, x=4)])
        self.assertEqual([(hit.marker_id, hit.center_y) for hit in ordered], [(10, 15.0), (10, 155.0), (12, 305.0)])

        frame = _stack([10, 10, 12])
        hits = detect_markers(frame)
        self.assertEqual([hit.marker_id for hit in hits], [10, 10, 12])
        self.assertLess(hits[0].center_y, hits[1].center_y)
        self.assertLess(hits[1].center_y, hits[2].center_y)

    def test_manifest_lists_dictionary_label_syntax_and_key(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = write_marker_pack(Path(directory), side=180)
            import json

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["dictionary"], ARUCO_DICTIONARY)
        self.assertEqual(
            [(row["marker_id"], row["label"], row["syntax"], row["action_key"]) for row in manifest["markers"]],
            list(EXPECTED),
        )
        self.assertIn("facing", manifest["print_instructions"].lower())
        self.assertIn("square", manifest["print_instructions"].lower())


class SequenceTests(unittest.TestCase):
    def test_duplicate_ids_stay_in_order(self):
        resolved = resolve_hits([_hit(10, 0), _hit(10, 40), _hit(12, 80)])
        self.assertTrue(resolved.ok)
        self.assertEqual([step.action_key for step in resolved.steps], ["look_left", "look_left", "nod"])
        self.assertEqual(natural_language(resolved.steps), "1. Look left\n2. Look left\n3. Nod yes")
        self.assertEqual(code_listing(resolved.steps), "head.turn_left()\nhead.turn_left()\nhead.nod()")

    def test_listing_comes_from_the_registry_and_is_not_evaluated(self):
        import builtins

        evil = BlockAction(10, "Look left", "__import__('os').system('echo pwned')", "look_left")
        seen = []
        original_eval = builtins.eval
        original_exec = builtins.exec

        def guard_eval(*args, **kwargs):
            seen.append("eval")
            return original_eval(*args, **kwargs)

        def guard_exec(*args, **kwargs):
            seen.append("exec")
            return original_exec(*args, **kwargs)

        builtins.eval = guard_eval
        builtins.exec = guard_exec
        try:
            text = code_listing([evil, BY_ID[12]])
        finally:
            builtins.eval = original_eval
            builtins.exec = original_exec
        self.assertEqual(text, evil.syntax + "\n" + BY_ID[12].syntax)
        self.assertEqual(seen, [])

        roots = list(PACKAGE.glob("*.py")) + [REPO / "tools" / "generate_markers.py"]
        for path in roots:
            if path.name == "test_lego.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                self.assertNotIn(name, {"eval", "exec"}, path.name)

    def test_empty_and_unknown_sequences_do_not_move(self):
        motion = RecordingMotion()
        empty = run_frame(np.full((320, 320), 255, np.uint8), motion)
        self.assertEqual(empty.state, AppState.ERROR)
        self.assertEqual(motion.performed, [])
        self.assertEqual(motion.celebrations, 0)

        unknown_motion = RecordingMotion()
        unknown = run_frame(_stack([10, 42, 12]), unknown_motion)
        self.assertEqual(unknown.state, AppState.ERROR)
        self.assertIn("42", unknown.message)
        self.assertEqual(unknown.performed, ())
        self.assertEqual(unknown_motion.performed, [])
        self.assertEqual(unknown_motion.celebrations, 0)

        def refuse(action_key):
            raise AssertionError(action_key)

        motion.perform = refuse
        motion.celebrate = lambda: refuse("celebrate")
        rejected = perform_sequence(
            Analysis(False, "no", (), (), None),
            motion,
        )
        self.assertEqual(rejected.state, AppState.ERROR)

    def test_valid_sequence_runs_keys_then_celebrates(self):
        motion = RecordingMotion()
        seen = []

        def on_update(update):
            seen.append((update.state, len(motion.performed), motion.celebrations))

        result = run_frame(cv2.cvtColor(_stack([11, 15]), cv2.COLOR_GRAY2BGR), motion, on_update)
        self.assertEqual(result.state, AppState.READY)
        self.assertEqual(result.message, "Dance complete!")
        self.assertEqual(motion.performed, ["look_right", "body_twist_left"])
        self.assertTrue(all("(" not in key for key in motion.performed))
        self.assertEqual(motion.celebrations, 1)
        self.assertEqual(code_listing(result.steps), "head.turn_right()\nbody.turn_left()")
        self.assertEqual(seen[0][0], AppState.SCANNING)
        self.assertLess(
            [item[0] for item in seen].index(AppState.SHOWING_SEQUENCE),
            [item[0] for item in seen].index(AppState.RUNNING),
        )
        showing = next(item for item in seen if item[0] == AppState.SHOWING_SEQUENCE)
        self.assertEqual(showing[1], 0)
        celebrating = next(item for item in seen if item[0] == AppState.CELEBRATING)
        self.assertEqual(celebrating[2], 0)
        self.assertEqual(seen[-1], (AppState.READY, 2, 1))

    def test_movement_error_stops_before_celebration(self):
        steps = (BY_ID[10], BY_ID[12], BY_ID[13])
        motion = RecordingMotion(fail_on=1)
        result = perform_sequence(Analysis(True, "", steps, (), None), motion)
        self.assertEqual(result.state, AppState.ERROR)
        self.assertEqual(motion.performed, ["look_left"])
        self.assertEqual(motion.celebrations, 0)

    def test_scan_button_policy(self):
        self.assertTrue(scan_allowed(AppState.READY, True))
        self.assertTrue(scan_allowed(AppState.ERROR, True))
        self.assertFalse(scan_allowed(AppState.READY, False))
        for state in (AppState.SCANNING, AppState.SHOWING_SEQUENCE, AppState.RUNNING, AppState.CELEBRATING):
            self.assertFalse(scan_allowed(state, True))


class MotionCommandTests(unittest.TestCase):
    def test_action_keys_match_the_registry(self):
        from reachy_lego.reachy_adapter import ACTION_HANDLERS

        self.assertEqual(set(ACTION_HANDLERS), set(BY_KEY))
        self.assertNotIn("celebrate", ACTION_HANDLERS)

    def test_commanded_signs_match_the_course_helpers(self):
        try:
            R, create_head_pose = _load_sdk_pose_helpers()
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))
        pose = create_head_pose(yaw=18, pitch=12, roll=0, degrees=True)
        roll, pitch, yaw = (
            float(value) for value in R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
        )
        self.assertAlmostEqual(yaw, 18, delta=0.2)
        self.assertAlmostEqual(pitch, 12, delta=0.2)
        self.assertAlmostEqual(roll, 0, delta=0.2)
        from reachy_lego.reachy_adapter import (
            ANTENNA_CAP_RAD,
            BODY_TWIST_RAD,
            BODY_YAW_CAP_RAD,
            HEAD_YAW_DEG,
            ReachyMotion,
        )

        class FakeMini:
            def __init__(self):
                self.calls = []

            def goto_target(self, head=None, antennas=None, body_yaw=0.0, duration=0.5):
                yaw = pitch = roll = None
                if head is not None:
                    roll, pitch, yaw = (
                        float(value)
                        for value in R.from_matrix(head[:3, :3]).as_euler("xyz", degrees=True)
                    )
                self.calls.append(
                    {
                        "yaw": yaw,
                        "pitch": pitch,
                        "roll": roll,
                        "antennas": None if antennas is None else [float(value) for value in antennas],
                        "body_yaw": float(body_yaw),
                        "duration": float(duration),
                    }
                )

        def run(method_name: str):
            mini = FakeMini()
            getattr(ReachyMotion(mini), method_name)()
            return mini.calls

        def assert_bounded(calls):
            for call in calls:
                self.assertGreaterEqual(call["duration"], 0.5)
                self.assertLessEqual(abs(call["body_yaw"]), BODY_YAW_CAP_RAD + 1e-9)
                for antenna in call["antennas"]:
                    self.assertLessEqual(abs(antenna), ANTENNA_CAP_RAD + 1e-9)
                self.assertLessEqual(abs(call["yaw"]), HEAD_YAW_DEG + 0.2)
                self.assertLessEqual(abs(call["pitch"]), HEAD_YAW_DEG + 0.2)

        left = run("look_left")
        assert_bounded(left)
        self.assertGreater(left[0]["yaw"], 10)
        self.assertAlmostEqual(left[-1]["yaw"], 0, delta=0.2)
        self.assertEqual(left[0]["body_yaw"], 0)

        right = run("look_right")
        assert_bounded(right)
        self.assertLess(right[0]["yaw"], -10)

        nod = run("nod")
        assert_bounded(nod)
        self.assertGreater(nod[0]["pitch"], 5)
        self.assertAlmostEqual(nod[-1]["pitch"], 0, delta=0.2)

        shake = run("shake_head")
        assert_bounded(shake)
        self.assertGreater(shake[0]["yaw"], 10)
        self.assertLess(shake[1]["yaw"], -10)

        wiggle = run("wiggle_antennas")
        assert_bounded(wiggle)
        self.assertAlmostEqual(wiggle[0]["antennas"][0], 0.40, delta=0.001)
        self.assertAlmostEqual(wiggle[0]["antennas"][1], -0.40, delta=0.001)
        self.assertAlmostEqual(wiggle[-1]["antennas"][0], 0, delta=0.001)

        twist_left = run("body_twist_left")
        assert_bounded(twist_left)
        self.assertAlmostEqual(twist_left[0]["body_yaw"], BODY_TWIST_RAD, delta=0.001)
        self.assertAlmostEqual(twist_left[0]["yaw"], 0, delta=0.2)
        self.assertAlmostEqual(twist_left[-1]["body_yaw"], 0, delta=0.001)

        twist_right = run("body_twist_right")
        assert_bounded(twist_right)
        self.assertAlmostEqual(twist_right[0]["body_yaw"], -BODY_TWIST_RAD, delta=0.001)

        dance = run("celebrate")
        assert_bounded(dance)
        last = dance[-1]
        self.assertAlmostEqual(last["yaw"], 0, delta=0.2)
        self.assertAlmostEqual(last["pitch"], 0, delta=0.2)
        self.assertAlmostEqual(last["body_yaw"], 0, delta=0.001)
        self.assertEqual(last["antennas"], [0.0, 0.0])


class ConnectionTests(unittest.TestCase):
    def test_no_address_is_hard_coded(self):
        for path in PACKAGE.glob("*.py"):
            if path.name.startswith("test_"):
                continue
            self.assertNotIn("192.168", path.read_text(encoding="utf-8"), path.name)

    def test_host_comes_from_the_cli_or_environment(self):
        try:
            from reachy_lego import connection
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))
        self.assertEqual(connection.resolve_robot_host("10.1.2.3"), "10.1.2.3")
        with patch.dict(os.environ, {"REACHY_IP": "10.9.8.7"}):
            self.assertEqual(connection.resolve_robot_host("  "), "10.9.8.7")
        with patch.object(connection, "load_dotenv", lambda *args, **kwargs: False):
            with patch.dict(os.environ, {"REACHY_IP": ""}):
                with self.assertRaises(RuntimeError) as caught:
                    connection.resolve_robot_host("")
        self.assertIn("REACHY_IP", str(caught.exception))

    def test_runtime_path_does_not_save_frames(self):
        runtime = (
            "detection.py",
            "runner.py",
            "gui.py",
            "frames.py",
            "connection.py",
            "reachy_adapter.py",
            "sequence.py",
            "block_library.py",
            "motion.py",
            "__main__.py",
        )
        for name in runtime:
            text = (PACKAGE / name).read_text(encoding="utf-8")
            self.assertNotIn("imwrite", text, name)
            self.assertNotIn("save_frame", text, name)


class GuiTests(unittest.TestCase):
    def test_dry_run_window_shows_the_sequence_and_records_keys(self):
        try:
            import tkinter
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))
        from reachy_lego.frames import StillFrames
        from reachy_lego.gui import LegoApp

        frame = cv2.cvtColor(_stack([10, 12]), cv2.COLOR_GRAY2BGR)
        motion = RecordingMotion(pause_s=0)
        try:
            app = LegoApp(StillFrames(frame), motion, dry_run=True, camera_ready=True)
        except tkinter.TclError as exc:
            self.skipTest(str(exc))

        def finish_if_done():
            done = (
                AppState.SCANNING in app.state_log
                and not app._worker_running
                and app.state in (AppState.READY, AppState.ERROR)
            )
            if done:
                app.snapshot = {
                    "message": app.message_var.get(),
                    "program": app.program.get("1.0", "end"),
                    "recorded": app.recorded_var.get(),
                    "button": str(app.scan_button["state"]),
                }
                app.after(150, app.destroy)
                return
            app.after(40, finish_if_done)

        app.after(100, app._on_scan)
        app.after(40, finish_if_done)
        app.after(10000, app.destroy)
        app.mainloop()

        self.assertEqual(app.state, AppState.READY)
        self.assertEqual(app.snapshot["message"], "Dance complete!")
        self.assertEqual(motion.performed, ["look_left", "nod"])
        self.assertEqual(motion.celebrations, 1)
        self.assertIn("Look left", app.snapshot["program"])
        self.assertIn("head.turn_left()", app.snapshot["program"])
        self.assertIn("head.nod()", app.snapshot["program"])
        self.assertIn("look_left", app.snapshot["recorded"])
        self.assertEqual(app.snapshot["button"], "normal")
        names = [state for state in app.state_log]
        self.assertLess(names.index(AppState.SCANNING), names.index(AppState.SHOWING_SEQUENCE))
        self.assertLess(names.index(AppState.SHOWING_SEQUENCE), names.index(AppState.RUNNING))
        self.assertLess(names.index(AppState.RUNNING), names.index(AppState.CELEBRATING))
        self.assertLess(names.index(AppState.CELEBRATING), len(names) - 1 - names[::-1].index(AppState.READY))


def _load_sdk_pose_helpers():
    """Load ``create_head_pose`` without importing ReachyMini.

    Importing the top-level package pulls in GStreamer. Pose checks only
    need the SDK's Euler helper, which does not.
    """
    import importlib.util
    import sys
    import types

    util_dir = None
    for entry in sys.path:
        candidate = Path(entry) / "reachy_mini" / "utils"
        if (candidate / "rotation.py").is_file():
            util_dir = candidate
            break
    if util_dir is None:
        raise ModuleNotFoundError("reachy_mini.utils")

    package_name = "reachy_mini"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(util_dir.parent)]
        package.__package__ = package_name
        sys.modules[package_name] = package

    module_name = "reachy_mini.utils"
    spec = importlib.util.spec_from_file_location(
        module_name,
        util_dir / "__init__.py",
        submodule_search_locations=[str(util_dir)],
    )
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(module_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.R, module.create_head_pose


if __name__ == "__main__":
    unittest.main()
