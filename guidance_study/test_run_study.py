"""Mocked tests. They do not open a robot, camera, model API, or participant log."""

from __future__ import annotations

import argparse
import csv
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import models
import robot_io
import run_study
from models import ActionResult, CueResult


ROOT = Path(__file__).resolve().parent


def args(**overrides):
    values = dict(
        participant="P01",
        order=1,
        simulate=True,
        simulate_cue="observed",
        simulate_action="no_nudge",
        model=models.DEFAULT_REASONING_MODEL,
        vlm_model=models.DEFAULT_VLM_MODEL,
        tts_model=models.DEFAULT_TTS_MODEL,
        reasoning_effort=models.DEFAULT_REASONING_EFFORT,
        output_dir="logs",
        robot_ip="",
        check=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class Script:
    def __init__(self, replies):
        self.replies = list(replies)

    def __call__(self, prompt):
        if not self.replies:
            raise AssertionError(f"unexpected prompt: {prompt}")
        return self.replies.pop(0)


class Materials(unittest.TestCase):
    def test_file_and_protocol_agree(self):
        materials = run_study.load_materials()
        run_study.validate_materials(materials)
        protocol = (ROOT / "protocol.md").read_text(encoding="utf-8")
        counterbalancing = (ROOT / "counterbalancing.md").read_text(encoding="utf-8")
        self.assertIn(materials["comparison_note"], protocol)
        self.assertIn(materials["scoring_rule"], protocol)
        for prompt in materials["prompts"].values():
            self.assertIn(prompt, protocol)
        for case in materials["cases"]:
            self.assertIn(case["title"], protocol)
            for turn in case["turns"]:
                if turn["narration"]:
                    self.assertIn(turn["narration"], protocol)
                checkpoint = turn["checkpoint"]
                if not checkpoint:
                    continue
                for field in ("question", "hint", "explanation_criterion"):
                    self.assertIn(checkpoint[field], protocol)
                self.assertIn(checkpoint["options"]["A"], protocol)
                self.assertIn(checkpoint["options"]["B"], protocol)
                self.assertNotIn(checkpoint["hint"], run_study.story_context(case, turn["turn_id"]))
        self.assertIn("Briarwick Watch / fixed-point, then Ashcombe Star / camera-informed", counterbalancing)
        self.assertNotIn("You look confused", "".join(materials["prompts"].values()))
        self.assertNotIn("You seem certain", "".join(materials["prompts"].values()))

    def test_orders_and_answers(self):
        materials = run_study.load_materials()
        expected = {
            "A-C1": "A",
            "A-C2": "A",
            "A-C3": "B",
            "B-C1": "A",
            "B-C2": "A",
            "B-C3": "B",
        }
        found = {}
        for case in materials["cases"]:
            ids = [turn["turn_id"] for turn in case["turns"]]
            self.assertEqual(ids, [f"{case['case_id']}-T{number}" for number in range(1, 7)])
            for turn in case["turns"]:
                checkpoint = turn["checkpoint"]
                if checkpoint:
                    found[checkpoint["checkpoint_id"]] = checkpoint["correct_option"]
                    self.assertIn(turn["turn_id"], {"A-T2", "A-T4", "A-T6", "B-T2", "B-T4", "B-T6"})
        self.assertEqual(found, expected)
        self.assertEqual(run_study.next_participant(Path("missing-logs")), "P01")
        self.assertEqual(run_study.default_order("P01"), 1)
        self.assertEqual(run_study.default_order("P04"), 4)
        self.assertEqual(run_study.default_order("P05"), 1)

    def test_model_requests(self):
        kwargs = models.chat_kwargs("gpt-6-luna", "none", [], models.ACTION_FORMAT, 40)
        local = models.chat_kwargs("llama3.2", "", [], models.ACTION_FORMAT, 40, structured=False)
        self.assertEqual(set(local), {"model", "messages"})
        self.assertEqual(kwargs["reasoning_effort"], "none")
        self.assertEqual(kwargs["temperature"], 0)
        self.assertEqual(kwargs["response_format"]["json_schema"]["schema"]["properties"]["action"]["enum"], list(models.ACTIONS))
        image = models.vlm_messages(
            "story",
            [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA"}}],
            "B",
            "I think Bea had the key",
        )
        self.assertIn("I think Bea had the key", image[1]["content"][0]["text"])
        self.assertEqual(image[1]["content"][1]["type"], "image_url")
        text = models.reasoning_messages("story", "A", "because", "hands on the table")
        self.assertNotIn("image_url", text[1]["content"])
        self.assertNotIn("Separate the last confirmed sighting", text[1]["content"])
        self.assertIn("appears confused", models.VLM_INSTRUCTIONS)
        self.assertIn("Do not write a sentence for Reachy to say aloud.", models.VLM_INSTRUCTIONS)
        self.assertNotIn("Describe the emotion", models.VLM_INSTRUCTIONS)
        self.assertIn("typed answer expresses uncertainty", models.REASONING_INSTRUCTIONS)
        previous = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-test-secret"
        try:
            hidden = models.redact("failed sk-test-secret data:image/jpeg;base64,AAAA")
        finally:
            if previous is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = previous
        self.assertNotIn("sk-test-secret", hidden)
        self.assertNotIn("base64", hidden)
        self.assertIn("[image omitted]", hidden)

    def test_terminal_layout_wraps_without_changing_speech(self):
        materials = run_study.load_materials()
        checkpoint = materials["cases"][0]["turns"][1]["checkpoint"]
        spoken = run_study.question_line(checkpoint)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            robot_io.show_reachy(spoken)
        shown = buffer.getvalue()
        self.assertIn("\n  A. ", shown)
        self.assertIn("\n  B. ", shown)
        self.assertTrue(all(len(line) <= robot_io.LINE_WIDTH for line in shown.splitlines()))
        self.assertIn(checkpoint["question"], spoken)
        self.assertNotIn("\n", spoken)
        source = (ROOT / "robot_io.py").read_text(encoding="utf-8")
        self.assertNotIn("last_frame", source)
        self.assertIsNone(robot_io.save_frame({"not": "an image"}, Path("frames"), "skip"))


class Guidance(unittest.TestCase):
    def test_next_participant_follows_hardware_logs_only(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name, mode in (
                ("P01_order1_sim.csv", "simulate"),
                ("P01_order1_real.csv", "hardware"),
                ("P03_order3_real.csv", "hardware"),
            ):
                with (folder / name).open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["session_mode"])
                    writer.writeheader()
                    writer.writerow({"session_mode": mode})
            self.assertEqual(run_study.next_participant(folder), "P04")

    def test_explicit_help_and_scoring(self):
        self.assertTrue(run_study.explicit_help_requested("I need a hint"))
        self.assertFalse(run_study.explicit_help_requested("Ada returned the key"))
        self.assertEqual(run_study.change_type("B", "A", "A"), "incorrect-to-correct")
        self.assertEqual(run_study.change_type("A", "B", "A"), "correct-to-incorrect")
        self.assertEqual(run_study.change_type("A", "A", "A"), "unchanged")
        checkpoints = {
            turn["checkpoint"]["checkpoint_id"]: turn["checkpoint"]
            for case in run_study.load_materials()["cases"]
            for turn in case["turns"]
            if turn["checkpoint"]
        }
        self.assertEqual(run_study.score_explanation(checkpoints["A-C1"], "case 3 at 8:00 p.m."), "1")
        self.assertEqual(run_study.score_explanation(checkpoints["A-C1"], "I guessed"), "0")
        self.assertEqual(run_study.score_explanation(checkpoints["A-C3"], "recorded access is not proof"), "1")
        self.assertEqual(run_study.score_explanation(checkpoints["B-C1"], "empty cabinet"), "0")
        self.assertEqual(run_study.score_explanation(checkpoints["B-C2"], "Nia from 6:22 to 6:27"), "1")
        self.assertEqual(run_study.score_explanation(checkpoints["B-C3"], "ticking does not identify who"), "1")

    def test_fixed_point_never_captures(self):
        def capture():
            raise AssertionError("camera requested")

        plan = run_study.plan_guidance("fixed", "A", "the ledger", "story", capture, None, None)
        self.assertFalse(plan["camera_accessed"])
        self.assertTrue(plan["offer"])
        self.assertFalse(plan["deliver_hint"])

        explicit = run_study.plan_guidance("fixed", "A", "help", "story", capture, None, None)
        self.assertTrue(explicit["deliver_hint"])
        self.assertFalse(explicit["offer"])

    def test_camera_branches(self):
        calls = {"capture": 0, "describe": 0, "decide": 0}

        def capture():
            calls["capture"] += 1
            return object()

        def describe(frame, story, choice="", explanation=""):
            del frame, story, choice, explanation
            calls["describe"] += 1
            return CueResult("unclear", "", "", 2)

        def decide(*_args):
            calls["decide"] += 1
            return ActionResult("offer_hint", "", 2)

        unclear = run_study.plan_guidance("camera", "A", "because", "story", capture, describe, decide)
        self.assertEqual(unclear["nudge"], "fallback_neutral")
        self.assertTrue(unclear["speak_neutral"])
        self.assertFalse(unclear["offer"])
        self.assertEqual(calls["decide"], 0)

        def missing():
            return None

        failed = run_study.plan_guidance("camera", "A", "because", "story", missing, describe, decide)
        self.assertEqual(failed["vlm_error"], "NO_FRAME")
        self.assertEqual(failed["nudge"], "fallback_neutral")
        self.assertEqual(calls["describe"], 1)

        def observed(frame, story, choice="", explanation=""):
            del frame, story, choice, explanation
            return CueResult("observed", "gaze toward the table", "", 3)

        def broken(*_args):
            return ActionResult("", "timeout", 4)

        fallback = run_study.plan_guidance("camera", "A", "because", "story", capture, observed, broken)
        self.assertEqual(fallback["nudge"], "fallback_neutral")
        self.assertFalse(fallback["offer"])

        offered = run_study.plan_guidance("camera", "A", "because", "story", capture, observed, decide)
        self.assertEqual(offered["nudge"], "offer_hint")
        self.assertTrue(offered["offer"])

        quiet = run_study.plan_guidance(
            "camera", "A", "because", "story", capture, observed,
            lambda *_args: ActionResult("no_nudge", "", 1),
        )
        self.assertEqual(quiet["nudge"], "no_nudge")
        self.assertFalse(quiet["speak_neutral"])

        asked = run_study.plan_guidance("camera", "B", "please help", "story", capture, observed, decide)
        self.assertTrue(asked["explicit_help"])
        self.assertTrue(asked["deliver_hint"])
        self.assertFalse(asked["offer"])
        self.assertTrue(asked["camera_accessed"])

        typed = run_study.plan_guidance("camera", "B", "i don't know", "story", capture, describe, decide)
        self.assertEqual(typed["nudge"], "typed_confusion")
        self.assertTrue(typed["offer"])

        def looks_confused(frame, story, choice="", explanation=""):
            del frame, story, choice, explanation
            return CueResult("observed", "furrowed brow", "", 3, "yes")

        visual = run_study.plan_guidance(
            "camera", "A", "Nia had the key", "story", capture, looks_confused,
            lambda *_args: ActionResult("no_nudge", "", 1),
        )
        self.assertEqual(visual["nudge"], "visual_confusion")
        self.assertTrue(visual["offer"])


class Session(unittest.TestCase):
    def test_simulated_order_one(self):
        spoken = []
        calls = {"capture": 0, "decide": 0}

        def speak(text):
            spoken.append(text)
            return True

        def capture():
            calls["capture"] += 1
            return {"simulated_frame": True}

        def describe(frame, story, choice="", explanation=""):
            del frame, story, choice, explanation
            if calls["capture"] == 1:
                return CueResult("unclear", "", "", 5)
            return CueResult("observed", "hands resting on the table", "", 6)

        def decide(story, choice, explanation, cue):
            del story, choice, explanation, cue
            calls["decide"] += 1
            return ActionResult("offer_hint", "", 7)

        script = Script([
            "yes",
            "A", "case 3 at 8:00 p.m.", "no", "keep", "case 3 at 8:00 p.m.", "2",
            "B", "Bea had the key", "yes", "change", "A", "Ada signed the ledger from 8:02 to 8:07", "3",
            "B", "I need a hint", "keep", "recorded access is not proof", "4",
            "4", "5", "3",
            "B", "it was empty at 6:35", "keep", "empty cabinet", "1",
            "A", "Nia checked out the key", "yes", "keep", "Nia from 6:22 to 6:27", "2",
            "A", "please help", "keep", "ticking does not identify who", "5",
            "2", "2", "2",
        ])
        screen = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(screen):
            path = run_study.run_session(
                args(output_dir=directory),
                input_fn=script,
                speak_fn=speak,
                capture=capture,
                describe=describe,
                decide=decide,
            )
            self.assertEqual(script.replies, [])
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            text = path.read_text(encoding="utf-8")

        self.assertNotIn("sk-secret", text)
        self.assertNotIn("simulated_frame", text)
        self.assertNotIn(".jpg", text)
        self.assertTrue(all(row["frame_saved"] == "False" for row in rows))
        self.assertEqual(calls["capture"], 3)
        self.assertEqual(calls["decide"], 1)

        checkpoints = [row for row in rows if row["row_type"] == "checkpoint"]
        self.assertEqual([row["checkpoint_id"] for row in checkpoints], ["A-C1", "A-C2", "A-C3", "B-C1", "B-C2", "B-C3"])
        for row in checkpoints:
            if row["condition"] == "fixed":
                self.assertEqual(row["camera_accessed"], "False")
                self.assertEqual(row["vlm_cue_status"], "")
            else:
                self.assertEqual(row["camera_accessed"], "True")
        for row in rows:
            if row["row_type"] != "checkpoint":
                self.assertEqual(row["camera_accessed"], "False")

        by_id = {row["checkpoint_id"]: row for row in checkpoints}
        self.assertEqual(by_id["A-C1"]["hint_accepted"], "no")
        self.assertEqual(by_id["A-C1"]["hint_delivered"], "False")
        self.assertEqual(by_id["A-C1"]["change_type"], "unchanged")
        self.assertEqual(by_id["A-C1"]["initial_correct"], "True")
        self.assertEqual(by_id["A-C2"]["change_type"], "incorrect-to-correct")
        self.assertEqual(by_id["A-C2"]["hint_delivered"], "True")
        self.assertEqual(by_id["A-C2"]["final_correct"], "True")
        self.assertEqual(by_id["A-C3"]["explicit_help"], "True")
        self.assertEqual(by_id["A-C3"]["hint_offered"], "False")
        self.assertEqual(by_id["A-C3"]["hint_delivered"], "True")
        self.assertEqual(by_id["B-C1"]["vlm_cue_status"], "unclear")
        self.assertEqual(by_id["B-C1"]["nudge"], "fallback_neutral")
        self.assertEqual(by_id["B-C1"]["hint_delivered"], "False")
        self.assertEqual(by_id["B-C2"]["reasoning_action"], "offer_hint")
        self.assertEqual(by_id["B-C2"]["vlm_description"], "hands resting on the table")
        self.assertEqual(by_id["B-C3"]["nudge"], "explicit_help")
        self.assertEqual(by_id["A-C1"]["case_completion_ms"], "")
        self.assertNotEqual(by_id["A-C3"]["case_completion_ms"], "")
        self.assertNotEqual(by_id["B-C3"]["case_completion_ms"], "")
        self.assertEqual(by_id["A-C3"]["explanation_score"], "1")
        self.assertEqual(by_id["B-C1"]["explanation_score"], "0")

        materials = run_study.load_materials()
        hints = {
            checkpoint["checkpoint_id"]: checkpoint["hint"]
            for case in materials["cases"]
            for turn in case["turns"]
            if (checkpoint := turn["checkpoint"])
        }
        joined = "\n".join(spoken)
        self.assertNotIn(hints["A-C1"], joined)
        self.assertIn(hints["A-C2"], joined)
        self.assertIn(hints["A-C3"], joined)
        self.assertNotIn(hints["B-C1"], joined)
        self.assertIn(hints["B-C2"], joined)
        self.assertIn(hints["B-C3"], joined)
        self.assertEqual(joined.count(materials["prompts"]["offer"]), 3)
        self.assertEqual(joined.count(materials["prompts"]["neutral"]), 1)
        self.assertNotIn("You look confused", joined)
        self.assertNotIn("You seem certain", joined)
        self.assertNotIn("Operator", screen.getvalue())
        self.assertNotIn("Score 1 only", screen.getvalue())
        self.assertNotIn("Type the initial choice, A or B.", joined)
        self.assertIn(materials["prompts"]["welcome"], joined)
        self.assertIn(materials["prompts"]["confusion"], joined)
        self.assertIn(materials["prompts"]["thanks"], joined)
        for item in materials["perception_ratings"]:
            self.assertEqual(joined.count(item["prompt"]), 2)
        self.assertTrue(all(row["speech_delivery"] == "simulated" for row in rows))
        self.assertTrue(all(row["robot_delivery_success"] == "False" for row in rows))
        self.assertTrue(all(row["explanation_scored_by"] == "program" for row in checkpoints))
        self.assertTrue(all(row["capture_status"] == "not_requested" for row in checkpoints if row["condition"] == "fixed"))
        self.assertTrue(all(row["capture_status"] == "simulated" for row in checkpoints if row["condition"] == "camera"))
        self.assertTrue(all(row["capture_status"] == "not_requested" for row in rows if row["row_type"] == "story"))
        self.assertIn(materials["prompts"]["thanks"], joined)


class ReachyPlayback(unittest.TestCase):
    def _tts_client(self, sink):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def stream_to_file(self, path):
                import wave
                with wave.open(str(path), "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(8000)
                    handle.writeframes(b"\x00\x00" * 8)

        class Streaming:
            def create(self, **kwargs):
                sink["request"] = kwargs
                return Response()

        class Speech:
            def __init__(self):
                self.with_streaming_response = Streaming()

        class Audio:
            def __init__(self):
                self.speech = Speech()

        class Client:
            def __init__(self):
                self.audio = Audio()

        return Client()

    def test_same_text_is_sent_to_tts_and_reachy(self):
        import tempfile
        sink = {}
        played = []

        class Mini:
            def __init__(self):
                self.media = self

            def play_sound(self, path):
                played.append(path)

        line = "Would a hint help?"
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            screen = io.StringIO()
            with redirect_stdout(screen):
                delivered = robot_io.speak(Mini(), self._tts_client(sink), line, "gpt-4o-mini-tts", "alloy", cache)
                delivered_again = robot_io.speak(Mini(), self._tts_client(sink), line, "gpt-4o-mini-tts", "alloy", cache)
        self.assertTrue(delivered)
        self.assertTrue(delivered_again)
        self.assertEqual(len(played), 2)
        self.assertEqual(played[0], played[1])
        self.assertEqual(sink["request"]["input"], line)
        self.assertEqual(sink["request"]["model"], "gpt-4o-mini-tts")
        self.assertEqual(sink["request"]["voice"], "alloy")
        self.assertLess(screen.getvalue().find("Reachy"), screen.getvalue().find("Would a hint help?"))

    def test_camera_frame_is_saved_as_jpeg(self):
        import numpy as np
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            saved = robot_io.save_frame(image, Path(directory), "P01_order1_B-C1")
            self.assertEqual(saved, Path(directory) / "P01_order1_B-C1.jpg")
            self.assertTrue(saved.is_file())
            self.assertGreater(saved.stat().st_size, 0)

    def test_streamed_wav_wait_uses_the_bytes_on_disk(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "streamed.wav"
            payload = b"\x00\x00" * 24000
            header = bytearray(44)
            header[0:4] = b"RIFF"
            header[4:8] = b"\xff\xff\xff\xff"
            header[8:12] = b"WAVE"
            header[12:16] = b"fmt "
            header[16:20] = (16).to_bytes(4, "little")
            header[20:22] = (1).to_bytes(2, "little")
            header[22:24] = (1).to_bytes(2, "little")
            header[24:28] = (24000).to_bytes(4, "little")
            header[32:34] = (2).to_bytes(2, "little")
            header[34:36] = (16).to_bytes(2, "little")
            header[36:40] = b"data"
            header[40:44] = b"\xff\xff\xff\xff"
            path.write_bytes(bytes(header) + payload)
            self.assertAlmostEqual(robot_io.wav_duration(path), 1.0, places=2)

    def test_playback_failure_is_not_success(self):
        import tempfile

        class Mini:
            def __init__(self):
                self.media = self

            def play_sound(self, path):
                raise RuntimeError("speaker unavailable")

        with tempfile.TemporaryDirectory() as directory:
            delivered = robot_io.speak(
                Mini(), self._tts_client({}), "Take your time—you're making progress.", "gpt-4o-mini-tts", "alloy", Path(directory)
            )
        self.assertFalse(delivered)

    def test_missing_frame_does_not_offer_a_hint(self):
        def refuse(*_args):
            raise AssertionError("model called without a frame")

        plan = run_study.plan_guidance("camera", "A", "because", "story", lambda: None, refuse, refuse)
        self.assertEqual(plan["vlm_error"], "NO_FRAME")
        self.assertFalse(plan["offer"])
        self.assertEqual(plan["nudge"], "fallback_neutral")

    def test_cached_line_is_not_sent_to_tts_again(self):
        sink = {"calls": 0}

        class Response:
            def __enter__(self):
                sink["calls"] += 1
                return self

            def __exit__(self, *_args):
                return False

            def stream_to_file(self, path):
                import wave
                with wave.open(str(path), "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(8000)
                    handle.writeframes(b"\x00\x00" * 8)

        class Streaming:
            def create(self, **kwargs):
                del kwargs
                return Response()

        class Speech:
            def __init__(self):
                self.with_streaming_response = Streaming()

        class Audio:
            def __init__(self):
                self.speech = Speech()

        class Client:
            def __init__(self):
                self.audio = Audio()

        with tempfile.TemporaryDirectory() as directory:
            cache = robot_io.SpeechCache(Client(), "gpt-4o-mini-tts", "alloy", Path(directory))
            cache.prepare(["Would a hint help?", "Would a hint help?"])
            cache.prepare(["Would a hint help?"])
        self.assertEqual(sink["calls"], 1)

    def test_spoken_lines_cover_reachy_and_skip_typing_prompts(self):
        materials = run_study.load_materials()
        lines = run_study.spoken_lines(materials, 1, False)
        self.assertIn(materials["prompts"]["welcome"], lines)
        self.assertIn(materials["cases"][0]["turns"][1]["checkpoint"]["hint"], lines)
        self.assertNotIn("Type the initial choice, A or B.", lines)


class ComponentCheck(unittest.TestCase):
    def _ok(self, frame=object()):
        seen = {}

        def connect():
            seen["connected"] = True
            return object()

        def capture():
            seen["captured"] = True
            return frame

        def describe(got):
            seen["described"] = got
            return CueResult("unclear", "", "", 12)

        def decide():
            seen["decided"] = True
            return ActionResult("no_nudge", "", 8)

        def speak(text):
            seen["spoken"] = text
            return True

        return seen, dict(connect=connect, capture=capture, describe=describe, decide=decide, speak=speak)

    def test_each_component_is_called_once(self):
        frame = object()
        seen, calls = self._ok(frame)
        self.assertTrue(run_study.check_components(**calls))
        self.assertEqual(seen["spoken"], "This is a Reachy component check.")
        self.assertIs(seen["described"], frame)
        self.assertTrue(seen["decided"])
        self.assertTrue(seen["captured"])

    def test_missing_frame_does_not_call_the_vision_model(self):
        seen, calls = self._ok(None)

        def describe(frame):
            raise AssertionError(frame)

        calls["describe"] = describe
        self.assertFalse(run_study.check_components(**calls))
        self.assertNotIn("described", seen)

    def test_playback_failure_fails_the_check(self):
        seen, calls = self._ok()

        def speak(text):
            seen["spoken"] = text
            return False

        calls["speak"] = speak
        self.assertFalse(run_study.check_components(**calls))


if __name__ == "__main__":
    unittest.main()
