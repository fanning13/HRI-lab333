"""Fixed-point versus camera-informed hint guidance on Reachy Mini.

Participants type every response. Reachy speaks from the task file.
The fixed-point condition never requests a camera frame. The
camera-informed condition requests one frame at each checkpoint and
lets the reasoning model choose offer_hint, neutral_encouragement, or
no_nudge. An explicit typed request for help delivers that checkpoint's
hint in either condition.

    python run_study.py --check
    python run_study.py --participant P01 --order 1 --simulate
    python run_study.py --participant P01 --order 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

from models import (
    DEFAULT_REASONING_EFFORT,
    DEFAULT_REASONING_MODEL,
    DEFAULT_ROBOT_IP,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    DEFAULT_VLM_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_LLM_MODEL,
    OLLAMA_VLM_MODEL,
    ActionResult,
    CueResult,
)

ROOT = Path(__file__).resolve().parent
MATERIALS_PATH = ROOT / "task_materials.json"
HELP_PATTERN = re.compile(r"\b(help|hint|clue|stuck)\b", re.IGNORECASE)
CHOICE_PATTERN = re.compile(r"^([ab])\b", re.IGNORECASE)
YES_PATTERN = re.compile(r"^(y|yes|yeah|yep)\b", re.IGNORECASE)
NO_PATTERN = re.compile(r"^(n|no|nope)\b", re.IGNORECASE)
KEEP_PATTERN = re.compile(r"^(keep|k|same)\b", re.IGNORECASE)
CHANGE_PATTERN = re.compile(r"^(change|c|switch)\b", re.IGNORECASE)

FIELDNAMES = [
    "participant_id",
    "order_id",
    "session_mode",
    "row_type",
    "case_id",
    "case_title",
    "condition",
    "turn_id",
    "checkpoint_id",
    "initial_choice",
    "initial_explanation",
    "initial_correct",
    "final_choice",
    "final_explanation",
    "final_correct",
    "answer_changed",
    "change_type",
    "explanation_score",
    "explanation_scored_by",
    "confusion_rating",
    "perception_item",
    "perception_rating",
    "camera_accessed",
    "capture_status",
    "frame_saved",
    "vlm_model",
    "vlm_cue_status",
    "vlm_description",
    "vlm_error",
    "vlm_latency_ms",
    "reasoning_model",
    "reasoning_effort",
    "reasoning_action",
    "reasoning_error",
    "reasoning_latency_ms",
    "nudge",
    "hint_offered",
    "hint_accepted",
    "hint_delivered",
    "explicit_help",
    "response_latency_ms",
    "case_completion_ms",
    "speech_delivery",
    "robot_delivery_success",
    "tts_model",
    "model_output_status",
]


class SessionStop(Exception):
    """The participant typed q."""


def load_materials() -> dict:
    return json.loads(MATERIALS_PATH.read_text(encoding="utf-8"))


def validate_materials(materials: dict) -> None:
    prompts = materials["prompts"]
    for key in ("welcome", "consent", "offer", "neutral", "revision", "confusion", "fixed_intro", "camera_intro", "thanks"):
        if not prompts[key].strip():
            raise ValueError(f"missing prompt {key}")
    if len(materials["perception_ratings"]) != 3:
        raise ValueError("expected three perception ratings")
    if [item["order_id"] for item in materials["orders"]] != [1, 2, 3, 4]:
        raise ValueError("orders must be 1 through 4")
    for order in materials["orders"]:
        cases = order["cases"]
        if {item["case_id"] for item in cases} != {"A", "B"}:
            raise ValueError(f"order {order['order_id']} must contain both cases")
        if {item["condition"] for item in cases} != {"fixed", "camera"}:
            raise ValueError(f"order {order['order_id']} must contain both conditions")
    cases = {case["case_id"]: case for case in materials["cases"]}
    if set(cases) != {"A", "B"}:
        raise ValueError("expected cases A and B")
    for case in cases.values():
        if len(case["turns"]) != 6:
            raise ValueError(f"{case['case_id']} must have six turns")
        for turn in case["turns"]:
            checkpoint = turn["checkpoint"]
            if checkpoint is None:
                continue
            if checkpoint["narration"] != turn["narration"]:
                raise ValueError(f"{checkpoint['checkpoint_id']} narration does not match its turn")
            if set(checkpoint["options"]) != {"A", "B"}:
                raise ValueError(f"{checkpoint['checkpoint_id']} needs options A and B")
            if checkpoint["correct_option"] not in {"A", "B"}:
                raise ValueError(f"{checkpoint['checkpoint_id']} has a bad correct option")
            for option in checkpoint["options"].values():
                if "(correct)" in option.lower():
                    raise ValueError(f"{checkpoint['checkpoint_id']} marks an option as correct in the text")
            if checkpoint["hint"] in story_context(case, turn["turn_id"]):
                raise ValueError(f"{checkpoint['checkpoint_id']} leaks its hint into the story context")
            checks = checkpoint.get("evidence_checks")
            if not checks or any(not group or any(not str(phrase).strip() for phrase in group) for group in checks):
                raise ValueError(f"{checkpoint['checkpoint_id']} needs evidence checks for the program score")
    fourth = materials["orders"][3]["cases"]
    if [ (item["case_id"], item["condition"]) for item in fourth ] != [("B", "fixed"), ("A", "camera")]:
        raise ValueError("order 4 must be Briarwick fixed-point, then Ashcombe camera-informed")


def default_order(participant: str) -> int | None:
    match = re.fullmatch(r"P(\d+)", participant, re.IGNORECASE)
    if not match:
        return None
    return ((int(match.group(1)) - 1) % 4) + 1


def explicit_help_requested(text: str) -> bool:
    return HELP_PATTERN.search(text or "") is not None


def story_context(case: dict, turn_id: str) -> str:
    """Story text up to this turn, plus the question and both options.

    The hint and the correct-option field are omitted.
    """
    parts: list[str] = []
    for turn in case["turns"]:
        if turn["narration"]:
            parts.append(turn["narration"])
        if turn["turn_id"] == turn_id:
            checkpoint = turn["checkpoint"]
            if checkpoint:
                parts.append(checkpoint["question"])
                parts.append(f"A. {checkpoint['options']['A']}")
                parts.append(f"B. {checkpoint['options']['B']}")
            break
    return "\n".join(parts)


def question_line(checkpoint: dict) -> str:
    return (
        f"{checkpoint['question']} "
        f"A. {checkpoint['options']['A']} "
        f"B. {checkpoint['options']['B']}"
    )


def score_explanation(checkpoint: dict, explanation: str) -> str:
    """Apply the checkpoint's evidence checks. 1 if any check is fully present."""
    text = " ".join((explanation or "").lower().split())
    for group in checkpoint["evidence_checks"]:
        if all(phrase.lower() in text for phrase in group):
            return "1"
    return "0"


def choice_correct(choice: str, correct: str) -> bool:
    return choice.upper() == correct.upper()


def change_type(initial: str, final: str, correct: str) -> str:
    if initial.upper() == final.upper():
        return "unchanged"
    initial_ok = choice_correct(initial, correct)
    final_ok = choice_correct(final, correct)
    if not initial_ok and final_ok:
        return "incorrect-to-correct"
    if initial_ok and not final_ok:
        return "correct-to-incorrect"
    return "other"


def plan_guidance(condition: str, choice: str, explanation: str, story: str, capture, describe, decide) -> dict:
    """Decide the guidance event. Fixed-point never calls capture."""
    plan = {
        "camera_accessed": False,
        "frame_saved": False,
        "vlm_cue_status": "",
        "vlm_description": "",
        "vlm_error": "",
        "vlm_latency_ms": "",
        "reasoning_action": "",
        "reasoning_error": "",
        "reasoning_latency_ms": "",
        "nudge": "",
        "offer": False,
        "deliver_hint": False,
        "explicit_help": False,
        "speak_neutral": False,
    }
    asked = explicit_help_requested(f"{choice}\n{explanation}")
    if condition == "fixed":
        if asked:
            plan["explicit_help"] = True
            plan["deliver_hint"] = True
            plan["nudge"] = "explicit_help"
        else:
            plan["offer"] = True
            plan["nudge"] = "fixed_offer"
        return plan

    plan["camera_accessed"] = True
    frame = capture()
    cue = None
    if frame is None:
        plan["vlm_error"] = "NO_FRAME"
    else:
        cue = describe(frame, story, choice, explanation)
        plan["vlm_cue_status"] = cue.status
        plan["vlm_description"] = cue.description
        plan["vlm_error"] = cue.error
        plan["vlm_latency_ms"] = cue.latency_ms
    if asked:
        plan["explicit_help"] = True
        plan["deliver_hint"] = True
        plan["nudge"] = "explicit_help"
        return plan
    if frame is None or cue is None or cue.error or cue.status != "observed":
        plan["nudge"] = "fallback_neutral"
        plan["speak_neutral"] = True
        return plan
    action = decide(story, choice, explanation, cue.description)
    plan["reasoning_action"] = action.action
    plan["reasoning_error"] = action.error
    plan["reasoning_latency_ms"] = action.latency_ms
    if action.error or action.action not in {"offer_hint", "neutral_encouragement", "no_nudge"}:
        plan["nudge"] = "fallback_neutral"
        plan["speak_neutral"] = True
        return plan
    if action.action == "offer_hint":
        plan["offer"] = True
        plan["nudge"] = "offer_hint"
    elif action.action == "neutral_encouragement":
        plan["speak_neutral"] = True
        plan["nudge"] = "neutral_encouragement"
    else:
        plan["nudge"] = "no_nudge"
    return plan


class Log:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.rows: list[dict] = []
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES, extrasaction="ignore")
        self._writer.writeheader()

    def write(self, row: dict) -> None:
        full = {key: "" for key in FIELDNAMES}
        full.update(row)
        full["frame_saved"] = False
        self._writer.writerow(full)
        self._file.flush()
        self.rows.append(full)

    def close(self) -> None:
        self._file.close()


def _read(input_fn, prompt: str) -> str:
    shown = textwrap.fill(prompt, width=72, initial_indent="  ", subsequent_indent="  ")
    value = input_fn(f"You\n{shown}\n> ")
    if value is None:
        value = ""
    value = str(value).strip()
    if value.lower() == "q":
        raise SessionStop()
    return value


def read_choice(input_fn, prompt: str) -> str:
    while True:
        value = _read(input_fn, prompt)
        match = CHOICE_PATTERN.match(value)
        if match:
            return match.group(1).upper()
        print("Type A or B.")


def read_yes(input_fn, prompt: str) -> bool:
    while True:
        value = _read(input_fn, prompt)
        if YES_PATTERN.match(value):
            return True
        if NO_PATTERN.match(value):
            return False
        print("Type yes or no.")


def read_keep(input_fn, prompt: str) -> bool:
    while True:
        value = _read(input_fn, prompt)
        if KEEP_PATTERN.match(value):
            return True
        if CHANGE_PATTERN.match(value):
            return False
        print("Type keep or change.")


def read_scale(input_fn, prompt: str, low: int, high: int) -> str:
    while True:
        value = _read(input_fn, prompt)
        if value.isdigit() and low <= int(value) <= high:
            return value
        print(f"Type a whole number from {low} to {high}.")


def read_text(input_fn, prompt: str) -> str:
    while True:
        value = _read(input_fn, prompt)
        if value:
            return value
        print("Type a response.")


def speech_fields(simulate: bool, delivered: bool) -> dict:
    """Real playback is recorded only after the Reachy speaker call succeeds."""
    if simulate:
        return {"speech_delivery": "simulated", "robot_delivery_success": False}
    if delivered:
        return {"speech_delivery": "played", "robot_delivery_success": True}
    return {"speech_delivery": "failed", "robot_delivery_success": False}


def capture_status(simulate: bool, condition: str, plan: dict | None = None) -> str:
    if condition != "camera":
        return "not_requested"
    if simulate:
        return "simulated"
    if plan and plan.get("vlm_error") == "NO_FRAME":
        return "failed"
    return "captured"


class Speaker:
    def __init__(self, speak_fn):
        self.speak_fn = speak_fn
        self.ok = True
        self.lines: list[str] = []

    def say(self, text: str) -> None:
        self.lines.append(text)
        if not self.speak_fn(text):
            self.ok = False

    def reset(self) -> None:
        self.ok = True


def run_session(args, input_fn=input, speak_fn=None, capture=None, describe=None, decide=None) -> Path:
    materials = load_materials()
    validate_materials(materials)
    prompts = materials["prompts"]
    cases = {case["case_id"]: case for case in materials["cases"]}
    order = next(item for item in materials["orders"] if item["order_id"] == args.order)
    mode = "simulate" if args.simulate else "hardware"
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = Log(output / f"{args.participant}_order{args.order}_{stamp}.csv")
    speaker = Speaker(speak_fn or _print_speak)
    status = "simulated_not_participant_data" if args.simulate else "uncertain_cue_not_ground_truth"

    def base(**extra) -> dict:
        row = {
            "participant_id": args.participant,
            "order_id": args.order,
            "session_mode": mode,
            "tts_model": "" if args.simulate else args.tts_model,
            "vlm_model": args.vlm_model,
            "reasoning_model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "model_output_status": status,
            "frame_saved": False,
        }
        row.update(extra)
        return row

    try:
        speaker.say(prompts["welcome"])
        speaker.say(prompts["consent_api"] if getattr(args, "api", False) else prompts["consent"])
        consent = read_yes(input_fn, "Type yes to confirm informed consent, or no to stop.")
        if not consent:
            print("Stopped before either case. No camera frame was requested.")
            return log.path
        for assignment in order["cases"]:
            case = cases[assignment["case_id"]]
            condition = assignment["condition"]
            _run_case(
                args, materials, case, condition, input_fn, speaker, capture, describe, decide, log, base, status
            )
        speaker.say(prompts["thanks"])
        log.write(base(
            row_type="session_end",
            camera_accessed=False,
            capture_status="not_requested",
            **speech_fields(args.simulate, speaker.ok),
        ))
    except SessionStop:
        print("Stopped. The log contains the rows recorded so far.")
    finally:
        log.close()
    return log.path


def _run_case(args, materials, case, condition, input_fn, speaker, capture, describe, decide, log, base, status) -> None:
    prompts = materials["prompts"]
    intro = prompts["fixed_intro"] if condition == "fixed" else prompts["camera_intro"]
    case_started = time.perf_counter()
    speaker.reset()
    speaker.say(f"Case. {case['title']}.")
    speaker.say(intro)
    log.write(base(
        row_type="case_start",
        case_id=case["case_id"],
        case_title=case["title"],
        condition=condition,
        camera_accessed=False,
        capture_status="not_requested",
        **speech_fields(args.simulate, speaker.ok),
        model_output_status="not_used",
    ))
    completion_ms = ""
    for turn in case["turns"]:
        checkpoint = turn["checkpoint"]
        if checkpoint is None:
            speaker.reset()
            if turn["narration"]:
                speaker.say(turn["narration"])
            log.write(base(
                row_type="story",
                case_id=case["case_id"],
                case_title=case["title"],
                condition=condition,
                turn_id=turn["turn_id"],
                camera_accessed=False,
                capture_status="not_requested",
                **speech_fields(args.simulate, speaker.ok),
                model_output_status="not_used" if condition == "fixed" else status,
            ))
            continue
        row = _run_checkpoint(
            args, materials, case, condition, turn, input_fn, speaker, capture, describe, decide, status, case_started
        )
        if turn["turn_id"].endswith("T6"):
            completion_ms = row["case_completion_ms"]
        row["case_completion_ms"] = completion_ms if turn["turn_id"].endswith("T6") else ""
        log.write(row)
    for item in materials["perception_ratings"]:
        speaker.reset()
        speaker.say(item["prompt"])
        rating = read_scale(input_fn, item["prompt"], 1, 5)
        log.write(base(
            row_type="perception",
            case_id=case["case_id"],
            case_title=case["title"],
            condition=condition,
            perception_item=item["id"],
            perception_rating=rating,
            camera_accessed=False,
            capture_status="not_requested",
            case_completion_ms=completion_ms,
            **speech_fields(args.simulate, speaker.ok),
            model_output_status="not_used",
        ))


def _run_checkpoint(args, materials, case, condition, turn, input_fn, speaker, capture, describe, decide, status, case_started) -> dict:
    prompts = materials["prompts"]
    checkpoint = turn["checkpoint"]
    speaker.reset()
    if turn["narration"]:
        speaker.say(turn["narration"])
    speaker.say(question_line(checkpoint))
    answered_at = time.perf_counter()
    choice = read_choice(input_fn, "Type the initial choice, A or B.")
    explanation = read_text(input_fn, "Type the initial explanation.")
    response_ms = int((time.perf_counter() - answered_at) * 1000)
    story = story_context(case, turn["turn_id"])
    if condition == "fixed":
        def _fixed_capture():
            raise RuntimeError("fixed-point condition requested a camera frame")
        capture_fn = _fixed_capture
    else:
        capture_fn = capture
    plan = plan_guidance(condition, choice, explanation, story, capture_fn, describe, decide)
    hint_accepted = ""
    if plan["speak_neutral"]:
        speaker.say(prompts["neutral"])
    if plan["offer"]:
        speaker.say(prompts["offer"])
        accepted = read_yes(input_fn, "Type yes to hear the hint, or no to continue.")
        hint_accepted = "yes" if accepted else "no"
        if accepted:
            speaker.say(checkpoint["hint"])
            plan["deliver_hint"] = True
    elif plan["deliver_hint"]:
        speaker.say(checkpoint["hint"])
    speaker.say(prompts["revision"])
    keep = read_keep(input_fn, "Type keep or change.")
    if keep:
        final_choice = choice
    else:
        final_choice = read_choice(input_fn, "Type the final choice, A or B.")
    final_explanation = read_text(input_fn, "Type the final explanation.")
    completion_ms = int((time.perf_counter() - case_started) * 1000)
    speaker.say(prompts["confusion"])
    confusion = read_scale(input_fn, prompts["confusion"], 1, 5)
    explanation_score = score_explanation(checkpoint, final_explanation)
    model_status = "not_used" if condition == "fixed" else status
    return {
        "participant_id": args.participant,
        "order_id": args.order,
        "session_mode": "simulate" if args.simulate else "hardware",
        "row_type": "checkpoint",
        "case_id": case["case_id"],
        "case_title": case["title"],
        "condition": condition,
        "turn_id": turn["turn_id"],
        "checkpoint_id": checkpoint["checkpoint_id"],
        "initial_choice": choice,
        "initial_explanation": explanation,
        "initial_correct": choice_correct(choice, checkpoint["correct_option"]),
        "final_choice": final_choice,
        "final_explanation": final_explanation,
        "final_correct": choice_correct(final_choice, checkpoint["correct_option"]),
        "answer_changed": choice.upper() != final_choice.upper(),
        "change_type": change_type(choice, final_choice, checkpoint["correct_option"]),
        "explanation_score": explanation_score,
        "explanation_scored_by": "program",
        "confusion_rating": confusion,
        "camera_accessed": plan["camera_accessed"],
        "capture_status": capture_status(args.simulate, condition, plan),
        "frame_saved": False,
        "vlm_model": "" if condition == "fixed" else args.vlm_model,
        "vlm_cue_status": plan["vlm_cue_status"],
        "vlm_description": plan["vlm_description"],
        "vlm_error": plan["vlm_error"],
        "vlm_latency_ms": plan["vlm_latency_ms"],
        "reasoning_model": "" if condition == "fixed" else args.model,
        "reasoning_effort": "" if condition == "fixed" else args.reasoning_effort,
        "reasoning_action": plan["reasoning_action"],
        "reasoning_error": plan["reasoning_error"],
        "reasoning_latency_ms": plan["reasoning_latency_ms"],
        "nudge": plan["nudge"],
        "hint_offered": plan["offer"],
        "hint_accepted": hint_accepted,
        "hint_delivered": plan["deliver_hint"],
        "explicit_help": plan["explicit_help"],
        "response_latency_ms": response_ms,
        "case_completion_ms": completion_ms,
        **speech_fields(args.simulate, speaker.ok),
        "tts_model": "" if args.simulate else args.tts_model,
        "model_output_status": model_status,
    }


def _print_speak(text: str) -> bool:
    from robot_io import show_reachy

    show_reachy(text)
    return True


def _simulated_cue(status: str):
    def describe(frame, story: str, choice: str = "", explanation: str = "") -> CueResult:
        del frame, story, choice, explanation
        if status == "unclear":
            return CueResult("unclear", "", "", 0)
        if status == "error":
            return CueResult("error", "", "simulated model failure", 0)
        return CueResult("observed", "Simulated cue for a walkthrough. Not a camera observation.", "", 0)

    return describe


def _simulated_action(action: str):
    def decide(story: str, choice: str, explanation: str, cue: str) -> ActionResult:
        del story, choice, explanation, cue
        if action == "error":
            return ActionResult("", "simulated model failure", 0)
        return ActionResult(action, "", 0)

    return decide


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the guidance-strategy study.")
    parser.add_argument("--participant", default="P01")
    parser.add_argument("--order", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--check", action="store_true", help="Validate the task file and print the four orders.")
    parser.add_argument("--api", action="store_true", help="Use OpenAI instead of the Lab 3 Ollama models.")
    parser.add_argument("--simulate", action="store_true", help="Print speech and skip the robot, camera, and models.")
    parser.add_argument("--simulate-cue", choices=("observed", "unclear", "error"), default="observed")
    parser.add_argument("--simulate-action", choices=("offer_hint", "neutral_encouragement", "no_nudge", "error"), default="no_nudge")
    parser.add_argument("--model", default="")
    parser.add_argument("--vlm-model", default="")
    parser.add_argument("--tts-model", default="")
    parser.add_argument("--voice", default=DEFAULT_TTS_VOICE)
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--output-dir", default="logs")
    parser.add_argument("--robot-ip", default="")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    materials = load_materials()
    validate_materials(materials)
    if args.order is None:
        args.order = default_order(args.participant)
    if args.order is None:
        raise SystemExit("Pass --order 1, 2, 3, or 4.")
    if args.check:
        for order in materials["orders"]:
            pair = ", then ".join(f"{item['case_id']}/{item['condition']}" for item in order["cases"])
            print(f"Order {order['order_id']}: {pair}")
        print(materials["comparison_note"])
        return
    if args.simulate:
        print("Simulation. No robot, camera, or model API will be used.")
        run_session(
            args,
            capture=lambda: object(),
            describe=_simulated_cue(args.simulate_cue),
            decide=_simulated_action(args.simulate_action),
        )
        return

    print("Loading the session.")
    from dotenv import load_dotenv
    from openai import OpenAI

    from robot_io import connect_robot, get_frame, image_content, speak as robot_speak

    from models import choose_action, describe_frame

    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing. Add it to guidance_study/.env for Reachy speech.")
    tts_client = OpenAI()
    args.tts_model = args.tts_model or DEFAULT_TTS_MODEL
    if args.api:
        args.model = args.model or DEFAULT_REASONING_MODEL
        args.vlm_model = args.vlm_model or DEFAULT_VLM_MODEL
        args.reasoning_effort = args.reasoning_effort or DEFAULT_REASONING_EFFORT
        chat_client = tts_client
        structured = True
    else:
        args.model = args.model or OLLAMA_LLM_MODEL
        args.vlm_model = args.vlm_model or OLLAMA_VLM_MODEL
        args.reasoning_effort = ""
        chat_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        structured = False
        print(f"Ollama. Decisions: {args.model}. Camera cues: {args.vlm_model}.")
    print(f"Reachy speaker. TTS: {args.tts_model}. Voice: {args.voice}.")
    host = args.robot_ip or os.environ.get("REACHY_IP", "") or DEFAULT_ROBOT_IP
    wav_path = ROOT / args.output_dir / "last_speech.wav"
    print(f"Connecting to Reachy at {host}.")
    mini = connect_robot(host)
    try:
        def speak_fn(text: str) -> bool:
            return robot_speak(mini, tts_client, text, args.tts_model, args.voice, wav_path)

        def capture():
            return get_frame(mini)

        def describe(frame, story: str, choice: str = "", explanation: str = "") -> CueResult:
            return describe_frame(
                chat_client, args.vlm_model, args.reasoning_effort, frame, story, image_content,
                choice, explanation, structured=structured,
            )

        def decide(story: str, choice: str, explanation: str, cue: str) -> ActionResult:
            return choose_action(
                chat_client, args.model, args.reasoning_effort, story, choice, explanation, cue, structured=structured
            )

        run_session(args, speak_fn=speak_fn, capture=capture, describe=describe, decide=decide)
    finally:
        closer = getattr(mini, "__exit__", None)
        if closer:
            closer(None, None, None)


if __name__ == "__main__":
    main()
