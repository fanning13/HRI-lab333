"""Turn-based Reachy Mini confusion-aware collaborative storytelling study."""
from __future__ import annotations

import argparse
import base64
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

LABELS = {"CONFUSED", "NOT_CONFUSED", "UNCERTAIN"}
SOFT_CHECK_IN = "Take your time. Would you like to keep going, or would a small hint help?"


def _turn(narration: str, prompt: str, fact: str, clarification: str, formal: bool = False) -> dict:
    return {"narration": narration, "prompt": prompt, "checkpoint_fact": fact,
            "clarification": clarification, "formal": formal}


STORIES = {
    "A": [
        _turn("Emma, Noah, and Lily planned a picnic beside the old greenhouse. Emma locked the supply box with a small brass key and handed it to Noah.", "What should they pack first?", "Noah has the brass key.", "Emma gave the brass key to Noah."),
        _turn("Noah slipped the key into the zipped pocket of his green backpack while Lily carried the picnic blanket.", "What might happen on their walk?", "The key is in Noah's green backpack.", "Noah put the key in the zipped pocket of his green backpack.", True),
        _turn("At the garden, Emma reached into her coat for the key, but it was missing. Lily wondered whether it had fallen near the fountain.", "Where should they look?", "Emma does not have the key; Noah stored it.", "The key is not lost: Noah put it in his green backpack."),
        _turn("Noah remembered the backpack, but it was no longer on his shoulders. Lily recalled setting a green bag beside the greenhouse bench.", "How should they solve this?", "The backpack was left by the greenhouse bench.", "Lily remembers that the green backpack was placed beside the greenhouse bench.", True),
        _turn("They returned to the bench and found the backpack under the blanket. The zipped pocket was still closed.", "What should Noah do next?", "The key remains in the zipped pocket.", "They found Noah's backpack, and the key should still be in its zipped pocket."),
        _turn("Noah opened the pocket, found the brass key, and Emma unlocked the supply box. Lily discovered cups and a note inviting them to plant flowers after lunch.", "How would you end the story?", "They recovered the key and opened the box.", "Noah recovered the key from his backpack and Emma opened the supply box.", True),
    ],
    "B": [
        _turn("On a crisp fall morning in New York City, Sophie, Daniel, and Mia met in Central Park to make sketches. Mia brought her favorite blue sketchbook.", "What should they draw first?", "Mia brought a blue sketchbook.", "Mia brought her blue sketchbook to Central Park."),
        _turn("After drawing the orange leaves, Mia handed the sketchbook to Daniel so he could add the skyline. Sophie carried their pencils.", "What could Daniel add to the picture?", "Daniel now has Mia's sketchbook.", "Mia handed her blue sketchbook to Daniel.", True),
        _turn("They walked toward Bethesda Terrace. Mia checked her tote bag and could not find the sketchbook.", "Where should they look?", "The sketchbook is not in Mia's tote because Daniel has it.", "Mia's sketchbook is with Daniel, not in her tote bag."),
        _turn("Daniel remembered placing the sketchbook on a bench when a gust scattered Sophie's pencils. The bench was beside a musician wearing a red scarf.", "How can they find the right bench?", "The sketchbook was left on the bench beside the red-scarf musician.", "Daniel left the sketchbook on a bench beside a musician in a red scarf.", True),
        _turn("They followed the music and found the bench, but the sketchbook was not on the seat. The musician pointed to a park attendant holding a blue book.", "What should they do next?", "A park attendant picked up the blue sketchbook.", "The musician saw a park attendant pick up the blue sketchbook."),
        _turn("The attendant returned Mia's sketchbook. Inside, Daniel's skyline drawing was safe, and Sophie added a bright red scarf to the scene.", "How would you end their fall day in New York?", "The attendant returned the sketchbook.", "Mia got her sketchbook back from the park attendant.", True),
    ],
}

CLASSIFICATION_PROMPT = """Classify observable evidence during collaborative storytelling.
Return only JSON: {"label":"CONFUSED"}, {"label":"NOT_CONFUSED"}, or {"label":"UNCERTAIN"}.
CONFUSED means the response clearly shows misunderstanding of the story, asks for clarification, or expresses uncertainty about story facts.
NOT_CONFUSED means the response clearly follows the story and continues it coherently.
UNCERTAIN means evidence is missing or ambiguous. Creative hesitation, pauses, short answers, neutral expression, or taking time to invent an idea are not by themselves confusion. Do not infer a person's internal state from appearance."""

BRIDGE_PROMPT = """Continue a controlled child-friendly collaborative story. Write exactly 1-2 short sentences that connect the participant's idea to the supplied next fixed narration. Do not add, change, reveal, or contradict facts about who has the missing object or where it is. Do not introduce new named characters. Return only the bridge text."""

COMPREHENSION = {
    "A": [("Who did Emma give the key to?", "Noah"), ("Where was the backpack left?", "beside the greenhouse bench"), ("Where was the key found?", "the zipped pocket of Noah's green backpack")],
    "B": [("Who did Mia give the sketchbook to?", "Daniel"), ("What identified the bench?", "a musician wearing a red scarf"), ("Who returned the sketchbook?", "the park attendant")],
}


@dataclass
class Prediction:
    label: str = "UNCERTAIN"
    milliseconds: float | None = None
    error: str = ""


@dataclass
class AudioCapture:
    wav: io.BytesIO
    speech_start_delay_ms: float | None
    user_speech_duration_ms: float | None
    end_silence_ms: float | None
    method: str


def parse_label(raw: str) -> str:
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        match = re.search(r"\{[^{}]*\}", raw or "")
        try:
            data = json.loads(match.group(0)) if match else {}
        except ValueError:
            data = {}
    label = data.get("label", "") if isinstance(data, dict) else ""
    return label if label in LABELS else "UNCERTAIN"


def explicit_request(text: str) -> bool:
    return bool(re.search(r"\b(explain|clarify|confused|don't understand|do not understand|what do you mean|lost|hint|help me)\b", text, re.I))


def interaction_stall(text: str, speech_start_delay_ms: float | None, user_speech_duration_ms: float | None, threshold_ms: float) -> bool:
    if not text.strip():
        return True
    return bool((speech_start_delay_ms is not None and speech_start_delay_ms >= threshold_ms) or
                (user_speech_duration_ms is not None and user_speech_duration_ms < 500))


def assistance_decision(label: str, stalled: bool, explicit: bool, condition: str) -> str:
    if explicit:
        return "clarification"
    if condition != "assistance":
        return "neutral"
    if label == "CONFUSED":
        return "clarification"
    if label == "UNCERTAIN" and stalled:
        return "soft_check_in"
    return "neutral"


def rating(prompt: str, maximum: int, input_fn: Callable[[str], str] = input) -> str:
    while True:
        value = input_fn(prompt).strip()
        if not value:
            return ""
        if value.isdigit() and 1 <= int(value) <= maximum:
            return value
        print(f"  Please enter 1–{maximum} or press Enter to skip.")


def jpg_as_data_url(frame: Any) -> str:
    import cv2
    height, width = frame.shape[:2]
    if width > 640:
        frame = cv2.resize(frame, (640, int(height * 640 / width)))
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
    if not ok:
        raise RuntimeError("Could not encode camera frame")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def classify(client: Any, model: str, transcript: str, context: str, frame: Any = None) -> Prediction:
    started = time.perf_counter()
    try:
        prompt = f"Story context:\n{context}\n\nParticipant response:\n{transcript}\n\nClassify the evidence."
        content: Any = prompt
        if frame is not None:
            content = [{"type": "text", "text": prompt + "\nUse visible cues only as weak supporting evidence."},
                       {"type": "image_url", "image_url": {"url": jpg_as_data_url(frame)}}]
        response = client.chat.completions.create(model=model, messages=[{"role": "system", "content": CLASSIFICATION_PROMPT}, {"role": "user", "content": content}], max_tokens=60, response_format={"type": "json_object"})
        return Prediction(parse_label(response.choices[0].message.content or ""), (time.perf_counter() - started) * 1000)
    except Exception as exc:
        return Prediction("UNCERTAIN", (time.perf_counter() - started) * 1000, str(exc)[:200])


def generate_bridge(client: Any, model: str, transcript: str, next_narration: str, locked_facts: str) -> tuple[str, float, str]:
    if not transcript.strip() or not next_narration:
        return "", 0.0, ""
    started = time.perf_counter()
    try:
        prompt = f"Participant idea: {transcript}\nLocked facts: {locked_facts}\nNext fixed narration: {next_narration}"
        response = client.chat.completions.create(model=model, messages=[{"role": "system", "content": BRIDGE_PROMPT}, {"role": "user", "content": prompt}], max_tokens=80, temperature=0.3)
        return (response.choices[0].message.content or "").strip(), (time.perf_counter() - started) * 1000, ""
    except Exception as exc:
        return "", (time.perf_counter() - started) * 1000, str(exc)[:200]


def _wav(samples: Any, rate: int) -> io.BytesIO:
    import numpy as np
    audio = np.concatenate(samples, axis=0)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate); wf.writeframes(pcm.tobytes())
    buf.seek(0); buf.name = "utterance.wav"
    return buf


def fixed_recording(mini: Any, seconds: float) -> AudioCapture:
    samples, started = [], time.monotonic()
    mini.media.start_recording()
    try:
        while time.monotonic() - started < seconds:
            sample = mini.media.get_audio_sample()
            if sample is not None: samples.append(sample)
            else: time.sleep(0.01)
    finally:
        mini.media.stop_recording()
    if not samples: raise RuntimeError("No microphone samples received")
    return AudioCapture(_wav(samples, int(mini.media.get_input_audio_samplerate())), None, seconds * 1000, None, "fixed_fallback")


def vad_recording(mini: Any, max_seconds: float, silence_ms: int = 1000, aggressiveness: int = 2) -> AudioCapture:
    """Local WebRTC VAD; automatically falls back to fixed recording on any VAD/backend error."""
    try:
        import numpy as np
        import webrtcvad
        rate = int(mini.media.get_input_audio_samplerate())
        if rate not in {8000, 16000, 32000, 48000}: raise ValueError(f"Unsupported VAD sample rate: {rate}")
        vad, frame_ms = webrtcvad.Vad(aggressiveness), 30
        needed = int(rate * frame_ms / 1000)
        samples, pending, voiced = [], np.array([], dtype=np.float32), False
        started, speech_start, last_voice = time.monotonic(), None, None
        mini.media.start_recording()
        try:
            while time.monotonic() - started < max_seconds:
                chunk = mini.media.get_audio_sample()
                if chunk is None: time.sleep(0.005); continue
                mono = np.asarray(chunk).mean(axis=1) if np.asarray(chunk).ndim > 1 else np.asarray(chunk)
                samples.append(mono); pending = np.concatenate((pending, mono.astype(np.float32)))
                while len(pending) >= needed:
                    frame, pending = pending[:needed], pending[needed:]
                    pcm = (np.clip(frame, -1, 1) * 32767).astype("<i2").tobytes()
                    now = time.monotonic()
                    if vad.is_speech(pcm, rate):
                        voiced = True; speech_start = speech_start or now; last_voice = now
                if voiced and last_voice and (time.monotonic() - last_voice) * 1000 >= silence_ms: break
        finally:
            mini.media.stop_recording()
        if not samples: raise RuntimeError("No microphone samples received")
        end = time.monotonic()
        return AudioCapture(_wav(samples, rate), ((speech_start - started) * 1000 if speech_start else None),
                            ((last_voice - speech_start) * 1000 if speech_start and last_voice else None),
                            ((end - last_voice) * 1000 if last_voice else None), "webrtc_vad")
    except Exception as exc:
        print(f"  [VAD unavailable: {exc}; using fixed {max_seconds:g}s recording]")
        return fixed_recording(mini, max_seconds)


def transcribe(client: Any, capture: AudioCapture, model: str) -> str:
    return client.audio.transcriptions.create(model=model, file=("utterance.wav", capture.wav, "audio/wav")).text.strip()


def say(client: Any, mini: Any, text: str, args: argparse.Namespace) -> bool:
    print(f"\nReachy > {text}")
    if args.demo or args.no_tts or mini is None: return False
    fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    try:
        with client.audio.speech.with_streaming_response.create(model=args.tts_model, voice=args.voice, input=text, response_format="wav") as response:
            response.stream_to_file(path)
        mini.media.play_sound(path)
        return True
    except Exception as exc:
        print(f"  [TTS error] {exc}"); return False
    finally:
        try: os.remove(path)
        except OSError: pass


def mock_predict(text: str) -> Prediction:
    if explicit_request(text): return Prediction("CONFUSED", 0, "MOCK - not research data")
    if not text.strip(): return Prediction("UNCERTAIN", 0, "MOCK - not research data")
    return Prediction("NOT_CONFUSED", 0, "MOCK - not research data")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    if not args.demo and not os.environ.get("OPENAI_API_KEY"): raise SystemExit("OPENAI_API_KEY missing; use --demo for a software-only run")
    if args.input == "mic" and (args.demo or args.no_robot): raise SystemExit("Microphone mode requires a connected robot and non-demo run")
    client = None
    if not args.demo:
        from openai import OpenAI
        client = OpenAI()
    robot_context = None
    if not args.demo and not args.no_robot:
        from reachy_mini import ReachyMini
        robot_context = ReachyMini(host=args.robot_ip, connection_mode="network", media_backend="default")
    story, rows = STORIES[args.story], []
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.output_dir) / f"{args.participant}_{args.story}_{args.condition}_{session_id}.csv"

    def session(mini: Any) -> None:
        if mini is not None: mini.enable_motors()
        say(client, mini, "Let's create a story together. I'll tell one part, then you add an idea.", args)
        bridge = ""
        context_parts = []
        for index, turn in enumerate(story, 1):
            if bridge: say(client, mini, bridge, args)
            say(client, mini, turn["narration"] + " " + turn["prompt"], args)
            context_parts.append(turn["narration"])
            ready = time.perf_counter(); capture = None
            if args.input == "typed" or args.demo:
                transcript = input("Participant (q to quit)> ").strip()
                if transcript.lower() == "q": break
                start_delay = (time.perf_counter() - ready) * 1000
            else:
                input("Press Enter when the participant is ready...")
                capture = vad_recording(mini, args.record_seconds, args.end_silence_ms, args.vad_aggressiveness)
                transcript = transcribe(client, capture, args.stt_model)
                start_delay = capture.speech_start_delay_ms
                print(f"Recognized speech: {transcript!r}")
            frame = None
            if mini is not None:
                try: frame = mini.media.get_frame()
                except Exception as exc: print(f"  [Camera unavailable] {exc}")
            context = " ".join(context_parts)
            if args.demo:
                llm, vlm = mock_predict(transcript), (mock_predict(transcript) if frame is not None else Prediction(error="NO_IMAGE"))
            else:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    a = pool.submit(classify, client, args.model, transcript, context)
                    b = pool.submit(classify, client, args.model, transcript, context, frame) if frame is not None else None
                    llm, vlm = a.result(), (b.result() if b else Prediction(error="NO_IMAGE"))
            duration = capture.user_speech_duration_ms if capture else None
            stalled = interaction_stall(transcript, start_delay, duration, args.stall_threshold_ms)
            decision = assistance_decision(llm.label, stalled, explicit_request(transcript), args.condition)
            offered, accepted, clarification = decision != "neutral", "", ""
            if decision == "soft_check_in":
                say(client, mini, SOFT_CHECK_IN, args)
                answer = input("Participant response to check-in> ").strip()
                accepted = int(bool(re.search(r"\b(hint|help|yes|please)\b", answer, re.I)))
                if accepted: clarification = turn["clarification"]; say(client, mini, clarification, args)
            elif decision == "clarification":
                clarification = turn["clarification"]; accepted = 1; say(client, mini, clarification, args)
            else:
                say(client, mini, "Thanks for your idea. Let's continue.", args)
            next_text = story[index]["narration"] if index < len(story) else ""
            if args.demo:
                bridge, bridge_ms, bridge_error = (("Your idea leads them onward.", 0.0, "MOCK - not research data") if next_text else ("", 0.0, ""))
            else:
                bridge, bridge_ms, bridge_error = generate_bridge(client, args.model, transcript, next_text, " ".join(x["checkpoint_fact"] for x in story[:index + 1]))
            confusion_score = rating("How confused were you during this turn? (1–5, Enter skips): ", 5) if turn["formal"] and not args.skip_survey else ""
            rows.append({"participant_id": args.participant, "session_id": session_id, "story": args.story, "condition": args.condition,
                "turn": index, "formal_checkpoint": int(turn["formal"]), "checkpoint_fact": turn["checkpoint_fact"], "story_bridge": bridge,
                "story_generation_ms": round(bridge_ms, 1), "story_generation_error": bridge_error, "llm_label": llm.label,
                "vlm_label": vlm.label if frame is not None else "MISSING", "llm_model_ms": round(llm.milliseconds or 0, 1),
                "vlm_model_ms": round(vlm.milliseconds or 0, 1) if frame is not None else "", "model_error_llm": llm.error,
                "model_error_vlm": vlm.error, "interaction_stall": int(stalled), "response_kind": decision,
                "assistance_offered": int(offered), "assistance_accepted": accepted, "clarification_text": clarification,
                "recording_method": capture.method if capture else "typed", "speech_start_delay_ms": round(start_delay, 1) if start_delay is not None else "",
                "user_speech_duration_ms": round(duration, 1) if duration is not None else "", "end_silence_ms": round(capture.end_silence_ms, 1) if capture and capture.end_silence_ms is not None else "",
                "self_report_confusion_1to5": confusion_score, "self_report_support_1to7": "", "self_report_enjoyment_1to7": "",
                "self_report_continue_1to7": "", "comprehension_correct_out_of_3": ""})
        if rows and not args.skip_survey:
            support = rating("I felt supported by Reachy Mini. (1–7): ", 7)
            enjoyment = rating("The storytelling was enjoyable. (1–7): ", 7)
            continuing = rating("I would like to continue storytelling. (1–7): ", 7)
            correct = 0
            print("Comprehension questions:")
            for question, answer in COMPREHENSION[args.story]:
                response = input(question + " ").strip()
                print(f"  Scoring key: {answer}")
                correct += input("Mark correct? (y/N): ").strip().lower() == "y"
            for row in rows:
                row.update(self_report_support_1to7=support, self_report_enjoyment_1to7=enjoyment,
                           self_report_continue_1to7=continuing, comprehension_correct_out_of_3=correct)

    try:
        if robot_context:
            with robot_context as mini: session(mini)
        else: session(None)
    finally:
        write_csv(out, rows)
        print(f"Saved log: {out}\nNo raw audio, image, or transcript was saved.")
    return out


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--participant", required=True); p.add_argument("--condition", choices=["baseline", "assistance"], required=True)
    p.add_argument("--story", choices=["A", "B"], default="A"); p.add_argument("--input", choices=["typed", "mic"], default="typed")
    p.add_argument("--robot-ip", default=os.environ.get("REACHY_IP", "")); p.add_argument("--no-robot", action="store_true")
    p.add_argument("--demo", action="store_true"); p.add_argument("--no-tts", action="store_true"); p.add_argument("--skip-survey", action="store_true")
    p.add_argument("--record-seconds", type=float, default=7.0); p.add_argument("--end-silence-ms", type=int, default=1000)
    p.add_argument("--stall-threshold-ms", type=float, default=2500); p.add_argument("--vad-aggressiveness", type=int, choices=range(4), default=2)
    p.add_argument("--model", default=os.environ.get("CONFUSION_MODEL", "gpt-4.1-mini")); p.add_argument("--stt-model", default="gpt-4o-mini-transcribe")
    p.add_argument("--tts-model", default="gpt-4o-mini-tts"); p.add_argument("--voice", default="alloy"); p.add_argument("--output-dir", default="logs")
    args = p.parse_args()
    if args.demo: args.no_robot = True
    if not 1 <= args.record_seconds <= 30: p.error("--record-seconds must be 1-30")
    run(args)


if __name__ == "__main__": main()
