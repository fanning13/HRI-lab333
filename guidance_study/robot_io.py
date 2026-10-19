"""Reachy camera frame and TTS, following lab3/common.py.

Lab 3 encodes one BGR frame as a JPEG data URL and, in API mode, writes
a wav with gpt-4o-mini-tts, then calls play_sound. play_sound returns
before the speaker finishes, so this study waits for the wav duration
before the next prompt. Frames are encoded in memory and are not written
to disk.
"""

from __future__ import annotations

import textwrap
import time
import wave
from pathlib import Path

from models import redact

LINE_WIDTH = 72


def image_content(frame) -> list[dict]:
    """Return the image half of a chat message. Does not save the frame."""
    import base64

    import cv2

    image = frame
    height, width = image.shape[:2]
    if width > 640:
        image = cv2.resize(image, (640, height * 640 // width))
    ok, jpg = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Could not encode the camera frame")
    encoded = base64.b64encode(jpg.tobytes()).decode("ascii")
    return [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}]


def get_frame(mini):
    """Grab one BGR frame. Lab 3 polls because the stream can start late."""
    for _ in range(80):
        frame = mini.media.get_frame()
        if frame is not None:
            return frame
        time.sleep(0.25)
    return None


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate() or 1
    return frames / float(rate)


def _split_question(text: str) -> tuple[str, list[str]]:
    """Separate a spoken A/B question for display. The spoken line stays intact."""
    if " A. " not in text or " B. " not in text:
        return text, []
    question, rest = text.split(" A. ", 1)
    option_a, option_b = rest.split(" B. ", 1)
    return question.strip(), [f"A. {option_a.strip()}", f"B. {option_b.strip()}"]


def show_reachy(text: str) -> None:
    """Print one Reachy line wrapped to a readable width."""
    question, options = _split_question(text)
    print()
    print("Reachy")
    paragraphs = [part.strip() for part in question.split("\n") if part.strip()]
    for index, paragraph in enumerate(paragraphs):
        print(textwrap.fill(paragraph, width=LINE_WIDTH, initial_indent="  ", subsequent_indent="  "))
        if options or index < len(paragraphs) - 1:
            print()
    for option in options:
        print(textwrap.fill(option, width=LINE_WIDTH, initial_indent="  ", subsequent_indent="     "))


def speak(mini, client, text: str, tts_model: str, voice: str, wav_path: Path | None) -> bool:
    """Print the line and play the same text on Reachy. False means it was not delivered."""
    show_reachy(text)
    if mini is None or client is None or wav_path is None:
        print("Speech failed: Reachy playback was not available")
        return False
    try:
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        with client.audio.speech.with_streaming_response.create(
            model=tts_model,
            voice=voice,
            input=text,
            response_format="wav",
        ) as response:
            response.stream_to_file(wav_path)
        mini.media.play_sound(str(wav_path))
        time.sleep(wav_duration(wav_path) + 0.2)
        return True
    except Exception as exc:
        print(f"Speech failed: {redact(str(exc))}")
        return False


def connect_robot(host: str):
    """Open the network media connection Lab 3 uses for the camera and speaker.

    The SDK has no speaker-only backend: no_media also disables TTS, and the
    deprecated no-video aliases now resolve to full local media. Callers must
    still refrain from requesting frames when the procedure forbids capture.
    """
    from reachy_mini import ReachyMini
    from reachy_mini.utils import create_head_pose

    mini = ReachyMini(host=host, connection_mode="network", media_backend="default", timeout=10)
    mini.enable_motors()
    pose = create_head_pose(yaw=0, pitch=0, roll=0, degrees=True)
    mini.goto_target(head=pose, duration=0.5)
    mini.goto_target(antennas=[0, 0], duration=0.3)
    return mini
