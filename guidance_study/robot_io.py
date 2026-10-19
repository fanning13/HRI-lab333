"""Reachy camera frame and TTS, following lab3/common.py.

Lab 3 encodes one BGR frame as a JPEG data URL and, in API mode, writes
a wav with gpt-4o-mini-tts, then calls play_sound. play_sound returns
before the speaker finishes, so this study waits for the wav duration
before the next prompt. Frames are encoded in memory and are not written
to disk.
"""

from __future__ import annotations

import hashlib
import textwrap
import time
import wave
from concurrent.futures import ThreadPoolExecutor
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


def save_frame(frame, directory: Path, name: str) -> Path | None:
    """Write one camera frame as a JPEG. Non-image objects are skipped."""
    shape = getattr(frame, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    import cv2

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.jpg"
    if not cv2.imwrite(str(path), frame):
        return None
    return path


def get_frame(mini):
    """Grab one BGR frame. Lab 3 polls because the stream can start late."""
    for _ in range(80):
        frame = mini.media.get_frame()
        if frame is not None:
            return frame
        time.sleep(0.25)
    return None


def wav_duration(path: Path) -> float:
    """Seconds of audio in a wav file.

    Streamed speech files leave the length fields blank. The wave reader
    then reports a huge frame count, so the wait uses the bytes on disk.
    """
    size = path.stat().st_size
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate() or 1
        width = handle.getsampwidth() or 1
        channels = handle.getnchannels() or 1
        frames = handle.getnframes()
    bytes_per_frame = width * channels
    frames_on_disk = max(0, size - 44) // bytes_per_frame
    if frames > frames_on_disk:
        frames = frames_on_disk
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


def speech_cache_path(cache_dir: Path, tts_model: str, voice: str, text: str) -> Path:
    """One file per model, voice, and exact spoken line."""
    digest = hashlib.sha256(f"{tts_model}\n{voice}\n{text}".encode()).hexdigest()[:16]
    return cache_dir / f"{digest}.wav"


def _write_speech(client, text: str, tts_model: str, voice: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    with client.audio.speech.with_streaming_response.create(
        model=tts_model,
        voice=voice,
        input=text,
        response_format="wav",
    ) as response:
        response.stream_to_file(partial)
    partial.replace(path)


class SpeechCache:
    """Reuse prepared wav files so a line is not synthesized again at playback time."""

    def __init__(self, client, tts_model: str, voice: str, cache_dir: Path):
        self.client = client
        self.tts_model = tts_model
        self.voice = voice
        self.cache_dir = cache_dir
        self.remote: dict[str, str] = {}

    def path_for(self, text: str) -> Path:
        return speech_cache_path(self.cache_dir, self.tts_model, self.voice, text)

    def ensure(self, text: str) -> Path:
        path = self.path_for(text)
        if path.is_file() and path.stat().st_size > 44:
            return path
        _write_speech(self.client, text, self.tts_model, self.voice, path)
        return path

    def prepare(self, lines: list[str]) -> None:
        """Synthesize any missing lines before the participant hears the first one."""
        missing = []
        seen = set()
        for line in lines:
            if line in seen:
                continue
            seen.add(line)
            path = self.path_for(line)
            if not path.is_file() or path.stat().st_size <= 44:
                missing.append(line)
        if not missing:
            return
        print(f"Preparing {len(missing)} spoken lines so playback can start with the text.")
        workers = min(4, len(missing))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self.ensure, line) for line in missing]
            for future in futures:
                future.result()

    def upload(self, mini, lines: list[str]) -> None:
        """Send this session's files once. Later playback skips the upload."""
        audio = getattr(getattr(mini, "media", None), "audio", None)
        upload = getattr(audio, "upload_sound", None)
        if upload is None:
            return
        for line in dict.fromkeys(lines):
            path = self.path_for(line)
            if not path.is_file():
                continue
            try:
                remote = upload(str(path))
            except Exception as exc:
                print(f"Speech upload skipped: {redact(str(exc))}")
                continue
            self.remote[str(path)] = remote

    def speak(self, mini, text: str) -> bool:
        """Show the line and start the same text on Reachy. False means it was not delivered."""
        if mini is None or self.client is None:
            show_reachy(text)
            print("Speech failed: Reachy playback was not available")
            return False
        try:
            path = self.ensure(text)
        except Exception as exc:
            show_reachy(text)
            print(f"Speech failed: {redact(str(exc))}")
            return False
        show_reachy(text)
        try:
            target = self.remote.get(str(path), str(path))
            mini.media.play_sound(target)
            time.sleep(wav_duration(path) + 0.2)
            return True
        except Exception as exc:
            print(f"Speech failed: {redact(str(exc))}")
            return False


def speak(mini, client, text: str, tts_model: str, voice: str, cache_dir: Path | None) -> bool:
    """Print the line and play the same text on Reachy. False means it was not delivered."""
    if cache_dir is None:
        show_reachy(text)
        print("Speech failed: Reachy playback was not available")
        return False
    return SpeechCache(client, tts_model, voice, cache_dir).speak(mini, text)


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
