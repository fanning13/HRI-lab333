"""Frame sources. Preview grabbing is separate from marker detection."""

from __future__ import annotations

import threading
import time

import cv2


def fit_frame(frame, max_w: int = 720, max_h: int = 420):
    """Shrink a frame for display, keeping the aspect ratio."""
    height, width = frame.shape[:2]
    scale = min(max_w / float(width), max_h / float(height), 1.0)
    if scale >= 0.999:
        return frame
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def load_image(path: str):
    frame = cv2.imread(path)
    if frame is None:
        raise RuntimeError(f"Could not read image {path}")
    return frame


class StillFrames:
    """One image, used by --image. Not a live preview."""

    live = False

    def __init__(self, frame):
        self._frame = frame
        self._error = None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def error(self) -> str | None:
        return self._error

    def latest(self):
        return None if self._frame is None else self._frame.copy()

    def capture(self, timeout: float = 0.0):
        return self.latest()


class UnavailableFrames:
    """No camera. Scan stays disabled."""

    live = False

    def __init__(self, message: str):
        self._message = message

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def error(self) -> str | None:
        return self._message

    def latest(self):
        return None

    def capture(self, timeout: float = 0.0):
        return None


class _LiveFrames:
    """Background grabber. ``capture`` only copies the newest frame."""

    live = True

    def __init__(self):
        self._lock = threading.Lock()
        self._latest = None
        self._error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="lego-preview", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def error(self) -> str | None:
        with self._lock:
            return self._error

    def latest(self):
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def capture(self, timeout: float = 8.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame = self.latest()
            if frame is not None:
                return frame
            time.sleep(0.05)
        message = self.error() or "The camera did not return a frame."
        raise RuntimeError(message)

    def _store(self, frame, error: str | None) -> None:
        with self._lock:
            if frame is not None:
                self._latest = frame
                self._error = None
            elif error:
                self._error = error

    def _loop(self) -> None:
        raise NotImplementedError


class ReachyFrames(_LiveFrames):
    """Reachy camera preview. This loop does not run marker detection."""

    def __init__(self, mini):
        super().__init__()
        self._mini = mini

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._mini.media.get_frame()
            except Exception as exc:
                self._store(None, f"Camera error: {exc}")
                self._stop.wait(0.4)
                continue
            if frame is not None:
                self._store(frame, None)
            self._stop.wait(0.05)


class WebcamFrames(_LiveFrames):
    """This computer's webcam, for a dry run without Reachy's camera."""

    def __init__(self, index: int = 0):
        super().__init__()
        self._index = index
        self._cap = None

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            self._store(None, "Could not open the webcam.")
            self._cap.release()
            self._cap = None
            return
        super().start()

    def stop(self) -> None:
        super().stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _loop(self) -> None:
        cap = self._cap
        if cap is None:
            return
        while not self._stop.is_set():
            ok, frame = cap.read()
            if ok and frame is not None:
                self._store(frame, None)
            else:
                self._store(None, "The webcam did not return a frame.")
            self._stop.wait(0.03)
