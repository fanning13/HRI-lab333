"""Tkinter window for the LEGO action scanner."""

from __future__ import annotations

import base64
import queue
import threading
import tkinter as tk

import cv2
import numpy as np

from reachy_lego.block_library import BlockAction
from reachy_lego.frames import fit_frame
from reachy_lego.runner import AppState, RunUpdate, analyze_frame, perform_sequence, scan_allowed


BG = "#f6f4f0"
INK = "#1c1c1c"
MUTED = "#5c584f"
PANEL = "#fffdf9"
STATE_COLOR = {
    AppState.READY: "#0f6e56",
    AppState.SCANNING: "#1d4e89",
    AppState.SHOWING_SEQUENCE: "#1d4e89",
    AppState.RUNNING: "#a15c07",
    AppState.CELEBRATING: "#6d28d9",
    AppState.ERROR: "#9b1c1c",
}


class FrozenFrame:
    """A captured frame the window should animate before showing results."""

    def __init__(self, frame):
        self.frame = frame


class WorkerFinished:
    """The scan thread has exited. The button may be enabled again."""


def bgr_photo(master, bgr):
    """Tk image from a BGR array.

    Tk 8.5 accepts raw PPM bytes. Tk 8.6 accepts base64 PPM, and PNG if
    the build includes it.
    """
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    height, width = rgb.shape[:2]
    ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + rgb.tobytes()
    last_error = None
    for payload in (ppm, base64.b64encode(ppm)):
        try:
            return tk.PhotoImage(master=master, data=payload)
        except tk.TclError as exc:
            last_error = exc
    ok, encoded = cv2.imencode(".png", bgr)
    if ok:
        try:
            return tk.PhotoImage(master=master, data=base64.b64encode(encoded.tobytes()))
        except tk.TclError as exc:
            last_error = exc
    raise RuntimeError(f"Could not prepare the preview image: {last_error}")


class LegoApp(tk.Tk):
    """Ready, scan, sequence, run, celebrate, or error."""

    def __init__(self, source, motion, dry_run: bool, camera_ready: bool, startup_message: str = ""):
        super().__init__()
        self.title("Reachy LEGO")
        self.configure(bg=BG)
        self.source = source
        self.motion = motion
        self.dry_run = dry_run
        self.camera_ready = camera_ready
        self.state = AppState.ERROR if startup_message and not camera_ready else AppState.READY
        self._queue: queue.Queue = queue.Queue()
        self._worker_running = False
        self._closed = False
        self._animation_done = threading.Event()
        self._photo = None
        self.state_log: list[AppState] = [self.state]

        self._build(startup_message)
        self._apply_chrome()
        first = source.latest()
        if first is not None:
            self._show_bgr(first)
        self.after(40, self._poll)
        self.after(90, self._preview_tick)
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _build(self, startup_message: str) -> None:
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=16, pady=(14, 4))
        tk.Label(
            header, text="Reachy LEGO", bg=BG, fg=INK, font=("Helvetica", 22, "bold"),
        ).pack(side="left")
        self.mode_label = tk.Label(header, bg=BG, fg=MUTED, font=("Helvetica", 13))
        self.mode_label.pack(side="right")

        self.status_label = tk.Label(self, bg=BG, fg=INK, font=("Helvetica", 13, "bold"), anchor="w")
        self.status_label.pack(fill="x", padx=16)

        self.message_var = tk.StringVar(value=startup_message or "Press Scan to read the stack. Detection runs once per press.")
        self.message_label = tk.Label(
            self, textvariable=self.message_var, bg=BG, fg=INK,
            font=("Helvetica", 14), anchor="w", justify="left", wraplength=720,
        )
        self.message_label.pack(fill="x", padx=16, pady=(4, 8))

        self.image_frame = tk.Frame(self, width=720, height=420, bg="#171717")
        self.image_frame.pack(padx=16)
        self.image_frame.pack_propagate(False)
        self.image_label = tk.Label(
            self.image_frame,
            bg="#171717",
            fg="#f3f1ec",
            text="Camera preview will appear here.\nMarker detection runs only when you press Scan.",
            font=("Helvetica", 13),
        )
        self.image_label.pack(expand=True, fill="both")

        tk.Label(
            self,
            text="Green outlines are detected ArUco markers, not the whole LEGO block.",
            bg=BG, fg=MUTED, font=("Helvetica", 11), anchor="w",
        ).pack(fill="x", padx=16, pady=(6, 4))

        tk.Label(self, text="Sequence", bg=BG, fg=INK, font=("Helvetica", 13, "bold"), anchor="w").pack(
            fill="x", padx=16,
        )
        self.program = tk.Text(
            self, height=10, width=72, bg=PANEL, fg=INK, relief="flat",
            font=("Menlo", 13), padx=10, pady=8, wrap="word",
        )
        self.program.pack(fill="x", padx=16, pady=(2, 6))
        self.program.tag_configure("current", background="#f3dfb0")
        self.program.configure(state="disabled")
        self._write_program(())

        self.recorded_var = tk.StringVar(value="Dry run records action keys here." if self.dry_run else "")
        tk.Label(
            self, textvariable=self.recorded_var, bg=BG, fg=MUTED,
            font=("Helvetica", 12), anchor="w", justify="left", wraplength=720,
        ).pack(fill="x", padx=16)

        self.scan_button = tk.Button(
            self, text="Scan", command=self._on_scan,
            font=("Helvetica", 16, "bold"), padx=28, pady=8,
        )
        self.scan_button.pack(pady=(8, 16))

    def _close(self) -> None:
        self._closed = True
        self._animation_done.set()
        try:
            self.source.stop()
        except Exception:
            pass
        self.destroy()

    def _set_state(self, state: AppState) -> None:
        self.state = state
        self.state_log.append(state)
        self._apply_chrome()

    def _apply_chrome(self) -> None:
        color = STATE_COLOR[self.state]
        self.status_label.configure(text=self.state.value, fg=color)
        mode = "Dry run — Reachy will not move" if self.dry_run else "Connected to Reachy"
        if not self.camera_ready:
            mode = "Camera unavailable"
        self.mode_label.configure(text=mode)
        allowed = scan_allowed(self.state, self.camera_ready) and not self._worker_running
        self.scan_button.configure(state="normal" if allowed else "disabled")

    def _set_message(self, text: str, state: AppState | None = None) -> None:
        self.message_var.set(text)
        if text == "Dance complete!":
            self.message_label.configure(font=("Helvetica", 22, "bold"), fg=STATE_COLOR[AppState.CELEBRATING])
        elif state == AppState.ERROR:
            self.message_label.configure(font=("Helvetica", 14), fg=STATE_COLOR[AppState.ERROR])
        else:
            self.message_label.configure(font=("Helvetica", 14), fg=INK)

    def _show_bgr(self, frame) -> None:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        shown = fit_frame(frame)
        self._photo = bgr_photo(self, shown)
        self.image_label.configure(image=self._photo, text="")
        self.image_label.image = self._photo

    def _write_program(self, steps: tuple[BlockAction, ...] | list[BlockAction], current: int | None = None) -> None:
        self.program.configure(state="normal")
        self.program.delete("1.0", "end")
        if not steps:
            self.program.insert("end", "The scanned sequence will appear here.")
        for index, step in enumerate(steps):
            start = self.program.index("end-1c")
            self.program.insert("end", f"{index + 1}. {step.label}\n    {step.syntax}\n\n")
            if current == index:
                self.program.tag_add("current", start, self.program.index("end-1c"))
        self.program.configure(state="disabled")

    def _recorded_text(self, performed: tuple[str, ...], celebrated: bool) -> str:
        if not performed and not celebrated:
            return "Dry run records action keys here." if self.dry_run else ""
        text = "Action keys: " + ", ".join(performed)
        if celebrated:
            text += ". Celebration recorded (not a block)."
        elif self.dry_run:
            text += "."
        return text

    def _preview_tick(self) -> None:
        if self._closed:
            return
        if (
            self.state == AppState.READY
            and not self._worker_running
            and getattr(self.source, "live", False)
        ):
            frame = self.source.latest()
            if frame is not None:
                self._show_bgr(frame)
            elif self.source.error():
                self._set_message(self.source.error(), AppState.ERROR)
        self.after(90, self._preview_tick)

    def _on_scan(self) -> None:
        if not scan_allowed(self.state, self.camera_ready) or self._worker_running:
            return
        self._worker_running = True
        self._animation_done = threading.Event()
        self._set_state(AppState.SCANNING)
        self._set_message("Scanning the stack.")
        self._write_program(())
        self.recorded_var.set("Dry run records action keys here." if self.dry_run else "")
        threading.Thread(target=self._scan_worker, name="lego-scan", daemon=True).start()

    def _scan_worker(self) -> None:
        try:
            frame = self.source.capture(timeout=8)
            if frame is None:
                self._queue.put(
                    RunUpdate(
                        AppState.ERROR,
                        "The camera did not return a frame. Scan again when a picture is visible.",
                    )
                )
                return
            self._queue.put(FrozenFrame(fit_frame(frame)))
            analysis = analyze_frame(frame)
            self._animation_done.wait(timeout=4)
            if self._closed:
                return
            if not analysis.ok:
                self._queue.put(RunUpdate(AppState.ERROR, analysis.message, annotated=analysis.annotated))
                return
            perform_sequence(analysis, self.motion, on_update=self._queue.put)
        except Exception as exc:
            self._queue.put(RunUpdate(AppState.ERROR, f"Scan failed: {exc}"))
        finally:
            self._queue.put(WorkerFinished())

    def _animate(self, frame, step: int, steps: int) -> None:
        if self._closed:
            self._animation_done.set()
            return
        shown = frame.copy()
        y = int((shown.shape[0] - 1) * step / max(steps - 1, 1))
        cv2.line(shown, (0, y), (shown.shape[1] - 1, y), (40, 170, 255), 2)
        self._show_bgr(shown)
        if step + 1 < steps:
            self.after(28, lambda: self._animate(frame, step + 1, steps))
        else:
            self._animation_done.set()

    def _apply_update(self, update: RunUpdate) -> None:
        self._set_state(update.state)
        if update.annotated is not None and update.state != AppState.SCANNING:
            self._show_bgr(update.annotated)
        self._write_program(update.steps, update.current_index)
        if update.state == AppState.RUNNING and update.current_index is not None:
            step = update.steps[update.current_index]
            self._set_message(f"Running: {step.label}    {step.syntax}")
        elif update.message:
            self._set_message(update.message, update.state)
        self.recorded_var.set(self._recorded_text(update.performed, update.celebrated))

    def _poll(self) -> None:
        if self._closed:
            return
        try:
            while True:
                item = self._queue.get_nowait()
                if isinstance(item, FrozenFrame):
                    self._animate(item.frame, 0, 24)
                elif isinstance(item, WorkerFinished):
                    self._worker_running = False
                    self._apply_chrome()
                elif isinstance(item, RunUpdate):
                    self._apply_update(item)
        except queue.Empty:
            pass
        self.after(30, self._poll)


def needs_drawn_window() -> bool:
    """Apple's Tk 8.5 paints native buttons and leaves every other widget blank."""
    return tk.TkVersion < 8.6


VIEW_W = 960
PREVIEW_W = 920
PREVIEW_H = 420


def compose_view(
    state: AppState,
    message: str,
    mode: str,
    frame,
    steps: tuple[BlockAction, ...] | list[BlockAction],
    current: int | None,
    recorded: str,
    scan_enabled: bool,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Draw the scanner as pixels.

    Tk 8.5 on current macOS shows the Scan button and nothing else, so the
    live window is an image: title, status, camera, sequence, and button.
    """
    header_h = 76
    message_h = 58
    preview_h = PREVIEW_H
    sequence_h = 210
    button_h = 78
    height = header_h + message_h + preview_h + sequence_h + button_h
    image = np.full((height, VIEW_W, 3), (240, 244, 246), dtype=np.uint8)

    state_bgr = {
        AppState.READY: (86, 110, 15),
        AppState.SCANNING: (137, 78, 29),
        AppState.SHOWING_SEQUENCE: (137, 78, 29),
        AppState.RUNNING: (7, 92, 161),
        AppState.CELEBRATING: (217, 40, 109),
        AppState.ERROR: (28, 28, 155),
    }[state]
    image[:header_h] = state_bgr
    _text(image, "Reachy LEGO", (24, 48), 1.15, (255, 255, 255), 2)
    _text(image, state.value, (VIEW_W - 24 - 18 * len(state.value), 36), 0.8, (255, 255, 255), 2)
    _text(image, mode, (VIEW_W - 24 - 11 * len(mode), 62), 0.5, (255, 255, 255), 1)

    shown = message or "Hold the stack in front of Reachy, then press Scan."
    for index, line in enumerate(_wrap(shown, 78)[:2]):
        color = (28, 28, 155) if state == AppState.ERROR else (28, 28, 28)
        _text(image, line, (24, header_h + 28 + index * 24), 0.62, color, 1)

    preview_top = header_h + message_h
    image[preview_top:preview_top + preview_h] = (23, 23, 23)
    if frame is not None:
        shown_frame = fit_frame(frame, PREVIEW_W, preview_h - 16)
        y0 = preview_top + (preview_h - shown_frame.shape[0]) // 2
        x0 = (VIEW_W - shown_frame.shape[1]) // 2
        image[y0:y0 + shown_frame.shape[0], x0:x0 + shown_frame.shape[1]] = shown_frame
    else:
        _text(
            image,
            "Waiting for Reachy's camera...",
            (250, preview_top + preview_h // 2),
            0.8,
            (236, 241, 243),
            2,
        )
    if state == AppState.SCANNING:
        _text(image, "SCANNING", (24, preview_top + 42), 1.1, (255, 190, 40), 2)

    seq_top = preview_top + preview_h
    _text(image, "Sequence", (24, seq_top + 32), 0.7, (28, 28, 28), 2)
    if not steps:
        _text(image, "The scanned sequence will appear here.", (24, seq_top + 64), 0.55, (79, 88, 92), 1)
    for index, step in enumerate(steps[:6]):
        y = seq_top + 64 + index * 22
        if current == index:
            image[y - 16:y + 6, 16:VIEW_W - 16] = (176, 223, 243)
        mark = ">" if current == index else " "
        _text(image, f"{mark} {index + 1}. {step.label}    {step.syntax}", (24, y), 0.55, (28, 28, 28), 1)
    if recorded:
        _text(image, _wrap(recorded, 90)[0], (24, seq_top + sequence_h - 16), 0.48, (79, 88, 92), 1)

    button_top = seq_top + sequence_h
    if state == AppState.SCANNING:
        label = "Scanning..."
    elif scan_enabled:
        label = "Scan"
    else:
        label = "Please wait"
    button_w, button_h_px = 280, 46
    x0 = (VIEW_W - button_w) // 2
    y0 = button_top + (button_h - button_h_px) // 2
    x1, y1 = x0 + button_w, y0 + button_h_px
    fill = (255, 255, 255) if scan_enabled else (196, 196, 196)
    image[y0:y1, x0:x1] = fill
    image[y0:y0 + 2, x0:x1] = (28, 28, 28)
    image[y1 - 2:y1, x0:x1] = (28, 28, 28)
    image[y0:y1, x0:x0 + 2] = (28, 28, 28)
    image[y0:y1, x1 - 2:x1] = (28, 28, 28)
    text_x = x0 + max(16, (button_w - int(len(label) * 16)) // 2)
    _text(image, label, (text_x, y0 + 31), 0.8, (28, 28, 28), 2)
    return image, (x0, y0, x1, y1)


def _text(image, text: str, origin: tuple[int, int], scale: float, color: tuple[int, int, int], thickness: int) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if len(trial) <= width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


class DrawnWindow:
    """Camera, status, sequence, and Scan control drawn into one picture."""

    def __init__(self, source, motion, dry_run: bool, camera_ready: bool, startup_message: str = ""):
        self.source = source
        self.motion = motion
        self.dry_run = dry_run
        self.camera_ready = camera_ready
        self.state = AppState.ERROR if startup_message and not camera_ready else AppState.READY
        self._message = startup_message or "Hold the stack in front of Reachy, then press Scan."
        self._steps: tuple[BlockAction, ...] = ()
        self._current: int | None = None
        self._recorded = "Dry run records action keys here." if dry_run else ""
        self._still = None
        self._queue: queue.Queue = queue.Queue()
        self._worker_running = False
        self._closed = False
        self._animation_done = threading.Event()
        self._anim_frame = None
        self._anim_step = 0
        self._button = (0, 0, 0, 0)
        self._scan_click = False
        self.state_log: list[AppState] = [self.state]

    def _mode(self) -> str:
        if not self.camera_ready:
            return "Camera unavailable"
        if self.dry_run:
            return "Dry run. Reachy will not move"
        return "Connected to Reachy"

    def _on_mouse(self, event, x, y, _flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        x0, y0, x1, y1 = self._button
        if x0 <= x < x1 and y0 <= y < y1:
            self._scan_click = True

    def _on_scan(self) -> None:
        if not scan_allowed(self.state, self.camera_ready) or self._worker_running:
            return
        self._worker_running = True
        self._animation_done = threading.Event()
        self._set_state(AppState.SCANNING)
        self._message = "Scanning the stack."
        self._steps = ()
        self._current = None
        self._recorded = "Dry run records action keys here." if self.dry_run else ""
        self._still = None
        print("SCANNING", flush=True)
        threading.Thread(target=self._scan_worker, name="lego-scan", daemon=True).start()

    def _scan_worker(self) -> None:
        try:
            frame = self.source.capture(timeout=8)
            if frame is None:
                self._queue.put(
                    RunUpdate(
                        AppState.ERROR,
                        "The camera did not return a frame. Scan again when a picture is visible.",
                    )
                )
                return
            self._queue.put(FrozenFrame(fit_frame(frame, PREVIEW_W, PREVIEW_H - 16)))
            analysis = analyze_frame(frame)
            self._animation_done.wait(timeout=4)
            if self._closed:
                return
            if not analysis.ok:
                self._queue.put(RunUpdate(AppState.ERROR, analysis.message, annotated=analysis.annotated))
                return
            perform_sequence(analysis, self.motion, on_update=self._queue.put)
        except Exception as exc:
            self._queue.put(RunUpdate(AppState.ERROR, f"Scan failed: {exc}"))
        finally:
            self._queue.put(WorkerFinished())

    def _set_state(self, state: AppState) -> None:
        self.state = state
        self.state_log.append(state)

    def _apply_update(self, update: RunUpdate) -> None:
        self._set_state(update.state)
        if update.annotated is not None and update.state != AppState.SCANNING:
            self._still = update.annotated
        if update.state == AppState.READY:
            self._still = None
        self._steps = tuple(update.steps)
        self._current = update.current_index
        if update.state == AppState.RUNNING and update.current_index is not None:
            step = update.steps[update.current_index]
            self._message = f"Running: {step.label}    {step.syntax}"
        elif update.message:
            self._message = update.message
        elif update.state == AppState.SHOWING_SEQUENCE:
            count = len(update.steps)
            self._message = f"Found {count} action{'s' if count != 1 else ''}. Reachy plays them from top to bottom."
        performed = update.performed
        if performed or update.celebrated:
            text = "Action keys: " + ", ".join(performed)
            if update.celebrated:
                text += ". Celebration recorded (not a block)."
            self._recorded = text
        print(f"{update.state.value}: {self._message}", flush=True)

    def _drain(self) -> None:
        try:
            while True:
                item = self._queue.get_nowait()
                if isinstance(item, FrozenFrame):
                    self._anim_frame = item.frame
                    self._anim_step = 0
                elif isinstance(item, WorkerFinished):
                    self._worker_running = False
                elif isinstance(item, RunUpdate):
                    self._apply_update(item)
        except queue.Empty:
            pass

    def _preview(self):
        if self._anim_frame is not None:
            shown = self._anim_frame.copy()
            steps = 24
            y = int((shown.shape[0] - 1) * self._anim_step / max(steps - 1, 1))
            cv2.line(shown, (0, y), (shown.shape[1] - 1, y), (40, 170, 255), 2)
            self._anim_step += 1
            if self._anim_step >= steps:
                self._anim_frame = None
                self._animation_done.set()
            return shown
        if self.state == AppState.READY and not self._worker_running and getattr(self.source, "live", False):
            return self.source.latest()
        if self._still is not None:
            return self._still
        return self.source.latest()

    def run(self) -> None:
        title = "Reachy LEGO"
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, self._on_mouse)
        print("Reachy LEGO window is open. Press Scan, or the S key. Esc closes it.", flush=True)
        seen = False
        try:
            while not self._closed:
                self._drain()
                if self._scan_click:
                    self._scan_click = False
                    self._on_scan()
                frame = self._preview()
                if frame is None and self.source.error() and self.state != AppState.SCANNING:
                    self._message = self.source.error()
                image, self._button = compose_view(
                    self.state,
                    self._message,
                    self._mode(),
                    frame,
                    self._steps,
                    self._current,
                    self._recorded,
                    scan_allowed(self.state, self.camera_ready) and not self._worker_running,
                )
                cv2.imshow(title, image)
                key = cv2.waitKey(30)
                if key != -1 and (key & 0xFF) in (27, ord("q")):
                    break
                if key != -1 and (key & 0xFF) in (ord("s"), ord("S")):
                    self._on_scan()
                try:
                    visible = cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE)
                except cv2.error:
                    break
                if visible >= 1:
                    seen = True
                elif seen:
                    break
        finally:
            self._closed = True
            self._animation_done.set()
            try:
                self.source.stop()
            except Exception:
                pass
            cv2.destroyAllWindows()


def launch_gui(source, motion, dry_run: bool, camera_ready: bool, startup_message: str = "") -> None:
    if needs_drawn_window():
        DrawnWindow(source, motion, dry_run, camera_ready, startup_message).run()
        return
    app = LegoApp(source, motion, dry_run, camera_ready, startup_message)
    app.mainloop()
