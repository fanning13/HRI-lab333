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


def launch_gui(source, motion, dry_run: bool, camera_ready: bool, startup_message: str = "") -> None:
    app = LegoApp(source, motion, dry_run, camera_ready, startup_message)
    app.mainloop()
