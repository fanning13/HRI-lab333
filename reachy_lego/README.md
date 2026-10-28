# Reachy LEGO scanner

The guidance study stays in `guidance_study/`. This package is a separate
scanner: students stack LEGO-like blocks, each with an ArUco marker, and
Reachy plays the actions from top to bottom.

It does not run missions, check answers, or call a model. Marker IDs are
read with OpenCV ArUco. The code shown on screen comes from
`block_library.py` and is never executed.

## Actions

Dictionary `DICT_6X6_250`.

| ID | Label | On screen | Action key |
| ---: | --- | --- | --- |
| 10 | Look left | `head.turn_left()` | `look_left` |
| 11 | Look right | `head.turn_right()` | `look_right` |
| 12 | Nod yes | `head.nod()` | `nod` |
| 13 | Shake head | `head.shake_no()` | `shake_head` |
| 14 | Wiggle antennas | `antennas.wiggle_both()` | `wiggle_antennas` |
| 15 | Twist body left | `body.turn_left()` | `body_twist_left` |
| 16 | Twist body right | `body.turn_right()` | `body_twist_right` |

Repeated blocks may share an ID. Each visible copy is kept, in order from
top to bottom.

## OpenCV

This workspace already has one OpenCV wheel, `opencv-python` 5.0.0, and
that build includes `cv2.aruco.generateImageMarker` and `ArucoDetector`.
Do not also install `opencv-contrib-python`. The two wheels conflict.

On OpenCV 4.7 through 4.x, ArUco lives in the contrib package. Install
`opencv-contrib-python` instead of `opencv-python`, still only one wheel.

## Print the markers

From the repository root, with the project virtualenv active:

```bash
python tools/generate_markers.py
```

Sheets land in `reachy_lego/markers/` as `10_look_left.png` and the rest,
plus `manifest.json`. The default marker square is 600 pixels. Change it
with `--side 800`. Scale the page uniformly in the print dialog if the
block face needs a different size. Do not crop the black border or the
white margin, and do not stretch the square.

Attach each print flat on the camera-facing side. The camera has to see
every marker in the stack. A hidden marker cannot be decoded. The green
outlines in the app are the marker squares ArUco found, not the edges of
the plastic blocks.

## Run

From the repository root:

```bash
python -m reachy_lego --check
python -m unittest reachy_lego.test_lego
python -m reachy_lego --dry-run --image reachy_lego/markers/10_look_left.png
python -m reachy_lego
```

`--check` prints the registry and does not open a window.

The window is Tkinter. Press Scan once. The app freezes that frame, plays
a short scan line, then reads the markers. It shows the labels and the
display syntax, runs each mapped move, plays a short fixed celebration,
and shows “Dance complete!” Scan stays disabled until the run is over.

Reachy is reached the same way as the guidance study: `REACHY_IP` in the
environment or in `guidance_study/.env`, or `--robot-ip`. The address is
not hard-coded. The connection is network mode with the default media
backend. Camera frames are not written to disk.

`--dry-run` decodes the stack and records the action keys it would run.
It does not connect to Reachy and does not move it. Add `--image` or
`--webcam` so Scan has a picture. `--webcam` is this computer's camera,
not Reachy's.

The project `.venv` is Python 3.11.9 from pyenv and was built without Tk,
so `python -m reachy_lego` exits with a short message in that interpreter.
Tests do not need a window. A Python that includes Tk can run the same
module; this machine's pyenv 3.10.13 has Tk 8.5 plus OpenCV and can run
the dry-run window. The Reachy SDK used here is installed in `.venv`.

If a real connection reports `No module named 'gi'`, the GStreamer path
files are hidden because this repo is on an iCloud Desktop. The guidance
study README has the `chflags nohidden` command for this virtualenv.

## What still needs a physical Reachy Mini

The tests check marker generation, ordering, registry lookup, the dry-run
recorder, and the poses sent to `goto_target`. They do not move hardware.

On a robot, confirm:

- The camera preview appears, and Scan uses one frame rather than detecting continuously.
- Look left and twist body left match the labels. The commanded signs follow the course helpers (positive head yaw and positive body yaw are left; positive pitch looks down; antennas are `[right, left]` radians). A live check is still the confirmation.
- The moves feel small. Caps are 18° of head yaw, 12° nod, 0.40 rad antennas, and 0.18 rad body yaw, each at least 0.5 s.
- A stack with a repeated marker runs both copies, then the celebration, then returns to Ready.
- An empty scan or an unknown marker leaves the robot still.
