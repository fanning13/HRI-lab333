# Reachy LEGO scanner

Students stack LEGO-like blocks. Each block has an ArUco marker. Press **Scan** once and Reachy Mini plays those actions from top to bottom, then does a short celebration.

This package does not change the guidance study. It does not run missions, score answers, or call a model. The syntax on screen comes from `block_library.py` and is never executed.

## What you need

- The Reachy Mini on the same network as your computer. Its daemon is already running on the robot. Do not start a daemon on your laptop.
- Python 3.10 or newer that includes Tk. The repo `.venv` is Python 3.11.9 built without Tk, so leave that virtualenv deactivated for this program.
- One OpenCV install that includes ArUco. Do not install both `opencv-python` and `opencv-contrib-python`.

## 1. Install

From the repository root, with `.venv` **not** active:

```bash
python -c "import tkinter; print(tkinter.TkVersion)"
python -m pip install -r reachy_lego/requirements.txt
python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'aruco'))"
```

The Tk check must print a version, not an import error. The OpenCV check must print `True`.

`reachy-mini` 1.12 matches the robot and requires Python 3.11 or newer. On Python 3.10, pip installs 1.8.4 instead. The program still runs. The terminal will warn that the SDK and daemon versions differ.

If `hasattr(cv2, 'aruco')` is `False` and you do not already have `opencv-python` installed, install `opencv-contrib-python`. If `opencv-python` is already installed, do not add the second wheel.

## 2. Point at the robot

`guidance_study/.env` is gitignored. Add the robot’s address there, or pass it on the command line. Do not commit that file.

```bash
REACHY_IP=192.168.8.199
```

`192.168.8.199` is the lab Reachy used with the guidance study. If your robot is different, use its address. Confirm the daemon before opening the scanner:

```bash
curl http://192.168.8.199:8000/api/daemon/status
```

You can also skip the env file and run `python -m reachy_lego --robot-ip 192.168.8.199`.

## 3. Run

From the repository root, not from inside `reachy_lego/`:

```bash
python -m reachy_lego
```

The terminal prints `Connecting to Reachy at ...` first. The window opens after that connection finishes. It is titled **Reachy LEGO** and shows the robot’s camera.

1. Stack the printed markers so Reachy’s camera can see every square. A covered marker is skipped.
2. Press **Scan**, or the **S** key. Holding a marker in view does not start a scan.
3. The window says **SCANNING**, then lists the actions from top to bottom and Reachy plays them.
4. Reachy celebrates and the window says **Dance complete!**
5. Press **Esc**, or close the window, to quit. **Ctrl+C** in the terminal also stops it.

Only one computer should connect at a time.

These lines are normal and do not mean the scan failed:

- `No Reachy Mini Audio USB device found` — this scanner does not use the microphone.
- `SDK=1.8.4, daemon=1.12.0` — expected when the interpreter is Python 3.10.

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

Two blocks may use the same ID. Each visible copy is kept, ordered from top to bottom. An empty view or an unknown marker does not move the robot.

## Print the markers

Sheets are already in `reachy_lego/markers/` (`10_look_left.png` through `16_body_twist_right.png`). Print at 100% scale. Do not crop the black border or the white margin, and do not stretch the square. Tape each sheet flat on the side that faces the camera.

Regenerate them from the repository root if you need a different size:

```bash
python tools/generate_markers.py
python tools/generate_markers.py --side 800
```

## Without the robot

```bash
python -m reachy_lego --check
python -m unittest reachy_lego.test_lego
python -m reachy_lego --dry-run --image reachy_lego/markers/10_look_left.png
```

`--check` prints the registry and exits. `--dry-run` does not connect and does not move Reachy. Add `--image` or `--webcam` so Scan has a picture. `--webcam` is this computer’s camera, not Reachy’s.
