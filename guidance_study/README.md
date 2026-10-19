# Guidance study

How do participants perceive Reachy Mini's guidance when it offers hints at fixed checkpoints, compared with when it uses a camera cue to decide when to offer support?

Each person completes two fictional mysteries, one in each condition. The camera-informed condition changes both camera access and the moment support is offered, so a difference is a difference between the two strategies.

## Study design

| | |
| --- | --- |
| Cases | The Ashcombe Star and The Briarwick Watch |
| Turns | Six per case. Checkpoints are turns 2, 4, and 6. Turns 4 and 6 add no new narration. |
| Input | The participant types every choice, explanation, revision, and rating. |
| Reachy | Speaks the story, questions, hint offers, hints, the neutral line, ratings, and the closing. The same text is shown in the terminal. |

**Fixed-point.** Reachy does not capture a frame. At each checkpoint it asks "Would a hint help?" and speaks that checkpoint's hint only if the person accepts.

**Camera-informed.** As soon as the question is asked, Reachy captures one frame and saves it in `guidance_study/frames/`. While the participant types the initial choice and explanation, the vision model judges whether the person appears confused. That judgment is an uncertain cue, and Reachy does not say it aloud. After the explanation is submitted:

- `offer_hint` asks "Would a hint help?" and speaks the hint only if accepted.
- `neutral_encouragement` says "Take your time—you're making progress."
- `no_nudge` adds no guidance line.

A missing frame, an unclear cue, or a failed model call is logged and uses the neutral line. It does not offer a camera-based hint. A typed "I don't know," "not sure," or "confused" still asks "Would a hint help?" In either condition, a typed request that contains help, hint, clue, or stuck speaks that checkpoint's hint once.

Both conditions then ask whether to keep or change the answer, collect the final choice and explanation, and ask for a confusion rating from 1 to 5 about the moment before Reachy responded. After each case, Reachy asks three ratings: helpfulness, timing, and match to when the person wanted help.

The program scores the letter choice against the answer key. The explanation score is secondary: 1 when the final explanation contains that checkpoint's evidence check, otherwise 0. The scoring rule is not shown during the session. Case time runs from the case introduction through the turn-6 final explanation.

Orders cross case and condition. `P01` is order 1. The four orders are in `counterbalancing.md`. Full wording and the pilot checklist are in `protocol.md`.

## Files

| File | Role |
| --- | --- |
| `task_materials.json` | Cases, checkpoints, hints, orders, and spoken lines |
| `protocol.md` | Procedure and canonical wording |
| `counterbalancing.md` | The four orders |
| `run_study.py` | Session runner |
| `models.py` | Vision and reasoning requests |
| `robot_io.py` | Camera frame and Reachy speech |
| `test_run_study.py` | Mocked tests |
| `requirements.txt` | Packages |

## Set up

Use Python 3.11. From the repo root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r guidance_study/requirements.txt
```

Install [Ollama](https://ollama.com) and pull the local models:

```bash
ollama pull llama3.2
ollama pull gemma3:4b
```

`llama3.2` chooses the dialogue action. `gemma3:4b` reads the camera frame. The fixed-point condition calls neither model.

Create `guidance_study/.env`. It is gitignored. Speech on a real run uses Reachy's speaker through `gpt-4o-mini-tts`, voice `alloy`. Change those with `--tts-model` and `--voice`.

```bash
OPENAI_API_KEY=your-key
REACHY_IP=192.168.8.199
```

`REACHY_IP` is optional. `--api` sends the vision and reasoning calls to `gpt-6-luna` as well. Do not commit `.env`, `logs/`, `frames/`, or generated `.wav` files. The `frames/` folder holds participant images.

## Run

From `guidance_study`, with `.venv` active.

Check the task file and the mocked tests. These do not use the robot, camera, or a model API.

```bash
python -m unittest test_run_study.py
python run_study.py --check
```

Walk through a session on screen only. The log is marked simulated.

```bash
python run_study.py --participant P01 --order 1 --simulate
```

`--simulate-cue unclear` and `--simulate-action offer_hint` try the other camera branches. Type `q` to stop.

Check the live robot, speaker, camera, and both models once. Reachy says "This is a Reachy component check." The frame is not saved.

```bash
python run_study.py --components
```

Run a session. The first run prepares the spoken lines and stores them in `speech_cache/`. Later runs reuse them. Type answers in the terminal. `P01` hears Ashcombe Star with fixed-point hints, then Briarwick Watch with the camera.

```bash
python run_study.py --participant P01 --order 1
```

Confirm the Reachy daemon first: `curl http://192.168.8.199:8000/api/daemon/status`. Use `--order 2`, `3`, or `4` for the other assignments, and a new participant ID for each person. Use `--robot-ip` when the robot is not at `192.168.8.199`.

`No Reachy Mini Audio USB device found` means this computer does not have the robot's USB microphone. Speech is sent over the network. A line was not played only if the terminal prints `Speech failed`.

If Python reports `No module named 'gi'`, the GStreamer path files are hidden. This happens when the repo is on an iCloud Desktop. Clear the flag and start in one command:

```bash
chflags nohidden \
  "../.venv/lib/python3.11/site-packages/gstreamer_bundle.pth" \
  "../.venv/lib/python3.11/site-packages/gstreamer_ext_runtime.pth" && \
python run_study.py --participant P01 --order 1
```

Use the same prefix for `--components`. Logs go to `guidance_study/logs/` and are gitignored because they contain answers. Use the pilot checklist in `protocol.md` before treating a session as data.
