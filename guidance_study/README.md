# Guidance study

Within-participant comparison of two ways Reachy Mini guides a fictional mystery: a hint offer at every checkpoint, or a camera-informed choice among a hint offer, a neutral line, and no nudge.

The camera-informed condition changes both camera access and the trigger for adaptive support. The comparison tests two guidance strategies, not the isolated causal effect of camera use.

Lab 3 is not modified. This folder copies the working patterns from `lab3/common.py`, `lab3/demo1_llm.py`, and `lab3/demo2_vlm.py`.

| Lab 3 pattern | Use here |
| --- | --- |
| `ask()` chat completion | VLM cue and reasoning action |
| `image_content()` JPEG data URL, width capped at 640 | One in-memory frame per camera checkpoint. Nothing is written with `imwrite`. Demo 2's `last_frame.jpg` save is not used. |
| `get_frame()` poll | Camera-informed checkpoints only |
| Ollama at `http://localhost:11434/v1` | Default. `llama3.2` chooses the dialogue action. `gemma3:4b` reads the camera frame. |
| Lab 3 `speak()` wav, then `play_sound` | Every Reachy line is shown and played on the Reachy speaker. Default TTS is `gpt-4o-mini-tts`, voice `alloy`. |
| Demo 1 typed chat | Participants type every response. No speech-to-text and no voice-activity detection. |
| Demo 3 fixed action list | The reasoning model may return only `offer_hint`, `neutral_encouragement`, or `no_nudge`. Any other result becomes the neutral fallback. The study does not run Demo 3's head-movement policy. |

Demo 2's sample question asks the model to describe emotion. That sentence is a demo prompt, not a detector. This study's vision prompt forbids emotion, confidence, and mental-state claims and allows `unclear`.

## Models

The default chat models follow Lab 3 without `--api`. Ollama serves `llama3.2` for the text decision and `gemma3:4b` for the one camera frame. Those calls send only the model and the messages. The fixed-point condition does not call either model and does not capture a frame.

Speech always uses Reachy's speaker on a real run. The TTS model defaults to `gpt-4o-mini-tts` and the voice to `alloy`, the Lab 3 speaker settings. Change them with `--tts-model` and `--voice`. `OPENAI_API_KEY` is read from `guidance_study/.env` and is not printed. `--api` also sends the chat calls to `gpt-6-luna` with `reasoning_effort=none`. Do not send `none` to `gpt-6.1-sol`.

A missing frame, an unreadable cue, or a failed model call logs the error and uses the neutral line. It does not offer a camera-based hint. If playback fails, the log says `speech_delivery=failed` and does not mark the line as played. Simulation prints the lines only, sets `speech_delivery=simulated`, and does not call the camera, Ollama, or the speech API.

## Files

| File | Role |
| --- | --- |
| `task_materials.json` | Cases, checkpoints, hints, orders, and spoken lines |
| `protocol.md` | Procedure, illustrative dialogues, and the same canonical wording |
| `counterbalancing.md` | The four orders |
| `models.py` | VLM and reasoning requests |
| `robot_io.py` | Camera frame and TTS |
| `run_study.py` | Session runner |
| `test_run_study.py` | Mocked tests |
| `requirements.txt` | Packages for a hardware session |

## Set up

Use Python 3.11. From the repo root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r guidance_study/requirements.txt
```

Install [Ollama](https://ollama.com), then pull the two local models:

```bash
ollama pull llama3.2
ollama pull gemma3:4b
```

Create `guidance_study/.env`. It is gitignored. The speech key is required for a real session. `REACHY_IP` is optional and defaults to `192.168.8.199`.

```bash
OPENAI_API_KEY=your-key
REACHY_IP=192.168.8.199
```

Do not commit `.env`, `logs/`, or generated `.wav` files. Those can contain the key, answers, or spoken audio.

## Run the checks

From `guidance_study`, with `.venv` active:

```bash
python -m unittest test_run_study.py
python run_study.py --check
```

The tests do not contact a robot, camera, or model API, and they do not read participant data. `--check` prints the four condition orders.

## Simulation walkthrough

```bash
python run_study.py --participant P01 --order 1 --simulate
```

Simulation prints Reachy's lines and does not connect to the robot or the API. The default camera branch uses a fake observed cue and chooses `no_nudge`. Try the other branches with `--simulate-cue unclear` or `--simulate-action offer_hint`. Those logs are marked `simulated_not_participant_data`.

`P01` is order 1: Ashcombe Star in the fixed-point condition, then Briarwick Watch in the camera-informed condition. Order 4 is Briarwick Watch / fixed-point, then Ashcombe Star / camera-informed, so every order contains both strategies. See `counterbalancing.md`.

Type `q` to stop. Rows already written stay in `logs/`.

## Hardware session

1. Start Ollama so `llama3.2` and `gemma3:4b` are available. Confirm `guidance_study/.env` contains `OPENAI_API_KEY`.
2. Confirm the Reachy daemon is running and `no_media` is false. From this computer: `curl http://192.168.8.199:8000/api/daemon/status`.
3. From `guidance_study`, with `.venv` active:

```bash
python run_study.py --participant P01 --order 1
```

The first line is `Loading the session.` Then the terminal prints the Ollama models, `Reachy speaker. TTS: gpt-4o-mini-tts. Voice: alloy.`, and the robot address. Every Reachy line is printed and sent to the robot speaker. Type answers, explanations, and ratings in the terminal. Type `q` to stop. `P01` uses order 1: Ashcombe Star with fixed-point hints, then Briarwick Watch with the camera. The camera captures one frame only after the initial answer at a camera checkpoint. Fixed-point checkpoints never call `get_frame`.

`No Reachy Mini Audio USB device found` is the laptop looking for the robot's USB microphone board. Speech does not use that device. A line was not played only if the terminal prints `Speech failed`.

If Python reports `No module named 'gi'`, the GStreamer path files in `.venv` are hidden. This happens when the repo is on an iCloud Desktop. Clear the flag and start the session in one command:

```bash
chflags nohidden \
  "../.venv/lib/python3.11/site-packages/gstreamer_bundle.pth" \
  "../.venv/lib/python3.11/site-packages/gstreamer_ext_runtime.pth" && \
python run_study.py --participant P01 --order 1
```

Other orders are `--order 2`, `--order 3`, and `--order 4`. Use a new participant ID for each person. Use `--robot-ip` when the robot is not at `192.168.8.199`. Use `--voice` and `--tts-model` to change the speaker voice.

Use the pilot checklist in `protocol.md` before treating a session as data. Logs are written to `guidance_study/logs/` and are gitignored because they contain answers.
