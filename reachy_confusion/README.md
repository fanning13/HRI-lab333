# Lab 3 — Reachy Mini confusion-aware storytelling

This prototype runs **turn-based**, not continuous, confusion detection during collaborative storytelling. Each story has six participant turns; turns 2, 4, and 6 are formal evaluation checkpoints. At every turn it records a text-only prediction and, when a camera frame is available, a text-plus-image prediction: `CONFUSED`, `NOT_CONFUSED`, or `UNCERTAIN`.

Story A follows Emma, Noah, and Lily searching for a missing key. Story B is set during fall in New York City and follows Sophie, Daniel, and Mia searching for Mia's missing sketchbook. A tightly constrained LLM bridge may connect the participant's idea to the next fixed segment, but it may not change locked facts about the missing object.

## Setup and checks

Use Python 3.11 or 3.12 from this directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest test_confusion_storytelling.py
.\.venv\Scripts\python.exe -m py_compile confusion_storytelling.py
```

Software-only demonstration (synthetic labels, no API or hardware):

```powershell
.\.venv\Scripts\python.exe confusion_storytelling.py --demo --participant TEST --condition assistance --story A --skip-survey
```

Create an untracked `.env` containing `OPENAI_API_KEY` and optionally `REACHY_IP`. Never commit participant media or API keys.

## Run the study

```powershell
.\.venv\Scripts\python.exe confusion_storytelling.py --participant P01 --condition assistance --story B --input mic --robot-ip 192.168.1.10
```

Microphone mode uses local WebRTC VAD and ends after about 1 second of silence (`--end-silence-ms 1000`). If VAD is unavailable or encounters incompatible audio, it visibly falls back to fixed-duration recording (`--record-seconds 7`). Typed and `--no-robot` modes are available for API-only checks.

In the assistance condition, `CONFUSED` leads to the fixed factual clarification. `UNCERTAIN` alone does not: only `UNCERTAIN` plus an interaction stall triggers the soft check-in “Take your time. Would you like to keep going, or would a small hint help?” Direct requests for help are answered in either condition. The baseline condition never offers proactive help.

Formal checkpoints collect confusion self-report on a 1–5 scale. Post-session support, enjoyment, and desire to continue use 1–7 scales. The program contains three comprehension questions per story and asks the operator to mark each response against a displayed scoring key.

## Logging and privacy

CSV output includes the story/condition/turn, `formal_checkpoint`, `checkpoint_fact`, both prediction labels and latency/errors, `story_bridge`, `story_generation_ms`, stall decision, `assistance_offered`, `assistance_accepted`, `clarification_text`, `speech_start_delay_ms`, `user_speech_duration_ms`, `end_silence_ms`, recording method, ratings, and comprehension score.

Raw audio, images, and transcripts are not saved by default. Obtain informed consent before sending speech or images to an API. Model output is an operational label about observable evidence, not a diagnosis or ground truth about internal mental state. Creative hesitation alone is explicitly not treated as confusion.

## Hardware status

The local unit/static checks do not validate Reachy Mini motors, network, camera, microphone, speaker, SDK media backend, or audible latency. Run a consented hardware pilot before collecting study data, confirm the daemon and IP, verify VAD behavior and camera framing, and lock all prompts/model/settings before formal sessions.
