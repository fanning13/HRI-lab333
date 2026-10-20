# Lab 3 — GenAI for Human–Robot Interaction

**INFO 5356-030 · Fall 2026 · Cornell Tech**

## Project overview

**Evaluating Camera-Informed Adaptive Guidance in Human–Robot Mystery-Solving**

This study investigates when a Reachy Mini robot should offer hints while a person solves two short fictional mysteries. We compare two **complete guidance strategies**:

- **Fixed-point guidance:** at predefined checkpoints, Reachy always asks whether the participant wants a hint.
- **Camera-informed guidance:** at checkpoints, Reachy uses a single camera frame and typed-response evidence to select among an optional hint offer, neutral encouragement, or no additional prompt.

The strategies differ in both camera access and the guidance-triggering policy. **The comparison does not isolate the causal effect of camera access alone.**

## Research questions

1. How do fixed-point and camera-informed guidance differ in perceived **helpfulness, timing, and match to desired assistance**?
2. How do the strategies differ in **final-answer accuracy and initial-to-final answer improvement**?

## Repository structure

| Path | Purpose |
| --- | --- |
| [`guidance_study/`](guidance_study/) | Current experiment and study materials |
| [`guidance_study/README.md`](guidance_study/README.md) | Detailed setup and run instructions |
| [`guidance_study/run_study.py`](guidance_study/run_study.py) | Main experimental session runner |
| [`guidance_study/models.py`](guidance_study/models.py) | Vision and reasoning model calls |
| [`guidance_study/robot_io.py`](guidance_study/robot_io.py) | Reachy camera and speech interfaces |
| [`guidance_study/task_materials.json`](guidance_study/task_materials.json) | Mystery cases, prompts, and answer keys |
| [`guidance_study/protocol.md`](guidance_study/protocol.md) | Study workflow and measures |
| [`guidance_study/counterbalancing.md`](guidance_study/counterbalancing.md) | Four case/condition sequences |
| [`guidance_study/test_run_study.py`](guidance_study/test_run_study.py) | Mocked automated tests |
| [`guidance_study/analysis_results.py`](guidance_study/analysis_results.py) | Descriptive analysis of a local CSV ZIP |
| [`LaTex Paper/lab3_guidance_submission.tex`](LaTex%20Paper/lab3_guidance_submission.tex) | **Current report source** |
| [`LaTex Paper/README.md`](LaTex%20Paper/README.md) | Notes on paper versions |

**Important:** `guidance_study/` is the experiment used for the current submission. `lab3_confusion_storytelling_draft.tex` documents an earlier design and should not be interpreted as the evaluated procedure. `LaTex Paper/run_file.tex` is the original template, not the current paper.

## Installation and running the experiment

Use Python 3.11 and follow the complete setup instructions in [`guidance_study/README.md`](guidance_study/README.md), including Reachy Mini, Ollama, and TTS configuration.

Install the study's Python dependencies from the repository root:

```bash
python -m pip install -r guidance_study/requirements.txt
```

Run the checks and mock tests without requiring the robot:

```bash
cd guidance_study
python run_study.py --check
python -m unittest test_run_study.py
python run_study.py --simulate
```

For a real session, first complete the hardware setup and run `python run_study.py --components`, then `python run_study.py`. Refer to the study README for details and optional flags.

## Report and demo video

- **Current LaTeX report:** [`LaTex Paper/lab3_guidance_submission.tex`](LaTex%20Paper/lab3_guidance_submission.tex)
- **PDF build:** install a TeX distribution with `pdflatex`, then run `bash build_paper.sh` from the repo root. It produces `LaTex Paper/lab3_guidance_submission.pdf`.
- **Robot demonstration:** [Google Drive video folder](https://drive.google.com/drive/folders/1KQmbK1nYUo2-PXA0Jl994M648heiw137). Please ensure the instructor has view access.

The built PDF must be reviewed and submitted through the required course submission channel; keeping source files on GitHub alone is not a substitute for submission.

## Data analysis and privacy

The system records answers, ratings, correctness, hint events, and timing measurements in session CSVs. The local analyzer accepts a ZIP-file path:

```bash
python -m pip install pandas
python guidance_study/analysis_results.py /path/to/authorized-session-logs.zip
```

**Compatibility note:** As currently written, `analysis_results.py` expects `session_mode=Result`. Hardware-export CSVs may instead use `session_mode=hardware`; the analyzer needs an explicit, provenance-preserving compatibility update before running on those files. Do not alter source labels merely to bypass the validation.

The public repository intentionally excludes participant-level answers, camera frames, identifiable recordings, and private API keys. Raw data and video must be shared only where participant permission and the lab's data-handling rules allow.

## Limitations

A single camera snapshot is an uncertain behavioral cue rather than ground truth about a participant's mental state. The two experimental conditions also vary the guidance policy, so any differences are attributed only to the strategies as implemented. See the report for additional design and measurement limitations.
