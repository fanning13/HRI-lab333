"""Patch the Oct 17 reachy_confusion/confusion_storytelling.py protocol safely.

Run this script *inside* reachy_confusion. It NEVER overwrites the group source;
creates confusion_storytelling_audiofix.py next to the original.
"""
from pathlib import Path
import ast

src = Path('confusion_storytelling.py')
dst = Path('confusion_storytelling_audiofix.py')
if not src.is_file():
    raise SystemExit('ERROR: run from HRI-lab333/reachy_confusion (original source not found).')
if dst.exists():
    raise SystemExit(f'ERROR: {dst} already exists; please move/rename it first, to protect existing work.')
text = src.read_text(encoding='utf-8')


def replace_once(before: str, after: str, label: str) -> None:
    global text
    n = text.count(before)
    if n != 1:
        raise SystemExit(f'ABORT: expected 1 occurrence of {label}, found {n}. '
                         'Repo may have changed; do not use this repair script on a newer version.')
    text = text.replace(before, after, 1)

# 1: Accurate English ASR: hint names, lock language.
replace_once(
    '''def transcribe(client: Any, capture: AudioCapture, model: str) -> str:
    return client.audio.transcriptions.create(model=model, file=("utterance.wav", capture.wav, "audio/wav")).text.strip()
''',
    '''def transcribe(client: Any, capture: AudioCapture, model: str) -> str:
    """Transcribe English storytelling, with context for proper names."""
    capture.wav.seek(0)
    result = client.audio.transcriptions.create(
        model=model,
        file=("utterance.wav", capture.wav, "audio/wav"),
        language="en",
        prompt=("English collaborative storytelling conversation. Characters include "
                "Emma, Noah, Lily, Sophie, Daniel, Mia. Objects include a brass key, "
                "a green backpack, greenhouse, and blue sketchbook."),
    )
    return result.text.strip()
''',
    'transcribe',
)

# 2: Serial audio playback. Remote play_sound triggers async playback; do not
#    dispatch another sentence (or start participant mic) until this has ended.
replace_once(
    '''def say(client: Any, mini: Any, text: str, args: argparse.Namespace) -> bool:
    print(f"\\nReachy > {text}")
    if args.demo or args.no_tts or mini is None: return False
    fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    try:
        with client.audio.speech.with_streaming_response.create(model=args.tts_model, voice=args.voice, input=text, response_format="wav") as response:
            response.stream_to_file(path)
        mini.media.play_sound(path)
        return True
    except Exception as exc:
        print(f"  [TTS error] {exc}"); return False
    finally:
        try: os.remove(path)
        except OSError: pass
''',
    '''def say(client: Any, mini: Any, text: str, args: argparse.Namespace) -> bool:
    """Play one full utterance before the next TTS call or microphone turn."""
    print(f"\\nReachy > {text}", flush=True)
    if args.demo or args.no_tts or mini is None:
        return False
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        start = time.perf_counter()
        with client.audio.speech.with_streaming_response.create(
            model=args.tts_model, voice=args.voice, input=text, response_format="wav"
        ) as response:
            response.stream_to_file(path)
        with wave.open(path, "rb") as wf:
            seconds = wf.getnframes() / max(1, wf.getframerate())
        print(f"  [TTS generated in {time.perf_counter()-start:.1f}s; "
              f"speaking for ~{seconds:.1f}s]", flush=True)
        mini.media.play_sound(path)
        # play_sound triggers remote playback and can return before the speaker
        # finishes. Prevent out-of-order story/audio and self-transcription.
        time.sleep(seconds + args.playback_tail_seconds)
        return True
    except Exception as exc:
        print(f"  [TTS error] {exc}", flush=True)
        return False
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
''',
    'say / playback',
)

# 2b: Log the time spent on speech-to-text, separately from model inference.
replace_once(
    '            ready = time.perf_counter(); capture = None\n',
    '            ready = time.perf_counter(); capture = None\n            stt_ms = 0.0\n',
    'initialize stt timer',
)
replace_once(
    '                transcript = transcribe(client, capture, args.stt_model)\n',
    '                stt_started = time.perf_counter()\n                transcript = transcribe(client, capture, args.stt_model)\n                stt_ms = (time.perf_counter() - stt_started) * 1000\n                print(f"  [STT inference: {stt_ms:.0f}ms]", flush=True)\n',
    'STT timing',
)
replace_once(
    '                "recording_method": capture.method if capture else "typed", "speech_start_delay_ms": round(start_delay, 1) if start_delay is not None else "",\n',
    '                "recording_method": capture.method if capture else "typed", "stt_ms": round(stt_ms, 1), "speech_start_delay_ms": round(start_delay, 1) if start_delay is not None else "",\n',
    'STT CSV logging',
)

# 3: Avoid two TTS files for the bridge and the next segment.
replace_once(
    '''            if bridge: say(client, mini, bridge, args)
            say(client, mini, turn["narration"] + " " + turn["prompt"], args)
''',
    '''            utterance = ((bridge + " ") if bridge else "") + turn["narration"] + " " + turn["prompt"]
            say(client, mini, utterance, args)
''',
    'merged bridge/story utterance',
)

# 4: Classify and generate the next story bridge concurrently. Never wait for
#    the shadow-mode VLM before deciding the audio reply based on the LLM.
replace_once(
    '''            if args.demo:
                llm, vlm = mock_predict(transcript), (mock_predict(transcript) if frame is not None else Prediction(error="NO_IMAGE"))
            else:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    a = pool.submit(classify, client, args.model, transcript, context)
                    b = pool.submit(classify, client, args.model, transcript, context, frame) if frame is not None else None
                    llm, vlm = a.result(), (b.result() if b else Prediction(error="NO_IMAGE"))
''',
    '''            next_text = story[index]["narration"] if index < len(story) else ""
            locked_facts = " ".join(x["checkpoint_fact"] for x in story[:index + 1])
            pool = None
            if args.demo:
                llm = mock_predict(transcript)
                vlm = mock_predict(transcript) if frame is not None else Prediction(error="NO_IMAGE")
                bridge, bridge_ms, bridge_error = (
                    ("Your idea leads them onward.", 0.0, "MOCK - not research data")
                    if next_text else ("", 0.0, "")
                )
            else:
                pool = ThreadPoolExecutor(max_workers=3)
                llm_job = pool.submit(classify, client, args.model, transcript, context)
                vlm_job = (pool.submit(classify, client, args.model, transcript, context, frame)
                           if frame is not None else None)
                bridge_job = pool.submit(generate_bridge, client, args.model, transcript,
                                         next_text, locked_facts)
                # The LLM is the only classifier used to choose a robot action.
                llm = llm_job.result()
                if llm.error:
                    print(f"  [LLM error] {llm.error}", flush=True)
''',
    'parallel LLM/VLM/story jobs',
)

replace_once(
    '''            else:
                say(client, mini, "Thanks for your idea. Let's continue.", args)
            next_text = story[index]["narration"] if index < len(story) else ""
            if args.demo:
                bridge, bridge_ms, bridge_error = (("Your idea leads them onward.", 0.0, "MOCK - not research data") if next_text else ("", 0.0, ""))
            else:
                bridge, bridge_ms, bridge_error = generate_bridge(client, args.model, transcript, next_text, " ".join(x["checkpoint_fact"] for x in story[:index + 1]))
''',
    '''            else:
                # Avoid an extra slow, scripted TTS request between each turn.
                # The generated bridge will acknowledge the participant's idea.
                print("  [No clarification needed; moving to story continuation]", flush=True)
            if pool is not None:
                bridge, bridge_ms, bridge_error = bridge_job.result()
                vlm = vlm_job.result() if vlm_job is not None else Prediction(error="NO_IMAGE")
                pool.shutdown(wait=True)
                if bridge_error:
                    print(f"  [Story bridge error] {bridge_error}", flush=True)
                if vlm.error and frame is not None:
                    print(f"  [VLM error] {vlm.error}", flush=True)
''',
    'nonblocking assistance + finish parallel jobs',
)

# 5: Surface suspicious transcripts so pilot data is not silently treated as English.
replace_once(
    '                print(f"Recognized speech: {transcript!r}")\n',
    '                print(f"Recognized speech: {transcript!r}")\n                if re.search(r"[\\uac00-\\ud7a3]", transcript):\n                    print("  [WARNING: Korean characters detected in English study. "\n                          "Check audio capture or rerecord this pilot turn.]", flush=True)\n',
    'non-English transcript warning',
)

# Note that typed response latency must not be interpreted as human hesitation.
replace_once(
    '''            stalled = interaction_stall(transcript, start_delay, duration, args.stall_threshold_ms)
''',
    '''            # Keyboard typing speed is not an HRI speech hesitation signal.
            stalled = (interaction_stall(transcript, start_delay, duration, args.stall_threshold_ms)
                       if args.input == "mic" else not bool(transcript.strip()))
''',
    'typed-stall correction',
)

replace_once(
    '''    p.add_argument("--record-seconds", type=float, default=7.0); p.add_argument("--end-silence-ms", type=int, default=1000)
''',
    '''    p.add_argument("--record-seconds", type=float, default=10.0); p.add_argument("--end-silence-ms", type=int, default=1500)
    p.add_argument("--playback-tail-seconds", type=float, default=0.8,
                   help="Extra pause after expected speaker playback (wireless jitter margin)")
''',
    'VAD and playback settings',
)

replace_once(
    '''    if not 1 <= args.record_seconds <= 30: p.error("--record-seconds must be 1-30")
''',
    '''    if not 1 <= args.record_seconds <= 30: p.error("--record-seconds must be 1-30")
    if not 0 <= args.playback_tail_seconds <= 5:
        p.error("--playback-tail-seconds must be 0-5")
''',
    'validate playback tail',
)

try:
    ast.parse(text, filename=str(dst))
except SyntaxError as e:
    raise SystemExit(f'ABORT: generated code invalid: {e}')
dst.write_text(text, encoding='utf-8')
print(f'Created {dst} with 12 targeted repairs. Original {src} untouched.')
print('Test with: .\\.venv\\Scripts\\python.exe -m py_compile confusion_storytelling_audiofix.py')
