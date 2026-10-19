"""VLM cue description and reasoning-model action choice.

Lab 3 sends text, or text plus one image, through chat completions
(`lab3/common.py`). Demo 3 then keeps only an action that appears in a
fixed list. This study uses the same two ideas: one image-plus-text call
for an uncertain visual cue, then one text call that must return one of
three dialogue actions.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_LLM_MODEL = "llama3.2"
OLLAMA_VLM_MODEL = "gemma3:4b"
OLLAMA_TTS = "pyttsx3"
DEFAULT_REASONING_MODEL = "gpt-6-luna"
DEFAULT_VLM_MODEL = "gpt-6-luna"
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_TTS_VOICE = "alloy"
DEFAULT_REASONING_EFFORT = "none"
DEFAULT_ROBOT_IP = "192.168.8.199"

ACTIONS = ("offer_hint", "neutral_encouragement", "no_nudge")
CUE_STATUSES = ("observed", "unclear")

VLM_INSTRUCTIONS = (
    "You describe one camera frame for a researcher. Report only observable, "
    "non-identifying cues that might suggest a person needs more time or support, "
    "such as gaze directed away from the workspace or a hand covering the mouth. "
    "Do not name, identify, or describe a face in identifying detail. Do not claim "
    "to know the person's emotion, confidence, or mental state. Do not say the "
    "person looks confused or seems certain. Do not solve the mystery and do not "
    "offer a hint. If the frame does not support a relevant observation, set "
    "cue_status to unclear and keep the description empty."
)

REASONING_INSTRUCTIONS = (
    "You choose Reachy's next dialogue action during a mystery question. "
    "You receive the story so far, the typed answer, and an uncertain visual cue. "
    "The cue is not ground truth and is not a measure of emotion. "
    "Choose offer_hint only when the typed answer or the cue gives a reason to "
    "offer support. Choose neutral_encouragement when a brief non-evaluative "
    "line is enough. Choose no_nudge when you should continue without comment. "
    "Do not claim the person is confused or certain. Do not reveal the answer."
)


@dataclass
class CueResult:
    status: str
    description: str
    error: str
    latency_ms: int


@dataclass
class ActionResult:
    action: str
    error: str
    latency_ms: int


def _schema(name: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


CUE_FORMAT = _schema(
    "visual_cue",
    {
        "cue_status": {"type": "string", "enum": list(CUE_STATUSES)},
        "description": {"type": "string"},
    },
    ["cue_status", "description"],
)

ACTION_FORMAT = _schema(
    "guidance_action",
    {"action": {"type": "string", "enum": list(ACTIONS)}},
    ["action"],
)


def chat_kwargs(model: str, effort: str, messages: list[dict], schema: dict, max_tokens: int, structured: bool = True) -> dict:
    """Parameters for one chat call.

    Ollama uses the same client as Lab 3, with only model and messages.
    Structured output and `reasoning_effort` are for the OpenAI `--api` path.
    `gpt-6-luna` accepts `none`. Do not send `none` to `gpt-6.1-sol`.
    """
    if not structured:
        return {"model": model, "messages": messages}
    kwargs = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "response_format": schema,
    }
    if effort:
        kwargs["reasoning_effort"] = effort
    if effort == "none":
        kwargs["temperature"] = 0
    return kwargs


def vlm_user_text(story_context: str, choice: str = "", explanation: str = "") -> str:
    return (
        "The participant just typed this answer. Describe only the current frame.\n\n"
        f"Typed choice: {choice}\n"
        f"Typed explanation: {explanation}\n\n"
        f"{story_context}"
    )


def vlm_messages(story_context: str, image_parts: list[dict], choice: str = "", explanation: str = "") -> list[dict]:
    return [
        {"role": "system", "content": VLM_INSTRUCTIONS},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": vlm_user_text(story_context, choice, explanation)},
                *image_parts,
            ],
        },
    ]


def reasoning_messages(story_context: str, choice: str, explanation: str, cue_description: str) -> list[dict]:
    user = (
        f"Story context:\n{story_context}\n\n"
        f"Typed choice: {choice}\n"
        f"Typed explanation: {explanation}\n\n"
        f"Uncertain visual cue: {cue_description or 'none'}"
    )
    return [
        {"role": "system", "content": REASONING_INSTRUCTIONS},
        {"role": "user", "content": user},
    ]


def redact(text: str) -> str:
    import os
    import re

    key = os.environ.get("OPENAI_API_KEY", "")
    if key and key in text:
        text = text.replace(key, "[redacted]")
    return re.sub(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+", "[image omitted]", text)


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _json_object(text: str) -> dict | None:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _local_messages(messages: list[dict], instruction: str) -> list[dict]:
    copied = [dict(message) for message in messages]
    content = copied[-1]["content"]
    if isinstance(content, str):
        copied[-1]["content"] = f"{content}\n\n{instruction}"
    else:
        copied[-1]["content"] = [*content, {"type": "text", "text": instruction}]
    return copied


def describe_frame(client, model: str, effort: str, frame, story_context: str, encode_image, choice: str = "", explanation: str = "", structured: bool = True) -> CueResult:
    started = time.perf_counter()
    try:
        image_parts = encode_image(frame)
        messages = vlm_messages(story_context, image_parts, choice, explanation)
        if not structured:
            messages = _local_messages(
                messages,
                'Reply with one JSON object and no other text: {"cue_status":"observed" or "unclear","description":"short observable cue or empty"}',
            )
        response = client.chat.completions.create(
            **chat_kwargs(model, effort, messages, CUE_FORMAT, max_tokens=160, structured=structured)
        )
        raw = response.choices[0].message.content or ""
        payload = _json_object(raw) or {}
        status = payload.get("cue_status", "")
        description = (payload.get("description") or "").strip()
        if status == "unclear":
            return CueResult("unclear", "", "", _elapsed_ms(started))
        if status == "observed" and description:
            return CueResult("observed", description, "", _elapsed_ms(started))
        if not structured and "unclear" in raw.lower():
            return CueResult("unclear", "", "", _elapsed_ms(started))
        return CueResult("error", "", "could not read the visual cue", _elapsed_ms(started))
    except Exception as exc:
        return CueResult("error", "", redact(str(exc)), _elapsed_ms(started))


def choose_action(
    client,
    model: str,
    effort: str,
    story_context: str,
    choice: str,
    explanation: str,
    cue_description: str,
    structured: bool = True,
) -> ActionResult:
    started = time.perf_counter()
    try:
        messages = reasoning_messages(story_context, choice, explanation, cue_description)
        if not structured:
            messages = _local_messages(
                messages,
                'Reply with one JSON object and no other text: {"action":"offer_hint" or "neutral_encouragement" or "no_nudge"}',
            )
        response = client.chat.completions.create(
            **chat_kwargs(model, effort, messages, ACTION_FORMAT, max_tokens=40, structured=structured)
        )
        raw = response.choices[0].message.content or ""
        payload = _json_object(raw) or {}
        action = payload.get("action", "")
        if action not in ACTIONS and not structured:
            found = [name for name in ACTIONS if name in raw.lower()]
            action = found[0] if len(found) == 1 else ""
        if action not in ACTIONS:
            return ActionResult("", f"unexpected action: {raw[:120]}", _elapsed_ms(started))
        return ActionResult(action, "", _elapsed_ms(started))
    except Exception as exc:
        return ActionResult("", redact(str(exc)), _elapsed_ms(started))
