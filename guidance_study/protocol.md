# Guidance strategy protocol

## Question

How do participants perceive Reachy Mini's guidance when it offers hints at fixed checkpoints versus when it uses camera-informed cues to decide when to offer support?

The camera-informed condition changes both camera access and the trigger for adaptive support. The comparison tests two guidance strategies, not the isolated causal effect of camera use.

Each person completes two fictional cases, one in each condition. Orders are listed in `counterbalancing.md`. There are six story turns in each case. Checkpoints are turns 2, 4, and 6. Participants type every choice, explanation, revision, rating, and perception response. Reachy speaks story text, questions, nudges, and hints. This procedure does not record participant speech and does not run speech-to-text or voice-activity detection.

Turns 4 and 6 have no new narration. The previous turn is the story for that question. Reachy does not restate it in different words.

## What Reachy does

Reachy welcomes the participant and explains the two mysteries. The participant confirms informed consent before either case. The default Ollama session processes one camera image per camera-informed checkpoint on this computer. Each captured frame is saved in `guidance_study/frames/`. Pass `--api` only when the image will instead be sent to the model provider. If consent is declined, the session stops and no frame is requested.

For a story turn, Reachy speaks the narration and does not use the camera.

At a checkpoint, Reachy speaks any narration for that turn, then the question and the two choices. The participant types an initial choice and explanation.

Fixed-point checkpoints do not request a camera frame. Reachy asks "Would a hint help?" every time, unless the typed response already asks for help. A yes speaks that checkpoint's hint. A no continues. An explicit request for help speaks the same hint and does not ask the offer question again. A request is explicit when the initial choice or explanation contains the word help, hint, clue, or stuck.

Camera-informed checkpoints request one frame as soon as the question has been spoken, then the participant types the initial choice and explanation. During that typing, the vision call judges whether the person appears confused. The judgment is an uncertain cue. Reachy does not say it aloud and does not identify the person. A typed expression of confusion, such as "I don't know," or a cue that the person appears confused, asks "Would a hint help?" If the frame is missing, the cue is unclear, and the typed answer does not express confusion, Reachy uses the neutral line and does not offer a camera-based hint. Otherwise the reasoning model reads the cue, the typed response, and the story so far, and it must choose `offer_hint`, `neutral_encouragement`, or `no_nudge`.

- `offer_hint`: Reachy asks "Would a hint help?" and speaks the fixed hint only if the person accepts.
- `neutral_encouragement`: Reachy says "Take your time—you're making progress."
- `no_nudge`: Reachy adds no guidance line.

If the frame is missing, the cue is unclear, or either model call fails, Reachy does not offer a camera-triggered hint. The failure is logged and Reachy uses the same neutral line, then continues. An explicit help request still delivers the checkpoint hint. A typed expression of confusion still asks whether a hint would help. The frame is still requested so the procedure matches the condition, but a failed image does not block a help request the person already typed.

Neither condition says "You look confused" or "You seem certain." Model output is an uncertain cue, not ground truth about the participant.

After the guidance step, both conditions ask "Would you like to keep your answer or change it?" Reachy then collects the final choice and final explanation. Reachy asks how confused the participant felt immediately before Reachy responded, from 1 (not confused) to 5 (extremely confused).

After each case, Reachy collects three perception ratings: helpfulness, timing, and how well the support matched when the person wanted help. After the second case, Reachy thanks the participant and the log is already on disk.

The media connection used for the robot speaker also owns the camera pipeline. There is no speaker-only mode in the current SDK. Fixed-point still does not request a frame and does not send an image. Do not describe that condition as proof that the camera hardware was powered down.

## Illustrative checkpoint: Ashcombe key ledger

The question is "Whose checkout of the case key is recorded between the last confirmation and discovery?" Choice A is Ada. Choice B is Bea. The hint is "Compare the key record's times with the interval between 8:00 and 8:15." Briarwick's key-ledger checkpoint uses the same pattern with Nia and Theo.

### Fixed-point

Reachy speaks the ledger narration on the previous turn, then the question and both choices. The participant types `B` and `Bea had a key too`. Reachy offers a hint even though nothing was inferred about the person:

Reachy: Would a hint help?

The participant types yes. Reachy speaks the fixed hint, then asks whether to keep or change the answer. The participant may switch to Ada and type a new explanation. Reachy then asks for the 1–5 confusion rating about the moment before Reachy responded.

### Camera-informed, hint offered

Reachy asks the question and captures one frame while the participant types `B` and an explanation. Suppose the cue says the person appears confused, or the explanation says "I don't know." Reachy asks "Would a hint help?" and speaks the same fixed hint only after a yes. Reachy does not say what it thinks the person feels.

### Camera-informed, unclear or failed

The participant types an initial answer. The frame is missing, the cue is unclear, or a model call fails. Reachy does not offer a hint. Reachy says:

Reachy: Take your time—you're making progress.

The revision question and the confusion rating still follow. A typed request such as "I need a hint" would instead speak the fixed hint in either condition.

## Measures

Score the A/B choice correct only when it matches correct_option. Score the explanation 1 only when the final explanation meets that checkpoint's explanation_criterion; otherwise score it 0. The explanation score is secondary to choice accuracy.

The program applies that checkpoint's evidence checks to the final explanation and records 1 or 0. It does not show the criterion or the checks during the session. A paraphrase that avoids those words scores 0. The participant's confusion rating is not an explanation score.

`case_completion_ms` runs from the start of the case introduction through submission of the turn-6 final explanation. It excludes that checkpoint's confusion rating and the three perception ratings. `response_latency_ms` runs from the end of the question speech through submission of the initial explanation.

Logged fields cover participant ID, order, case, condition, checkpoint, initial and final choice and explanation, correctness, explanation score, confusion rating, perception ratings, VLM cue, error, and latency, reasoning action, error, and latency, nudge, hint offered, accepted, and delivered, explicit help, response and case timing, and whether each spoken line in that step succeeded. The CSV records whether the frame was saved and its file name. The image itself stays in `guidance_study/frames/`.

## Canonical task text

### The Ashcombe Star

At Ashcombe Museum's candlelit preview, the participant assists detective Rowan Vale. The silver Star brooch is displayed in locked case 3 in the East Gallery. Curator Ada confirms it is there at 8:00 p.m. Conservator Bea and night guard Tomas are also working in the museum.

At 8:15 p.m., the case is found empty. The glass is intact and the lock shows no damage.

Where and when was the brooch last confirmed to be present?

A. in case 3 in the East Gallery at 8:00 p.m.

B. when it was found empty at 8:15 p.m.

Hint: Separate the last confirmed sighting from the time the absence was discovered.

Score 1 only if the final explanation cites case 3, the East Gallery, or 8:00 p.m. as the last confirmation that the brooch was present, or distinguishes that confirmation from the empty case at 8:15 p.m. Otherwise score 0.

The key ledger shows Ada signed out the case key at 8:02 p.m. and returned it at 8:07 p.m. No other checkout is recorded during the interval. There is no camera view inside the case.

Whose checkout of the case key is recorded between the last confirmation and discovery?

A. Ada

B. Bea

Hint: Compare the key record's times with the interval between 8:00 and 8:15.

Score 1 only if the final explanation cites Ada or Ada's 8:02 p.m. to 8:07 p.m. checkout as the recorded key access between 8:00 p.m. and 8:15 p.m. Otherwise score 0.

A visitor recalls seeing a silver glint near the East Gallery exit at 8:09 p.m., but could not identify the person or object. Staff and guests use that exit.

Which conclusion is best supported?

A. Ada definitely stole the brooch because she checked out the key

B. Ada had recorded key access, but the evidence does not establish who took the brooch

Hint: Distinguish having an opportunity from proof of who took the object.

Score 1 only if the final explanation distinguishes Ada's recorded key access from proof of who took the brooch, or states that the 8:09 p.m. silver glint does not identify the person or the object. Otherwise score 0.

### The Briarwick Watch

At Briarwick Hotel's old archive, the participant assists detective Rowan Vale. The silver pocket watch is kept in a locked cabinet in the West Reading Room. Curator Nia confirms it is there at 6:20 p.m. Archive assistant Theo and visiting historian Dr. Vale are also in the building.

At 6:35 p.m., the cabinet is found empty. The glass is intact and the lock shows no damage.

Where and when was the watch last confirmed to be present?

A. in the West Reading Room cabinet at 6:20 p.m.

B. when it was found empty at 6:35 p.m.

Hint: Identify the last timed observation that confirms the watch was there.

Score 1 only if the final explanation cites the West Reading Room cabinet or 6:20 p.m. as the last confirmation that the watch was present, or distinguishes that confirmation from the empty cabinet at 6:35 p.m. Otherwise score 0.

The key ledger shows Nia signed out the cabinet key at 6:22 p.m. and returned it at 6:27 p.m. No other checkout is recorded. There is no camera view inside the cabinet.

Whose checkout of the cabinet key is recorded between the last confirmation and discovery?

A. Nia

B. Theo

Hint: Compare the key record's times with the interval between 6:20 and 6:35.

Score 1 only if the final explanation cites Nia or Nia's 6:22 p.m. to 6:27 p.m. checkout as the recorded key access between 6:20 p.m. and 6:35 p.m. Otherwise score 0.

A porter heard ticking near the West Reading Room at 6:30 p.m., but an old mantel clock ticks in an adjoining room. The porter could not identify the source or who was nearby.

Which conclusion is best supported?

A. Nia definitely took the watch because she checked out the key

B. Nia had recorded key access, but the evidence does not establish who took the watch

Hint: A clue may show opportunity without identifying who took the watch.

Score 1 only if the final explanation distinguishes Nia's recorded key access from proof of who took the watch, or states that the 6:30 p.m. ticking does not identify the source or who was nearby. Otherwise score 0.

## Spoken lines

I am Reachy. We will look at two short fictional mysteries. I will tell you what was found, then ask you to choose A or B and explain why. Type every answer in the terminal. You may ask for a hint.

In the camera-informed case I capture one image at each question, process it with a model on this computer, and save that image in the study folder. My spoken lines are generated by the configured speech service and played on my speaker. A description of the image is an uncertain cue, not a fact about how you feel. Type yes to confirm informed consent, or no to stop.

In the camera-informed case I capture one image at each question, send that image to the model provider, and save a copy in the study folder on this computer. A description of the image is an uncertain cue, not a fact about how you feel. Type yes to confirm informed consent, or no to stop.

I will offer a hint at each question.

At each question I may use the camera once, then choose whether to offer a hint. I will not say that I know how you feel.

Would a hint help?

Take your time—you're making progress.

Would you like to keep your answer or change it?

How confused did you feel immediately before I responded? Type 1 for not confused through 5 for extremely confused.

How helpful was Reachy's guidance during this case? Type 1 for not helpful through 5 for extremely helpful.

How well timed was Reachy's support during this case? Type 1 for poorly timed through 5 for very well timed.

How well did Reachy's support match when you wanted help? Type 1 for not at all through 5 for very well.

Thank you. That is the end of the two cases.

## Privacy

Use a pseudonymous participant ID. The default models run locally through Ollama. Do not commit `.env` or `logs/`. A camera frame is encoded in memory for the local model and then discarded. The vision-language description is an uncertain cue. The spoken consent line says the image is processed on this computer. `--api` instead requires `OPENAI_API_KEY` and tells the participant the image is sent to the model provider.

## Pilot checklist

Run one or two consented pilots before a data session.

- Task ambiguity and difficulty: each checkpoint has a defensible A and B, and turn 6 is not obvious from the key checkout alone.
- Camera framing: the person and workspace are in view, and the image does not show documents that identify them.
- Cue uncertainty: unclear and failed calls produce the neutral line and do not offer a hint.
- Hint delivery: an accepted offer and an explicit help request each speak the checkpoint hint once; a decline does not.
- Answer revision: keep and change both record a final choice and explanation.
- Timing: the question is audible before the typing prompt, and case completion excludes the final confusion rating.
- Robot speech success: every spoken line is marked in the log, including a TTS failure that still prints the line in the terminal.

## Analysis

With a small pilot sample, summarize accuracy, answer changes, hint events, and perception ratings within each person and condition. Treat model labels as estimates. Do not claim that a condition difference isolates the camera.
