# Readback Guard: checking pilot readbacks with speech recognition and an LLM

Abdul, atomcamp AI Bootcamp final project (NLP/LLM track), October 2026. Supervisor: Mr. Muhammad Umair.

> Draft written with Claude from the project's notes. Every number marked **[T]** comes from `results/tables.md` after the final scoring notebook runs; replace the marker with the value.

## 1. Problem

A pilot must read back safety-critical parts of a clearance (level, heading, speed, frequency, squawk, runway) so the controller can catch a mishearing. Controllers miss wrong readbacks under workload ("hearback" errors). Readback Guard listens to a controller instruction and the pilot's reply and says whether the readback is correct, which item is wrong or missing, or asks "say again" when it cannot hear a number clearly.

**Research question:** when the system raises a false alarm or misses an error, how much of that comes from speech recognition, and how much from language understanding?

## 2. Data

- **UWB-ATCC** (Prague, CC BY-NC-SA 4.0, non-commercial; fine for coursework). Its test split (2,822 segments, 677 recordings) was merged into speaker turns (same speaker, gap ≤ 1.0 s) and paired controller turn → pilot turn: 774 candidate pairs, split by recording into dev (40%) and test (60%) so no conversation is in both.
- **Gold labels:** 158 pairs kept (52 dev, 106 test) with callsign, items and verdict for each side. *Drafted by Claude, verified by the author row by row*, then frozen in git (commit 5f23788) before any system was scored. Verdicts: 119 CORRECT, 30 MISSING_ITEM, 5 WRONG_CALLSIGN, 4 WRONG_VALUE.
- **Injected errors:** real wrong readbacks are rare, so each correct pair got one copy with the last spoken digit of a value or callsign changed (117 copies: dev 35, test 82). They exist only as text.
- **Own recordings:** 20 controller/pilot pairs voiced by the author from scripts drafted by Claude (`labels/own_recordings.csv`): 10 correct, 5 wrong value, 2 wrong callsign, 3 missing item. The only real speech containing real errors.

## 3. System

```
audio → Whisper (ATC large-v3, word confidences) → Gemini 3.5 Flash-Lite (clearance as JSON)
      → Python normalise + compare → confidence gate → verdict + reason
```

- **Speech:** `jlvdoorn/whisper-large-v3-atco2-asr`, greedy decoding on the Mac GPU; each word's confidence is the lowest token probability inside it.
- **Extraction:** Gemini returns `{callsign, items[type, value, unit, qualifier]}` validated by a Pydantic schema; answers are cached by model + prompt version + text. Prompt v1 was kept after a v2 attempt gave more false alarms on dev.
- **Rules:** values are normalised (e.g. `FL120`, `090`, `121.9`) and compared. Precedence: WRONG_CALLSIGN > WRONG_VALUE > MISSING_ITEM > CORRECT. A callsign matches if one is a shortened or cut-off form of the other. 58 unit tests.
- **Gate:** if any number word on either side was heard below a confidence threshold, the verdict becomes SAY_AGAIN. The threshold was chosen on dev (highest detection minus false-alarm rate) and frozen before the test split.
- **Agent:** a 3-node LangGraph graph (extract → compare → gate) with retries on invalid JSON or server errors, wrapping the same functions as the experiment pipeline. The Streamlit demo runs it.

## 4. Experiments and results

Detection rate = errors flagged (anything but CORRECT) / real errors. False-alarm rate = correct readbacks flagged / correct readbacks. 95% Wilson intervals.

**Experiment 1 (dev, human transcripts, 87 cases):** with gold fields fed straight to the rules, 50/50 errors detected, 0/37 false alarms (the rules are sound; 86/87 exact verdicts, the one difference is a label conflict, see limits). With Gemini extraction: 47/50 detected (0.94, CI 0.84–0.98), 2/37 false alarms (0.054, CI 0.015–0.18). Almost every mistake was Gemini on the pilot side: completing cut-off numbers, inventing a runway, merging words into a callsign.

**Speech recognition quality:** WER after the same normalisation on both sides: dev 27.2% (controller 17.9%, pilot 43.5%); test **[T]**. Critical-value error rate (values spoken digit by digit that Whisper did not reproduce exactly): dev 14.7% of 177; test **[T]**. Pilot callsigns are the weakest field.

**Main table (test split, scored once):**

| Text source | Detection | False alarms | Say again |
|---|---|---|---|
| Human text, real + injected (Exp 1b) | [T] | [T] | – |
| Human text, real pairs | [T] | [T] | – |
| Whisper text, real pairs (Exp 2) | [T] | [T] | – |
| Whisper + gate, real pairs (Exp 3) | [T] | [T] | [T] |

**Attribution (test, real pairs).** For each pair: right on both texts; wrong only on Whisper text (caused by speech recognition); wrong on both (caused by language understanding, since the rules are near-perfect on gold fields); wrong only on human text. [paste table from results/tables.md]. McNemar's exact test on the discordant pairs: [T].

**Gate trade-off (dev):** see `results/figures/gate_curve_dev.png`. [one sentence: what raising the threshold does to detection, false alarms and say-again rate; the picked value].

**Own recordings:** [T] per-recording table; how many of the 10 spoken errors were caught on script text, on Whisper text and with the gate.

## 5. Discussion

[Answer the research question in three sentences using the attribution table: which source causes more false alarms, which causes more misses, and whether the gate helps.]

## 6. Limits

- Small samples: 24 real errors in the test split (22 of them missing items, 0 wrong values), hence wide intervals. Injected errors are text-only, so they cannot test the audio path; the own recordings (20 pairs, one speaker, quiet room) are the only real speech with real errors.
- p0736 is labelled WRONG_VALUE, but the pilot also mangled the callsign, so the precedence rule gives WRONG_CALLSIGN. Labels stayed frozen.
- While debugging the error injector in Module 2, the rules were run once on the gold fields of the test split. It exposed an injector bug (fixed) and that some test labels write one callsign in two forms ("russia 221" vs "sdm 221"). No rule or prompt was changed because of it.
- The gate cannot catch words Whisper dropped (a deletion has no confidence) and only watches number words, not callsign letters.
- Free-tier Gemini limits and model names change; the cache and a single wrapper file keep results reproducible.

## 7. Deviations from the Blueprint

One ASR model instead of two; 158 gold pairs; runway scored (6 types); Gemini 3.5 Flash-Lite instead of 3.8 Flash (time-outs with JSON output); about 20 own recordings instead of 150; regex baseline dropped; NOT_A_READBACK filtered at labelling; SAY_AGAIN overrides other verdicts when a number is uncertain (Blueprint precedence put it below the error verdicts).

## 8. Who did what

The author designed the project, verified every gold label, wrote the normalisation and comparison logic, recorded the own set and ran every experiment. Claude (an AI assistant) drafted the gold labels and own-recording scripts, wrote the confidence gate (gate.py) at the author's request, wrote boilerplate code (data loading, caching, batch scripts, scoring notebooks, the Streamlit layout) and helped draft this report.
