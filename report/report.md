# Readback Guard: checking pilot readbacks with speech recognition and an LLM

Abdul, atomcamp AI Bootcamp final project (NLP/LLM track), October 2026. Supervisor: Mr. Muhammad Umair.

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
- **Gate:** if any number word on either side was heard below a confidence threshold, the verdict becomes SAY_AGAIN. The threshold was chosen on dev and frozen before the test split. The rule written down first (highest detection minus false-alarm rate) counts a "say again" as a false alarm, so it can never reward the gate and picked 0.0 (gate off). After seeing the dev curve, the rule was changed to: fewest false accusations plus missed errors, with "say again" on at most 20% of pairs. This picked 0.4. The change was made on dev only; the test split was untouched.
- **Agent:** a 3-node LangGraph graph (extract → compare → gate) with retries on invalid JSON or server errors, wrapping the same functions as the experiment pipeline. The Streamlit demo runs it.

## 4. Experiments and results

Detection rate = errors flagged (anything but CORRECT) / real errors. False-alarm rate = correct readbacks flagged / correct readbacks. 95% Wilson intervals.

**Experiment 1 (dev, human transcripts, 87 cases):** with gold fields fed straight to the rules, 50/50 errors detected, 0/37 false alarms (the rules are sound; 86/87 exact verdicts, the one difference is a label conflict, see limits). With Gemini extraction: 47/50 detected (0.94, CI 0.84–0.98), 2/37 false alarms (0.054, CI 0.015–0.18). Almost every mistake was Gemini on the pilot side: completing cut-off numbers, inventing a runway, merging words into a callsign.

**Speech recognition quality:** WER after the same normalisation on both sides: dev 27.2% (controller 17.9%, pilot 43.5%); test 25.0% (controller 17.4%, pilot 38.5%). Critical-value error rate (values spoken digit by digit that Whisper did not reproduce exactly): dev 14.7% of 177; test 16.1% of 366. Pilot callsigns are the weakest field.

**Main table (test split, scored once):**

| Text source | Detection | False alarms (any flag) | False accusations | Say again | Exact verdict |
|---|---|---|---|---|---|
| Human text, real + injected (Exp 1b), 188 cases | 100/106 = 0.94 (0.88–0.97) | 17/82 = 0.21 (0.13–0.31) | 17 | – | 0.81 |
| Human text, real pairs, 106 | 23/24 = 0.96 (0.80–0.99) | 17/82 = 0.21 (0.13–0.31) | 17 | – | 0.77 |
| Whisper text, real pairs (Exp 2), 106 | 24/24 = 1.00 (0.86–1.00) | 51/82 = 0.62 (0.51–0.72) | 51 | – | 0.42 |
| Whisper + gate 0.4, real pairs (Exp 3), 106 | 24/24 = 1.00 (0.86–1.00) | 56/82 = 0.68 (0.58–0.77) | 45 | 19 (0.18) | 0.34 |

"False accusations" are correct readbacks given an error verdict (WRONG_VALUE, WRONG_CALLSIGN or MISSING_ITEM); "say again" is counted separately. Perfect detection on Whisper text is not good news on its own: the system flags so much that real errors are caught partly by accident, which the exact-verdict rate (0.42 vs 0.77) shows.

**Attribution (test, real pairs).** For each pair: right on both texts; wrong only on Whisper text (caused by speech recognition); wrong on both (caused by language understanding, since the rules are near-perfect on gold fields); wrong only on human text.

| Gold label | Right on both | Speech recognition | Language understanding | Right only on Whisper | Total |
|---|---|---|---|---|---|
| Correct readback (wrong = false alarm) | 25 | 40 | 11 | 6 | 82 |
| Real error (wrong = miss) | 23 | 0 | 0 | 1 | 24 |
| Total | 48 | 40 | 11 | 7 | 106 |

McNemar's exact test on the discordant pairs (40 right only on human text, 7 right only on Whisper text): p < 0.0001. With the gate the counts are 43 and 5 (p < 0.0001); the gate does not change who causes the mistakes, it changes what the system says about them.

On human text, the 17 false alarms are language understanding (Gemini) and labels: an extracted runway with an extra "C" (3 cases), a different runway number on the pilot side (3), a missing item (3), and 8 callsign mismatches, including callsign forms the rule cannot match such as the known telephony vs ICAO case ("russia 221" vs "sdm 221", 2 cases). Which of these are Gemini errors and which are label problems was not checked case by case. On Whisper text, 31 of the 51 false alarms are WRONG_CALLSIGN, mostly from misheard airline words or callsign digits (inferred from the reasons, not counted one by one).

**Gate trade-off (dev):** see `results/figures/gate_curve_dev.png`. On Whisper text, false alarms rose from 2/37 (human text) to 23/37, mostly wrong-callsign verdicts. Raising the threshold turns false accusations into "say again": 23 at threshold 0 (gate off), 16 at 0.4 with 10 of 52 pairs answered "say again", 2 at 0.95 with 43 of 52 answered "say again". Missed errors stay at 2 of 15 until 0.7.

**Own recordings (20 pairs, real speech, real errors):** on script text, 10/10 errors detected with the right verdict and 0/10 false alarms. On Whisper text, still 10/10 detected but 4/10 false alarms, all WRONG_CALLSIGN from misheard airline names on the controller side ("Liner" for Ryanair, "Visor"/"Vizor" for Wizz Air, "str" for Austrian, "smart wing" for Smartwings). Two real WRONG_VALUE errors (r02, r10) were reported as WRONG_CALLSIGN for the same reason. The gate never fired: all numbers in these clean recordings were heard with confidence above 0.4. McNemar: 4 vs 0 discordant, p = 0.125 (too few pairs to be significant).

## 5. Discussion

Most errors produced by the system are false alarms, with speech recognition being the primary source of these mistakes. In the test split, 40 readbacks that were correctly identified using human-transcribed text were incorrectly flagged when Whisper-generated text was used, compared with 11 cases where both sources failed due to language-understanding errors and 7 cases where the opposite occurred (McNemar's test, *p* < 0.0001). Missed errors were uncommon in both cases, with only 1 out of 24 occurring on human-transcribed text and none on Whisper text. These results suggest that the main concern is an overly sensitive checker that raises incorrect warnings rather than one that fails to detect genuine errors. Callsign recognition appears to be the main weakness: Whisper struggles considerably more with airline names and pilot callsigns than with numerical information. Although the critical-value error rate is 16%, 31 of the 51 false alarms are attributed to `WRONG_CALLSIGN`. The existing confidence gate, which evaluates only numerical values, converted 6 of the 51 false accusations into "say again" responses, while generating 19 such responses across 106 pairs. While this provides some improvement, a more effective solution would involve a confidence mechanism that also accounts for callsign recognition or an ASR model trained to recognize the airline names and callsign vocabulary specific to the airport.

## 6. Limits

- Small samples: 24 real errors in the test split (22 of them missing items, 2 wrong callsigns, 0 wrong values), hence wide intervals. The test split's false-alarm rate on human text (0.21) is much higher than on dev (0.05), so dev-based choices (prompt v1, gate threshold) may not transfer perfectly. Injected errors are text-only, so they cannot test the audio path; the own recordings (20 pairs, one speaker, quiet room) are the only real speech with real errors.
- p0736 is labelled WRONG_VALUE, but the pilot also mangled the callsign, so the precedence rule gives WRONG_CALLSIGN. Labels stayed frozen.
- While debugging the error injector in Module 2, the rules were run once on the gold fields of the test split. It exposed an injector bug (fixed) and that some test labels write one callsign in two forms ("russia 221" vs "sdm 221"). No rule or prompt was changed because of it.
- The gate cannot catch words Whisper dropped (a deletion has no confidence) and only watches number words, not callsign words, which is where most Whisper false alarms come from.
- The gate threshold rule was changed after seeing the dev curve (still before the test split was scored).
- The own recordings were voiced by one speaker in a quiet room and are much cleaner than real radio; both sides were read by the author.
- Free-tier Gemini limits and model names change; the cache and a single wrapper file keep results reproducible.

## 7. Deviations from the Blueprint

One ASR model instead of two; 158 gold pairs; runway scored (6 types); Gemini 3.5 Flash-Lite instead of 3.8 Flash (time-outs with JSON output); about 20 own recordings instead of 150; regex baseline dropped; NOT_A_READBACK filtered at labelling; SAY_AGAIN overrides other verdicts when a number is uncertain (Blueprint precedence put it below the error verdicts); the gate threshold rule was changed on dev after the first rule picked 0.0.

## 8. Who did what

The author designed the project, verified every gold label, wrote the normalisation and comparison logic, recorded the own set and ran every experiment. Claude (an AI assistant) drafted the gold labels and own-recording scripts, wrote the confidence gate (gate.py) at the author's request, wrote boilerplate code (data loading, caching, batch scripts, scoring notebooks, the Streamlit layout) and helped draft this report.
