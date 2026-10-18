# Readback Guard

An LLM-based agent that checks whether a pilot read back an air traffic control instruction correctly.
atomcamp AI Bootcamp final project (NLP/LLM track), October 2026. Author: Abdul. Supervisor: Mr. Muhammad Umair.

**Full report:** [`report/Readback_Guard_Report.pdf`](report/Readback_Guard_Report.pdf)

## What it does

```
controller audio + pilot audio
  → Whisper (ATC-tuned large-v3, per-word confidence)
  → Gemini 3.5 Flash-Lite (clearance as JSON: callsign, level, heading, speed, frequency, squawk, runway)
  → Python rules (normalise + compare)
  → confidence gate ("say again" if a number was heard too uncertainly)
  → verdict: CORRECT / WRONG_VALUE / WRONG_CALLSIGN / MISSING_ITEM / SAY_AGAIN + a reason
```

The same functions run as a plain pipeline for the experiments and as a 3-node LangGraph agent (extract → compare → gate, with retries) for the Streamlit demo.

**Research question:** how many false alarms and missed errors come from speech recognition, and how many from language understanding?

## Results (UWB-ATCC test split, 106 real pairs, scored once)

| Text source | Errors caught | Correct readbacks flagged | Say again |
|---|---|---|---|
| Human transcript | 23/24 | 17/82 (21%) | – |
| Whisper transcript | 24/24 | 51/82 (62%) | – |
| Whisper + gate (0.4) | 24/24 | 45 accused + 11 say again | 19/106 |

- 40 pairs were right on human text and wrong on Whisper text (speech recognition), 11 wrong on both (language understanding), 7 the other way round. Exact McNemar test p < 0.0001.
- Whisper WER 25% (controller 17%, pilot 39%); 16% of critical values misheard; most false alarms are wrong callsigns.
- 20 readbacks recorded by the author (10 with deliberate errors): all 10 errors caught on both script and Whisper text; 4 false alarms on Whisper text from misheard airline names.

## Repository layout

| Path | What |
|---|---|
| `src/readback_guard/` | `ingest` (turns, pairs, splits), `transcribe` (Whisper), `extract` + `llm` + `cache` (Gemini), `normalize` + `compare` (rules), `gate`, `graph` (LangGraph), `asr` (WER, critical values), `score`, `evaluate` (Wilson CIs, McNemar, attribution), `batch`, `inject` |
| `tests/` | 58 pytest tests (normalise, compare, gate, ingest) |
| `labels/` | Gold labels (frozen before scoring), injected errors, own-recording scripts |
| `notebooks/` | `00` spikes, `01` ingest, `02` text checker (Exp 1), `03` gold labels, `04` audio path and gate on dev (Exp 2, 3), `05` final scoring |
| `app/` | `streamlit_app.py` (demo), `record_app.py` (records the own set) |
| `results/` | `log.csv` (every result with its git commit), `tables.md`, `figures/` |
| `report/` | Final report (PDF and Markdown) |
| `configs/default.yaml` | Model names, paths, gate threshold |

## Running it

Python 3.12 (conda), Apple Silicon GPU (`mps`) for Whisper. Data and caches live in `data/`, which is not in git.

```bash
conda create -n readback python=3.12 && conda activate readback
pip install -r requirements.txt && pip install -e .
echo "GEMINI_API_KEY=..." > .env
pytest -q
python -m readback_guard.transcribe            # Whisper on the gold pairs (downloads UWB-ATCC once)
python -m readback_guard.batch dev             # Gemini extraction, cached; then test, own
streamlit run app/streamlit_app.py             # demo
```

Notebooks `02`, `04` and `05` reproduce the experiments in that order.

## Notes on honesty and data

- Gold labels and the own-recording scripts were drafted by Claude (an AI assistant) and verified row by row by the author. Claude also wrote boilerplate code, the gate and most of the report; see section 9 of the report.
- The test split was scored once; prompts and the gate threshold were tuned on dev only. The threshold rule was changed on dev after the first rule picked 0.0 (documented in the report).
- UWB-ATCC is CC BY-NC-SA 4.0 (non-commercial). It is downloaded from Hugging Face (`Jzuluaga/uwb_atcc`), not included here.
