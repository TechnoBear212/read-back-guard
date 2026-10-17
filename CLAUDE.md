# Readback Guard: notes for Claude Code

I'm a beginner building this to learn (atomcamp capstone, deadline 18 Oct 2026).

## How to work with me
- Explain the idea before changing code. Make small changes I can review.
- Don't write the core logic for me: turn/pair rules, normalize.py, compare.py, gate.py.
  Boilerplate (loading files, plots, Streamlit layout) is fine.
- Keep code minimal and readable. Explanations go in chat, not in heavy comments.

## Project rules
- The design is fixed by the Blueprint. Flag deviations; don't silently work around them.
- Model names, paths and thresholds live in configs/default.yaml, never in code.
- data/ and .env are never committed. labels/ is committed and frozen before scoring.
- Tune on the dev split only. The test split is scored once at the end.

## Environment
- conda env `readback`, Python 3.12, MacBook Air M4 16 GB, Jupyter kernel "Python (readback)".
- Whisper: jlvdoorn/whisper-large-v3-atco2-asr on mps, float16, generate(..., use_cache=False)
  (transformers 5.17-5.19 crash on mps otherwise; fallback: transformers==5.16.1).
- Gemini: gemini-3.5-flash-lite via google-genai, always with
  genai.Client(http_options=types.HttpOptions(timeout=60_000)).