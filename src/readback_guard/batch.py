import sys

import pandas as pd

from readback_guard.asr import load_asr_cases, load_own_cases
from readback_guard.config import load_config
from readback_guard.evaluate import load_cases
from readback_guard.extract import extract


def texts_for(which, cfg):
    """Every text the experiments for one set will need, human and Whisper."""
    if which == "dev":
        frames = [load_asr_cases(cfg, "dev")]
    elif which == "test":
        frames = [load_cases(cfg, "test"), load_asr_cases(cfg, "test")]
    elif which == "own":
        frames = [load_own_cases(cfg, "human"), load_own_cases(cfg, "asr")]
    else:
        raise SystemExit("usage: python -m readback_guard.batch dev|test|own")
    texts = pd.concat([pd.concat([f.atco_text, f.pilot_text]) for f in frames])
    return list(pd.unique(texts))


def run(which, cfg):
    texts = texts_for(which, cfg)
    failed = []
    for n, text in enumerate(texts, 1):
        try:
            extract(text, cfg)
        except Exception as e:
            failed.append(text)
            print("FAILED:", type(e).__name__, str(e)[:120])
        if n % 20 == 0 or n == len(texts):
            print(f"{n}/{len(texts)}")
    print(f"done: {len(texts) - len(failed)} extracted, {len(failed)} failed (run again to retry them)")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "", load_config())
