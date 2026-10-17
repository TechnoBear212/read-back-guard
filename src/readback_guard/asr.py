import json
import re

import jiwer
import pandas as pd

from readback_guard.config import ROOT
from readback_guard.evaluate import load_cases
from readback_guard.inject import DIGITS

SPELLING = {"alpha": "alfa", "juliet": "juliett", "niner": "nine", "x-ray": "xray"}


def norm_text(text):
    """The same text normalisation for human and Whisper text, before WER and before Gemini."""
    words = []
    for w in text.lower().split():
        w = SPELLING.get(w.strip(".,?!"), w)
        w = re.sub(r"[^\w\s]", " ", w)
        words.extend(w.split())
    return " ".join(SPELLING.get(w, w) for w in words)


def load_transcripts(cfg):
    with open(ROOT / cfg["asr"]["out_file"]) as f:
        return {r["utt_id"]: r for r in map(json.loads, f)}


def turn_asr(cfg):
    """One row per turn: Whisper text and word confidences, segments joined in time order."""
    tr = load_transcripts(cfg)
    turns = pd.read_parquet(ROOT / cfg["data"]["interim_dir"] / "turns.parquet")
    rows = {}
    for t in turns.itertuples():
        if not all(u in tr for u in t.utt_ids):
            continue
        words = [w for u in t.utt_ids for w in words_of(tr[u])]
        rows[t.turn_id] = {"asr_text": " ".join(w["word"] for w in words), "words": words}
    return rows


def load_own_transcripts(cfg):
    path = ROOT / cfg["own"]["out_file"]
    if not path.exists():
        return {}
    with open(path) as f:
        return {r["utt_id"]: r for r in map(json.loads, f)}


def words_of(record):
    return [{"word": piece, "conf": w["conf"]} for w in record["words"] for piece in norm_text(w["word"]).split()]


def load_own_cases(cfg, source="asr"):
    """Your own recordings, as cases. source="human" uses the script text, "asr" the Whisper text."""
    cases = pd.read_csv(ROOT / cfg["labels"]["own"], dtype=str).fillna("")
    cases["case_id"], cases["kind"] = cases["rec_id"], "own"
    tr = load_own_transcripts(cfg)
    for side in ("atco", "pilot"):
        cases[f"{side}_ref"] = cases[f"{side}_text"].map(norm_text)
        ids = cases["rec_id"] + "_" + side
        if source == "human":
            cases[f"{side}_text"] = cases[f"{side}_ref"]
            cases[f"{side}_words"] = [[] for _ in ids]
        else:
            missing = [i for i in ids if i not in tr]
            if missing:
                raise FileNotFoundError(f"not transcribed yet: {missing[:3]} ... run python -m readback_guard.transcribe own")
            cases[f"{side}_text"] = [norm_text(tr[i]["hyp_text"]) for i in ids]
            cases[f"{side}_words"] = [words_of(tr[i]) for i in ids]
    return cases


def load_asr_cases(cfg, split):
    """Real gold pairs of one split with the human text replaced by Whisper text.

    Injected copies are left out: their error exists only in the text, there is no audio of it.
    """
    cases = load_cases(cfg, split)
    cases = cases[cases["kind"] == "real"].copy()
    pairs = pd.read_parquet(ROOT / cfg["data"]["interim_dir"] / "candidate_pairs.parquet")
    cases = cases.merge(pairs[["pair_id", "atco_turn_id", "pilot_turn_id"]], on="pair_id")
    asr = turn_asr(cfg)
    for side in ("atco", "pilot"):
        turn = cases[f"{side}_turn_id"]
        cases[f"{side}_ref"] = cases[f"{side}_text"]
        cases[f"{side}_text"] = [asr[t]["asr_text"] for t in turn]
        cases[f"{side}_words"] = [asr[t]["words"] for t in turn]
    return cases


def wer_table(cases):
    """WER per side after the same normalisation on both sides."""
    out = {}
    for side in ("atco", "pilot"):
        ref = [norm_text(t) for t in cases[f"{side}_ref"]]
        hyp = [norm_text(t) for t in cases[f"{side}_text"]]
        keep = [i for i, r in enumerate(ref) if r]
        out[side] = round(jiwer.wer([ref[i] for i in keep], [hyp[i] for i in keep]), 3)
    both_ref = [norm_text(t) for side in ("atco", "pilot") for t in cases[f"{side}_ref"]]
    both_hyp = [norm_text(t) for side in ("atco", "pilot") for t in cases[f"{side}_text"]]
    keep = [i for i, r in enumerate(both_ref) if r]
    out["all"] = round(jiwer.wer([both_ref[i] for i in keep], [both_hyp[i] for i in keep]), 3)
    return out


def spoken_in(text, digits):
    """True if the digits appear in text spoken one word per digit, e.g. 1404 -> 'one four zero four'."""
    target = " ".join(DIGITS[int(c)] for c in digits)
    text = " ".join(w for w in text.split() if w not in ("decimal", "point"))
    return f" {target} " in f" {text} "


def critical_values(cases):
    """Every gold value and callsign number, with whether Whisper got its spoken digits right.

    A value counts only if its digits are spoken one by one in the human transcript
    ("one sixty" for 160 can't be checked this way and is skipped).
    """
    rows = []
    for r in cases.itertuples():
        for side in ("atco", "pilot"):
            ref, hyp = norm_text(getattr(r, f"{side}_ref")), getattr(r, f"{side}_text")
            values = [p.split("=") for p in getattr(r, f"{side}_items").split(";") if p.strip()]
            callsign_digits = re.findall(r"\d+", getattr(r, f"{side}_callsign"))
            values += [("callsign", d) for d in callsign_digits]
            for kind, value in values:
                digits = re.sub(r"\D", "", value)
                if digits and spoken_in(ref, digits):
                    rows.append({"case_id": r.case_id, "side": side, "type": kind.strip(),
                                 "value": value, "asr_ok": spoken_in(hyp, digits)})
    return pd.DataFrame(rows)
