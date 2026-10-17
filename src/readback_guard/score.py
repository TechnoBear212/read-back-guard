import pandas as pd

from readback_guard.compare import compare
from readback_guard.evaluate import rates
from readback_guard.extract import extract
from readback_guard.gate import apply_gate


def run_cases(cases, cfg, threshold=None):
    """Extract, compare and (if a threshold is given) gate every case. Adds pred and reason."""
    out = cases.copy()
    verdicts = []
    for r in cases.itertuples():
        v = compare(extract(r.atco_text, cfg), extract(r.pilot_text, cfg))
        if threshold is not None:
            v = apply_gate(v, r.atco_words, r.pilot_words, threshold)
        verdicts.append(v)
    out["pred"] = [v.verdict for v in verdicts]
    out["reason"] = [v.reason for v in verdicts]
    return out


def sweep(cases, cfg, thresholds):
    """One row per gate threshold: detection, false alarms and how often it says 'say again'."""
    rows = []
    for t in thresholds:
        m = rates(run_cases(cases, cfg, t))
        rows.append({"threshold": t, "detection_rate": m["detection_rate"],
                     "false_alarm_rate": m["false_alarm_rate"], "say_again_rate": m["say_again_rate"],
                     "youden": round(m["detection_rate"] - m["false_alarm_rate"], 3)})
    return pd.DataFrame(rows)


def pick_threshold(curve):
    """Highest detection minus false-alarm rate on dev; on a tie, the lowest threshold (fewest say-agains)."""
    best = curve["youden"].max()
    return float(curve.loc[curve["youden"] == best, "threshold"].min())
