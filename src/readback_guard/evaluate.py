import csv
import json
import subprocess
from datetime import datetime, timezone
from math import sqrt

import pandas as pd
from scipy.stats import binomtest

from readback_guard.config import ROOT
from readback_guard.schemas import Clearance, Item


def parse_items(text):
    items = []
    for part in filter(None, (p.strip() for p in text.split(";"))):
        kind, value = part.split("=")
        unit = None
        if kind == "level":
            unit = "ft" if value.startswith("A") else "FL"
            value = value.removeprefix("A").removeprefix("FL")
        items.append(Item(type=kind, value=value, unit=unit))
    return items


def gold_clearance(callsign, items):
    return Clearance(callsign=callsign or None, items=parse_items(items))


def load_cases(cfg, split):
    cols = ["case_id", "pair_id", "kind", "atco_text", "pilot_text",
            "atco_callsign", "pilot_callsign", "atco_items", "pilot_items", "verdict"]
    gold = pd.read_csv(ROOT / cfg["labels"]["gold"], dtype=str).fillna("")
    gold = gold[(gold["keep"] == "y") & (gold["split"] == split)].copy()
    gold["case_id"], gold["kind"] = gold["pair_id"], "real"

    inj = pd.read_csv(ROOT / cfg["labels"]["injected"], dtype=str).fillna("")
    inj = inj[inj["split"] == split].merge(
        gold[["pair_id", "atco_text", "atco_callsign", "atco_items"]], on="pair_id")
    inj["case_id"], inj["kind"] = inj["inj_id"], "injected"
    return pd.concat([gold[cols], inj[cols]], ignore_index=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(centre - half, 3), round(centre + half, 3)


def rates(df):
    """df needs columns verdict (gold) and pred (system)."""
    errors = df[df["verdict"] != "CORRECT"]
    correct = df[df["verdict"] == "CORRECT"]
    det_k = int((errors["pred"] != "CORRECT").sum())
    fa_k = int((correct["pred"] != "CORRECT").sum())
    exact_k = int((df["pred"] == df["verdict"]).sum())
    say_again = int((df["pred"] == "SAY_AGAIN").sum())
    return {
        "n_errors": len(errors), "detected": det_k,
        "detection_rate": round(det_k / max(len(errors), 1), 3), "detection_ci": wilson(det_k, len(errors)),
        "n_correct": len(correct), "false_alarms": fa_k,
        "false_alarm_rate": round(fa_k / max(len(correct), 1), 3), "false_alarm_ci": wilson(fa_k, len(correct)),
        "exact_verdict_rate": round(exact_k / len(df), 3),
        "say_again": say_again, "say_again_rate": round(say_again / len(df), 3),
    }


def log_result(experiment, split, text_source, metrics, cfg, notes=""):
    path = ROOT / "results" / "log.csv"
    path.parent.mkdir(exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    row = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": commit,
           "experiment": experiment, "split": split, "text_source": text_source,
           "llm": cfg["llm"]["model"], "metrics": json.dumps(metrics), "notes": notes}
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def attribution(human, asr):
    """Same cases run on human text and on Whisper text. Who caused each mistake?

    right = the system's flag (anything but CORRECT) agrees with the gold label.
    """
    m = human[["case_id", "verdict", "pred"]].merge(asr[["case_id", "pred"]], on="case_id",
                                                    suffixes=("_human", "_asr"))
    is_error = m["verdict"] != "CORRECT"
    m["human_right"] = (m["pred_human"] != "CORRECT") == is_error
    m["asr_right"] = (m["pred_asr"] != "CORRECT") == is_error
    m["cause"] = "right on both"
    m.loc[m.human_right & ~m.asr_right, "cause"] = "speech recognition"
    m.loc[~m.human_right & ~m.asr_right, "cause"] = "language understanding"
    m.loc[~m.human_right & m.asr_right, "cause"] = "right only on Whisper text"
    m["gold"] = is_error.map({True: "real error (miss if wrong)", False: "correct readback (false alarm if wrong)"})
    table = pd.crosstab(m["gold"], m["cause"], margins=True, margins_name="total")
    return m, table


def mcnemar(m):
    """Exact McNemar test on the discordant pairs of attribution()."""
    b = int((m.human_right & ~m.asr_right).sum())
    c = int((~m.human_right & m.asr_right).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"human_only_right": b, "asr_only_right": c, "p_value": round(float(p), 4)}


def md_table(df):
    """A pandas table as Markdown, for pasting into the report."""
    df = df.reset_index()
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)
