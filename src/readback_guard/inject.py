import random
import re

import pandas as pd

from readback_guard.config import ROOT, load_config

DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def spoken(digits):
    return [DIGITS[int(c)] for c in digits]


def change_last_digit(text, digits, rng):
    """Find the digits spoken word by word in text and change the last one."""
    words, target = text.split(), spoken(digits)
    for i in range(len(words) - len(target), -1, -1):
        if words[i:i + len(target)] == target:
            old = int(digits[-1])
            new = rng.choice([d for d in range(10) if d != old])
            words[i + len(target) - 1] = DIGITS[new]
            return " ".join(words), digits[:-1] + str(new)
    return None, None


def inject_one(row, rng):
    options = []
    for item in filter(None, row["pilot_items"].split(";")):
        kind, value = item.strip().split("=")
        digits = value.replace(".", "")
        if not digits.isdigit():
            continue
        text, new_digits = change_last_digit(row["pilot_text"], digits, rng)
        if text:
            new_value = value[:-1] + new_digits[-1]
            items = row["pilot_items"].replace(f"{kind}={value}", f"{kind}={new_value}")
            options.append(("WRONG_VALUE", kind, value, new_value, text, items, row["pilot_callsign"]))

    run = re.search(r"\d+", row["pilot_callsign"])
    if run and row["atco_callsign"]:
        text, new_digits = change_last_digit(row["pilot_text"], run.group(), rng)
        if text:
            callsign = row["pilot_callsign"].replace(run.group(), new_digits, 1)
            options.append(("WRONG_CALLSIGN", "callsign", row["pilot_callsign"], callsign,
                             text, row["pilot_items"], callsign))

    if not options:
        return None
    values = [o for o in options if o[0] == "WRONG_VALUE"]
    callsigns = [o for o in options if o[0] == "WRONG_CALLSIGN"]
    pick = callsigns if callsigns and (not values or rng.random() < 1 / 3) else values
    error, field, old, new, text, items, callsign = rng.choice(pick)
    return {"inj_id": row["pair_id"] + "_inj", "pair_id": row["pair_id"], "split": row["split"],
            "error_type": error, "field": field, "old": old, "new": new,
            "pilot_text": text, "pilot_items": items, "pilot_callsign": callsign, "verdict": error}


def run(cfg):
    gold = pd.read_csv(ROOT / cfg["labels"]["gold"], dtype=str).fillna("")
    correct = gold[(gold["keep"] == "y") & (gold["verdict"] == "CORRECT")]
    rng = random.Random(cfg["inject"]["seed"])
    rows = [r for r in (inject_one(row, rng) for _, row in correct.iterrows()) if r]
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / cfg["labels"]["injected"], index=False)
    return correct, out


if __name__ == "__main__":
    correct, out = run(load_config())
    print(len(out), "injected copies from", len(correct), "correct pairs")
    print(out.groupby(["split", "error_type"]).size())
