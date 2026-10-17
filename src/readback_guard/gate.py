from readback_guard.schemas import Verdict

NUMBER_WORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "decimal", "hundred", "thousand"}


def lowest_number_conf(words):
    """Lowest confidence among the number words in one transmission. None if there is no number word."""
    confs = [w["conf"] for w in words if w["word"] in NUMBER_WORDS]
    return min(confs) if confs else None


def apply_gate(verdict, atco_words, pilot_words, threshold):
    """Return SAY_AGAIN if a number word on either side is below threshold, else the verdict unchanged."""
    for side, words in (("controller", atco_words), ("pilot", pilot_words)):
        c = lowest_number_conf(words)
        if c is not None and c < threshold:
            return Verdict(verdict="SAY_AGAIN", items=verdict.items,
                           reason=f"a number in the {side} transmission was heard with confidence {c:.2f}")
    return verdict
