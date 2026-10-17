from readback_guard.gate import apply_gate, lowest_number_conf
from readback_guard.schemas import Verdict


def w(*pairs):
    return [{"word": word, "conf": conf} for word, conf in pairs]


CORRECT = Verdict(verdict="CORRECT", reason="all items read back")
WRONG = Verdict(verdict="WRONG_VALUE", reason="squawk 1404 read back as 1402")
SURE = w(("squawk", 0.99), ("one", 0.98), ("four", 0.97), ("zero", 0.99), ("four", 0.96))
UNSURE = w(("squawk", 0.99), ("one", 0.98), ("four", 0.97), ("zero", 0.99), ("two", 0.31))


def test_lowest_number_conf_ignores_other_words():
    assert lowest_number_conf(w(("hello", 0.10), ("five", 0.80), ("two", 0.95))) == 0.80


def test_lowest_number_conf_none_without_numbers():
    assert lowest_number_conf(w(("roger", 0.5), ("bye", 0.2))) is None
    assert lowest_number_conf([]) is None


def test_confident_numbers_keep_verdict():
    assert apply_gate(CORRECT, SURE, SURE, 0.5).verdict == "CORRECT"


def test_uncertain_pilot_number_says_again():
    v = apply_gate(CORRECT, SURE, UNSURE, 0.5)
    assert v.verdict == "SAY_AGAIN"
    assert "pilot" in v.reason


def test_uncertain_controller_number_says_again():
    assert apply_gate(CORRECT, UNSURE, SURE, 0.5).verdict == "SAY_AGAIN"


def test_uncertain_number_overrides_an_error_verdict():
    v = apply_gate(WRONG, SURE, UNSURE, 0.5)
    assert v.verdict == "SAY_AGAIN"
    assert v.items == WRONG.items


def test_threshold_zero_never_fires():
    assert apply_gate(WRONG, UNSURE, UNSURE, 0.0).verdict == "WRONG_VALUE"


def test_no_numbers_no_gate():
    assert apply_gate(CORRECT, w(("roger", 0.1)), w(("bye", 0.1)), 0.9).verdict == "CORRECT"


def test_empty_words_no_gate():
    assert apply_gate(CORRECT, [], [], 0.9).verdict == "CORRECT"
