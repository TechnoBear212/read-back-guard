from readback_guard.compare import compare
from readback_guard.schemas import Clearance, Item


def clr(callsign, *items):
    return Clearance(callsign=callsign, items=[Item(type=t, value=v, unit=u) for t, v, u in items])


def test_correct_readback():
    atco = clr("csa 536", ("runway", "13", None))
    pilot = clr("csa 536", ("runway", "13", None))
    assert compare(atco, pilot).verdict == "CORRECT"


def test_wrong_squawk():
    atco = clr("sky travel 1102", ("runway", "31", None), ("squawk", "1404", None))
    pilot = clr("sky travel 1102", ("runway", "31", None), ("squawk", "1402", None))
    v = compare(atco, pilot)
    assert v.verdict == "WRONG_VALUE"
    assert [i.result for i in v.items] == ["OK", "WRONG"]


def test_missing_frequency():
    atco = clr("csa 481", ("frequency", "121.9", None))
    pilot = clr("csa 481")
    assert compare(atco, pilot).verdict == "MISSING_ITEM"


def test_wrong_callsign():
    atco = clr("csa 838w", ("speed", "160", None))
    pilot = clr("csa 88w", ("speed", "160", None))
    assert compare(atco, pilot).verdict == "WRONG_CALLSIGN"


def test_wrong_callsign_beats_wrong_value():
    atco = clr("lot 525", ("speed", "160", None))
    pilot = clr("lot 1525", ("speed", "170", None))
    assert compare(atco, pilot).verdict == "WRONG_CALLSIGN"


def test_wrong_value_beats_missing_item():
    atco = clr("csa 1", ("runway", "31", None), ("squawk", "1404", None))
    pilot = clr("csa 1", ("squawk", "1402", None))
    assert compare(atco, pilot).verdict == "WRONG_VALUE"


def test_formatting_differences_are_not_errors():
    atco = clr("CSA 5PC", ("frequency", "121.900", None), ("heading", "90", None))
    pilot = clr("csa 5pc", ("frequency", "121.9", None), ("heading", "090", None))
    assert compare(atco, pilot).verdict == "CORRECT"


def test_no_callsign_in_readback_is_fine():
    atco = clr("csa 024", ("runway", "31", None))
    pilot = clr(None, ("runway", "31", None))
    assert compare(atco, pilot).verdict == "CORRECT"


def test_qnh_is_not_scored():
    atco = clr("csa 1", ("qnh", "1013", None), ("runway", "31", None))
    pilot = clr("csa 1", ("runway", "31", None))
    assert compare(atco, pilot).verdict == "CORRECT"


def test_level_unit_matters():
    atco = clr("csa 1", ("level", "100", "FL"))
    pilot = clr("csa 1", ("level", "100", "ft"))
    assert compare(atco, pilot).verdict == "WRONG_VALUE"


def test_reason_is_filled():
    atco = clr("csa 481", ("frequency", "121.9", None))
    pilot = clr("csa 481")
    assert compare(atco, pilot).reason != ""


def test_value_without_digits_is_ignored():
    atco = clr("csa 1", ("heading", "left", None), ("runway", "31", None))
    pilot = clr("csa 1", ("runway", "31", None))
    assert compare(atco, pilot).verdict == "CORRECT"
