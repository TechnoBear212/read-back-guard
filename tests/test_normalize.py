import pytest

from readback_guard.normalize import callsigns_match, norm_callsign, norm_value


@pytest.mark.parametrize("raw, expected", [
    ("CSA 5PC", "csa5pc"),
    ("csa 5pc", "csa5pc"),
    ("Lufthansa 452", "lufthansa452"),
    ("wizz-air 530", "wizzair530"),
    (None, ""),
    ("", ""),
])
def test_norm_callsign(raw, expected):
    assert norm_callsign(raw) == expected


@pytest.mark.parametrize("atco, pilot, ok", [
    ("csa 536", "csa 536", True),
    ("csa 536", "CSA536", True),
    ("lufthansa 6ma", "6ma", True),         # pilot shortened it
    ("303", "csa 303", True),               # controller shortened it
    ("csa 536", "csa 53", True),            # pilot's text cut off
    ("csa 536", "", True),                  # no callsign in the readback is not WRONG_CALLSIGN
    ("csa 536", None, True),
    ("csa 838w", "csa 88w", False),
    ("lot 525", "lot 1525", False),
    ("swiss 460t", "swiss 160", False),
])
def test_callsigns_match(atco, pilot, ok):
    assert callsigns_match(atco, pilot) == ok


@pytest.mark.parametrize("kind, value, unit, expected", [
    ("level", "120", "FL", "FL120"),
    ("level", "080", "FL", "FL80"),
    ("level", "3000", "ft", "A3000"),
    ("heading", "90", None, "090"),
    ("heading", "270", None, "270"),
    ("speed", "160", None, "160"),
    ("speed", "0160", None, "160"),
    ("frequency", "121.9", None, "121.9"),
    ("frequency", "121.900", None, "121.9"),
    ("frequency", "127.575", None, "127.575"),
    ("frequency", "119", None, "119"),
    ("squawk", "1404", None, "1404"),
    ("squawk", "403", None, "0403"),
    ("runway", "31", None, "31"),
    ("runway", "6", None, "06"),
    ("runway", "24l", None, "24L"),
    ("runway", "RWY 24L", None, "24L"),
    ("qnh", "1013", None, "1013"),
])
def test_norm_value(kind, value, unit, expected):
    assert norm_value(kind, value, unit) == expected


def test_value_without_digits_is_empty():
    assert norm_value("heading", "left") == ""
    assert norm_value("runway", "papa") == ""
