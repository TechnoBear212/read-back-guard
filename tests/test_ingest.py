import pandas as pd

from readback_guard.ingest import build_turns, find_pairs, to_utterances

ROWS = [
    ("uwb-atcc_TWR-A_000000_000500_AT", "csa one two three line up runway three one"),
    ("uwb-atcc_TWR-A_000550_000700_AT", "and wind calm"),
    ("uwb-atcc_TWR-A_000800_001000_PI", "line up runway three one"),
    ("uwb-atcc_TWR-A_001040_001100_PI", "csa one two three"),
    ("uwb-atcc_TWR-A_001600_001700_PI", "tower"),
    ("uwb-atcc_TWR-B_000000_000300_AT", "contact ground one two one decimal nine"),
    ("uwb-atcc_TWR-B_000400_000600_PIAT", "one two one nine tower good day"),
    ("uwb-atcc_TWR-B_000700_000800_PI", "hello"),
]


def make_segments():
    raw = pd.DataFrame(ROWS, columns=["id", "text"])
    times = raw["id"].str.rsplit("_", n=3, expand=True)
    raw["segment_start_time"] = times[1].astype(int) / 100
    raw["segment_end_time"] = times[2].astype(int) / 100
    return to_utterances(raw)


def test_merges_close_same_speaker_segments():
    _, turns = build_turns(make_segments(), max_gap=1.0)
    assert turns["n_segments"].tolist() == [2, 2, 1, 1, 1, 1]


def test_piat_blocks_pairing():
    _, turns = build_turns(make_segments(), max_gap=1.0)
    pairs = find_pairs(turns)
    assert len(pairs) == 1
    assert pairs.loc[0, "pilot_text"] == "line up runway three one csa one two three"