import numpy as np
import pandas as pd

from readback_guard.config import ROOT, load_config

SPEAKERS = {"AT": "ATCO", "PI": "PILOT", "PIAT": "UNKNOWN"}

VALUE_WORDS = {
    "frequency": "decimal",
    "squawk": "squawk",
    "level": "flight level|altitude|feet",
    "heading": "heading",
    "speed": "speed",
    "runway": "runway",
}


def load_segments(dataset, hf_split):
    from datasets import load_dataset

    ds = load_dataset(dataset, split=hf_split, streaming=True)
    cols = ["id", "text", "segment_start_time", "segment_end_time"]
    return pd.DataFrame(list(ds.select_columns(cols)))


def to_utterances(raw):
    seg = raw.rename(columns={"id": "utt_id", "text": "ref_text",
                              "segment_start_time": "start", "segment_end_time": "end"})
    parts = seg["utt_id"].str.rsplit("_", n=3, expand=True)
    seg["recording_id"] = parts[0].str.removeprefix("uwb-atcc_")
    seg["speaker"] = parts[3].map(SPEAKERS)
    seg["source"] = "uwb_atcc"
    return seg.sort_values(["recording_id", "start"]).reset_index(drop=True)


def build_turns(seg, max_gap):
    seg = seg.copy()
    prev = seg.groupby("recording_id").shift(1)
    seg["gap"] = seg["start"] - prev["end"]
    new_turn = (
        (seg["recording_id"] != seg["recording_id"].shift())
        | (seg["speaker"] != seg["speaker"].shift())
        | (seg["gap"] > max_gap)
    )
    seg["turn_id"] = new_turn.cumsum()
    turns = (
        seg.groupby("turn_id")
           .agg(recording_id=("recording_id", "first"),
                speaker=("speaker", "first"),
                start=("start", "first"),
                end=("end", "last"),
                ref_text=("ref_text", " ".join),
                utt_ids=("utt_id", list),
                n_segments=("utt_id", "size"))
           .reset_index()
    )
    return seg.drop(columns="gap"), turns


def find_pairs(turns):
    nxt = turns.groupby("recording_id").shift(-1)
    is_pair = (turns["speaker"] == "ATCO") & (nxt["speaker"] == "PILOT")
    pairs = pd.DataFrame({
        "recording_id": turns.loc[is_pair, "recording_id"],
        "atco_turn_id": turns.loc[is_pair, "turn_id"],
        "pilot_turn_id": nxt.loc[is_pair, "turn_id"].astype(int),
        "atco_text": turns.loc[is_pair, "ref_text"],
        "pilot_text": nxt.loc[is_pair, "ref_text"],
    }).reset_index(drop=True)
    pairs.insert(0, "pair_id", [f"p{i:04d}" for i in range(len(pairs))])
    return pairs


def flag_values(pairs):
    pairs = pairs.copy()
    for name, pattern in VALUE_WORDS.items():
        pairs[name] = pairs["atco_text"].str.contains(pattern)
    pairs["has_value"] = pairs[list(VALUE_WORDS)].any(axis=1)
    return pairs


def dev_recordings(recording_ids, dev_fraction, seed):
    recs = np.sort(pd.unique(recording_ids))
    np.random.default_rng(seed).shuffle(recs)
    return set(recs[: round(dev_fraction * len(recs))])


def run(cfg):
    seg = to_utterances(load_segments(cfg["data"]["dataset"], cfg["data"]["hf_split"]))
    seg, turns = build_turns(seg, cfg["ingest"]["max_merge_gap_s"])
    pairs = flag_values(find_pairs(turns))

    dev = dev_recordings(seg["recording_id"], cfg["split"]["dev_fraction"], cfg["split"]["seed"])
    for df in (seg, turns, pairs):
        df["split"] = np.where(df["recording_id"].isin(dev), "dev", "test")

    out = ROOT / cfg["data"]["interim_dir"]
    out.mkdir(parents=True, exist_ok=True)
    seg[seg["speaker"] != "UNKNOWN"].to_parquet(out / "utterances.parquet", index=False)
    turns[turns["speaker"] != "UNKNOWN"].to_parquet(out / "turns.parquet", index=False)
    pairs.to_parquet(out / "candidate_pairs.parquet", index=False)
    return seg, turns, pairs


if __name__ == "__main__":
    seg, turns, pairs = run(load_config())
    print(len(seg), "segments,", len(turns), "turns,", len(pairs), "pairs")
    print(pairs.groupby("split")["has_value"].agg(["size", "sum"]))