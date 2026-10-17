import json
import time

import pandas as pd

from readback_guard.config import ROOT, load_config


def gold_segment_ids(cfg):
    gold = pd.read_csv(ROOT / cfg["labels"]["gold"], dtype=str).fillna("")
    gold = gold[gold["keep"] == "y"]
    interim = ROOT / cfg["data"]["interim_dir"]
    pairs = pd.read_parquet(interim / "candidate_pairs.parquet")
    turns = pd.read_parquet(interim / "turns.parquet")

    pairs = pairs[pairs["pair_id"].isin(gold["pair_id"])]
    turn_ids = set(pairs["atco_turn_id"]) | set(pairs["pilot_turn_id"])
    wanted = turns[turns["turn_id"].isin(turn_ids)]
    return [u for ids in wanted["utt_ids"] for u in ids]


def done_ids(path):
    if not path.exists():
        return set()
    with open(path) as f:
        return {json.loads(line)["utt_id"] for line in f}


def load_model(model_id, device):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=torch.float16).to(device)
    model.eval()
    return processor, model


def transcribe(array, processor, model):
    import torch

    f = processor(array, sampling_rate=16000, return_tensors="pt").input_features
    f = f.to(model.device, torch.float16)
    o = model.generate(f, language="en", task="transcribe", num_beams=1, do_sample=False,
                       use_cache=False, return_dict_in_generate=True, output_scores=True)
    text = processor.batch_decode(o.sequences, skip_special_tokens=True)[0].strip()

    special = set(processor.tokenizer.all_special_ids)
    words = []
    for step in o.scores:
        p, tok = step[0].float().softmax(-1).max(-1)
        if tok.item() in special:
            continue
        piece = processor.decode([tok.item()])
        if piece.startswith(" ") or not words:
            words.append({"word": piece.strip(), "conf": p.item()})
        else:
            words[-1]["word"] += piece
            words[-1]["conf"] = min(words[-1]["conf"], p.item())
    for w in words:
        w["conf"] = round(w["conf"], 4)
    return text, words


def run(cfg):
    from datasets import load_dataset

    out = ROOT / cfg["asr"]["out_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    todo = set(gold_segment_ids(cfg)) - done_ids(out)
    print(len(todo), "segments to transcribe")
    if not todo:
        return

    ds = load_dataset(cfg["data"]["dataset"], split=cfg["data"]["hf_split"])
    ds = ds.filter(lambda i: i in todo, input_columns="id")
    processor, model = load_model(cfg["asr"]["model"], cfg["asr"]["device"])

    for n, row in enumerate(ds, 1):
        start = time.perf_counter()
        samples = row["audio"].get_all_samples()
        assert samples.sample_rate == 16000
        text, words = transcribe(samples.data[0].numpy(), processor, model)
        record = {"utt_id": row["id"], "asr_model": cfg["asr"]["model"], "hyp_text": text,
                  "words": words, "seconds": round(time.perf_counter() - start, 2)}
        with open(out, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"{n}/{len(ds)}  {record['seconds']:.1f}s  {text}")


def run_own(cfg):
    """Transcribe your own recordings (data/own/*.wav) into their own cache file."""
    import librosa

    out = ROOT / cfg["own"]["out_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    todo = [p for p in sorted((ROOT / cfg["own"]["audio_dir"]).glob("*.wav")) if p.stem not in done_ids(out)]
    print(len(todo), "recordings to transcribe")
    if not todo:
        return
    processor, model = load_model(cfg["asr"]["model"], cfg["asr"]["device"])
    for n, path in enumerate(todo, 1):
        start = time.perf_counter()
        array, _ = librosa.load(path, sr=16000, mono=True)
        text, words = transcribe(array, processor, model)
        record = {"utt_id": path.stem, "asr_model": cfg["asr"]["model"], "hyp_text": text,
                  "words": words, "seconds": round(time.perf_counter() - start, 2)}
        with open(out, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"{n}/{len(todo)}  {record['seconds']:.1f}s  {path.stem}: {text}")


if __name__ == "__main__":
    import sys

    cfg = load_config()
    run_own(cfg) if sys.argv[1:] == ["own"] else run(cfg)
