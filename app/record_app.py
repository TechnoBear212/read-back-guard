from pathlib import Path

import pandas as pd
import streamlit as st

from readback_guard.config import ROOT, load_config

cfg = load_config()
scripts = pd.read_csv(ROOT / cfg["labels"]["own"], dtype=str).fillna("")
audio_dir = ROOT / cfg["own"]["audio_dir"]
audio_dir.mkdir(parents=True, exist_ok=True)


def path(rec_id, side):
    return audio_dir / f"{rec_id}_{side}.wav"


st.title("Record your readbacks")
done = [r for r in scripts.rec_id if path(r, "atco").exists() and path(r, "pilot").exists()]
st.progress(len(done) / len(scripts), text=f"{len(done)} of {len(scripts)} pairs recorded")

rec_id = st.selectbox("Script", scripts.rec_id, format_func=lambda r: f"{r} {'✓' if r in done else ''}")
row = scripts.set_index("rec_id").loc[rec_id]

for side, label in (("atco", "Controller"), ("pilot", "Pilot")):
    st.subheader(label)
    st.markdown(f"> {row[f'{side}_text']}")
    clip = st.audio_input(f"Record the {label.lower()}", key=f"{rec_id}_{side}")
    if clip is not None:
        path(rec_id, side).write_bytes(clip.getvalue())
        st.success(f"saved {path(rec_id, side).name}")
    elif path(rec_id, side).exists():
        st.audio(str(path(rec_id, side)))
        st.caption("already saved; record again to replace it")
