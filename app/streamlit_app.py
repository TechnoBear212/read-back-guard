import io

import pandas as pd
import streamlit as st

from readback_guard.asr import load_asr_cases, load_own_cases, norm_text
from readback_guard.config import load_config
from readback_guard.graph import build_graph

cfg = load_config()
COLOURS = {"CORRECT": "green", "SAY_AGAIN": "orange"}


@st.cache_resource
def graph():
    return build_graph(cfg)


@st.cache_resource
def whisper():
    from readback_guard.transcribe import load_model
    return load_model(cfg["asr"]["model"], cfg["asr"]["device"])


@st.cache_data
def cases():
    frames = [load_asr_cases(cfg, "dev")]
    try:
        frames.append(load_own_cases(cfg, "asr"))
    except FileNotFoundError:
        pass
    return pd.concat(frames, ignore_index=True)


def transcribe_clip(clip):
    import librosa
    from readback_guard.transcribe import transcribe

    array, _ = librosa.load(io.BytesIO(clip.getvalue()), sr=16000, mono=True)
    text, words = transcribe(array, *whisper())
    words = [{"word": p, "conf": w["conf"]} for w in words for p in norm_text(w["word"]).split()]
    return norm_text(text), words


def show(state, title):
    v = state["verdict"]
    st.markdown(f"**{title}:** :{COLOURS.get(v.verdict, 'red')}[{v.verdict}]  \n{v.reason}")
    if v.items:
        st.dataframe(pd.DataFrame([i.model_dump() for i in v.items]), hide_index=True)
    with st.expander("What Gemini extracted"):
        st.json({"controller": state["atco"].model_dump(), "pilot": state["pilot"].model_dump()})


def words_line(words):
    t = cfg["gate"]["threshold"] or 0
    return " ".join(f"**{w['word']}**({w['conf']:.2f})" if w["conf"] < t else w["word"]
                    for w in words)


st.title("Readback Guard")
st.caption(f"Whisper ATC large-v3, {cfg['llm']['model']}, Python rules, say-again gate at "
           f"confidence {cfg['gate']['threshold']}")
mode = st.radio("Input", ["Record a new pair", "Pick a pair from the data"], horizontal=True)

if mode == "Record a new pair":
    atco_clip = st.audio_input("Controller")
    pilot_clip = st.audio_input("Pilot readback")
    if atco_clip and pilot_clip and st.button("Check readback", type="primary"):
        with st.spinner("Transcribing with Whisper..."):
            atco_text, atco_words = transcribe_clip(atco_clip)
            pilot_text, pilot_words = transcribe_clip(pilot_clip)
        st.markdown(f"**Controller heard:** {words_line(atco_words)}")
        st.markdown(f"**Pilot heard:** {words_line(pilot_words)}")
        state = graph().invoke({"atco_text": atco_text, "pilot_text": pilot_text,
                                "atco_words": atco_words, "pilot_words": pilot_words})
        show(state, "Verdict")
else:
    df = cases()
    case_id = st.selectbox("Pair", df.case_id, format_func=lambda c: f"{c} (gold: {df.set_index('case_id').loc[c, 'verdict']})")
    r = df.set_index("case_id").loc[case_id]
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Human transcript**  \nController: {r.atco_ref}  \nPilot: {r.pilot_ref}")
        show(graph().invoke({"atco_text": r.atco_ref, "pilot_text": r.pilot_ref}), "On human text")
    with right:
        st.markdown(f"**Whisper transcript**  \nController: {words_line(r.atco_words)}  \nPilot: {words_line(r.pilot_words)}")
        show(graph().invoke({"atco_text": r.atco_text, "pilot_text": r.pilot_text,
                             "atco_words": r.atco_words, "pilot_words": r.pilot_words}), "On Whisper text")
    st.caption(f"Gold verdict: {r.verdict}")
