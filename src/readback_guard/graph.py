from typing import TypedDict

from google.genai import errors
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy
from pydantic import ValidationError

from readback_guard.compare import compare
from readback_guard.extract import extract
from readback_guard.gate import apply_gate
from readback_guard.schemas import Clearance, Verdict


class State(TypedDict, total=False):
    atco_text: str
    pilot_text: str
    atco: Clearance
    pilot: Clearance
    verdict: Verdict
    atco_words: list
    pilot_words: list


def build_graph(cfg):
    def extract_node(state):
        return {"atco": extract(state["atco_text"], cfg), "pilot": extract(state["pilot_text"], cfg)}

    def compare_node(state):
        return {"verdict": compare(state["atco"], state["pilot"])}

    def gate_node(state):
        threshold = cfg["gate"]["threshold"]
        if threshold is None or not (state.get("atco_words") or state.get("pilot_words")):
            return {}
        return {"verdict": apply_gate(state["verdict"], state.get("atco_words", []),
                                      state.get("pilot_words", []), threshold)}

    retry = RetryPolicy(max_attempts=3, retry_on=(ValidationError, errors.ServerError))
    g = StateGraph(State)
    g.add_node("extract", extract_node, retry_policy=retry)
    g.add_node("compare", compare_node)
    g.add_node("gate", gate_node)
    g.add_edge(START, "extract")
    g.add_edge("extract", "compare")
    g.add_edge("compare", "gate")
    g.add_edge("gate", END)
    return g.compile()
