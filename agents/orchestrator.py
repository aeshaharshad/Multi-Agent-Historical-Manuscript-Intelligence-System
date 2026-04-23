from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

from agents.entity_agent import extract_entities
from agents.event_agent import extract_events
from agents.timeline_agent import build_timeline


# Shared state that flows between agents
class GraphState(TypedDict):
    raw_text: str
    chunks: List[str]
    persons: List[dict]
    locations: List[dict]
    events: List[dict]
    timeline: List[dict]
    final_output: Dict[str, Any]


# ---- Nodes ----
def split_text_node(state: GraphState) -> dict:
    """Orchestrator: split text into manageable chunks."""
    text = state["raw_text"]
    size = 2000
    chunks = [text[i:i + size] for i in range(0, len(text), size)]
    return {"chunks": chunks}


def entity_node(state: GraphState) -> dict:
    persons, locations = [], []
    for chunk in state["chunks"]:
        result = extract_entities(chunk)
        persons.extend(result["persons"])
        locations.extend(result["locations"])

    # Deduplicate by name (keep highest confidence)
    persons = list({p["name"]: p for p in persons}.values())
    locations = list({l["name"]: l for l in locations}.values())
    return {"persons": persons, "locations": locations}


def event_node(state: GraphState) -> dict:
    all_events = []
    for chunk in state["chunks"]:
        all_events.extend(extract_events(chunk))
    # Deduplicate by event name
    all_events = list({e["event"]: e for e in all_events}.values())
    return {"events": all_events}


def timeline_node(state: GraphState) -> dict:
    return {"timeline": build_timeline(state["events"])}


def finalize_node(state: GraphState) -> dict:
    output = {
        "entities": {
            "persons":   [p["name"] for p in state["persons"]],
            "locations": [l["name"] for l in state["locations"]],
            "events":    [e["event"] for e in state["events"]],
        },
        "timeline": state["timeline"],
        "detailed": {
            "persons":   state["persons"],
            "locations": state["locations"],
            "events":    state["events"],
        },
    }
    return {"final_output": output}


# ---- Build the graph ----
def build_workflow():
    wf = StateGraph(GraphState)

    wf.add_node("split_text",        split_text_node)
    wf.add_node("extract_entities",  entity_node)
    wf.add_node("extract_events",    event_node)
    wf.add_node("build_timeline",    timeline_node)
    wf.add_node("finalize",          finalize_node)

    wf.set_entry_point("split_text")
    wf.add_edge("split_text",        "extract_entities")
    wf.add_edge("extract_entities",  "extract_events")
    wf.add_edge("extract_events",    "build_timeline")
    wf.add_edge("build_timeline",    "finalize")
    wf.add_edge("finalize",          END)

    return wf.compile()