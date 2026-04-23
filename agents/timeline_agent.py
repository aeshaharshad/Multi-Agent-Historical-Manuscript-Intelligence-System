from typing import List


def build_timeline(events: List[dict]) -> List[dict]:
    """Timeline Agent — keeps only events with a year, sorts them."""
    dated = [e for e in events if e.get("year") is not None]
    dated.sort(key=lambda x: x["year"])

    return [
        {
            "event":        e["event"],
            "year":         e["year"],
            "location":     e.get("location"),
            "participants": e.get("participants", []),
        }
        for e in dated
    ]