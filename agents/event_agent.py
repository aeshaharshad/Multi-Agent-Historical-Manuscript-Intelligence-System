from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional
from agents.entity_agent import get_llm


class Event(BaseModel):
    event: str = Field(description="Short name/description of the event")
    year: Optional[int] = Field(default=None, description="Year if mentioned")
    date: Optional[str] = Field(default=None, description="Full date if available")
    location: Optional[str] = Field(default=None, description="Where it happened")
    participants: List[str] = Field(default_factory=list, description="People involved")
    confidence: float = Field(default=0.9)


class EventList(BaseModel):
    events: List[Event]


def extract_events(text: str) -> List[dict]:
    """Event Agent — finds battles, treaties, coronations, revolutions, etc."""
    llm = get_llm().with_structured_output(EventList)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a historian. Extract every significant HISTORICAL EVENT from the text. "
         "For each event include: event name, year (if known), location, participants, "
         "and a confidence score 0.0–1.0. "
         "Examples of events: battles, treaties, coronations, revolutions, sieges."),
        ("user", "Text:\n{text}"),
    ])

    chain = prompt | llm
    result = chain.invoke({"text": text})
    return [e.model_dump() for e in result.events]