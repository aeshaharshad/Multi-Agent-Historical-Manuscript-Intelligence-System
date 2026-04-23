from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()


class Person(BaseModel):
    name: str = Field(description="Full name of the person")
    role: str = Field(default="", description="Their role (e.g. General, Emperor)")
    confidence: float = Field(default=0.9, description="Confidence 0-1")


class Location(BaseModel):
    name: str = Field(description="Name of the place")
    type: str = Field(default="", description="city, country, battlefield, region…")
    confidence: float = Field(default=0.9, description="Confidence 0-1")


class EntityExtraction(BaseModel):
    persons: List[Person]
    locations: List[Location]


def get_llm():
    """Shared LLM used by all agents."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        api_key=os.getenv("GOOGLE_API_KEY"),
    )


def extract_entities(text: str) -> dict:
    """Entity Agent — finds PEOPLE and PLACES in a chunk of text."""
    llm = get_llm().with_structured_output(EntityExtraction)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a historical manuscript analyst. "
         "Extract EVERY person (rulers, generals, scholars, etc.) and EVERY location "
         "(cities, countries, regions, battlefields) mentioned in the text. "
         "Assign a confidence score between 0.0 and 1.0 for each. "
         "Return nothing else."),
        ("user", "Text:\n{text}"),
    ])

    chain = prompt | llm
    result = chain.invoke({"text": text})
    return {
        "persons":   [p.model_dump() for p in result.persons],
        "locations": [l.model_dump() for l in result.locations],
    }