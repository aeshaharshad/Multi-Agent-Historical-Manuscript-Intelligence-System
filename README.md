# Historical Manuscript Intelligence System

A multi-agent natural language processing pipeline that transforms unstructured historical text into a structured, queryable knowledge graph. The system uses specialized language-model agents — orchestrated through LangGraph — to extract named entities, historical events, and temporal relationships from raw manuscripts, then persists the result as a property graph in Neo4j for downstream querying and analysis.

Input
<img width="747" height="213" alt="image" src="https://github.com/user-attachments/assets/4f5f503f-e25a-49ae-a705-ca527699363f" />
<img width="877" height="514" alt="image" src="https://github.com/user-attachments/assets/b3fca31c-8163-4e64-ab23-77ae0ea3acaa" />
<img width="907" height="566" alt="image" src="https://github.com/user-attachments/assets/9323abc4-a6a2-4b5d-aa3e-c5b2b5ea2402" />
<img width="872" height="626" alt="image" src="https://github.com/user-attachments/assets/eab2cdc7-4bde-49be-a22f-52a5657c025d" />



**Live demo:** https://cayvgvfdkmbqzdrrldu2mt.streamlit.app/

---

## Motivation

Historical documents — speeches, chronicles, memoirs, scholarly articles — contain dense networks of people, places, and events, but this information is locked inside prose. Traditional named-entity recognition systems extract surface-level entities but miss the relational structure that gives a historical text its meaning. This project explores whether a cooperative multi-agent architecture, with each agent specialized for a narrow extraction task, can produce richer and more reliable structured output than a single monolithic prompt.

The broader question this prototype is meant to probe: **how far can lightweight agent orchestration substitute for fine-tuned domain models in low-resource information-extraction settings?**

---

## System Architecture

The system follows a directed-acyclic-graph (DAG) workflow implemented in LangGraph, where each node is a specialized agent operating on a shared state object.

```
                     ┌──────────────────────┐
   Raw manuscript ──▶│ Orchestrator Agent   │
                     │ (chunking + routing) │
                     └──────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
          ┌──────────────────┐    ┌──────────────────┐
          │  Entity Agent    │    │  Event Agent     │
          │  Persons + Locs  │    │  Dated events    │
          └────────┬─────────┘    └─────────┬────────┘
                   │                        │
                   └───────────┬────────────┘
                               ▼
                     ┌──────────────────────┐
                     │  Timeline Agent      │
                     │  Chronological sort  │
                     └──────────┬───────────┘
                                ▼
                     ┌──────────────────────┐
                     │  Finalizer           │
                     │  JSON + graph build  │
                     └──────────┬───────────┘
                                │
                   ┌────────────┴────────────┐
                   ▼                         ▼
            Structured JSON           Neo4j Knowledge Graph
            (for downstream use)      (for querying / visualization)
```

### Agent Responsibilities

| Agent | Input | Output | Model call |
|-------|-------|--------|------------|
| **Orchestrator** | Raw text | List of fixed-size text chunks | None (deterministic) |
| **Entity Agent** | Text chunk | `Person` and `Location` records with confidence scores | Gemini 2.5 Flash with structured output |
| **Event Agent** | Text chunk | `Event` records with year, location, and participants | Gemini 2.5 Flash with structured output |
| **Timeline Agent** | All extracted events | Chronologically sorted, year-filtered timeline | None (deterministic) |
| **Finalizer** | All prior state | Unified JSON payload | None (deterministic) |

The deterministic nodes (Orchestrator, Timeline, Finalizer) act as the "glue" that keeps LLM calls bounded and cheap. Only two of the five nodes incur model cost, and both use structured output (Pydantic-validated) to reduce parsing failures.

### Why LangGraph?

Unlike a plain chain, LangGraph maintains a typed shared state (`GraphState`) across nodes. This gives three practical benefits in a research setting:

1. **Inspectability** — every intermediate state is serializable, which matters when you're debugging why an event got dropped.
2. **Reproducibility** — the graph topology is explicit, not buried inside prompt text.
3. **Incremental extension** — adding a relationship-extraction agent or a retrieval step is one new node, not a rewrite.

---

## Knowledge Graph Schema

The extracted output is materialized as a property graph in Neo4j.

### Node labels

| Label | Properties |
|-------|-----------|
| `Person` | `name`, `role`, `confidence` |
| `Location` | `name`, `type`, `confidence` |
| `Event` | `name`, `year`, `confidence` |

### Relationship types

| Relationship | Direction | Semantics |
|-------------|-----------|-----------|
| `PARTICIPATED_IN` | `(Person) → (Event)` | A named individual is attested as a participant |
| `OCCURRED_IN` | `(Event) → (Location)` | The event is attested to have happened at this location |

### Example Cypher queries

Retrieve all dated events in chronological order:
```cypher
MATCH (e:Event) WHERE e.year IS NOT NULL
RETURN e.name, e.year ORDER BY e.year;
```

Find people who participated in the same event (co-occurrence):
```cypher
MATCH (p1:Person)-[:PARTICIPATED_IN]->(e:Event)<-[:PARTICIPATED_IN]-(p2:Person)
WHERE p1.name < p2.name
RETURN p1.name, p2.name, e.name;
```

Discover cross-document connections through shared locations:
```cypher
MATCH (p:Person)-[:PARTICIPATED_IN]->(e:Event)-[:OCCURRED_IN]->(l:Location {name: "France"})
RETURN p.name, e.name, e.year ORDER BY e.year;
```
// See everything
MATCH (n) RETURN n LIMIT 100;

// All people who participated in events
MATCH (p:Person)-[:PARTICIPATED_IN]->(e:Event)
RETURN p.name, e.name, e.year
ORDER BY e.year;

// Events that happened in each location
MATCH (e:Event)-[:OCCURRED_IN]->(l:Location)
RETURN l.name AS place, collect(e.name) AS events;

// Who fought alongside whom? (people in the same event)
MATCH (p1:Person)-[:PARTICIPATED_IN]->(e:Event)<-[:PARTICIPATED_IN]-(p2:Person)
WHERE p1.name < p2.name
RETURN p1.name, p2.name, e.name;

// Timeline view
MATCH (e:Event)
WHERE e.year IS NOT NULL
RETURN e.name, e.year
ORDER BY e.year;
### Graph persistence semantics

The `build_graph` routine uses `MERGE` rather than `CREATE`, which means entities with identical names are deduplicated across sessions. The UI exposes a "clear graph" toggle that controls whether each run replaces the existing graph or accumulates into it. Accumulation is the more interesting mode — it allows the graph to grow into a cross-document corpus where shared entities (a country, a major figure) become natural join points between independently processed texts.

---

## Technical Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Agent orchestration | LangGraph | Typed stateful DAG, lightweight, good debuggability |
| Language model | Google Gemini 2.5 Flash | Free tier with generous quota, fast, native structured output |
| Schema enforcement | Pydantic v2 | Compile-time guarantees on LLM output shape |
| Graph database | Neo4j AuraDB (managed) | Industry-standard property graph, native Cypher support |
| Interface | Streamlit | Minimal boilerplate, supports file upload and JSON rendering |
| Document parsing | pypdf | Reliable text extraction from PDF manuscripts |

---

## Repository Layout

```
manuscript-ai/
├── app.py                      Streamlit UI and workflow driver
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example                Template for required environment variables
│
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py         LangGraph workflow definition and node functions
│   ├── entity_agent.py         Person/Location extraction + shared LLM factory
│   ├── event_agent.py          Event extraction with temporal metadata
│   └── timeline_agent.py       Chronological ordering (deterministic)
│
├── graph/
│   ├── __init__.py
│   └── neo4j_client.py         Neo4j driver wrapper and graph-building logic
│
└── data/
    └── napoleon.txt            Sample historical text for testing
```

---

## Setup and Installation

### Prerequisites

- Python 3.10 or higher
- A Google AI Studio API key (free, no billing required) — https://aistudio.google.com/app/apikey
- A Neo4j AuraDB Free instance — https://console.neo4j.io

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/manuscript-ai.git
cd manuscript-ai

python -m venv venv
source venv/bin/activate           # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_key_here
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
```

A template is provided in `.env.example`.

### Running

```bash
streamlit run app.py
```

The UI will open at `http://localhost:8501`. Upload a `.txt` or `.pdf` file, or paste text directly, and click **Run Analysis**.

---

## Evaluation Approach

The system was tested qualitatively on three categories of input:

1. **Encyclopedic prose** (Wikipedia articles on Napoleon, the French Revolution, World War I)
2. **Primary-source style text** (excerpts from memoirs and contemporaneous accounts from Project Gutenberg)
3. **Dense factual passages** (compressed summaries with many named entities per sentence)

For each input, the extracted entities were manually compared against the source text to assess:

- **Recall** — were significant entities missed?
- **Precision** — were any non-entities incorrectly labeled?
- **Relationship correctness** — do `PARTICIPATED_IN` edges reflect what the text actually says?
- **Temporal correctness** — are years extracted faithfully, or fabricated?

A rigorous quantitative evaluation on an annotated benchmark (e.g., a subset of HIPE-2022 or a hand-annotated set) is a natural next step but is out of scope for this prototype.

---

## Deployment

The application is deployable on Streamlit Community Cloud at no cost. Configuration involves pushing the repository to GitHub and supplying the four environment variables as secrets in the Streamlit Cloud dashboard. The Neo4j AuraDB Free instance persists independently and is reachable from the deployed app over TLS.

---

## Acknowledgments and References

This project builds on and was informed by:

- **LangGraph** — Harrison Chase et al., LangChain AI. https://langchain-ai.github.io/langgraph/
- **Google Gemini API** — Google AI Studio. https://ai.google.dev/
- **Neo4j** — Neo4j Inc. https://neo4j.com/docs/
- The broader literature on information extraction with LLMs, particularly the line of work on structured prompting and constrained decoding.

Sample text was drawn from the public-domain Wikipedia article on Napoleon Bonaparte and from Project Gutenberg.

---

