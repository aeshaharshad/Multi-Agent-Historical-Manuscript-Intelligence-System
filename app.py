import streamlit as st
from agents.orchestrator import build_workflow
from graph.neo4j_client import KnowledgeGraph

st.set_page_config(
    page_title="Manuscript Intelligence System",
    layout="wide",
)

# --- Header ---
st.title("Historical Manuscript Intelligence System")
st.caption(
    "A multi-agent pipeline that extracts structured knowledge — people, places, "
    "events, and timelines — from unstructured historical text, and builds a Neo4j "
    "knowledge graph for querying."
)
st.divider()

# --- Sidebar ---
with st.sidebar:
    st.subheader("Configuration")
    push_to_neo4j = st.checkbox("Build Neo4j knowledge graph", value=True)
    clear_first = st.checkbox("Clear existing graph before building", value=True)

    st.divider()
    st.subheader("Pipeline")
    st.markdown(
        """
        **Orchestrator Agent** — splits the document into chunks

        **Entity Agent** — extracts persons and locations

        **Event Agent** — identifies historical events

        **Timeline Agent** — orders events chronologically
        """
    )

    st.divider()
    st.caption("Built with LangGraph, Google Gemini, and Neo4j AuraDB.")

# --- Input section ---
st.subheader("Input")

col_upload, col_paste = st.columns(2)

with col_upload:
    uploaded = st.file_uploader(
        "Upload a document",
        type=["txt", "pdf"],
        help="Supports .txt and .pdf files.",
    )

with col_paste:
    text_input = st.text_area(
        "Or paste text directly",
        height=180,
        placeholder="Paste a historical passage — a speech, chronicle, or encyclopedia article...",
    )

# Resolve input
text = ""
if uploaded:
    if uploaded.name.endswith(".txt"):
        text = uploaded.read().decode("utf-8", errors="ignore")
    elif uploaded.name.endswith(".pdf"):
        import pypdf
        reader = pypdf.PdfReader(uploaded)
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
elif text_input:
    text = text_input

if text:
    st.caption(f"Input loaded: {len(text):,} characters")

st.divider()

# --- Run ---
analyze_clicked = st.button("Run Analysis", type="primary", key="analyze_btn")

if analyze_clicked:
    if not text:
        st.warning("Please upload a file or paste text before running the analysis.")
    else:
        with st.spinner("Running multi-agent analysis..."):
            workflow = build_workflow()
            result = workflow.invoke({"raw_text": text})
            output = result["final_output"]

        st.success("Analysis complete.")
        st.divider()

        # --- Summary metrics ---
        st.subheader("Extraction Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("Persons identified", len(output["entities"]["persons"]))
        m2.metric("Locations identified", len(output["entities"]["locations"]))
        m3.metric("Events identified", len(output["entities"]["events"]))

        st.divider()

        # --- Results tabs ---
        tab_overview, tab_timeline, tab_details, tab_json = st.tabs(
            ["Overview", "Timeline", "Detailed Records", "Structured Output (JSON)"]
        )

        with tab_overview:
            st.markdown("**Persons**")
            persons = output["entities"]["persons"]
            st.write(", ".join(persons) if persons else "_No persons identified._")

            st.markdown("**Locations**")
            locations = output["entities"]["locations"]
            st.write(", ".join(locations) if locations else "_No locations identified._")

            st.markdown("**Events**")
            events = output["entities"]["events"]
            if events:
                for e in events:
                    st.markdown(f"- {e}")
            else:
                st.write("_No events identified._")

        with tab_timeline:
            if not output["timeline"]:
                st.info("No dated events were extracted from the text.")
            else:
                for item in output["timeline"]:
                    location_note = f"  —  {item['location']}" if item.get("location") else ""
                    st.markdown(
                        f"**{item['year']}**  |  {item['event']}{location_note}"
                    )

        with tab_details:
            st.markdown("**Persons**")
            if output["detailed"]["persons"]:
                st.dataframe(output["detailed"]["persons"], use_container_width=True)
            else:
                st.caption("No records.")

            st.markdown("**Locations**")
            if output["detailed"]["locations"]:
                st.dataframe(output["detailed"]["locations"], use_container_width=True)
            else:
                st.caption("No records.")

            st.markdown("**Events**")
            if output["detailed"]["events"]:
                st.dataframe(output["detailed"]["events"], use_container_width=True)
            else:
                st.caption("No records.")

        with tab_json:
            st.json(output)
            import json
            st.download_button(
                "Download JSON",
                data=json.dumps(output, indent=2, ensure_ascii=False).encode("utf-8"),
                file_name="manuscript_output.json",
                mime="application/json",
            )

        # --- Neo4j ---
        if push_to_neo4j:
            st.divider()
            st.subheader("Knowledge Graph")
            with st.spinner("Writing nodes and relationships to Neo4j..."):
                try:
                    kg = KnowledgeGraph()
                    if clear_first:
                        kg.clear_graph()
                    kg.build_graph(output)
                    kg.close()
                    st.success(
                        "Knowledge graph written successfully. "
                        "Open the Neo4j Browser and run `MATCH (n) RETURN n LIMIT 100` "
                        "to visualize the graph."
                    )
                except Exception as e:
                    st.error(f"Could not write to Neo4j: {e}")