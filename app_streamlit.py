"""
app_streamlit.py
----------------
Local web interface to query the three systems (Parametric LLM, Document RAG,
Graph RAG) and see their answers side by side, instead of from the terminal.

Run with:
  streamlit run app_streamlit.py
"""

import streamlit as st
import importlib
systems = importlib.import_module("03_systems")

system1_parametric_llm = systems.system1_parametric_llm
DocumentRAG = systems.DocumentRAG
GraphRAG = systems.GraphRAG
save_result = systems.save_result
load_results = systems.load_results

st.set_page_config(page_title="AntibioticKG-RAG", layout="wide")
st.title("AntibioticKG-RAG — Comparing the Three Systems")

@st.cache_resource
def init_systems():
    doc_rag = DocumentRAG()
    graph_rag = GraphRAG()
    return doc_rag, graph_rag

with st.spinner("Loading systems (Document RAG + Graph RAG)..."):
    doc_rag, graph_rag = init_systems()

st.success("Systems ready.")

question = st.text_input("Enter your question:", placeholder="E.g.: Who are the authors of the Stokes et al. 2020 paper?")
hop_level = st.selectbox("Hop level (optional, only used to label the saved result):",
                          ["", "1-hop", "2-hop", "3-hop"])

if st.button("Ask all three systems") and question.strip():
    with st.spinner("The three systems are answering..."):
        entry = save_result(question, doc_rag, graph_rag, hop_level)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("System 1 — Parametric LLM")
        st.write(entry["system1"]["answer"])

    with col2:
        st.subheader("System 2 — Document RAG")
        st.write(entry["system2"]["answer"])
        with st.expander("Retrieved chunks"):
            for c in entry["system2"]["retrieved_chunks"]:
                st.text(f"• {c}")

    with col3:
        st.subheader("System 3 — Graph RAG")
        st.write(entry["system3"]["answer"])
        st.caption(f"Query valid: {entry['system3']['query_valid']} | Results: {entry['system3']['results_count']}")
        with st.expander("Generated SPARQL query"):
            st.code(entry["system3"]["sparql_query"], language="sparql")

    st.info("Result saved to results/live_results.json (keyed by question)")

st.divider()
st.subheader("Previously saved questions")
saved = load_results()
if saved:
    for q, data in saved.items():
        with st.expander(f"[{data.get('hop_level', '')}] {q}"):
            st.write("**System 1:**", data["system1"]["answer"])
            st.write("**System 2:**", data["system2"]["answer"])
            st.write("**System 3:**", data["system3"]["answer"])
else:
    st.write("No questions saved yet.")