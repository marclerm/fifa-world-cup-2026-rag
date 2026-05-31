"""Streamlit UI for the World Cup 2026 RAG assistant."""

from __future__ import annotations

import streamlit as st

from wc2026_rag.config import CHAT_MODEL, SUGGESTED_QUESTIONS, VECTOR_DB_DIR
from wc2026_rag.rag import answer_question
from wc2026_rag.simulation import describe_advancement, simulate_group
from wc2026_rag.visualization import vector_figure


def render_context(docs) -> None:
    for doc in docs:
        source = doc.metadata.get("source_name") or doc.metadata.get("source", "unknown")
        with st.expander(source):
            st.write(doc.page_content)


def ask(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Retrieving World Cup context..."):
        answer, docs = answer_question(question, st.session_state.messages[:-1])
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_docs = docs


def chat_tab() -> None:
    st.caption(f"Model: {CHAT_MODEL} | Vector store: {VECTOR_DB_DIR}")

    cols = st.columns(3)
    for idx, question in enumerate(SUGGESTED_QUESTIONS):
        if cols[idx % 3].button(question, use_container_width=True):
            ask(question)
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about fixtures, groups, likely qualifiers, or dataset evidence")
    if prompt:
        ask(prompt)
        st.rerun()

    if st.session_state.get("last_docs"):
        st.subheader("Retrieved Evidence")
        render_context(st.session_state.last_docs)


def vectors_tab() -> None:
    st.write("Inspect the Chroma vector store by projecting embeddings into three dimensions.")
    limit = st.slider("Vectors to display", min_value=50, max_value=1000, value=400, step=50)
    try:
        st.plotly_chart(vector_figure(limit=limit), use_container_width=True)
    except Exception as exc:
        st.warning(f"Vector explorer is not ready yet: {exc}")
        st.info("Run `wc2026-ingest` after setting `OPENAI_API_KEY`.")


def simulator_tab() -> None:
    st.write("Run a lightweight scenario simulation. Replace these teams with any group or custom path.")
    default_teams = "Mexico, South Africa, South Korea, Czechia"
    teams_text = st.text_input("Teams", value=default_teams)
    seed = st.number_input("Simulation seed", min_value=0, max_value=9999, value=7, step=1)
    teams = [team.strip() for team in teams_text.split(",") if team.strip()]
    if len(teams) < 2:
        st.warning("Enter at least two teams.")
        return

    standings, results = simulate_group(teams, seed=int(seed))
    st.dataframe(standings, use_container_width=True, hide_index=True)
    st.write(describe_advancement(standings))
    with st.expander("Match results"):
        for result in results:
            st.write(
                f"{result.home_team} {result.home_goals}-{result.away_goals} {result.away_team}"
            )


def main() -> None:
    st.set_page_config(page_title="World Cup 2026 RAG", layout="wide")
    st.title("FIFA World Cup 2026 Conversational AI")
    st.caption("RAG assistant, scenario simulator, and Chroma vector explorer.")

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_docs", [])

    chat, vectors, simulator = st.tabs(["Chat", "3D Vectors", "Simulator"])
    with chat:
        chat_tab()
    with vectors:
        vectors_tab()
    with simulator:
        simulator_tab()


if __name__ == "__main__":
    main()
