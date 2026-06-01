"""Gradio UI for the World Cup 2026 RAG assistant."""

from __future__ import annotations

import gradio as gr

from wc2026_rag.config import CHAT_MODEL, SUGGESTED_QUESTIONS, VECTOR_DB_DIR
from wc2026_rag.rag import answer_question, vector_count
from wc2026_rag.simulation import describe_advancement, simulate_group
from wc2026_rag.visualization import vector_figure

WORLD_CUP_CSS = """
:root {
  --wc-green: #3CAC3B;
  --wc-blue: #2A398D;
  --wc-dark-gray: #474A4A;
}

.gradio-container {
  color: var(--wc-blue);
}

.gradio-container h1 {
  color: var(--wc-blue);
  border-bottom: 4px solid var(--wc-green);
  padding-bottom: 0.35rem;
}

.gradio-container h2,
.gradio-container h3 {
  color: var(--wc-blue);
}

button.primary,
.gradio-container .primary {
  background: var(--wc-green) !important;
  border-color: var(--wc-green) !important;
  color: var(--wc-blue) !important;
}

.gradio-container button.secondary {
  border-color: var(--wc-blue) !important;
  color: var(--wc-blue) !important;
}

.gradio-container button.secondary:hover {
  border-color: var(--wc-green) !important;
  color: var(--wc-green) !important;
}

.tab-nav button.selected {
  color: var(--wc-blue) !important;
  border-bottom-color: var(--wc-green) !important;
}

.message.user {
  border-left: 4px solid var(--wc-green) !important;
  color: var(--wc-blue) !important;
}

.message.bot {
  border-left: 4px solid var(--wc-blue) !important;
  color: var(--wc-blue) !important;
}

.wc-status,
.error,
.toast-wrap,
.toast-body {
  color: var(--wc-dark-gray) !important;
}

textarea:focus,
input:focus {
  border-color: var(--wc-green) !important;
  box-shadow: 0 0 0 1px var(--wc-green) !important;
}
"""


def message_text(content) -> str:
    """Normalize Gradio message content to plain text for the RAG layer."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


def format_context(docs) -> str:
    """Render retrieved chunks as Markdown."""
    if not docs:
        return "Retrieved context will appear here after the first answer."

    sections = ["## Retrieved context"]
    for doc in docs:
        source = doc.metadata.get("source_name") or doc.metadata.get("source", "unknown")
        sections.append(f"### {source}\n\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


def append_user_message(message: str, history: list[dict]) -> tuple[str, list[dict]]:
    """Echo the user message immediately before the slower model call."""
    if not message.strip():
        return "", history
    return "", history + [{"role": "user", "content": message.strip()}]


def respond(history: list[dict]) -> tuple[list[dict], str]:
    """Answer the latest chat turn and return updated chat plus evidence."""
    if not history:
        return history, format_context([])

    question = message_text(history[-1]["content"])
    prior = [
        {"role": item["role"], "content": message_text(item["content"])}
        for item in history[:-1]
    ]
    try:
        answer, docs = answer_question(question, prior)
    except Exception as exc:
        answer = f"Retrieval is not ready: {exc}"
        docs = []
    history = history + [{"role": "assistant", "content": answer}]
    return history, format_context(docs)


def ask_suggested(question: str, history: list[dict]) -> tuple[list[dict], str]:
    """Run a suggested question through the same chat path."""
    _, with_user = append_user_message(question, history)
    return respond(with_user)


def run_vector_plot(limit: int):
    """Build the 3D vector figure for Gradio."""
    try:
        return vector_figure(limit=int(limit))
    except Exception as exc:
        raise gr.Error(f"Vector explorer is not ready yet. Run wc2026-ingest first. Details: {exc}")


def run_simulation(teams_text: str, seed: int):
    """Run the group simulator and format results for display."""
    teams = [team.strip() for team in teams_text.split(",") if team.strip()]
    if len(teams) < 2:
        raise gr.Error("Enter at least two teams separated by commas.")

    standings, results = simulate_group(teams, seed=int(seed))
    result_lines = [
        f"- {result.home_team} {result.home_goals}-{result.away_goals} {result.away_team}"
        for result in results
    ]
    summary = describe_advancement(standings) + "\n\n" + "\n".join(result_lines)
    return standings, summary


def build_ui() -> gr.Blocks:
    """Construct the Gradio app."""
    theme = gr.themes.Soft(
        primary_hue="green",
        secondary_hue="green",
        neutral_hue="slate",
    )

    with gr.Blocks(title="FIFA World Cup 2026 RAG", theme=theme, css=WORLD_CUP_CSS) as demo:
        try:
            count_text = f"{vector_count():,} stored chunks"
        except Exception:
            count_text = "vector store not built"

        gr.Markdown(
            "# FIFA World Cup 2026 Conversational AI\n"
            f"<span class='wc-status'>Model: `{CHAT_MODEL}` | "
            f"Vector store: `{VECTOR_DB_DIR}` | {count_text}</span>"
        )

        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=6):
                    chatbot = gr.Chatbot(label="Conversation", height=620)
                    message = gr.Textbox(
                        placeholder="Ask about fixtures, groups, projections, or evidence...",
                        show_label=False,
                    )
                    with gr.Row():
                        send = gr.Button("Ask", variant="primary")
                        clear = gr.ClearButton([message, chatbot], value="Clear")
                with gr.Column(scale=4):
                    context = gr.Markdown(
                        value=format_context([]),
                        label="Retrieved context",
                        height=620,
                    )

            gr.Markdown("### Try a question")
            with gr.Row():
                suggested_buttons = [
                    gr.Button(question, size="sm") for question in SUGGESTED_QUESTIONS[:3]
                ]
            with gr.Row():
                suggested_buttons.extend(
                    gr.Button(question, size="sm") for question in SUGGESTED_QUESTIONS[3:]
                )

            submit_event = message.submit(
                append_user_message,
                inputs=[message, chatbot],
                outputs=[message, chatbot],
            )
            submit_event.then(respond, inputs=chatbot, outputs=[chatbot, context])
            send.click(
                append_user_message,
                inputs=[message, chatbot],
                outputs=[message, chatbot],
            ).then(respond, inputs=chatbot, outputs=[chatbot, context])

            for button, question in zip(suggested_buttons, SUGGESTED_QUESTIONS):
                button.click(
                    lambda history, q=question: ask_suggested(q, history),
                    inputs=chatbot,
                    outputs=[chatbot, context],
                )

        with gr.Tab("3D Vectors"):
            gr.Markdown("Project Chroma embeddings into 3D with PCA.")
            with gr.Row():
                limit = gr.Slider(50, 1000, value=400, step=50, label="Vectors to display")
                refresh_vectors = gr.Button("Refresh plot", variant="primary")
            vector_plot = gr.Plot(label="Chroma vectors")
            refresh_vectors.click(run_vector_plot, inputs=limit, outputs=vector_plot)

        with gr.Tab("Simulator"):
            gr.Markdown("Run a lightweight scenario simulation for a group or custom team set.")
            teams = gr.Textbox(
                label="Teams",
                value="Mexico, South Africa, South Korea, Czechia",
            )
            seed = gr.Number(label="Simulation seed", value=7, precision=0)
            run = gr.Button("Run simulation", variant="primary")
            standings = gr.Dataframe(label="Projected standings")
            summary = gr.Markdown(label="Scenario summary")
            run.click(run_simulation, inputs=[teams, seed], outputs=[standings, summary])

    return demo


def main() -> None:
    build_ui().launch(inbrowser=True)


if __name__ == "__main__":
    main()
