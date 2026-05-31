"""Conversational RAG layer for World Cup 2026 questions."""

from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, convert_to_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from wc2026_rag.config import CHAT_MODEL, COLLECTION_NAME, EMBEDDING_MODEL, VECTOR_DB_DIR

RETRIEVAL_K = 8

SYSTEM_PROMPT = """You are a careful FIFA World Cup 2026 conversational analyst.

Answer using the retrieved context first. Distinguish facts from projections.
For projections, describe the scenario assumptions and avoid presenting simulated outcomes as certainties.
If the retrieved context does not contain enough evidence, say what is missing and suggest what data should be added.

Retrieved context:
{context}
"""


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def vector_count() -> int:
    """Return the number of stored chunks in Chroma."""
    return int(get_vectorstore()._collection.count())


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=CHAT_MODEL, temperature=0.2)


def combine_question(question: str, history: list[dict] | None = None) -> str:
    """Bring prior user turns into retrieval for follow-up questions."""
    history = history or []
    prior = "\n".join(item["content"] for item in history if item.get("role") == "user")
    return f"{prior}\n{question}".strip()


def fetch_context(question: str, history: list[dict] | None = None, k: int = RETRIEVAL_K) -> list[Document]:
    vectorstore = get_vectorstore()
    if vectorstore._collection.count() == 0:
        raise RuntimeError(
            "The Chroma vector store is empty. Run `wc2026-ingest` before starting the app. "
            "If Kaggle credentials are not configured yet, ingestion will still index the built-in "
            "FIFA seed knowledge."
        )
    query = combine_question(question, history)
    return vectorstore.similarity_search(query, k=k)


def format_context(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source_name") or doc.metadata.get("source", "unknown")
        parts.append(f"Source: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(question: str, history: list[dict] | None = None) -> tuple[str, list[Document]]:
    """Answer a question and return the retrieved documents used as evidence."""
    history = history or []
    docs = fetch_context(question, history)
    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=format_context(docs)))]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = get_llm().invoke(messages)
    return str(response.content), docs
