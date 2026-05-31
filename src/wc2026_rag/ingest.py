"""Build the Chroma vector store for the FIFA World Cup 2026 RAG app."""

from __future__ import annotations

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from wc2026_rag.config import COLLECTION_NAME, EMBEDDING_MODEL, KNOWLEDGE_BASE_DIR, VECTOR_DB_DIR
from wc2026_rag.datasets import build_knowledge_base


def load_documents():
    """Load generated Markdown files into LangChain documents."""
    loader = DirectoryLoader(
        str(KNOWLEDGE_BASE_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    for doc in docs:
        source = Path(doc.metadata.get("source", "unknown"))
        doc.metadata["source_name"] = source.name
        doc.metadata["doc_type"] = "world_cup_2026"
    return docs


def split_documents(documents):
    """Split documents with overlap, following the Week Five RAG pattern."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=180)
    return splitter.split_documents(documents)


def create_vector_store(chunks, reset: bool = True) -> Chroma:
    """Embed chunks and persist them to Chroma."""
    if reset and VECTOR_DB_DIR.exists():
        shutil.rmtree(VECTOR_DB_DIR)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_DB_DIR),
        collection_name=COLLECTION_NAME,
    )
    return vectorstore


def main() -> None:
    """Build generated docs, split them, and persist embeddings."""
    generated = build_knowledge_base(download=True)
    documents = load_documents()
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks)
    sample = vectorstore._collection.get(limit=1, include=["embeddings"])
    embeddings = sample.get("embeddings")
    dimensions = len(embeddings[0]) if embeddings is not None and len(embeddings) else 0
    print(f"Generated {generated} knowledge files")
    print(f"Loaded {len(documents)} documents")
    print(f"Stored {len(chunks)} vector chunks with {dimensions} dimensions")
    print(f"Chroma path: {VECTOR_DB_DIR}")


if __name__ == "__main__":
    main()
