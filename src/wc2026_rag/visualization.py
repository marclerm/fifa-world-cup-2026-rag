"""Vector-store inspection and 3D Plotly visualization helpers."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
from langchain_openai import OpenAIEmbeddings
from sklearn.decomposition import PCA

from wc2026_rag.config import COLLECTION_NAME, EMBEDDING_MODEL, VECTOR_DB_DIR


def vector_dataframe(limit: int = 500) -> pd.DataFrame:
    """Return Chroma vectors projected to 3D with PCA."""
    from chromadb import PersistentClient

    client = PersistentClient(path=str(VECTOR_DB_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    result = collection.get(limit=limit, include=["embeddings", "documents", "metadatas"])
    embeddings = result.get("embeddings") or []
    if not embeddings:
        return pd.DataFrame(columns=["x", "y", "z", "source", "preview"])

    coords = PCA(n_components=3, random_state=42).fit_transform(embeddings)
    rows = []
    for idx, vector in enumerate(coords):
        metadata = result["metadatas"][idx] or {}
        text = result["documents"][idx] or ""
        rows.append(
            {
                "x": vector[0],
                "y": vector[1],
                "z": vector[2],
                "source": metadata.get("source_name", "unknown"),
                "preview": text[:260].replace("\n", " "),
            }
        )
    return pd.DataFrame(rows)


def vector_figure(limit: int = 500):
    """Create a 3D vector scatter plot."""
    df = vector_dataframe(limit=limit)
    if df.empty:
        return px.scatter_3d(title="No vectors found. Run wc2026-ingest first.")
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="source",
        hover_data=["preview"],
        title="Chroma Embeddings Projected to 3D",
        height=680,
    )
    fig.update_traces(marker={"size": 5, "opacity": 0.78})
    fig.update_layout(margin={"l": 0, "r": 0, "t": 42, "b": 0})
    return fig


def embed_texts_for_preview(texts: list[str]) -> list[list[float]]:
    """Embed arbitrary text snippets for demos and tests."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL).embed_documents(texts)

