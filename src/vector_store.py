"""
Pinecone integration: index creation, upsert, and metadata-filtered query.
cluster_id and is_noise are stored as metadata so queries can filter to a
specific audience segment or exclude noise points directly in Pinecone.
"""
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import pandas as pd

from src.config import settings
from src.embeddings import _as_list


def _safe_str(val, default: str = "") -> str:
    """
    Converts a value to a string, treating None/NaN as the default instead of
    stringifying them literally. Pinecone rejects null metadata values, and
    row.get("col", default) does NOT catch this -- pandas only falls back to
    default when the column is missing entirely, not when the cell is NaN.
    """
    if val is None:
        return default
    if isinstance(val, float) and pd.isna(val):
        return default
    return str(val)


def get_index():
    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing = [i["name"] for i in pc.list_indexes()]
    if settings.pinecone_index_name not in existing:
        # Dimension = SBERT dim (768 for all-mpnet-base-v2) + 3 behavioral dims
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=771,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=settings.pinecone_environment),
        )
    return pc.Index(settings.pinecone_index_name)


def upsert_records(df: pd.DataFrame, embeddings: np.ndarray, cluster_labels: np.ndarray, batch_size: int = 100):
    index = get_index()
    vectors = []
    for i, (_, row) in enumerate(df.iterrows()):
        vectors.append({
            "id": str(row["movieId"]),
            "values": embeddings[i].tolist(),
            "metadata": {
                "title": _safe_str(row.get("title"), default="Untitled"),
                "genres": _as_list(row.get("genres")),
                "release_year": _safe_str(row.get("release_year"), default="unknown"),
                "cluster_id": int(cluster_labels[i]),
                "is_noise": bool(cluster_labels[i] == -1),
            },
        })
        if len(vectors) >= batch_size:
            index.upsert(vectors=vectors)
            vectors = []
    if vectors:
        index.upsert(vectors=vectors)
    print(f"Upserted {len(df)} vectors to Pinecone index '{settings.pinecone_index_name}'")


def query_similar(embedding: np.ndarray, top_k: int = 5, exclude_noise: bool = True):
    index = get_index()
    filt = {"is_noise": {"$eq": False}} if exclude_noise else None
    result = index.query(
        vector=embedding.tolist(),
        top_k=top_k,
        include_metadata=True,
        filter=filt,
    )
    return result.get("matches", [])