"""
Builds combined content+behavior embeddings.

Text (content signal) -> SBERT sentence embedding.
Numeric behavioral features (avg_rating, rating_count, popularity) are scaled
and concatenated onto the SBERT vector, so the final vector reflects both
content semantics and behavioral signal -- matching the "metadata and
behavioral signals" framing rather than text alone.
"""
import pickle

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler

from src.config import settings

_model: SentenceTransformer | None = None

BEHAVIORAL_WEIGHT = 0.15
SCALER_PATH = "behavioral_scaler.pkl"


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def _as_list(val) -> list:
    """
    Normalizes genres/keywords into a plain Python list of strings.
    Parquet round-trips list columns as numpy arrays, not Python lists, and
    `numpy_array or []` raises "truth value of an array is ambiguous" --
    this handles arrays, None, NaN, and real lists uniformly instead.
    """
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, float) and pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    return []


def build_content_text(row) -> str:
    genres = " ".join(_as_list(row.get("genres")))
    keywords = " ".join(_as_list(row.get("keywords")))
    overview = row.get("overview", "") or ""
    tags = row.get("user_tags", "") or ""
    return f"{row.get('title', '')}. {overview} Genres: {genres}. Keywords: {keywords}. Audience tags: {tags}"


def embed_dataset(df: pd.DataFrame) -> np.ndarray:
    """
    Build-time embedding for the full corpus. Returns an (N, D) array: SBERT
    text embedding concatenated with scaled behavioral features (avg_rating,
    rating_count, popularity). The StandardScaler is FIT here, on the full
    dataset, and saved to disk -- see embed_single() for why that matters.
    """
    model = get_model()
    texts = df.apply(build_content_text, axis=1).tolist()
    text_embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    behavioral = df[["avg_rating", "rating_count", "popularity"]].fillna(0).to_numpy(dtype=float)
    scaler = StandardScaler().fit(behavioral)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    behavioral_scaled = scaler.transform(behavioral) * BEHAVIORAL_WEIGHT
    return np.concatenate([text_embeddings, behavioral_scaled], axis=1)


def embed_single(content: dict) -> np.ndarray:
    """
    Query-time embedding for one incoming content item (used by the /match
    endpoint). Loads the StandardScaler fit at build time instead of fitting
    a new one on a single row -- fitting StandardScaler on one row has zero
    variance and silently collapses every behavioral feature to 0, which
    would make the behavioral signal meaningless at inference. The scaler
    must be fit once, on the full corpus, and reused for every query.
    """
    model = get_model()
    text = build_content_text(content)
    text_emb = model.encode([text], normalize_embeddings=True)[0]

    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{SCALER_PATH} not found -- run scripts/build_index.py first "
            "to fit and save the behavioral scaler on the full corpus."
        ) from exc

    behavioral = np.array([[
        content.get("avg_rating", 0.0),
        content.get("rating_count", 0.0),
        content.get("popularity", 0.0),
    ]])
    behavioral_scaled = scaler.transform(behavioral)[0] * BEHAVIORAL_WEIGHT
    return np.concatenate([text_emb, behavioral_scaled])