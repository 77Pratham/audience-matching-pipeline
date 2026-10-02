import logging
import pickle

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from src.config import settings
from src.embeddings import embed_single
from src.matching import match_content
from src.clustering import load_reducer

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
logger = logging.getLogger("audience-matching")

app = FastAPI(title="Audience Matching & Embedding System v2")

# Loaded once at startup -- see scripts/build_index.py for how these files are produced.
_clusterer = None
_reducer = None


@app.on_event("startup")
def load_models():
    global _clusterer, _reducer
    try:
        with open("clusterer.pkl", "rb") as f:
            _clusterer = pickle.load(f)
        logger.info("Loaded fitted HDBSCAN clusterer")
    except FileNotFoundError:
        logger.warning("clusterer.pkl not found -- run scripts/build_index.py first")

    try:
        _reducer = load_reducer()
        logger.info("Loaded fitted UMAP reducer")
    except FileNotFoundError:
        logger.warning("umap_reducer.pkl not found -- run scripts/build_index.py first")


class ContentInput(BaseModel):
    title: str
    overview: str = ""
    genres: list[str] = []
    keywords: list[str] = []
    user_tags: str = ""
    avg_rating: float = 0.0
    rating_count: float = 0.0
    popularity: float = 0.0


def check_api_key(x_api_key: str = Header(...)):
    if x_api_key != settings.service_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.post("/match")
def match(content: ContentInput, x_api_key: str = Header(...)):
    check_api_key(x_api_key)
    if _clusterer is None or _reducer is None:
        raise HTTPException(status_code=503, detail="Clusterer or reducer not loaded")

    try:
        embedding = embed_single(content.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    result = match_content(embedding, _clusterer, reducer=_reducer)
    logger.info(f"match request title='{content.title}' auto_rejected={result.auto_rejected}")

    return {
        "auto_rejected": result.auto_rejected,
        "cluster_id": result.cluster_id,
        "matches": [
            {"id": m["id"], "score": m["score"], "metadata": m["metadata"]}
            for m in result.matches
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok", "clusterer_loaded": _clusterer is not None, "reducer_loaded": _reducer is not None}