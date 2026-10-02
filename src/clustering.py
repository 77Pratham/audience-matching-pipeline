"""
HDBSCAN clustering, replacing v1's K-Means -- with a UMAP dimensionality
reduction step first.

Why UMAP before HDBSCAN:
HDBSCAN (and most density-based clustering) breaks down in high-dimensional
spaces -- with 771 dimensions (768 SBERT + 3 behavioral), pairwise distances
between points become nearly uniform (the "curse of dimensionality"), so
HDBSCAN can't find real density differences and labels almost everything as
noise. Reducing to ~30-50 dimensions with UMAP first, which preserves local
neighborhood structure, is the standard fix and is what HDBSCAN's own docs
recommend for exactly this situation.

Why HDBSCAN over K-Means:
- No fixed K to guess/tune upfront -- it finds the natural number of audience
  segments in the embedding space.
- It naturally labels points that don't fit any dense cluster as noise (-1),
  which we reuse directly as the auto-reject signal in matching.py, instead
  of an arbitrary cosine-similarity threshold.
"""
import pickle

import numpy as np
import hdbscan
import umap

from src.config import settings

REDUCER_PATH = "umap_reducer.pkl"
REDUCED_DIMENSIONS = 30


def fit_clusters(embeddings: np.ndarray) -> tuple[np.ndarray, hdbscan.HDBSCAN]:
    """
    Reduces embeddings to REDUCED_DIMENSIONS with UMAP, fits HDBSCAN on the
    reduced space, and returns (cluster_labels, fitted_clusterer). The fitted
    UMAP reducer is saved to disk so query-time points go through the same
    transform before being assigned a cluster (see assign_new_point).
    cluster_labels == -1 means "noise" / no confident audience segment match.
    """
    print(f"Reducing {embeddings.shape[1]}D embeddings to {REDUCED_DIMENSIONS}D with UMAP...")
    reducer = umap.UMAP(n_components=REDUCED_DIMENSIONS, metric="cosine", random_state=42)
    reduced = reducer.fit_transform(embeddings)

    with open(REDUCER_PATH, "wb") as f:
        pickle.dump(reducer, f)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=settings.hdbscan_min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(reduced)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(f"HDBSCAN found {n_clusters} clusters; {n_noise} points labeled noise "
          f"({n_noise / len(labels):.1%} of dataset)")
    return labels, clusterer


def load_reducer() -> umap.UMAP:
    with open(REDUCER_PATH, "rb") as f:
        return pickle.load(f)


def assign_new_point(clusterer: hdbscan.HDBSCAN, embedding: np.ndarray, reducer: umap.UMAP = None) -> int:
    """
    Assigns a cluster label to a single new embedding at query time.
    The embedding must go through the SAME UMAP reduction fit at build time --
    reducer is required unless the caller has already reduced the embedding.
    """
    if reducer is not None:
        embedding = reducer.transform(embedding.reshape(1, -1))[0]
    labels, _ = hdbscan.approximate_predict(clusterer, embedding.reshape(1, -1))
    return int(labels[0])