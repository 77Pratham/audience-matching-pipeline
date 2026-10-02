"""
Query-time matching: embed incoming content, assign it a cluster, and either
auto-reject it (noise / no confident audience segment) or return its nearest
audience-matched content via Pinecone.
"""
from dataclasses import dataclass

import numpy as np

from src import vector_store
from src.clustering import assign_new_point


@dataclass
class MatchResult:
    auto_rejected: bool
    cluster_id: int
    matches: list


def match_content(embedding: np.ndarray, clusterer, reducer=None, top_k: int = 5) -> MatchResult:
    cluster_id = assign_new_point(clusterer, embedding, reducer=reducer)

    if cluster_id == -1:
        # No confident audience segment -- this is the auto-reject path.
        return MatchResult(auto_rejected=True, cluster_id=-1, matches=[])

    matches = vector_store.query_similar(embedding, top_k=top_k, exclude_noise=True)
    return MatchResult(auto_rejected=False, cluster_id=cluster_id, matches=matches)
