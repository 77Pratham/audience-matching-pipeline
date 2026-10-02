"""
Recall@K evaluation against a labeled subset -- documented explicitly so there's
no ambiguity later about methodology (unlike v1's resume/internal-notes mismatch).

Expected CSV format for --labeled-set (evaluation/labeled_pairs.csv):
    query_movie_id,relevant_movie_ids
    1,"2,34,110"
    5,"12,88"

Where relevant_movie_ids is a semicolon- or comma-separated list of movieIds
that a human reviewer confirmed are genuinely good audience/content matches
for the query movie. This file must be built manually or from real review-team
decisions -- do not fabricate it.

Usage:
    python evaluation/evaluate_recall.py --labeled-set evaluation/labeled_pairs.csv --k 5
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src import vector_store
from src.embeddings import embed_dataset

EMBEDDINGS_CACHE = "embeddings_cache.npy"


def recall_at_k(labeled_df: pd.DataFrame, content_df: pd.DataFrame, k: int) -> float:
    if os.path.exists(EMBEDDINGS_CACHE):
        print(f"Loading cached embeddings from {EMBEDDINGS_CACHE}...")
        embeddings = np.load(EMBEDDINGS_CACHE)
        if len(embeddings) != len(content_df):
            print("  [warn] cache size mismatch -- recomputing from scratch")
            embeddings = embed_dataset(content_df)
    else:
        embeddings = embed_dataset(content_df)

    id_to_idx = {mid: i for i, mid in enumerate(content_df["movieId"])}

    hits, total = 0, 0
    for _, row in labeled_df.iterrows():
        qid = row["query_movie_id"]
        if qid not in id_to_idx:
            continue
        relevant = set(str(row["relevant_movie_ids"]).replace(";", ",").split(","))
        relevant = {r.strip() for r in relevant if r.strip()}

        query_emb = embeddings[id_to_idx[qid]]
        results = vector_store.query_similar(query_emb, top_k=k, exclude_noise=False)
        retrieved_ids = {m["id"] for m in results}

        total += 1
        if retrieved_ids & relevant:
            hits += 1

    return hits / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-set", required=True)
    parser.add_argument("--content-data", default="data/joined_content.parquet")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    labeled_df = pd.read_csv(args.labeled_set)
    content_df = pd.read_parquet(args.content_data)

    score = recall_at_k(labeled_df, content_df, args.k)
    print(f"Recall@{args.k}: {score:.3f} (evaluated on {len(labeled_df)} labeled queries)")
    print("Methodology: retrieved = Pinecone top-K by cosine similarity; "
          "hit = at least one retrieved ID is in the human-labeled relevant set.")


if __name__ == "__main__":
    main()