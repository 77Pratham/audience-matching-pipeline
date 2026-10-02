"""
Popularity baseline for Recall@K -- a sanity check against the real pipeline's
score. If simply recommending the K globally most popular movies (completely
ignoring content/embeddings) scores close to the real pipeline's Recall@K on
the same labeled set, that proves the high score reflects popularity bias in
the ground truth (popular movies co-occur with almost everything), not
genuine content-based matching.

Usage:
    python evaluation/evaluate_popularity_baseline.py \
        --labeled-set evaluation/labeled_pairs.csv \
        --content-data data/joined_content.parquet \
        --k 5
"""
import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-set", required=True)
    parser.add_argument("--content-data", default="data/joined_content.parquet")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    labeled_df = pd.read_csv(args.labeled_set)
    content_df = pd.read_parquet(args.content_data)

    top_popular = (
        content_df.sort_values("rating_count", ascending=False)
        .head(args.k)["movieId"]
        .astype(str)
        .tolist()
    )
    print(f"Top-{args.k} most popular movies (by rating_count), used for EVERY query: {top_popular}")

    hits, total = 0, 0
    for _, row in labeled_df.iterrows():
        relevant = set(str(row["relevant_movie_ids"]).replace(";", ",").split(","))
        relevant = {r.strip() for r in relevant if r.strip()}

        total += 1
        if set(top_popular) & relevant:
            hits += 1

    score = hits / total if total else 0.0
    print(f"\nPopularity baseline Recall@{args.k}: {score:.3f} (evaluated on {total} labeled queries)")
    print("This baseline recommends the SAME top-K popular movies for every single "
          "query, using no content or embedding information at all. If this score is "
          "close to your pipeline's real Recall@K, the high number reflects popularity "
          "bias in the co-rating ground truth, not genuine content-based matching.")


if __name__ == "__main__":
    main()