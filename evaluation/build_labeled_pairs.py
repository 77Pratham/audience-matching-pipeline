"""
Builds a labeled ground-truth set for Recall@K evaluation using an INDEPENDENT
signal from the embedding pipeline: collaborative co-rating, ranked by LIFT
rather than raw co-occurrence count.

Why lift, not raw co-occurrence:
A raw co-occurrence count is dominated by globally popular movies -- if
almost everyone rates a blockbuster highly, it will show up as "co-rated"
with nearly every query movie, regardless of any real content/audience
relationship. A baseline check confirmed this: recommending the same 5 most
popular movies for every query scored HIGHER (1.000) than the actual
embedding pipeline (0.960) on the raw-co-occurrence version of this ground
truth -- proof the raw version wasn't measuring anything meaningful.

Lift corrects for this:
    lift(A, B) = P(both rated highly) / (P(A rated highly) * P(B rated highly))
               = (co_occurrence_count * total_users) / (users_A * users_B)

A lift > 1 means A and B are rated highly together MORE often than their
individual popularity would predict by chance -- i.e. genuine association,
not just "B is popular." A minimum co-occurrence count is still required so
lift isn't computed from tiny, noisy overlaps.

Usage:
    python evaluation/build_labeled_pairs.py \
        --movielens-dir data/raw/ml-latest-small \
        --content-data data/joined_content.parquet \
        --out evaluation/labeled_pairs.csv \
        --num-queries 50 \
        --min-relevant 3
"""
import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--movielens-dir", required=True)
    parser.add_argument("--content-data", required=True,
                         help="joined_content.parquet -- restricts candidates to movies actually in the index")
    parser.add_argument("--out", default="evaluation/labeled_pairs.csv")
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument("--min-relevant", type=int, default=3,
                         help="Skip query movies with fewer than this many co-rated relevant movies")
    parser.add_argument("--rating-threshold", type=float, default=4.0)
    parser.add_argument("--top-n-relevant", type=int, default=10,
                         help="Max relevant movies to keep per query")
    parser.add_argument("--min-overlap", type=int, default=3,
                         help="Minimum raw co-occurrence count before lift is trusted (avoids noisy small-sample lift)")
    parser.add_argument("--min-b-popularity", type=int, default=15,
                         help="Minimum total high-rating count a candidate relevant movie must have -- "
                              "without this, lift over-rewards obscure movies with tiny rating counts, "
                              "whose lift scores are noise, not signal")
    args = parser.parse_args()

    ratings = pd.read_csv(os.path.join(args.movielens_dir, "ratings.csv"))
    content_ids = set(pd.read_parquet(args.content_data)["movieId"])

    high_ratings = ratings[ratings["rating"] >= args.rating_threshold]
    high_ratings = high_ratings[high_ratings["movieId"].isin(content_ids)]

    total_users = high_ratings["userId"].nunique()
    # |users_B| for every movie -- the popularity denominator in the lift formula
    movie_rating_counts = high_ratings["movieId"].value_counts()
    candidate_queries = movie_rating_counts[movie_rating_counts >= 10].index.tolist()

    print(f"{len(candidate_queries)} candidate query movies with enough high-rating volume")
    print(f"{total_users} total users in the high-ratings pool")

    rows = []
    for movie_id in candidate_queries:
        if len(rows) >= args.num_queries:
            break

        users_a = set(high_ratings[high_ratings["movieId"] == movie_id]["userId"])
        n_users_a = len(users_a)
        if n_users_a < 5:
            continue

        co_rated = high_ratings[
            (high_ratings["userId"].isin(users_a)) & (high_ratings["movieId"] != movie_id)
        ]
        overlap_counts = co_rated["movieId"].value_counts()
        overlap_counts = overlap_counts[overlap_counts >= args.min_overlap]
        # Exclude candidates that are themselves too obscure -- a movie with
        # only a handful of raters total can get an extreme lift score from
        # pure small-sample noise, even with the min_overlap filter above.
        overlap_counts = overlap_counts[
            overlap_counts.index.map(lambda mid: movie_rating_counts.get(mid, 0) >= args.min_b_popularity)
        ]

        if overlap_counts.empty:
            continue

        # lift(A, B) = (overlap * total_users) / (|users_A| * |users_B|)
        lifts = {}
        for other_id, overlap in overlap_counts.items():
            n_users_b = movie_rating_counts[other_id]
            lifts[other_id] = (overlap * total_users) / (n_users_a * n_users_b)

        ranked = sorted(lifts.items(), key=lambda kv: kv[1], reverse=True)
        relevant = [movie_id_ for movie_id_, _ in ranked[:args.top_n_relevant]]

        if len(relevant) < args.min_relevant:
            continue

        rows.append({
            "query_movie_id": movie_id,
            "relevant_movie_ids": ";".join(str(m) for m in relevant),
        })

    if not rows:
        raise SystemExit(
            "No query movies had enough co-rating signal. Try a larger MovieLens "
            "dataset (ml-25m) or lower --min-relevant / --min-overlap."
        )

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} labeled query/relevant-set pairs to {args.out}")
    print(f"Methodology: for each query movie, 'relevant' = top {args.top_n_relevant} movies "
          f"by LIFT (co-occurrence corrected for individual popularity) among users who "
          f"rated the query movie >= {args.rating_threshold} stars, requiring >= "
          f"{args.min_overlap} raw co-occurrences before trusting the lift score. "
          "This corrects for the popularity bias a raw co-occurrence count would have.")

    # Diagnostic: confirms the --min-b-popularity filter is actually in effect.
    # If this fix applied correctly, the minimum should be >= args.min_b_popularity.
    all_relevant_ids = [int(m) for row in rows for m in row["relevant_movie_ids"].split(";")]
    popularities = [movie_rating_counts.get(mid, 0) for mid in all_relevant_ids]
    print(f"\n[diagnostic] Relevant-movie rating-count popularity: "
          f"min={min(popularities)}, max={max(popularities)}, "
          f"avg={sum(popularities) / len(popularities):.1f} "
          f"(min should be >= --min-b-popularity={args.min_b_popularity})")


if __name__ == "__main__":
    main()