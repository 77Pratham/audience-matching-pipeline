"""
Joins MovieLens behavioral data (ratings, tags) with TMDB content metadata
(genres, overview, keywords) via links.csv, so the resulting dataset genuinely
reflects both "content" and "behavioral" signals rather than ratings alone.

Usage:
    python data/prepare_dataset.py --movielens-dir data/raw/ml-latest-small \
        --out data/joined_content.parquet
"""
import argparse
import os
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()  # reads .env into the environment so TMDB_API_KEY is picked up below

TMDB_BASE = "https://api.themoviedb.org/3"


def build_session() -> requests.Session:
    """
    A persistent session with retry/backoff and a real User-Agent.
    Opening a fresh connection per request (the old approach) can trigger
    connection resets under any kind of TLS inspection, antivirus, or rate
    limiting -- reusing one connection and retrying transient failures fixes
    that instead of just logging and skipping the movie.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "audience-matching-v2/1.0"})
    retry = Retry(
        total=5,
        backoff_factor=1.5,  # 1.5s, 3s, 4.5s, 6s, 7.5s between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


def fetch_tmdb_metadata(session: requests.Session, tmdb_id: int, api_key: str) -> dict:
    """Fetch genres, overview, and keywords for one TMDB movie ID."""
    try:
        movie = session.get(
            f"{TMDB_BASE}/movie/{tmdb_id}",
            params={"api_key": api_key},
            timeout=15,
        ).json()
        keywords = session.get(
            f"{TMDB_BASE}/movie/{tmdb_id}/keywords",
            params={"api_key": api_key},
            timeout=15,
        ).json()
        return {
            "tmdbId": tmdb_id,
            "title": movie.get("title", ""),
            "overview": movie.get("overview", ""),
            "genres": [g["name"] for g in movie.get("genres", [])],
            "keywords": [k["name"] for k in keywords.get("keywords", [])],
            "release_year": (movie.get("release_date") or "")[:4],
            "popularity": movie.get("popularity", 0.0),
        }
    except Exception as exc:  # noqa: BLE001 - log and continue on per-item failure
        print(f"  [warn] tmdbId={tmdb_id} failed: {exc}")
        return {"tmdbId": tmdb_id}


def build_behavioral_features(ratings: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """Aggregate MovieLens ratings + tags into per-movie behavioral features."""
    rating_agg = ratings.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count"),
    ).reset_index()

    if tags is not None and not tags.empty:
        tag_agg = tags.groupby("movieId")["tag"].apply(
            lambda vals: " ".join(sorted(set(str(v) for v in vals)))
        ).reset_index().rename(columns={"tag": "user_tags"})
        rating_agg = rating_agg.merge(tag_agg, on="movieId", how="left")
    else:
        rating_agg["user_tags"] = ""

    rating_agg["user_tags"] = rating_agg["user_tags"].fillna("")
    return rating_agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--movielens-dir", required=True, help="Path to unzipped MovieLens dir")
    parser.add_argument("--out", required=True, help="Output parquet path")
    parser.add_argument("--tmdb-api-key", default=os.getenv("TMDB_API_KEY", ""))
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for testing")
    parser.add_argument("--sleep", type=float, default=0.25, help="Delay between TMDB calls")
    args = parser.parse_args()

    if not args.tmdb_api_key:
        raise SystemExit("TMDB API key required (--tmdb-api-key or TMDB_API_KEY env var)")

    links = pd.read_csv(os.path.join(args.movielens_dir, "links.csv"))
    ratings = pd.read_csv(os.path.join(args.movielens_dir, "ratings.csv"))
    tags_path = os.path.join(args.movielens_dir, "tags.csv")
    tags = pd.read_csv(tags_path) if os.path.exists(tags_path) else None

    links = links.dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)
    if args.limit:
        links = links.head(args.limit)

    print(f"Fetching TMDB metadata for {len(links)} movies...")
    session = build_session()
    tmdb_records = []
    for i, row in enumerate(links.itertuples(), 1):
        tmdb_records.append(fetch_tmdb_metadata(session, row.tmdbId, args.tmdb_api_key))
        if i % 50 == 0:
            print(f"  ...{i}/{len(links)}")
        time.sleep(args.sleep)

    tmdb_df = pd.DataFrame(tmdb_records)
    merged = links.merge(tmdb_df, on="tmdbId", how="inner")

    behavioral = build_behavioral_features(ratings, tags)
    final = merged.merge(behavioral, on="movieId", how="left")

    final["avg_rating"] = final["avg_rating"].fillna(final["avg_rating"].mean())
    final["rating_count"] = final["rating_count"].fillna(0)
    final["user_tags"] = final["user_tags"].fillna("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    final.to_parquet(args.out, index=False)
    print(f"Wrote {len(final)} joined content+behavior records to {args.out}")


if __name__ == "__main__":
    main()