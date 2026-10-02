# Audience Matching & Embedding System (v2)

A content-to-audience matching service that embeds **content metadata** (genre, overview,
keywords) and **behavioral signals** (rating patterns, popularity) into a shared vector
space, clusters audience segments with HDBSCAN, and serves matches through Pinecone +
FastAPI.

## Why this version is different from v1

v1 used raw MovieLens ratings and K-Means with a fixed K. That doesn't match the resume
framing of "content and behavioral signals for licensing," and K-Means forces an
arbitrary cluster count. This version:

- Joins **MovieLens** (behavioral: ratings, tags, watch patterns) to **TMDB** (content:
  genres, overview, keywords, cast) via `links.csv`, so the embedding genuinely reflects
  both content and behavior.
- Uses **HDBSCAN** instead of K-Means: no fixed K, and its noise label (`-1`) is reused
  directly as the auto-rejection signal for content that doesn't fit any real audience
  segment — a cleaner, more defensible story than an arbitrary similarity cutoff.
- Keeps **Pinecone** as the vector store (managed, scales past 25M+ records without you
  running your own index infrastructure).

## Architecture

```
MovieLens ratings/tags ──┐
                         ├─→ prepare_dataset.py ─→ joined_content.parquet
TMDB metadata ───────────┘
                                    │
                                    ▼
                          embeddings.py (SBERT)
                                    │
                                    ▼
                clustering.py (UMAP reduction → HDBSCAN, fit once)
                                    │
                                    ▼
                    vector_store.py (upsert to Pinecone,
                    cluster_id + genre + year as metadata)
                                    │
                                    ▼
                matching.py (query-time cosine similarity
                + cluster-based auto-reject)
                                    │
                                    ▼
                        main.py (FastAPI service)
```

## Results (verified against real data — see Evaluation below)

Run on the full `ml-latest-small` dataset (9,736 movies joined with TMDB metadata):

- **Clustering**: 8 audience clusters found, 0.3% noise (`HDBSCAN_MIN_CLUSTER_SIZE=10`).
  Tuning notes: `min_cluster_size=15` (default) found only 5 clusters covering ~1,950
  movies each — too coarse to be a meaningful audience segment. `min_cluster_size=8`
  overcorrected to 132 clusters with 53% noise. `10` and `12` both landed on 8 stable
  clusters with <0.5% noise, which is the value used in this build.
- **Retrieval quality**: Recall@5 = **0.240**, evaluated against a labeled ground-truth
  set built from an independent signal (MovieLens co-rating, weighted by lift — see
  Evaluation below) — not from the same content embeddings being evaluated, so this
  isn't circular. A popularity-only baseline (always recommending the 5 globally most
  popular movies, ignoring all content) scores **0.000** on the same ground truth,
  confirming the real system's score reflects genuine content/audience matching rather
  than popularity bias.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in PINECONE_API_KEY, TMDB_API_KEY
```

### 1. Build the dataset
Download `ml-latest-small` (or `ml-25m` for the full 25M-record scale) from
[MovieLens](https://grouplens.org/datasets/movielens/) into `data/raw/`, then:

```bash
python data/prepare_dataset.py --movielens-dir data/raw/ml-latest-small --out data/joined_content.parquet
```

This reads `links.csv`, fetches TMDB metadata for each linked `tmdbId`, and merges it
with aggregated MovieLens behavioral stats (mean rating, rating count, tag list) per
movie.

### 2. Build embeddings + clusters + Pinecone index

```bash
python scripts/build_index.py --data data/joined_content.parquet
```

This computes SBERT embeddings (cached to `embeddings_cache.npy` after the first run —
rerunning will reuse the cache instead of re-encoding everything, so an interrupted or
failed run downstream doesn't cost you another 20+ minutes; pass `--force-recompute` to
ignore the cache), fits a `StandardScaler` on the behavioral features (saved to
`behavioral_scaler.pkl` — reused at query time), reduces the 771-dimensional embeddings
to 30 dimensions with UMAP (saved to `umap_reducer.pkl` — HDBSCAN performs poorly in
high-dimensional spaces, so this reduction step is what makes clustering actually find
meaningful groups instead of labeling almost everything as noise), fits HDBSCAN on the
reduced space, and upserts the full-dimensional vectors + metadata (`cluster_id`,
`genres`, `year`, `is_noise`) into Pinecone.

### 3. Run the service

```bash
uvicorn src.main:app --reload
```

or via Docker:

```bash
docker build -t audience-matching .
docker run -p 8000:8000 --env-file .env audience-matching
```

### 4. Evaluate

Ground truth is built automatically from an independent signal — MovieLens co-rating
data — rather than hand-labeled or fabricated:

```bash
python evaluation/build_labeled_pairs.py --movielens-dir data/raw/ml-latest-small --content-data data/joined_content.parquet
python evaluation/evaluate_recall.py --labeled-set evaluation/labeled_pairs.csv --k 5
```

**Methodology, in full, so there's no ambiguity later (this is what caused the resume
discrepancy in v1):**

For each query movie, "relevant" movies are the top-10 movies by **lift** among movies
also rated ≥4.0 stars by the same users who rated the query movie ≥4.0 stars. Lift —
not raw co-occurrence count — is what makes this valid:

```
lift(A, B) = (co_occurrence_count * total_users) / (users_who_rated_A * users_who_rated_B)
```

A first version of this evaluation used raw co-occurrence count instead of lift, and
produced Recall@5 = 0.960 — which looked great, but a popularity-baseline sanity check
(always recommending the same 5 most popular movies) scored *higher* (1.000) on the same
ground truth. That proved the raw-co-occurrence version wasn't measuring content
matching at all — it was just rewarding blockbusters that co-occur with everything.
Switching to lift (which corrects for each movie's individual popularity) and requiring
candidate "relevant" movies to have ≥15 of their own high ratings (to prevent obscure,
low-sample movies from producing noisy, artificially extreme lift scores) fixed this:
the popularity baseline dropped to 0.000, and the real pipeline scored 0.240 — a result
that's actually attributable to the embeddings and clustering, not a ground-truth
artifact.

**Always run the popularity baseline alongside any future recall number, as a sanity
check that the ground truth isn't leaking popularity signal:**

```bash
python evaluation/evaluate_popularity_baseline.py --labeled-set evaluation/labeled_pairs.csv --content-data data/joined_content.parquet --k 5
```

## Auto-rejection logic

HDBSCAN assigns `-1` to points it considers noise — i.e., content that doesn't belong
convincingly to any discovered audience cluster. `matching.py` treats `cluster_id == -1`
as an automatic reject before a human reviewer sees it, which is the same operational
behavior as v1's "80% auto-rejection" but driven by the clustering algorithm's own
confidence rather than a hand-picked similarity threshold.

## Tests

```bash
pytest tests/
```