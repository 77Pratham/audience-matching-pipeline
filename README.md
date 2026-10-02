# Audience Matching & Embedding System

A content-to-audience matching service that embeds **content metadata** (genre, overview,
keywords) and **behavioral signals** (rating patterns, popularity) into a shared vector
space, discovers audience segments with HDBSCAN, and serves matches through Pinecone +
FastAPI — built and evaluated at scale on MovieLens + TMDB data.

## Design decisions

- **Content + behavior, jointly embedded.** MovieLens (behavioral: ratings, tags, watch
  patterns) is joined to TMDB (content: genres, overview, keywords, cast) via `links.csv`,
  so the embedding reflects both what a title is about and how audiences actually
  responded to it — not ratings alone.
- **HDBSCAN over K-Means.** No fixed cluster count to guess upfront — it finds the
  natural number of audience segments in the data. Its noise label (`-1`) is reused
  directly as the auto-rejection signal for content that doesn't fit any real segment,
  rather than an arbitrary similarity cutoff.
- **UMAP before HDBSCAN.** Density-based clustering breaks down in high-dimensional
  space (771 raw dimensions here) — pairwise distances become nearly uniform and
  HDBSCAN labels almost everything as noise. Reducing to 30 dimensions with UMAP first
  is what makes clustering find meaningful structure instead.
- **Pinecone** as the vector store — managed indexing and scaling without running your
  own infrastructure, which matters at a catalog of 22,000 titles and 2M+ ratings.
- **Lift-weighted evaluation, not raw co-occurrence.** Ground truth for Recall@K is
  built from an independent signal (MovieLens collaborative co-rating), not from the
  same embeddings being evaluated. Raw co-occurrence count is dominated by globally
  popular titles; lift corrects for each title's individual popularity so the score
  reflects genuine association, not blockbuster bias. See **Evaluation** below.

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

## Results

Run on MovieLens `ml-25m` joined with TMDB metadata — 22,000 titles, 2M ratings, 15,000
users:

- **Clustering**: 28 audience clusters found, 8% noise (`HDBSCAN_MIN_CLUSTER_SIZE=25`).
  Tuning across several values showed the same non-linear sensitivity seen at smaller
  scale — small changes to `min_cluster_size` produced large swings in cluster count and
  noise percentage before settling on a stable region.
- **Retrieval quality**: Recall@5 = **0.32**, evaluated against a labeled ground-truth
  set built from an independent signal (MovieLens co-rating, weighted by lift — see
  Evaluation below). A popularity-only baseline (always recommending the 5 globally most
  popular titles, ignoring all content) scores **0.03** on the same ground truth — the
  real pipeline beats it by roughly 10x, evidence the score reflects genuine
  content/audience matching rather than popularity bias.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in PINECONE_API_KEY, TMDB_API_KEY
```

### 1. Build the dataset

Download `ml-25m` from [MovieLens](https://grouplens.org/datasets/movielens/) into
`data/raw/`, then:

```bash
python data/prepare_dataset.py --movielens-dir data/raw/ml-25m --out data/joined_content.parquet
```

This reads `links.csv`, fetches TMDB metadata for each linked `tmdbId`, and merges it
with aggregated MovieLens behavioral stats (mean rating, rating count, tag list) per
title.

### 2. Build embeddings + clusters + Pinecone index

```bash
python scripts/build_index.py --data data/joined_content.parquet
```

This computes SBERT embeddings (cached to `embeddings_cache.npy` after the first run —
rerunning reuses the cache instead of re-encoding everything, so an interrupted or
failed downstream step doesn't cost the full encoding time again; pass
`--force-recompute` to ignore the cache), fits a `StandardScaler` on the behavioral
features (saved to `behavioral_scaler.pkl` — reused at query time), reduces the
771-dimensional embeddings to 30 dimensions with UMAP (saved to `umap_reducer.pkl`),
fits HDBSCAN on the reduced space, and upserts the full-dimensional vectors + metadata
(`cluster_id`, `genres`, `year`, `is_noise`) into Pinecone.

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
python evaluation/build_labeled_pairs.py --movielens-dir data/raw/ml-25m --content-data data/joined_content.parquet
python evaluation/evaluate_recall.py --labeled-set evaluation/labeled_pairs.csv --k 5
```

**Methodology, in full:**

For each query title, "relevant" titles are the top-N by **lift** among titles also
rated ≥4.0 stars by the same users who rated the query title ≥4.0 stars. Lift — not raw
co-occurrence count — is what makes this valid:

```
lift(A, B) = (co_occurrence_count * total_users) / (users_who_rated_A * users_who_rated_B)
```

Raw co-occurrence count is dominated by globally popular titles — a blockbuster shows
up as "co-rated" with nearly everything regardless of any real content/audience
relationship. An earlier raw-co-occurrence version of this evaluation produced a
misleadingly high recall score that a popularity-baseline sanity check then matched or
exceeded, proving the raw version wasn't measuring content matching at all. Lift
corrects for this by normalizing against each title's individual popularity, and
candidate "relevant" titles are additionally required to have a minimum amount of their
own rating volume (`--min-b-popularity`) so that obscure, low-sample titles can't
produce noisy, artificially extreme lift scores.

Run the popularity baseline alongside any recall number, as a standing sanity check
that the ground truth isn't leaking popularity signal:

```bash
python evaluation/evaluate_popularity_baseline.py --labeled-set evaluation/labeled_pairs.csv --content-data data/joined_content.parquet --k 5
```

## Auto-rejection logic

HDBSCAN assigns `-1` to points it considers noise — i.e., content that doesn't belong
convincingly to any discovered audience cluster. `matching.py` treats `cluster_id == -1`
as an automatic reject before a human reviewer sees it, driven by the clustering
algorithm's own confidence rather than a hand-picked similarity threshold.

## Tests

```bash
pytest tests/
```
