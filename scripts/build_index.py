"""
End-to-end pipeline: load joined dataset -> embed -> cluster -> upsert to Pinecone.
Also saves the fitted HDBSCAN clusterer so the FastAPI service can assign new
incoming content to a cluster at query time without refitting.

Usage:
    python scripts/build_index.py --data data/joined_content.parquet
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.embeddings import embed_dataset
from src.clustering import fit_clusters
from src import vector_store

EMBEDDINGS_CACHE = "embeddings_cache.npy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--clusterer-out", default="clusterer.pkl")
    parser.add_argument("--force-recompute", action="store_true",
                         help="Ignore any cached embeddings and recompute from scratch")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    print(f"Loaded {len(df)} records")

    if os.path.exists(EMBEDDINGS_CACHE) and not args.force_recompute:
        print(f"Found cached embeddings at {EMBEDDINGS_CACHE}, loading instead of recomputing "
              "(pass --force-recompute to ignore this cache)...")
        embeddings = np.load(EMBEDDINGS_CACHE)
        if len(embeddings) != len(df):
            print(f"  [warn] cached embeddings ({len(embeddings)}) don't match dataset size "
                  f"({len(df)}) -- recomputing instead.")
            embeddings = None
        else:
            print(f"Loaded {len(embeddings)} cached embeddings")
    else:
        embeddings = None

    if embeddings is None:
        print("Building embeddings (content + behavioral) -- this is the slow step, "
              f"progress will be saved to {EMBEDDINGS_CACHE} once done...")
        embeddings = embed_dataset(df)
        np.save(EMBEDDINGS_CACHE, embeddings)
        print(f"Cached embeddings to {EMBEDDINGS_CACHE} -- if anything below fails or is "
              "interrupted, rerunning this script will skip straight past this step.")

    print("Fitting HDBSCAN (with UMAP dimensionality reduction first)...")
    labels, clusterer = fit_clusters(embeddings)

    print("Upserting to Pinecone...")
    vector_store.upsert_records(df, embeddings, labels)

    with open(args.clusterer_out, "wb") as f:
        pickle.dump(clusterer, f)
    print(f"Saved fitted clusterer to {args.clusterer_out}")


if __name__ == "__main__":
    main()