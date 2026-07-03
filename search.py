"""
search.py  --  THE ONLINE PIPELINE (run per search)
----------------------------------------------------
  1. take your text query ("a dog sleeping")
  2. turn it into a vector with the SAME CLIP model used at ingestion
  3. ask ChromaDB for the nearest photo vectors
  4. print the ranked matching files + a similarity score

Run:
    python search.py "a dog"
    python search.py "sunset over water" --top 10
    python search.py "food on a plate" --open     # also opens the #1 result
"""

import argparse
import os
import chromadb

from clip_model import embed_text

DB_DIR = "photo_db"
COLLECTION = "photos"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="what to search for, in plain English")
    ap.add_argument("--top", type=int, default=5, help="how many results to show")
    ap.add_argument("--open", action="store_true", help="open the top result")
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        collection = client.get_collection(COLLECTION)
    except Exception:
        print("No database yet. Run:  python ingest.py")
        return
    if collection.count() == 0:
        print("Database is empty. Run:  python ingest.py")
        return

    qvec = embed_text(args.query)
    res = collection.query(query_embeddings=[qvec], n_results=args.top)

    ids = res["ids"][0]
    dists = res["distances"][0]

    print(f'\nTop {len(ids)} matches for: "{args.query}"\n')
    for rank, (path, dist) in enumerate(zip(ids, dists), 1):
        similarity = 1 - dist          # cosine distance -> similarity (higher = better)
        print(f"  {rank}. {similarity:.3f}   {path}")

    if args.open and ids:
        # Windows: open the best match in the default image viewer
        os.startfile(ids[0])  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
