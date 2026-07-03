"""
search_image.py  --  SEARCH BY IMAGE (image-to-image retrieval)
---------------------------------------------------------------
Instead of typing words, you hand it a photo and it finds the most visually similar
photos in your library.

Why this is almost free: the shared embedding space means an IMAGE vector and the
stored image vectors already live together. So "find similar images" is the exact
same nearest-vector search as text search -- we just build the query vector with
embed_image() instead of embed_text().

Run:
    python search_image.py path/to/photo.jpg
    python search_image.py test_photos/img_001.jpg --top 5
"""

import argparse
import os
import chromadb

from clip_model import embed_image

DB_DIR = "photo_db"
COLLECTION = "photos"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="path to the query image")
    ap.add_argument("--top", type=int, default=5, help="how many results to show")
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        print(f"Image not found: {args.image}")
        return

    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        collection = client.get_collection(COLLECTION)
    except Exception:
        print("No database yet. Run:  python ingest.py")
        return

    qvec = embed_image(args.image)
    # ask for one extra, in case the query image is itself in the library (it'll match
    # itself with score ~1.0, which we skip so the results are *other* similar photos).
    res = collection.query(query_embeddings=[qvec], n_results=args.top + 1)

    query_id = os.path.abspath(args.image)
    print(f'\nPhotos most similar to: {args.image}\n')
    shown = 0
    for path, dist in zip(res["ids"][0], res["distances"][0]):
        if path == query_id:
            continue                      # skip the query image matching itself
        shown += 1
        print(f"  {shown}. {1 - dist:.3f}   {path}")
        if shown >= args.top:
            break


if __name__ == "__main__":
    main()
