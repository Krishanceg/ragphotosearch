"""
ingest.py  --  THE OFFLINE PIPELINE (run once per photo)
--------------------------------------------------------
For every photo in a folder:
  1. open it
  2. turn it into a vector with CLIP (clip_model.embed_image)
  3. store {id, vector, file path} in a local ChromaDB database

After this runs you never touch the original photos during search -- only the vectors
are searched. Re-running skips photos already stored, so it's safe to run again after
adding new pictures.

Run:
    python ingest.py                  # ingests ./test_photos
    python ingest.py "C:/Users/Admin/Pictures"   # ingest your own folder
"""

import os
import sys
import chromadb

from clip_model import embed_image

DB_DIR = "photo_db"
COLLECTION = "photos"
VALID_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def find_images(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(VALID_EXT):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "test_photos"
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        print("Tip: run  python download_samples.py  first, or pass a real folder path.")
        return

    images = find_images(folder)
    if not images:
        print(f"No images found in {folder}")
        return
    print(f"Found {len(images)} images in {folder}")

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    already = set(collection.get()["ids"]) if collection.count() else set()

    added = 0
    for i, path in enumerate(images, 1):
        image_id = os.path.abspath(path)        # use full path as the unique id
        if image_id in already:
            continue
        try:
            vec = embed_image(path)
        except Exception as e:
            print(f"  skip {path}: {e}")
            continue
        collection.add(
            ids=[image_id],
            embeddings=[vec],
            metadatas=[{"path": image_id, "filename": os.path.basename(path)}],
        )
        added += 1
        print(f"[{i}/{len(images)}] embedded {os.path.basename(path)}")

    print(f"\nDone. Added {added} new photos. Database now holds {collection.count()} photos.")
    print("Next: python search.py \"a dog\"")


if __name__ == "__main__":
    main()
