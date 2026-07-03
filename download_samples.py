"""
download_samples.py
-------------------
Downloads a small set of FREE, copyright-friendly test photos into ./test_photos
so you have something to search before pointing the app at your own pictures.

Sources (no API key, no signup, no money needed):
  - Unsplash direct CDN  -> real themed photos (free Unsplash license)
  - Picsum               -> random photos, used to top up if any Unsplash URL fails

Note: the photo FILENAMES are deliberately generic (img_001.jpg, ...). That is the
whole point of this project -- the search does NOT rely on filenames. CLIP reads the
actual pixels, so it can find "a dog" even in a file called img_017.jpg.

Run:
    python download_samples.py
    python download_samples.py --count 60        # download more
    python download_samples.py --random-only     # skip Unsplash, use Picsum only
"""

import argparse
import os
import requests

OUT_DIR = "test_photos"

# A diverse spread of real Unsplash photos (free license, direct CDN, no API key).
# Variety matters more than the exact subject -- a mix of animals, nature, food,
# people, city and objects gives you obvious things to search for.
UNSPLASH_IDS = [
    "1543466835-00a7907e9de1", "1507525428034-b723cf961d3e",
    "1568605114967-8130f3a36994", "1530281700549-e82e7bf110d6",
    "1441974231531-c6227db76b6e", "1506744038136-46273834b3fb",
    "1518791841217-8f162f1e1131", "1425082661705-1834bfd09dca",
    "1552053831-71594a27632d",   "1546069901-ba9599a7e63c",
    "1414235077428-338989a2e8c0", "1504674900247-0877df9cc836",
    "1498936178812-4b2e558d2937", "1469474968028-56623f02e42e",
    "1500382017468-9049fed747ef", "1470071459604-3b5ec3a7fe05",
    "1493246507139-91e8fad9978e", "1518837695005-2083093ee35b",
    "1449824913935-59a10b8d2000", "1502082553048-f009c37129b9",
    "1444723121867-7a241cacace9", "1493809842364-78817add7ffb",
    "1465101046530-73398c7f28ca", "1519681393784-d120267933ba",
    "1518770660439-4636190af475", "1526374965328-7f61d4dc18c5",
    "1484591974057-265bb767ef71", "1485827404703-89b55fcc595e",
    "1517336714731-489689fd1ca8", "1531297484001-80022131f5a1",
]

UNSPLASH_URL = "https://images.unsplash.com/photo-{id}?w=800&q=80&fit=max"
PICSUM_URL = "https://picsum.photos/seed/{seed}/800/600"


def save(url, path, timeout=20):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200 and len(r.content) > 5000:  # skip tiny/error bodies
            with open(path, "wb") as f:
                f.write(r.content)
            return True
    except Exception as e:
        print(f"   ! failed: {e}")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=30, help="how many photos to download")
    ap.add_argument("--random-only", action="store_true", help="use Picsum only")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    saved = 0
    n = 1

    if not args.random_only:
        for pid in UNSPLASH_IDS:
            if saved >= args.count:
                break
            path = os.path.join(OUT_DIR, f"img_{n:03d}.jpg")
            if save(UNSPLASH_URL.format(id=pid), path):
                print(f"[{saved+1:>3}] saved {path}")
                saved += 1
                n += 1

    # Top up with random Picsum photos until we hit the requested count.
    seed = 1000
    while saved < args.count:
        path = os.path.join(OUT_DIR, f"img_{n:03d}.jpg")
        if save(PICSUM_URL.format(seed=seed), path):
            print(f"[{saved+1:>3}] saved {path}  (random)")
            saved += 1
            n += 1
        seed += 1
        if seed > 1000 + args.count * 3:  # safety stop
            break

    print(f"\nDone. {saved} photos in ./{OUT_DIR}/")
    print("Next: python ingest.py")


if __name__ == "__main__":
    main()
