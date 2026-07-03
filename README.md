# 📸 SnapSeek — Semantic Photo Search (Multimodal RAG)

**Describe it, find it.** Type a sentence, get the matching photos. No tags, no manual
labeling. The AI reads the **pixels**, so it finds "a dog" even in a file named
`IMG_20250718.jpg`.

## The free stack (no subscription, runs on your PC)

| Part | Tool | Cost |
|---|---|---|
| AI model (image + text → vectors) | `open_clip` (CLIP ViT-B-32) | Free, local |
| Vector database (store + search) | ChromaDB | Free, local |
| Photo storage | your disk | Free |

## How it works

```
INGEST (once):   photo  → CLIP → vector → ChromaDB
SEARCH (anytime): "a dog" → CLIP → vector → nearest vectors → matching photos
```

The golden rule: photos and queries are encoded by the **same** model (`clip_model.py`),
so their vectors live in the same space and "search" = "find nearest vectors".

## Setup (one time)

```bash
pip install -r requirements.txt
```

First run downloads the CLIP model weights (~600 MB) once, then caches them.

## Run it

```bash
# 1. get some free test photos (or skip and use your own folder)
python download_samples.py

# 2. embed the photos into the database
python ingest.py                 # uses ./test_photos
#   or your own pictures:
python ingest.py "C:/Users/Admin/Pictures"

# 3. search by TEXT
python search.py "a dog"
python search.py "sunset over water" --top 10
python search.py "food on a plate" --open

# 3b. search by IMAGE (find visually similar photos)
python search_image.py test_photos/img_001.jpg --top 5
```

## Files

| File | Role |
|---|---|
| `clip_model.py` | Loads CLIP once; `embed_image()` / `embed_text()` |
| `download_samples.py` | Grabs free test photos |
| `ingest.py` | Offline pipeline: photos → vectors → DB |
| `search.py` | Online pipeline: query → nearest photos |

## Phase 2 — Web UI (in browser) with login

```bash
python app.py
```
Then open **http://127.0.0.1:8500**. You'll hit a **login page** first — create an
account, then you're in. Type a query — results show as a clickable photo grid with
similarity scores. (Set a different port with `PORT=9000 python app.py`.)

### Accounts (MongoDB)
- User accounts are stored in **MongoDB** (Atlas cloud), collection `users`.
- Passwords are **bcrypt-hashed** — the plain password is never stored.
- Secrets live in **`.env`** (never commit it — it's in `.gitignore`):
  ```
  MONGO_URI=mongodb+srv://USER:PASS@cluster0.xxxx.mongodb.net/?appName=Cluster0
  MONGO_DB=photo_app
  SECRET_KEY=<random hex for signing session cookies>
  ```
- Every page and `/api/*` route requires login; sessions are signed cookies (7-day expiry).
- Auth endpoints: `GET/POST /login`, `GET/POST /register`, `GET /logout`.

| Endpoint | Purpose |
|---|---|
| `GET /` | the search page |
| `GET /api/search?q=...&top=12` | JSON results |
| `GET /image?id=<path>` | serves a photo (only files in the DB, for safety) |
| `GET /api/count` | how many photos are in the library |
| `POST /api/upload` | add photos from the browser (the **+ Add photos** button) |
| `POST /api/search_by_image` | find photos similar to an uploaded image (**🖼 Search by image** button) |

**Adding photos:** click **+ Add photos** in the UI to upload from your computer — they're
embedded and searchable instantly. Uploaded files are saved in `uploads/`. (Bulk-adding a
whole folder is still fastest via `python ingest.py "C:/path/to/folder"`.)

## What's next (later phases)

- **Phase 2+:** metadata filters (date/location from EXIF), image-to-image search
- **Phase 3 (true RAG):** feed top results into a vision model to caption / answer questions
  about your photos
