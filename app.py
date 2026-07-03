"""
app.py  --  PHASE 2: the web UI
-------------------------------
Runs a small web server so you can search your photos in a browser instead of the
command line. Reuses the EXACT same pieces as before:
  - clip_model.embed_text  -> turn the typed query into a vector
  - the same ChromaDB photo_db -> nearest-vector search

Endpoints:
  GET  /                      -> the search page (HTML)
  GET  /api/search?q=...&top= -> JSON results [{path, filename, score}]
  GET  /image?id=<path>       -> serves a photo file (only ones in the DB, for safety)

Run:
    python app.py
then open  http://127.0.0.1:8000  in your browser.
"""

import os
import shutil
import chromadb
from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from clip_model import embed_text, embed_image
import auth
import rag

DB_DIR = "photo_db"
COLLECTION = "photos"
UPLOAD_DIR = os.path.abspath("uploads")   # photos added via the web UI land here
VALID_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

app = FastAPI(title="SnapSeek")
# signed-cookie sessions; the secret comes from .env via auth.py
app.add_middleware(SessionMiddleware, secret_key=auth.SECRET_KEY, max_age=60 * 60 * 24 * 7)

_client = chromadb.PersistentClient(path=DB_DIR)


def current_user(request: Request):
    return request.session.get("user")


def require_api_user(request: Request):
    """For /api/* routes: 401 if not logged in."""
    user = current_user(request)
    if not user:
        raise HTTPException(401, "Not logged in")
    return user


def _collection():
    try:
        return _client.get_collection(COLLECTION)
    except Exception:
        raise HTTPException(503, "No database yet. Run: python ingest.py")


@app.get("/api/count")
def api_count(request: Request):
    require_api_user(request)
    return {"count": _collection().count()}


@app.post("/api/upload")
async def api_upload(request: Request, files: list[UploadFile] = File(...)):
    """Save uploaded photos, embed them, and add them to the database -- live."""
    require_api_user(request)
    col = _collection()
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    known = set(col.get()["ids"])
    added, skipped = [], []

    for f in files:
        if not f.filename.lower().endswith(VALID_EXT):
            skipped.append({"name": f.filename, "reason": "not an image"})
            continue
        dest = os.path.join(UPLOAD_DIR, os.path.basename(f.filename))
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)

        image_id = os.path.abspath(dest)
        if image_id in known:
            skipped.append({"name": f.filename, "reason": "already in library"})
            continue
        try:
            vec = embed_image(dest)
        except Exception as e:
            skipped.append({"name": f.filename, "reason": str(e)})
            continue
        col.add(
            ids=[image_id],
            embeddings=[vec],
            metadatas=[{"path": image_id, "filename": os.path.basename(dest)}],
        )
        added.append(f.filename)

    return {"added": added, "skipped": skipped, "total": col.count()}


@app.post("/api/search_by_image")
async def api_search_by_image(request: Request, file: UploadFile = File(...), top: int = 12):
    """Find photos visually similar to an uploaded query image (image-to-image)."""
    require_api_user(request)
    col = _collection()
    if col.count() == 0:
        return JSONResponse({"results": [], "message": "Database empty. Run ingest.py"})

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    tmp = os.path.join(UPLOAD_DIR, "_query_" + os.path.basename(file.filename or "q.jpg"))
    with open(tmp, "wb") as out:
        shutil.copyfileobj(file.file, out)
    try:
        qvec = embed_image(tmp)
    finally:
        if os.path.isfile(tmp):
            os.remove(tmp)              # query image is temporary, don't keep it

    res = col.query(query_embeddings=[qvec], n_results=min(top, col.count()))
    results = [
        {"path": p, "filename": os.path.basename(p), "score": round(1 - d, 3)}
        for p, d in zip(res["ids"][0], res["distances"][0])
    ]
    return {"query": "(uploaded image)", "results": results}


REL_MARGIN = 0.08    # keep photos whose score is within this of the BEST match (the "cliff")


@app.get("/api/search")
def api_search(request: Request, q: str, top: int = 12, min_score: float = 0.21):
    """
    Adaptive relevance filtering so only photos that truly match show up:
      1. floor  = min_score (the slider) -- if even the best match is below it, show nothing.
      2. cutoff = max(best_score - REL_MARGIN, floor) -- real matches cluster near the top,
         then drop off a cliff; we keep the cluster and cut everything past the cliff.
    """
    require_api_user(request)
    col = _collection()
    n = col.count()
    if n == 0:
        return JSONResponse({"results": [], "message": "Database empty. Run ingest.py"})

    # Prompt template "a photo of ..." sharpens the gap between the right subject and the
    # rest (standard CLIP technique). The displayed query stays what the user typed.
    qvec = embed_text(f"a photo of {q}")
    res = col.query(query_embeddings=[qvec], n_results=min(n, 200))

    scored = [(p, round(1 - d, 3)) for p, d in zip(res["ids"][0], res["distances"][0])]
    best = scored[0][1] if scored else 0.0

    results = []
    if best >= min_score:
        cutoff = max(best - REL_MARGIN, min_score)
        for path, score in scored:
            if score < cutoff:
                break
            results.append({"path": path, "filename": os.path.basename(path), "score": score})
            if len(results) >= top:
                break

    message = None
    if not results:
        message = (f'No photos clearly match "{q}". '
                   f'Try different wording, or lower the match strength.')
    return {"query": q, "results": results, "message": message}


@app.get("/api/ask")
def api_ask(request: Request, q: str, top: int = 4):
    """
    MULTIMODAL RAG: retrieve the most relevant photos for the question, then feed
    them to a vision model that answers grounded on what's actually in them.
    """
    require_api_user(request)
    if not rag.is_enabled():
        return {"answer": None, "used": [],
                "message": "AI answers are off. Add ANTHROPIC_API_KEY to .env to enable them."}

    col = _collection()
    if col.count() == 0:
        return {"answer": None, "used": [], "message": "No photos yet. Upload some first."}

    # 1) RETRIEVE: nearest photos to the question
    qvec = embed_text(f"a photo of {q}")
    res = col.query(query_embeddings=[qvec], n_results=min(col.count(), top))
    paths = [p for p in res["ids"][0] if os.path.isfile(p)]
    if not paths:
        return {"answer": None, "used": [], "message": "Couldn't find any matching photos."}

    # 2) GENERATE: vision model answers grounded on those photos
    try:
        answer, used = rag.generate(q, paths)
    except Exception as e:
        return {"answer": None, "used": [], "message": f"Generation failed: {e}"}

    used_meta = [{"path": p, "filename": os.path.basename(p)} for p in used]
    return {"answer": answer, "used": used_meta, "message": None}


@app.get("/image")
def image(request: Request, id: str):
    require_api_user(request)
    # SECURITY: only serve files that are actually in the database, never arbitrary paths.
    col = _collection()
    known = set(col.get()["ids"])
    if id not in known or not os.path.isfile(id):
        raise HTTPException(404, "Not found")
    return FileResponse(id)


# ----------------------------------------------------------------------------
# Authentication routes (login / register / logout)
# ----------------------------------------------------------------------------

@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    if current_user(request):
        return RedirectResponse("/", status_code=303)
    return auth_page("Create account", "/register", "Register", error=request.query_params.get("e"))


@app.post("/register")
def register_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    ok, msg = auth.create_user(username, password)
    if not ok:
        return RedirectResponse(f"/register?e={msg}", status_code=303)
    request.session["user"] = username.strip().lower()
    return RedirectResponse("/", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if current_user(request):
        return RedirectResponse("/", status_code=303)
    return auth_page("Sign in", "/login", "Login", error=request.query_params.get("e"))


@app.post("/login")
def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    user = auth.verify_user(username, password)
    if not user:
        return RedirectResponse("/login?e=Invalid username or password.", status_code=303)
    request.session["user"] = user
    return RedirectResponse("/", status_code=303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    user = current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    return HTML_PAGE.replace("{{USER}}", user)


def auth_page(title, action, button, error=None):
    other = ("/login", "Already have an account? Sign in") if action == "/register" \
        else ("/register", "New here? Create an account")
    err_html = f'<div class="err">{error}</div>' if error else ""
    return f"""
<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — SnapSeek</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ font-family:system-ui,sans-serif; background:#0f1115; color:#e8e8ea;
         display:flex; min-height:100vh; margin:0; align-items:center; justify-content:center; }}
  .box {{ background:#181b22; border:1px solid #222; border-radius:16px; padding:32px;
         width:min(360px,90vw); }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  p.sub {{ margin:0 0 20px; color:#8a8f98; font-size:13px; }}
  label {{ display:block; font-size:12px; color:#9aa0aa; margin:12px 0 4px; }}
  input {{ width:100%; box-sizing:border-box; padding:11px 12px; border-radius:9px;
          border:1px solid #333; background:#0f1115; color:#fff; font-size:14px; }}
  button {{ width:100%; margin-top:18px; padding:12px; border:0; border-radius:9px;
           background:#4f7cff; color:#fff; font-size:15px; cursor:pointer; }}
  a {{ color:#4f7cff; text-decoration:none; font-size:13px; }}
  .alt {{ text-align:center; margin-top:16px; }}
  .err {{ background:#3a1d22; color:#ff9aa6; border:1px solid #5a2730; padding:9px 12px;
         border-radius:9px; font-size:13px; margin-bottom:12px; }}
</style></head><body>
  <form class="box" method="post" action="{action}">
    <h1>📸 SnapSeek</h1>
    <p class="sub">{title}</p>
    {err_html}
    <label>Username</label>
    <input name="username" autocomplete="username" autofocus required>
    <label>Password</label>
    <input name="password" type="password" autocomplete="current-password" required>
    <button type="submit">{button}</button>
    <div class="alt"><a href="{other[0]}">{other[1]}</a></div>
  </form>
</body></html>
"""


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SnapSeek — describe it, find it</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: system-ui, sans-serif; margin: 0; background:#0f1115; color:#e8e8ea; }
  header { padding: 24px; text-align:center; border-bottom:1px solid #222; }
  h1 { margin:0 0 4px; font-size:22px; }
  p.sub { margin:0; color:#8a8f98; font-size:13px; }
  .searchbar { display:flex; gap:8px; justify-content:center; padding:20px; position:sticky; top:0; background:#0f1115; }
  input { width:min(520px,70vw); padding:12px 14px; border-radius:10px; border:1px solid #333; background:#181b22; color:#fff; font-size:15px; }
  button { padding:12px 20px; border:0; border-radius:10px; background:#4f7cff; color:#fff; font-size:15px; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:14px; padding:0 24px 40px; }
  .card { background:#181b22; border:1px solid #222; border-radius:12px; overflow:hidden; }
  .card img { width:100%; height:160px; object-fit:cover; display:block; cursor:zoom-in; }
  .meta { display:flex; justify-content:space-between; padding:8px 10px; font-size:12px; color:#9aa0aa; }
  .score { color:#6ee7a8; font-weight:600; }
  .low { color:#f0a868; }
  .hint { text-align:center; color:#8a8f98; padding:30px; }
  .examples { text-align:center; padding:0 24px 10px; color:#8a8f98; font-size:13px; }
  .examples a { color:#4f7cff; text-decoration:none; margin:0 6px; cursor:pointer; }
</style>
</head>
<body>
<header>
  <div style="position:absolute; right:20px; top:18px; font-size:13px; color:#9aa0aa;">
    👤 {{USER}} &nbsp;·&nbsp; <a href="/logout" style="color:#4f7cff; text-decoration:none;">Logout</a>
  </div>
  <h1>📸 SnapSeek</h1>
  <p class="sub">Describe it, find it — the AI reads the pixels, not the filenames.</p>
  <p class="sub" id="count">library: … photos</p>
</header>

<div class="searchbar">
  <input id="q" placeholder='try: "a dog", "snowy mountains", "food on a plate"' autofocus>
  <button id="go">Search</button>
  <button id="addBtn" style="background:#2b9e5f">+ Add photos</button>
  <button id="imgBtn" style="background:#7c5cff">🖼 Search by image</button>
  <input id="file" type="file" accept="image/*" multiple style="display:none">
  <input id="imgFile" type="file" accept="image/*" style="display:none">
</div>
<div style="text-align:center; color:#8a8f98; font-size:13px; padding:0 24px 4px;">
  match strength:
  <input id="thr" type="range" min="0.10" max="0.35" step="0.01" value="0.21" style="vertical-align:middle;">
  <span id="thrVal">0.21</span>
  &nbsp;<span style="color:#666;">(higher = stricter — fewer but more exact results)</span>
</div>
<div class="examples">
  examples:
  <a onclick="run('a dog')">a dog</a>
  <a onclick="run('a cat')">a cat</a>
  <a onclick="run('snowy mountains')">snowy mountains</a>
  <a onclick="run('food on a plate')">food on a plate</a>
  <a onclick="run('a person in nature')">a person in nature</a>
</div>

<div style="max-width:760px; margin:6px auto 0; padding:12px 24px;">
  <div style="display:flex; gap:8px;">
    <input id="ask" placeholder='Ask about your photos: "what animals are in my library?"'
           style="flex:1; padding:11px 13px; border-radius:10px; border:1px solid #333; background:#181b22; color:#fff; font-size:14px;">
    <button id="askBtn" style="padding:11px 18px; border:0; border-radius:10px; background:#c68a3a; color:#fff; cursor:pointer;">Ask AI</button>
  </div>
  <div id="answer" style="margin-top:10px;"></div>
</div>

<div id="status" class="hint">Enter a query to search your photo library.</div>
<div id="grid" class="grid"></div>

<script>
const q = document.getElementById('q');
const go = document.getElementById('go');
const grid = document.getElementById('grid');
const status = document.getElementById('status');
const thr = document.getElementById('thr');
const thrVal = document.getElementById('thrVal');
thr.addEventListener('input', () => { thrVal.textContent = (+thr.value).toFixed(2); });

function showResults(results, query){
  grid.innerHTML = '';
  for(const item of results){
    const card = document.createElement('div'); card.className = 'card';
    card.innerHTML =
      '<img loading="lazy" src="/image?id=' + encodeURIComponent(item.path) + '" '+
      'onclick="window.open(this.src)">' +
      '<div class="meta"><span>' + item.filename + '</span>' +
      '<span class="score">' + item.score.toFixed(3) + '</span></div>';
    grid.appendChild(card);
  }
}

async function run(text){
  if(text){ q.value = text; }
  const query = q.value.trim();
  if(!query) return;
  go.disabled = true; status.textContent = 'Searching for "' + query + '"...'; grid.innerHTML = '';
  try {
    const r = await fetch('/api/search?q=' + encodeURIComponent(query) +
                          '&top=12&min_score=' + thr.value);
    if(r.status === 401){ location = '/login'; return; }
    const data = await r.json();
    const results = data.results || [];
    if(results.length){
      status.textContent = results.length + ' match' + (results.length>1?'es':'') +
        ' for "' + query + '" (showing only strong matches ≥ ' + (+thr.value).toFixed(2) + ')';
    } else {
      status.textContent = data.message || ('No strong matches for "' + query + '".');
    }
    showResults(results, query);
  } catch(e){ status.textContent = 'Error: ' + e; }
  go.disabled = false;
}
go.onclick = () => run();
q.addEventListener('keydown', e => { if(e.key === 'Enter') run(); });

// ---- Ask AI (Multimodal RAG) ----
const ask = document.getElementById('ask');
const askBtn = document.getElementById('askBtn');
const answer = document.getElementById('answer');

async function runAsk(){
  const question = ask.value.trim();
  if(!question) return;
  askBtn.disabled = true;
  answer.innerHTML = '<div style="color:#8a8f98;">Reading your photos…</div>';
  try {
    const r = await fetch('/api/ask?q=' + encodeURIComponent(question));
    if(r.status === 401){ location = '/login'; return; }
    const d = await r.json();
    if(d.answer){
      const thumbs = (d.used||[]).map(u =>
        '<img src="/image?id=' + encodeURIComponent(u.path) + '" title="' + u.filename +
        '" style="width:70px;height:70px;object-fit:cover;border-radius:8px;margin:4px 4px 0 0;">').join('');
      answer.innerHTML =
        '<div style="background:#181b22;border:1px solid #2a2f3a;border-radius:12px;padding:14px;">' +
        '<div style="white-space:pre-wrap;">' + d.answer + '</div>' +
        '<div style="margin-top:8px;">' + thumbs + '</div>' +
        '<div style="color:#666;font-size:12px;margin-top:6px;">answer grounded on the photos above</div></div>';
    } else {
      answer.innerHTML = '<div style="color:#f0a868;">' + (d.message || 'No answer.') + '</div>';
    }
  } catch(e){ answer.innerHTML = '<div style="color:#ff9aa6;">Error: ' + e + '</div>'; }
  askBtn.disabled = false;
}
askBtn.onclick = runAsk;
ask.addEventListener('keydown', e => { if(e.key === 'Enter') runAsk(); });

// ---- Add photos (upload) ----
const addBtn = document.getElementById('addBtn');
const file = document.getElementById('file');
const countEl = document.getElementById('count');

async function refreshCount(){
  try { const r = await fetch('/api/count'); const d = await r.json();
        countEl.textContent = 'library: ' + d.count + ' photos'; } catch(e){}
}

addBtn.onclick = () => file.click();
file.onchange = async () => {
  if(!file.files.length) return;
  addBtn.disabled = true; addBtn.textContent = 'Uploading ' + file.files.length + '...';
  const fd = new FormData();
  for(const f of file.files) fd.append('files', f);
  try {
    const r = await fetch('/api/upload', { method:'POST', body: fd });
    const d = await r.json();
    status.textContent = 'Added ' + d.added.length + ' photo(s)' +
      (d.skipped.length ? ', skipped ' + d.skipped.length : '') +
      '. Library now has ' + d.total + ' photos. Try searching for them!';
  } catch(e){ status.textContent = 'Upload error: ' + e; }
  addBtn.disabled = false; addBtn.textContent = '+ Add photos'; file.value = '';
  refreshCount();
};

refreshCount();

// ---- Search by image (image-to-image) ----
const imgBtn = document.getElementById('imgBtn');
const imgFile = document.getElementById('imgFile');
imgBtn.onclick = () => imgFile.click();
imgFile.onchange = async () => {
  if(!imgFile.files.length) return;
  go.disabled = true; grid.innerHTML = '';
  status.textContent = 'Finding photos similar to "' + imgFile.files[0].name + '"...';
  const fd = new FormData(); fd.append('file', imgFile.files[0]);
  try {
    const r = await fetch('/api/search_by_image?top=12', { method:'POST', body: fd });
    const data = await r.json();
    const results = data.results || [];
    status.textContent = results.length + ' photos similar to your image (higher = more similar)';
    for(const item of results){
      const card = document.createElement('div'); card.className = 'card';
      const cls = item.score >= 0.40 ? 'score' : 'score low';
      card.innerHTML =
        '<img loading="lazy" src="/image?id=' + encodeURIComponent(item.path) + '" '+
        'onclick="window.open(this.src)">' +
        '<div class="meta"><span>' + item.filename + '</span>' +
        '<span class="' + cls + '">' + item.score.toFixed(3) + '</span></div>';
      grid.appendChild(card);
    }
  } catch(e){ status.textContent = 'Error: ' + e; }
  go.disabled = false; imgFile.value = '';
};
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    PORT = int(os.environ.get("PORT", "8500"))
    print(f"\n  Open http://127.0.0.1:{PORT} in your browser\n")
    uvicorn.run(app, host="127.0.0.1", port=PORT)
