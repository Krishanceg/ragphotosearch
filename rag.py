"""
rag.py  --  THE GENERATION LAYER (what makes this "Multimodal RAG")
------------------------------------------------------------------
Retrieval finds the right photos (clip_model + ChromaDB). This module is the
"Augmented Generation" half: it feeds those retrieved photos into a vision model
(Claude Opus 4.8) that can actually SEE them and write grounded text -- captions
or answers to questions about your pictures.

Retrieve-then-read, the multimodal way:
    text query -> nearest photos (retrieval)  ->  photos + question -> Claude (generation)

Needs an Anthropic API key. Put it in .env:
    ANTHROPIC_API_KEY=sk-ant-...
If the key is missing, the web app still runs -- the ask feature just reports that
the key isn't set, so search/upload/login all keep working for free.
"""

import base64
import os

import anthropic
import auth  # loads .env into os.environ (MONGO_URI, SECRET_KEY, ANTHROPIC_API_KEY, ...)

MODEL = "claude-opus-4-8"        # latest, most capable Claude vision model
MAX_IMAGES = 4                   # keep the evidence set small + high-precision
_MEDIA = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif",
}


def is_enabled():
    """True only if an Anthropic API key is configured."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _client():
    return anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from the environment


def _image_block(path):
    ext = os.path.splitext(path)[1].lower()
    media = _MEDIA.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}


def generate(question, image_paths):
    """
    Ground a vision model on the retrieved photos and answer the question.
    Returns (answer_text, used_paths). Raises RuntimeError if no key is set.
    """
    if not is_enabled():
        raise RuntimeError("ANTHROPIC_API_KEY not set. Add it to .env to enable answers.")

    used = image_paths[:MAX_IMAGES]
    # Build one user turn: all the photos, then the instruction. Images first is
    # the recommended ordering for Claude vision.
    content = [_image_block(p) for p in used]
    content.append({
        "type": "text",
        "text": (
            f"These are photos retrieved from the user's library. "
            f"Answer using ONLY what is visible in them; if they don't show it, say so. "
            f"Question: {question}"
        ),
    })

    client = _client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=(
            "You describe and answer questions about a user's personal photos. "
            "Be concise and factual. Never invent details that aren't visible."
        ),
        messages=[{"role": "user", "content": content}],
    )
    answer = "".join(b.text for b in resp.content if b.type == "text").strip()
    return answer, used
