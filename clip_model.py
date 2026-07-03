"""
clip_model.py
-------------
Loads the CLIP model ONCE and exposes two functions:
  - embed_image(path) -> vector for a photo
  - embed_text(text)  -> vector for a search query

CRITICAL RULE of this whole project: images and text queries must be encoded by the
SAME model, so their vectors live in the same "meaning space". Both ingest.py and
search.py import from here, which guarantees that.

Model: ViT-B-32 (CLIP) trained on LAION-2B. Small enough to run on CPU, no GPU or
paid API required. First run downloads the weights (~600 MB) once, then it's cached.
"""

import open_clip
import torch
from PIL import Image

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

# cosine similarity works on a fixed-size vector; CLIP ViT-B-32 outputs 512 dims.
EMBED_DIM = 512

_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[clip_model] loading {MODEL_NAME} on {_device} (first run downloads weights)...")

_model, _, _preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED
)
_model = _model.to(_device).eval()
_tokenizer = open_clip.get_tokenizer(MODEL_NAME)


def _normalize(vec):
    # normalize to unit length so dot-product == cosine similarity
    return (vec / vec.norm(dim=-1, keepdim=True)).cpu().numpy()[0].tolist()


def embed_image(path):
    img = Image.open(path).convert("RGB")
    tensor = _preprocess(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        vec = _model.encode_image(tensor)
    return _normalize(vec)


def embed_text(text):
    tokens = _tokenizer([text]).to(_device)
    with torch.no_grad():
        vec = _model.encode_text(tokens)
    return _normalize(vec)
