"""
auth.py  --  user accounts backed by MongoDB
--------------------------------------------
- Reads secrets from .env (MONGO_URI, MONGO_DB, SECRET_KEY).
- Stores users in MongoDB collection `users` with BCRYPT-HASHED passwords
  (we never store the plain password).
- Exposes: create_user(), verify_user(), get_user().

Why bcrypt: it's a slow, salted password hash. Even if the database leaks, the
plain passwords are not recoverable.
"""

import os
import datetime
import bcrypt
from pymongo import MongoClient, ASCENDING


# ---- tiny .env loader (avoids needing python-dotenv) ----
def _load_env(path=".env"):
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()

MONGO_URI = os.environ.get("MONGO_URI")
MONGO_DB = os.environ.get("MONGO_DB", "photo_app")
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-insecure-change-me")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set. Add it to your .env file.")

_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
_db = _client[MONGO_DB]
users = _db["users"]

# Ensure usernames are unique. Do this lazily and non-fatally: if Atlas is
# unreachable at startup (paused free cluster, network blip), the app must still
# boot and serve pages -- auth calls will surface a clear error only when used.
_index_ready = False


def _ensure_index():
    global _index_ready
    if not _index_ready:
        users.create_index([("username", ASCENDING)], unique=True)
        _index_ready = True


def create_user(username: str, password: str):
    """Returns (ok, message). Fails if username taken or input invalid."""
    username = (username or "").strip().lower()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password or "") < 6:
        return False, "Password must be at least 6 characters."
    _ensure_index()
    if users.find_one({"username": username}):
        return False, "That username is already taken."
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    users.insert_one({
        "username": username,
        "password_hash": pw_hash,                 # stored as BSON bytes
        "created_at": datetime.datetime.utcnow(),
    })
    return True, "Account created."


def verify_user(username: str, password: str):
    """Returns the username string if credentials are valid, else None."""
    username = (username or "").strip().lower()
    doc = users.find_one({"username": username})
    if not doc:
        return None
    stored = doc["password_hash"]
    if isinstance(stored, str):
        stored = stored.encode()
    if bcrypt.checkpw(password.encode(), stored):
        return username
    return None


def get_user(username: str):
    return users.find_one({"username": (username or "").strip().lower()}, {"password_hash": 0})
