import hashlib
from pathlib import Path


def cache_key(*parts):
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:20]


def cache_get(cache_dir, key):
    path = Path(cache_dir) / f"{key}.json"
    return path.read_text() if path.exists() else None


def cache_put(cache_dir, key, text):
    path = Path(cache_dir) / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
