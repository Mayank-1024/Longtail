"""Content-addressed response cache (plan M1) under $LONGTAIL_CACHE_DIR/llm.

A request's key is a hash of the model and the full request, so any change to the prompt, tools or inputs
is a miss. Replay mode (LONGTAIL_REPLAY=1) serves only from here, which makes the demo deterministic.
"""

# ASSUMPTION A-202: the cache root is $LONGTAIL_CACHE_DIR (default ./cache), shared with C3 (A-112)

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any


def cache_root() -> Path:
    return Path(os.environ.get("LONGTAIL_CACHE_DIR", "cache"))


def request_key(model: str, request: dict[str, Any]) -> str:
    raw = json.dumps({"model": model, "request": request}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:32]


class ResponseCache:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else cache_root() / "llm"

    def _path(self, key: str) -> Path:
        return self.root / "responses" / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        p = self._path(key)
        return json.loads(p.read_text()) if p.is_file() else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        # a name per process and thread: parallel writers of the same key must not share a temp file
        tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(value))
        tmp.replace(p)

    # Uploaded file ids, keyed by the file's content hash, so a clip is uploaded once.

    def _files(self) -> Path:
        return self.root / "file_ids.json"

    def file_id(self, digest: str) -> str | None:
        return json.loads(self._files().read_text()).get(digest) if self._files().is_file() else None

    def put_file_id(self, digest: str, file_id: str) -> None:
        known = json.loads(self._files().read_text()) if self._files().is_file() else {}
        known[digest] = file_id
        self._files().parent.mkdir(parents=True, exist_ok=True)
        self._files().write_text(json.dumps(known, indent=1))
