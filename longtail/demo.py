"""`longtail demo`: the recorded demo, replayed from the cache with no key and no paid call.

    uv run longtail demo --fetch-cache    # once, on a fresh clone: the replay cache (a GitHub release asset)
    uv run longtail demo --replay         # the web view on http://127.0.0.1:8000, every call from the cache
    uv run longtail demo --replay --query # the hero search, headless; or --query "<another recorded query>"

Needs the camera images of the index scenes (scripts/fetch_pandaset.py) and, for the web view, `npm run build`
in web/. A replay miss is an error, never a paid call.
"""

import argparse
import hashlib
import io
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

# ASSUMPTION A-243: the recorded demo's replay cache is a GitHub release asset, sha256-checked; it holds only
# clip indexes, SAM responses and parsed tracks for the index scenes, and the model responses (cache/llm)
DEMO_CACHE_URL = (
    "https://github.com/Mayank-1024/longtail/releases/download/demo-cache-v1/longtail-demo-cache.tgz"
)
DEMO_CACHE_SHA256 = ""  # set when the asset is published (scripts/pack_demo_cache.py prints it)
HERO_QUERY = "a pedestrian partly hidden by a car"


def register(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    p = sub.add_parser("demo", help="replay the recorded demo from the cache (no key, no paid call)")
    p.add_argument("--replay", action="store_true", help="serve every model and SAM call from the cache")
    p.add_argument("--fetch-cache", action="store_true", help="download and unpack the demo's replay cache")
    p.add_argument("--query", nargs="?", const=HERO_QUERY, help="one search headless, printing the findings")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=_demo)


def _safe_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    out = []
    for m in tar.getmembers():
        parts = Path(m.name).parts
        if not m.isfile() or Path(m.name).is_absolute() or ".." in parts or not parts or parts[0] != "cache":
            raise ValueError(f"unexpected entry in the demo cache: {m.name}")
        out.append(m)
    return out


def unpack(data: bytes, sha256: str, dest: Path = Path(".")) -> int:
    """Check the digest and unpack under dest/cache only; returns the number of files."""
    got = hashlib.sha256(data).hexdigest()
    if sha256 and got != sha256:
        raise ValueError(f"demo cache sha256 is {got}, expected {sha256}")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = _safe_members(tar)
        for m in members:
            target = dest / m.name
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(m)
            assert src is not None
            target.write_bytes(src.read())
    return len(members)


def _fetch(url: str) -> bytes:
    print(f"downloading {url}", file=sys.stderr)
    with urllib.request.urlopen(url) as r:  # noqa: S310 (a fixed https URL)
        return r.read()


def _replay_env() -> None:
    os.environ["LONGTAIL_MODE"] = "replay"
    os.environ["LONGTAIL_REPLAY"] = "1"
    os.environ["LONGTAIL_ALLOW_PAID"] = "0"


def _headless(query: str) -> int:
    from fastapi.testclient import TestClient

    from api.server import create_app_from_env

    with TestClient(create_app_from_env()) as http:
        job = http.post("/api/searches", json={"query": query}).json()["job_id"]
        while (status := http.get(f"/api/jobs/{job}").json())["status"] == "running":
            time.sleep(1)
        if status["status"] != "done":
            print(f"search {status['status']}: {status.get('error')}", file=sys.stderr)
            return 1
        search = http.get(f"/api/searches/{status['search_id']}").json()
    findings = search.get("findings") or search.get("result", {}).get("findings", [])
    for f in findings:
        span = f"{f['t0']:.2f}-{f['t1']:.2f}"
        print(f"VERIFIED {f['scene_id']} {span} score {f['score']:.3f} ({f['finding_id']})")
    print(f"{len(findings)} verified findings; saved runs/{status['search_id']}")
    return 0


def _demo(args: argparse.Namespace) -> int:
    if args.fetch_cache:
        n = unpack(_fetch(DEMO_CACHE_URL), DEMO_CACHE_SHA256)
        print(f"unpacked {n} cache files under cache/")
        if not (args.replay or args.query):
            return 0
    if not args.replay:
        print("the demo runs from the cache only: pass --replay (and --fetch-cache once on a fresh clone)")
        return 2
    _replay_env()
    if args.query:
        return _headless(args.query)
    from api.server import WEB_DIST

    if not WEB_DIST.is_dir():
        print("web/dist is missing: run `cd web && npm install && npm run build` first", file=sys.stderr)
        return 1
    import uvicorn

    print(f"LongTail demo (replay) on http://127.0.0.1:{args.port}", file=sys.stderr)
    uvicorn.run("api.server:create_app_from_env", factory=True, host="127.0.0.1", port=args.port)
    return 0
