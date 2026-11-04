"""Pack the replay cache the demo needs into one tarball, for a GitHub release asset.

    uv run python scripts/pack_demo_cache.py --concepts person car [--out longtail-demo-cache.tgz]

Takes, for every scene of the index (or --scenes): the full-frame clip indexes, the SAM responses and parsed
tracks for each concept, and all of cache/llm (the planner, agent and card calls). Clip MP4s are left out: a
clone rebuilds them from its own PandaSet images and gets the same content-addressed clip ids. Prints the
sha256 that `longtail demo --fetch-cache` checks.
"""

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

from longtail.data.index import load_index
from longtail.vision.clips import Clip, ClipIndex, cache_root
from longtail.vision.sam_client import SamClient
from longtail.vision.track_source import _tracks_path


def members(scenes: set[str], concepts: list[str]) -> tuple[list[Path], list[str]]:
    root = cache_root()
    client = SamClient(None, replay=True)
    files, missing = [], []
    for path in sorted((root / "clips").glob("*.clip_index.json")):
        index = ClipIndex.model_validate_json(path.read_text())
        if index.scene_id not in scenes or index.crop.name != "full":
            continue
        files.append(path)
        clip = Clip(path.with_name(f"{index.clip_id}.mp4"), index, cached=True)
        for concept in concepts:
            response = client.cache_dir / "responses" / f"{client._key(clip, concept)}.txt"
            if not response.is_file():
                missing.append(f"{index.scene_id}/{index.camera}/{concept}: no SAM response")
                continue
            files.append(response)
            tracks = _tracks_path(response.read_text(), clip, concept)
            if tracks.is_file():
                files.append(tracks)
    files += sorted(p for p in (root / "llm").rglob("*") if p.is_file())
    return files, missing


def search_members(search_id: str, run_root: Path = Path("runs")) -> list[Path]:
    """The recorded search (search.json, trace, cards) and each scene's verify_run.json; never viz/."""
    search_dir = run_root / search_id
    doc = json.loads((search_dir / "search.json").read_text())
    files = sorted(
        p for p in search_dir.rglob("*") if p.is_file() and "viz" not in p.relative_to(search_dir).parts
    )
    for rel in doc["run_dirs"].values():
        files.append(run_root / rel / "verify_run.json")
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--concepts", nargs="+", required=True, help="the SAM phrases the demo searches use")
    ap.add_argument("--scenes", nargs="*", help="scene ids (default: the scene index)")
    ap.add_argument("--out", type=Path, default=Path("longtail-demo-cache.tgz"))
    ap.add_argument("--search", nargs="*", default=[], help="recorded search ids (runs/search-*) to ship")
    args = ap.parse_args()
    scenes = set(args.scenes or [s["scene_id"] for s in load_index()["scenes"]])
    files, missing = members(scenes, args.concepts)
    runs = [f for sid in args.search for f in search_members(sid)]
    if missing:
        print("\n".join(missing), file=sys.stderr)
        print(f"{len(missing)} SAM responses missing: the replay would fail on them", file=sys.stderr)
        return 1
    root = cache_root().parent
    with tarfile.open(args.out, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=str(f.resolve().relative_to(root.resolve())))
        for f in runs:  # under runs/, where the web view and `longtail demo --list` look for them
            tar.add(f, arcname=str(Path("runs") / f.relative_to("runs")))
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    n = len(files) + len(runs)
    print(json.dumps({"out": str(args.out), "files": n, "scenes": len(scenes), "sha256": digest}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
