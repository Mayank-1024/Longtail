"""Verify one spec on scenes exactly as a search does (pipeline.search_scene: uniform windows, SAM replayed),
writing runs/<spec>-<scene>[-v2]/verify_run.json for a later search to resume from.

    LONGTAIL_REPLAY=1 LONGTAIL_ALLOW_PAID=0 [LONGTAIL_VERIFIER=v2] \\
        PYTHONPATH=. uv run python scripts/search_scene.py spec.json 001 005 [--runs runs]
"""

import argparse
import json
import sys
from pathlib import Path

from longtail.contracts import ScenarioSpec
from longtail.data.index import open_scene
from longtail.pipeline import scene_run_dir, search_scene
from longtail.verify.config import verifier_version
from longtail.vision.sam_client import SamClient, transport_from_env
from longtail.vision.track_source import sam_track_source


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("spec", type=Path, help="a ScenarioSpec as JSON (a verify_run.json spec)")
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    args = ap.parse_args()
    spec = ScenarioSpec.model_validate(json.loads(args.spec.read_text()))
    source = sam_track_source(SamClient(transport_from_env()))
    for sid in args.scenes:
        run = search_scene(spec, open_scene(sid), source, args.runs)
        verified = sum(f.verified for f in run.findings)
        print(f"{scene_run_dir(spec, sid, args.runs)} ({verifier_version()}): {verified} verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
