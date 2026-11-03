"""Which (scene, camera, concept) SAM responses the evaluation needs are missing from a cache.

Run from the repo root, on the machine that holds the cache:

    uv run python scripts/sam_coverage.py --cache cache --scenes-from data/pandaset
    uv run python scripts/sam_coverage.py --cache cache --scenes 001 005 018

Needs no images, no API key and no network. It reads:
- the clip indexes, `<cache>/clips/*.clip_index.json`, which sam_batch.py keeps after deleting each MP4
- the response files in `<cache>/sam/responses/`, named by the SAM client's cache key; their contents are
  read only to flag empty responses and responses whose last frame marker is short of the clip's last frame
  (a stream that may have been cut off)

The concepts are those sam_batch.py asks for: every concept of every gold spec (pipeline.concepts), on
full-frame clips, with include_confidence on. A (scene, camera) with no full-frame clip index counts as
missing for every concept. `--scenes-from` lists scenes the way the pipeline does, so excluded scenes
(A-240) are left out.
Exit status 1 when anything is missing, so a job script can stop before the evaluation.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from longtail.contracts import ScenarioSpec
from longtail.data.pandaset import CAMERA_ORDER, list_scenes
from longtail.pipeline import concepts
from longtail.vision.sam_client import SAM_MODEL

QUERIES = Path("longtail/eval/queries/queries.yaml")
FRAME_MARK = re.compile(r"^<(\d+)f>", re.M)


@dataclass
class Coverage:
    scenes: list[str]
    concepts: list[str]
    missing_clip: list[tuple[str, str]] = field(default_factory=list)  # (scene, camera): no full-frame index
    missing: list[tuple[str, str, str]] = field(default_factory=list)  # (scene, camera, concept)
    empty: list[tuple[str, str, str]] = field(default_factory=list)
    short: list[tuple[str, str, str, int, int]] = field(default_factory=list)  # ..., last marker, clip frames
    several_clips: list[tuple[str, str, int]] = field(default_factory=list)

    @property
    def needed(self) -> int:
        return len(self.scenes) * len(CAMERA_ORDER) * len(self.concepts)

    @property
    def gaps(self) -> int:
        return len(self.missing_clip) * len(self.concepts) + len(self.missing)

    def rerun_scenes(self) -> list[str]:
        return sorted({s for s, _ in self.missing_clip} | {s for s, _, _ in self.missing})


def response_key(clip_id: str, concept: str, include_confidence: bool = True) -> str:
    """The SAM client's cache key (SamClient._key); tests/test_sam_coverage.py checks the two agree."""
    raw = json.dumps([SAM_MODEL, clip_id, concept, include_confidence])
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def gold_concepts(queries: Path = QUERIES) -> list[str]:
    specs = [ScenarioSpec.model_validate(q["gold"]) for q in yaml.safe_load(queries.read_text())["queries"]]
    return sorted({c for s in specs for c in concepts(s)})


def check(cache: Path, scenes: list[str], phrases: list[str], include_confidence: bool = True) -> Coverage:
    cov = Coverage(sorted(scenes), sorted(phrases))
    clips: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for f in sorted((cache / "clips").glob("*.clip_index.json")):
        ix = json.loads(f.read_text())
        if ix["crop"]["name"] == "full":
            clips[(ix["scene_id"], ix["camera"])].append(ix)
    responses = cache / "sam" / "responses"
    have = {p.stem: p for p in responses.glob("*.txt")} if responses.is_dir() else {}

    for s in cov.scenes:
        for cam in CAMERA_ORDER:
            ixs = clips.get((s, cam), [])
            if not ixs:
                cov.missing_clip.append((s, cam))
                continue
            if len(ixs) > 1:
                cov.several_clips.append((s, cam, len(ixs)))
            for c in cov.concepts:
                hits = [
                    (ix, have[k])
                    for ix in ixs
                    if (k := response_key(ix["clip_id"], c, include_confidence)) in have
                ]
                if not hits:
                    cov.missing.append((s, cam, c))
                    continue
                ix, path = hits[0]
                text = path.read_text(errors="replace")
                if not text.strip():
                    cov.empty.append((s, cam, c))
                    continue
                marks = [int(m) for m in FRAME_MARK.findall(text)]
                if marks and max(marks) < len(ix["frames"]) - 1:
                    cov.short.append((s, cam, c, max(marks), len(ix["frames"])))
    return cov


def report(cov: Coverage) -> str:
    n = len(cov.concepts)
    lines = [
        f"{len(cov.scenes)} scenes x {len(CAMERA_ORDER)} cameras x {n} concepts {cov.concepts}"
        f" = {cov.needed} responses",
        f"present {cov.needed - cov.gaps}, missing {cov.gaps}",
    ]
    by_scene: dict[str, list[str]] = defaultdict(list)
    for s, cam in cov.missing_clip:
        by_scene[s].append(f"{cam}: no clip index (all {n} concepts)")
    for s, cam, c in cov.missing:
        by_scene[s].append(f"{cam}: {c}")
    lines += [f"  scene {s}: " + "; ".join(by_scene[s]) for s in sorted(by_scene)]
    if cov.several_clips:
        lines.append(
            f"several full-frame clip indexes for one (scene, camera), images changed?: {cov.several_clips}"
        )
    if cov.empty:
        lines.append(f"{len(cov.empty)} empty responses: {cov.empty[:20]}")
    if cov.short:
        lines.append(
            f"{len(cov.short)} responses whose last frame marker is short of the clip (possible cut-off):"
        )
        lines += [f"  {row}" for row in cov.short[:40]]
    lines.append("rerun scenes: " + (",".join(cov.rerun_scenes()) or "none"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, required=True, help="the directory holding clips/ and sam/")
    ap.add_argument("--queries", type=Path, default=QUERIES)
    ap.add_argument("--concepts", nargs="+", help="check these concepts instead of the gold specs' ones")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenes", nargs="+")
    group.add_argument("--scenes-from", type=Path, help="a PandaSet directory: every scene in it")
    args = ap.parse_args(argv)
    scenes = args.scenes or list_scenes(args.scenes_from)
    cov = check(args.cache, scenes, args.concepts or gold_concepts(args.queries))
    print(report(cov))
    return 1 if cov.gaps else 0


if __name__ == "__main__":
    sys.exit(main())
