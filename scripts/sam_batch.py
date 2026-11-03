"""Make every SAM 3.1 call the evaluation needs, on the laptop, so no API key goes on the cluster.

    LONGTAIL_ALLOW_PAID=1 LONGTAIL_SPEND_CAP_USD=60 uv run python scripts/sam_batch.py [--scenes 001,002]

For each scene (the 30 index scenes first): build its six full-frame clips, call SAM once per clip and concept
(every concept of every gold spec in the queries file), and save the responses in the SAM cache. With
--delete-clips (the default) a clip's MP4 is deleted once its calls are cached; its .clip_index.json stays.
Clips are content-addressed, so the cluster rebuilds the same clip ids from its own copy of the images, and
`rsync cache/sam/` lets it replay every response there with LONGTAIL_REPLAY=1.

Needs the camera images and JSON of each scene (scripts/fetch_pandaset.py --parts labels, then cameras).
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from longtail.contracts import ScenarioSpec
from longtail.data.index import load_index, open_scene, pandaset_root
from longtail.data.pandaset import list_scenes
from longtail.env import load_dotenv
from longtail.pipeline import concepts
from longtail.vision.clips import build_clip
from longtail.vision.sam_client import SamClient, transport_from_env

# ASSUMPTION A-237: paid SAM calls are made here, on the laptop with the keys; the cluster replays them
QUERIES = Path("longtail/eval/queries/queries.yaml")


def scene_order(only: list[str] | None) -> list[str]:
    on_disk = list_scenes(pandaset_root())
    index = [s["scene_id"] for s in load_index()["scenes"]]
    order = [s for s in index if s in on_disk] + sorted(set(on_disk) - set(index))
    return [s for s in order if only is None or s in only]


def one_scene(scene_id: str, phrases: list[str], client: SamClient, delete_clips: bool) -> tuple[int, float]:
    scene = open_scene(scene_id)
    calls = usd = 0
    for camera in scene.cameras:
        clip = build_clip(scene, camera)
        for concept in phrases:
            call = client.segment(clip, concept)
            calls += 0 if call.cache_hit else 1
            usd += call.usd
        if delete_clips:
            clip.path.unlink(missing_ok=True)
    return calls, usd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--queries", type=Path, default=QUERIES)
    ap.add_argument("--scenes", help="comma-separated scene ids (default: every scene on disk)")
    ap.add_argument("--workers", type=int, default=4, help="scenes in parallel")
    ap.add_argument("--keep-clips", action="store_true")
    args = ap.parse_args()
    load_dotenv()

    queries = yaml.safe_load(args.queries.read_text())["queries"]
    specs = [ScenarioSpec.model_validate(q["gold"]) for q in queries]
    phrases = sorted({c for s in specs for c in concepts(s)})
    order = scene_order(args.scenes.split(",") if args.scenes else None)
    client = SamClient(transport_from_env(), replay=False)
    print(f"{len(order)} scenes x 6 cameras x {len(phrases)} concepts {phrases}", flush=True)
    start, done, total_calls, total_usd, failed = time.time(), 0, 0, 0.0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one_scene, s, phrases, client, not args.keep_clips): s for s in order}
        for f in as_completed(futures):
            sid = futures[f]
            done += 1
            try:
                calls, usd = f.result()
            except Exception as e:  # e.g. the spend cap: the other scenes keep their cached calls
                failed.append(sid)
                print(f"  {sid} FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            total_calls, total_usd = total_calls + calls, total_usd + usd
            print(
                f"  {done}/{len(order)} {sid}: {calls} new calls, ${usd:.3f}; total ${total_usd:.2f}, "
                f"{(time.time() - start) / 60:.1f} min",
                flush=True,
            )
    print(f"done: {total_calls} new calls, ${total_usd:.2f}; failed scenes: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
