"""`longtail sam <scene> <camera> <concept>`: SAM 3.1 tracks for one camera of one scene, with their cost."""

import argparse
import sys
from pathlib import Path

from longtail.data.base import Scene
from longtail.vision.clips import build_clip
from longtail.vision.sam_client import NoKeyError, ReplayMissError, SamClient, transport_from_env
from longtail.vision.sam_parse import parse


def register(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    p = sub.add_parser("sam", help="SAM 3.1 tracks for one camera of a scene, and the cost")
    p.add_argument("scene", help="scene id, e.g. 002")
    p.add_argument("camera", help="camera name, e.g. front_camera")
    p.add_argument("concept", help='text concept, e.g. "person"')
    p.add_argument("--crop", choices=["full", "left", "right"], default="full")
    p.add_argument("--replay", action="store_true", help="serve only from the cache; fail on a miss")
    p.add_argument("--fixture", type=Path, help="parse this saved response instead of calling SAM")
    p.set_defaults(func=_sam)


def load_scene(scene_id: str) -> Scene:
    from longtail.data.index import open_scene

    return open_scene(scene_id)


def _sam(args: argparse.Namespace) -> int:
    # ASSUMPTION A-119: `longtail sam` without a key fails with a message; --replay and --fixture need none
    transport = None if (args.replay or args.fixture) else transport_from_env()
    if transport is None and not (args.replay or args.fixture):
        print(
            "MODEL_API_KEY is not set: SAM 3.1 calls need a Meta API key. "
            "Use --replay to serve from the cache, or --fixture to parse a saved response.",
            file=sys.stderr,
        )
        return 2
    scene = load_scene(args.scene)
    clip = build_clip(scene, args.camera, args.crop)
    if args.fixture:
        text, cost, note = args.fixture.read_text(), 0.0, f"fixture {args.fixture}"
    else:
        client = SamClient(transport, replay=args.replay)
        try:
            call = client.segment(clip, args.concept)
        except (ReplayMissError, NoKeyError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        text, cost = call.text, call.usd
        note = "cache hit, $0" if call.cache_hit else f"{call.frames} frames, ${call.usd:.4f}"
    tracks = parse(text, clip.index, args.concept)
    print(
        f"{args.scene} {args.camera} {args.concept!r}: {len(tracks)} track(s) "
        f"from {len(clip.index.frames)} frames "
        f"({note})"
    )
    for t in tracks:
        confs = [f.conf for f in t.frames if f.conf is not None]
        mean = f"{sum(confs) / len(confs):.2f}" if confs else "-"
        trunc = "  TRUNCATED (16-object cap)" if t.truncated else ""
        print(
            f"  object {t.obj_id}: {len(t.frames)} frames "
            f"({t.frames[0].frame_idx}-{t.frames[-1].frame_idx}), "
            f"mean conf {mean}{trunc}"
        )
    print(f"cost: ${cost:.4f}")
    return 0
