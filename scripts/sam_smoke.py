"""SAM 3.1 smoke test (plan C3 and the Oct 28 milestone): one clip, one concept, a real paid call.

Not run in CI and never without a key. It sends one PandaSet camera clip with one concept, saves the raw
response as a fixture, checks that our parser reads it, and prints the cost per frame.

Procedure, once you have a Meta key with a spend limit and PandaSet is downloaded:

1. Put the key in .env (git-ignored) as MODEL_API_KEY=..., then load it into the shell:
       set -a; source .env; set +a
2. Make sure the scene is on disk:  uv run longtail scenes ls   (or check data/pandaset/<scene>/)
3. Run, adding the OpenAI SDK for this run only (M1 will make it a dependency):
       uv run --with openai python scripts/sam_smoke.py --scene 002 --camera front_camera --concept person
   Cost: about $0.016 for one 80-frame clip ($0.20 per 1,000 frames, 50-frame minimum).
4. Read the output:
   - "file id" and "streamed N characters": upload and streaming work.
   - "parsed K tracks": our parser (meta-sam-parser) reads the real format. If parsing fails, the raw response
     is still saved; fix sam_parse.py against it.
   - "cost per frame": should print $0.000200.
   - "confidence present": tells whether include_confidence in metadata works (plan 3.1 marks it VERIFY).
5. Look at the saved fixture before committing it. It holds boxes and masks derived from PandaSet frames;
   commit it only if the dataset licence allows (ask Mayank, A-01). Then replace the SYNTHETIC fixture in the
   parser test with it, and record the cost in docs/costs.md and docs/build-log.md.
6. Re-run the same command: it must report a cache hit at $0.
"""

import argparse
import os
import sys
import time
from pathlib import Path

from longtail.data.index import open_scene
from longtail.vision.clips import build_clip
from longtail.vision.sam_client import USD_PER_FRAME, SamClient, transport_from_env
from longtail.vision.sam_parse import parse

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--camera", default="front_camera")
    ap.add_argument("--concept", default="person")
    args = ap.parse_args(argv)

    if not os.environ.get("MODEL_API_KEY", "").strip():
        print(
            "MODEL_API_KEY is not set; this script makes a paid call and needs a Meta key.", file=sys.stderr
        )
        return 2
    scene = open_scene(args.scene)
    clip = build_clip(scene, args.camera)
    print(f"clip {clip.path} ({len(clip.index.frames)} frames at {clip.index.fps} fps, cached={clip.cached})")

    client = SamClient(transport_from_env(), replay=False)
    start = time.perf_counter()
    file_id = client.file_id(clip)
    print(f"file id {file_id}")
    call = client.segment(clip, args.concept)
    print(
        f"streamed {len(call.text)} characters in {time.perf_counter() - start:.1f} s, "
        f"attempts {call.attempts}"
    )

    stem = f"sam31_{args.scene}_{args.camera}_{args.concept.replace(' ', '_')}"
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / f"{stem}.txt").write_text(call.text)
    (FIXTURES / f"{stem}.clip_index.json").write_text(clip.index.model_dump_json(indent=1) + "\n")
    print(f"saved {FIXTURES / stem}.txt and .clip_index.json")

    try:
        tracks = parse(call.text, clip.index, args.concept)
        confs = [f.conf for t in tracks for f in t.frames]
        print(f"parsed {len(tracks)} tracks; truncated={any(t.truncated for t in tracks)}")
        print(f"confidence present: {any(c is not None for c in confs)}")
    except Exception as e:  # keep the saved response for debugging the parser
        print(f"PARSE FAILED: {e}", file=sys.stderr)
    print(
        f"billed ${call.usd:.4f} for {call.frames} frames: cost per frame ${call.usd / call.frames:.6f} "
        f"(list price ${USD_PER_FRAME:.6f}, 50-frame minimum)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
