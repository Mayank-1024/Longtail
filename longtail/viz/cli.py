"""`longtail viz <run or finding>`: export a saved verification run for the web view."""

import argparse
import json
import sys
from pathlib import Path

from longtail.data.base import Scene
from longtail.verify.run import RUN_FILE, VerifyRun, load_run
from longtail.viz.export import export_run
from longtail.viz.schema import json_schema


def register(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    p = sub.add_parser("viz", help="export overlays, bird's-eye view and timelines for a verification run")
    p.add_argument(
        "target", type=Path, nargs="?", help="runs/<run> for every finding, or runs/<run>/<finding id>"
    )
    p.add_argument("--out", type=Path, help="output directory inside runs/ (default: <run>/viz)")
    p.add_argument(
        "--write-schema", type=Path, metavar="PATH", help="write the JSON Schema of the outputs and exit"
    )
    p.set_defaults(func=_viz)


def load_scene(run: VerifyRun) -> Scene:
    """The scene a run was verified on. Only PandaSet scenes can be loaded by id."""
    if run.dataset != "pandaset":
        raise ValueError(f"cannot load a {run.dataset!r} scene by id")
    from longtail.data.index import open_scene

    return open_scene(run.scene_id)


def _viz(args: argparse.Namespace) -> int:
    if args.write_schema:
        args.write_schema.write_text(json.dumps(json_schema(), indent=1) + "\n")
        print(f"wrote {args.write_schema}")
        return 0
    if args.target is None:
        print("give a run directory or a finding inside one", file=sys.stderr)
        return 2
    # ASSUMPTION A-103: the target is runs/<run> (every finding) or runs/<run>/<finding id> (one finding)
    if (args.target / RUN_FILE).is_file():
        run_dir, findings = args.target, None
    elif (args.target.parent / RUN_FILE).is_file():
        run_dir, findings = args.target.parent, [args.target.name]
    else:
        print(f"no {RUN_FILE} at {args.target} or its parent", file=sys.stderr)
        return 1
    run = load_run(run_dir)
    manifest = export_run(run, load_scene(run), args.out or run_dir / "viz", findings)
    for f in manifest.findings:
        status = "verified" if f.verified else "rejected"
        print(
            f"{f.finding_id} {status}: {len(f.overlays)} overlays, "
            f"bev {f.bev_bytes / 1e6:.2f} MB, {f.export_seconds:.2f} s"
        )
    print(f"manifest: {(args.out or run_dir / 'viz') / 'manifest.json'}")
    return 0
