"""`longtail baseline ...` and `longtail card ...`: Spark Open on Baseten (paid unless cached)."""

import argparse
import json
import sys
from pathlib import Path

import yaml

from longtail.data.index import load_index, open_scene
from longtail.eval.baseline_vlm import OUT_DIR, run_baseline
from longtail.explain.cards import make_card
from longtail.llm.baseten import spark_open_client
from longtail.llm.client import BudgetExceeded, NoKeyError, PaidCallsDisabled, ReplayMissError
from longtail.verify.run import load_run

GATED = (PaidCallsDisabled, NoKeyError, ReplayMissError, BudgetExceeded)


def register(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    b = sub.add_parser("baseline", help="ablation baseline (a): Spark Open describe-then-match")
    b.add_argument("--queries", type=Path, required=True, help="queries file (only query text is sent)")
    b.add_argument("--scenes", nargs="*", help="scene ids (default: the scene index)")
    b.add_argument("--out", type=Path, default=OUT_DIR)
    b.set_defaults(func=_baseline)
    c = sub.add_parser("card", help="the evidence card for one finding of a saved verify run")
    c.add_argument("run_dir", type=Path)
    c.add_argument("finding_id")
    c.add_argument("--images", nargs="*", type=Path, default=[], help="up to 3 keyframe overlays")
    c.add_argument("--template", action="store_true", help="no model call: the template card")
    c.set_defaults(func=_card)


def _baseline(args: argparse.Namespace) -> int:
    queries = [
        {"query_id": q["query_id"], "text": q["text"]}
        for q in yaml.safe_load(args.queries.read_text())["queries"]
    ]
    ids = args.scenes or [s["scene_id"] for s in load_index()["scenes"]]
    try:
        run_baseline([open_scene(s) for s in ids], queries, spark_open_client(), args.out)
    except GATED as e:
        print(e, file=sys.stderr)
        return 1
    print(f"wrote {args.out / 'claims.jsonl'} and {args.out / 'rankings.json'}")
    return 0


def _card(args: argparse.Namespace) -> int:
    run = load_run(args.run_dir)
    finding = next((f for f in run.findings if f.finding_id == args.finding_id), None)
    if finding is None:
        print(f"no finding {args.finding_id} in {args.run_dir}", file=sys.stderr)
        return 1
    try:
        card = make_card(
            finding, run.spec, None if args.template else spark_open_client(), images=args.images
        )
    except GATED as e:
        print(e, file=sys.stderr)
        return 1
    print(json.dumps(card.to_json(), indent=2))
    return 0
