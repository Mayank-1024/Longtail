"""`longtail gt build|spotcheck|motion-report`: ground truth from labels (plan C7)."""

import argparse
import json
import random
from pathlib import Path

from longtail.data.base import Scene
from longtail.data.guard import eval_ground_truth
from longtail.eval import gt


def register(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    # ASSUMPTION A-131: `longtail gt` is registered in the shared CLI dispatcher (A-34)
    p = sub.add_parser("gt", help="ground truth from the labels (evaluation only)")
    gsub = p.add_subparsers(dest="gt_command", required=True)

    b = gsub.add_parser("build", help="ground truth for every query in a queries file")
    b.add_argument("queries", type=Path, help="queries file (YAML or JSON, format longtail.queries/v1)")
    b.add_argument("--scenes", nargs="*", help="scene ids (default: every scene in the index)")
    b.add_argument("--out", type=Path, default=Path("gt"), help="output directory, named gt (default: gt)")
    b.set_defaults(func=_build)

    s = gsub.add_parser("spotcheck", help="random positives of one query, to check by eye")
    s.add_argument("query", help="query id")
    s.add_argument("--n", type=int, default=5)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--gt-dir", type=Path, default=Path("gt"))
    s.set_defaults(func=_spotcheck)

    m = gsub.add_parser("motion-report", help="geometric motion against the labelled motion attributes")
    m.add_argument("--scenes", nargs="*", help="scene ids (default: every scene in the index)")
    m.add_argument("--out", type=Path, default=Path("gt"))
    m.set_defaults(func=_motion_report)


def load_scene(scene_id: str) -> Scene:
    from longtail.data.index import open_scene

    return open_scene(scene_id)


def index_scene_ids() -> list[str]:
    from longtail.data.index import load_index

    return [s["scene_id"] for s in load_index()["scenes"]]


def _scenes(ids: list[str] | None) -> list[Scene]:
    return [load_scene(s) for s in (ids or index_scene_ids())]


def _build(args: argparse.Namespace) -> int:
    queries = gt.load_queries(args.queries)
    scenes = _scenes(args.scenes)
    with eval_ground_truth():
        summary = gt.build(queries, scenes, args.out)
    print(f"{'query':<16}{'split':<9}{'positives':>10}{'scenes':>8}")
    for r in summary["queries"]:
        flag = "  ZERO: replace before the freeze (tell Mayank)" if r["zero_positives"] else ""
        print(f"{r['query_id']:<16}{r['split']:<9}{r['positives']:>10}{r['scenes_with_positives']:>8}{flag}")
    print(f"{len(scenes)} scenes; motion source {summary['motion_source']}; written to {args.out}/")
    return 0


def _spotcheck(args: argparse.Namespace) -> int:
    # Reads gt/<query>.json only (already label-derived), so no label access is needed here.
    doc = gt.load_gt(args.gt_dir, args.query)
    picks = random.Random(args.seed).sample(doc.positives, min(args.n, len(doc.positives)))
    print(f"{doc.query_id}: {doc.query} ({len(doc.positives)} positives, showing {len(picks)})")
    for p in sorted(picks, key=lambda p: (p.scene_id, p.t0)):
        labels = ", ".join(f"{t.label_id} ({t.label})" for t in p.labels)
        print(
            f"scene {p.scene_id}  t {p.t0 - p.scene_start_ts:6.2f}-{p.t1 - p.scene_start_ts:6.2f} s "
            f"(ts {p.t0:.3f}-{p.t1:.3f})  labels: {labels}"
        )
    return 0 if picks else 1


def _motion_report(args: argparse.Namespace) -> int:
    gt.check_out_dir(args.out)
    report = {}
    with eval_ground_truth():
        for scene in _scenes(args.scenes):
            report[scene.scene_id] = gt.motion_agreement(gt.LabelScene(scene))
    totals: dict[str, dict[str, int]] = {}
    for per_class in report.values():
        for cls, counts in per_class.items():
            t = totals.setdefault(cls, {})
            for k, v in counts.items():
                t[k] = t.get(k, 0) + v
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "motion_agreement.json").write_text(json.dumps({"scenes": report, "total": totals}, indent=1))
    for cls, t in sorted(totals.items()):
        judged = t.get("agree", 0) + t.get("disagree", 0)
        rate = f"{t.get('agree', 0) / judged:.1%}" if judged else "-"
        print(f"{cls:<12} agreement {rate:>7} over {judged} judged frames; {t}")
    return 0
