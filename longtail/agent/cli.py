"""`longtail plan "<query>"`: print the ScenarioSpec the planner makes for a query."""

import argparse
import json
import sys
from pathlib import Path

from longtail.agent.planner import PlannerError, plan
from longtail.llm.client import BudgetExceeded, ModelClient, NoKeyError, PaidCallsDisabled, ReplayMissError
from longtail.llm.report import markdown, summarise


def register(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    p = sub.add_parser("plan", help="turn a query into a scenario spec (paid call unless cached)")
    p.add_argument("query")
    p.set_defaults(func=_plan)
    s = sub.add_parser("search", help="run the search agent on real scenes and print the verified findings")
    s.add_argument("query")
    s.add_argument("--scenes", nargs="*", help="scene ids (default: the scene index)")
    s.add_argument("--runs", type=Path, default=Path("runs"))
    s.set_defaults(func=_search)
    c = sub.add_parser("costs", help="spend by model and purpose, from runs/ledger.jsonl")
    c.add_argument("--ledger", type=Path, default=None)
    c.add_argument("--markdown", action="store_true", help="print a table for docs/costs.md")
    c.set_defaults(func=_costs)


def _plan(args: argparse.Namespace) -> int:
    try:
        result = plan(args.query, ModelClient.from_env())
    except (PaidCallsDisabled, NoKeyError, ReplayMissError, PlannerError) as e:
        print(e, file=sys.stderr)
        return 1
    print(result.spec.model_dump_json(indent=2))
    print(f"{result.attempts} attempt(s), ${result.usd:.4f}", file=sys.stderr)
    return 0


def _search(args: argparse.Namespace) -> int:
    from longtail.agent.orchestrator import SearchAgent
    from longtail.data.index import load_index, open_scene

    try:
        from longtail.vision.sam_client import SamClient, transport_from_env
        from longtail.vision.track_source import sam_track_source
    except ImportError:
        print("the SAM track source (C3, PR #9) is not merged yet", file=sys.stderr)
        return 1
    ids = args.scenes or [s["scene_id"] for s in load_index()["scenes"]]

    def show(e) -> None:
        print(f"[{e.step}] {e.kind:<5} {e.name:<14} ${e.usd:.4f}", file=sys.stderr)

    agent = SearchAgent(
        ModelClient.from_env(),
        ids,
        open_scene,
        sam_track_source(SamClient(transport_from_env())),
        args.runs,
        on_event=show,
    )
    try:
        result = agent.search(args.query)
    except (PaidCallsDisabled, NoKeyError, ReplayMissError, BudgetExceeded, PlannerError) as e:
        print(e, file=sys.stderr)
        return 1
    if result.spec.clarification:
        print(f"clarification needed: {result.spec.clarification}")
        return 0
    for f in result.findings:
        print(f"VERIFIED {f.scene_id} {f.t0:.2f}-{f.t1:.2f} score {f.score:.3f} ({f.finding_id})")
    print(
        f"{len(result.findings)} verified, {len(result.rejected)} rejected, stop: {result.stop_reason}, "
        f"${result.usd:.4f}; saved runs/{result.search_id}/search.json"
    )
    return 0


def _costs(args: argparse.Namespace) -> int:
    summary = summarise(args.ledger)
    print(markdown(summary) if args.markdown else json.dumps(summary, indent=1))
    return 0
