"""`longtail plan "<query>"`: print the ScenarioSpec the planner makes for a query."""

import argparse
import sys

from longtail.agent.planner import PlannerError, plan
from longtail.llm.client import ModelClient, NoKeyError, PaidCallsDisabled, ReplayMissError


def register(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    p = sub.add_parser("plan", help="turn a query into a scenario spec (paid call unless cached)")
    p.add_argument("query")
    p.set_defaults(func=_plan)


def _plan(args: argparse.Namespace) -> int:
    try:
        result = plan(args.query, ModelClient.from_env())
    except (PaidCallsDisabled, NoKeyError, ReplayMissError, PlannerError) as e:
        print(e, file=sys.stderr)
        return 1
    print(result.spec.model_dump_json(indent=2))
    print(f"{result.attempts} attempt(s), ${result.usd:.4f}", file=sys.stderr)
    return 0
