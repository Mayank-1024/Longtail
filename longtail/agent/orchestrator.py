"""The search agent (plan M3, second half): plan the query, then a tool loop on Wonder that decides which
scenes to search and when to stop, with a budget, a stop rule and a trace of every step.

Tools the model can call:
  list_scenes()                 the scene ids and their non-label metadata
  search_scenes(scene_ids)      runs the linear pipeline (pipeline.run) on those scenes
  finish(finding_ids, summary)  ends the search; only verified findings survive (A-17)

Stop rule: the model calls finish, or max_steps model calls pass, or the query budget runs out. The last two
finish automatically with every verified finding found so far, so a search always ends with an answer.
"""

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from longtail.agent.planner import DEFAULT_MODEL, plan
from longtail.contracts import Finding, ScenarioSpec
from longtail.data.base import Scene
from longtail.llm.client import Budget, BudgetExceeded, ModelClient
from longtail.pipeline import PipelineResult, SceneOpener, SceneResult, run, scene_run_dir, spec_run_id
from longtail.verify.engine import TrackSource

# ASSUMPTION A-215: the orchestrator runs on the planner's model (Wonder) with at most 6 model calls per query
MAX_STEPS = 6
# ASSUMPTION A-216: a search_scenes result shows the model at most 5 findings per scene, verified first
SHOWN_PER_SCENE = 5

SYSTEM = """You run a search over driving logs for one scenario. A verifier checks every candidate on
LiDAR and returns PASS, FAIL or UNVERIFIABLE per predicate; you cannot override it.

Work in few steps: list the scenes, search the ones worth searching (all of them if unsure), then finish with
the verified findings worth showing, best first. Never present an unverified finding as a result; finish drops
them anyway. Say in the summary what was searched and what failed verification. Every reply must be a tool
call, never plain text."""


def _tools() -> list[dict[str, Any]]:
    strings = {"type": "array", "items": {"type": "string"}}
    return [
        {
            "type": "function",
            "name": "list_scenes",
            "description": "List the scenes available to search, with their metadata.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "type": "function",
            "name": "search_scenes",
            "description": "Search these scenes; returns counts and the top findings per scene.",
            "parameters": {
                "type": "object",
                "properties": {"scene_ids": strings},
                "required": ["scene_ids"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "finish",
            "description": "End the search: the findings to show, best first, and a short summary.",
            "parameters": {
                "type": "object",
                "properties": {"finding_ids": strings, "summary": {"type": "string"}},
                "required": ["finding_ids", "summary"],
                "additionalProperties": False,
            },
        },
    ]


@dataclass
class TraceEvent:
    step: int
    kind: str  # plan | model | tool | stop
    name: str
    detail: dict[str, Any]
    usd: float = 0.0
    t: float = field(default_factory=time.time)


@dataclass
class SearchResult:
    query: str
    spec: ScenarioSpec
    findings: list[Finding]  # verified, in the order finish gave (or by score after an automatic finish)
    rejected: list[Finding]  # every unverified finding the verifier produced ("rejected by verifier" tab)
    summary: str
    stop_reason: str  # finish | max_steps | budget | clarification
    usd: float
    trace: list[TraceEvent]
    scenes_searched: list[str]
    search_id: str = ""
    run_dirs: dict[str, str] = field(default_factory=dict)  # scene id to its verify run, relative to run_root

    def to_json(self) -> dict[str, Any]:
        return {
            "search_id": self.search_id,
            "query": self.query,
            "spec": self.spec.model_dump(mode="json"),
            "findings": [f.model_dump(mode="json") for f in self.findings],
            "rejected": [f.model_dump(mode="json") for f in self.rejected],
            "summary": self.summary,
            "stop_reason": self.stop_reason,
            "usd": round(self.usd, 6),
            "scenes_searched": self.scenes_searched,
            "run_dirs": self.run_dirs,
            "trace": [e.__dict__ for e in self.trace],
        }


def _brief(f: Finding) -> dict[str, Any]:
    failing = [f"{r.name}:{r.status}:{r.evidence.get('reason')}" for r in f.results if r.status != "PASS"]
    return {
        "finding_id": f.finding_id,
        "scene_id": f.scene_id,
        "t0": round(f.t0, 2),
        "t1": round(f.t1, 2),
        "verified": f.verified,
        "score": round(f.score, 3),
        "not_passing": failing,
    }


def _scene_summary(s: SceneResult) -> dict[str, Any]:
    if s.error is not None:
        return {"scene_id": s.scene_id, "error": s.error}
    fs = s.findings
    return {
        "scene_id": s.scene_id,
        "windows": len(s.run.windows) if s.run else 0,
        "verified": sum(f.verified for f in fs),
        "unverified": sum(not f.verified for f in fs),
        "top": [_brief(f) for f in fs[:SHOWN_PER_SCENE]],
    }


class SearchAgent:
    def __init__(
        self,
        client: ModelClient,
        scene_ids: list[str],
        open_scene: SceneOpener,
        track_source: TrackSource,
        run_root: Path = Path("runs"),
        *,
        model: str | None = None,
        max_steps: int = MAX_STEPS,
        scene_meta: dict[str, dict[str, Any]] | None = None,
        on_event: Callable[[TraceEvent], None] | None = None,
    ) -> None:
        self.client, self.scene_ids, self.open_scene = client, list(scene_ids), open_scene
        self.track_source, self.run_root, self.model = track_source, Path(run_root), model
        self.max_steps, self.scene_meta = max_steps, scene_meta or {}
        self.on_event = on_event

    def _log(self, trace: list[TraceEvent], event: TraceEvent) -> None:
        trace.append(event)
        if self.on_event is not None:
            self.on_event(event)

    def search(self, query: str, budget: Budget | None = None) -> SearchResult:
        budget = budget or Budget.from_env()
        trace: list[TraceEvent] = []
        planned = plan(query, self.client, model=self.model, budget=budget)
        spec = planned.spec
        self._log(
            trace,
            TraceEvent(
                0,
                "plan",
                "planner",
                {"spec": spec.model_dump(mode="json"), "attempts": planned.attempts},
                planned.usd,
            ),
        )
        if spec.clarification is not None:
            return self._done(query, spec, {}, [], "", "clarification", budget, trace, [])
        model = self.model or os.environ.get("LONGTAIL_PLANNER_MODEL", DEFAULT_MODEL)
        history: list[dict[str, Any]] = [
            {"type": "message", "role": "system", "content": SYSTEM},
            {"type": "message", "role": "user", "content": _task(query, spec)},
        ]
        found: dict[str, Finding] = {}
        searched: list[str] = []
        for step in range(1, self.max_steps + 1):
            request = {
                "model": model,
                "input": history,
                "tools": _tools(),
                "tool_choice": "auto",  # A-212
                "max_output_tokens": 1500,
            }
            try:
                resp = self.client.respond(request, purpose="orchestrator", budget=budget)
            except BudgetExceeded as e:
                self._log(trace, TraceEvent(step, "stop", "budget", {"error": str(e)}))
                return self._auto_finish(query, spec, found, searched, "budget", budget, trace)
            calls = resp.function_calls()
            self._log(
                trace,
                TraceEvent(
                    step, "model", "orchestrator", {"tool_calls": [c.get("name") for c in calls]}, resp.usd
                ),
            )
            history += resp.output
            if not calls:
                history.append({"type": "message", "role": "user", "content": "Call one of the tools."})
                continue
            for call in calls:
                name = call.get("name")
                try:
                    args = json.loads(call.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = None
                if name == "finish" and isinstance(args, dict):
                    ids = [i for i in args.get("finding_ids", []) if isinstance(i, str)]
                    self._log(trace, TraceEvent(step, "tool", "finish", {"finding_ids": ids}))
                    return self._done(
                        query,
                        spec,
                        found,
                        ids,
                        str(args.get("summary", "")),
                        "finish",
                        budget,
                        trace,
                        searched,
                    )
                output = self._run_tool(name, args, spec, found, searched, step, trace)
                history.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.get("call_id", ""),
                        "output": json.dumps(output),
                    }
                )
        self._log(trace, TraceEvent(self.max_steps, "stop", "max_steps", {}))
        return self._auto_finish(query, spec, found, searched, "max_steps", budget, trace)

    def _run_tool(self, name, args, spec, found, searched, step, trace) -> dict[str, Any]:
        if args is None:
            out = {"error": "arguments were not valid JSON"}
        elif name == "list_scenes":
            out = {"scenes": [{"scene_id": s, **self.scene_meta.get(s, {})} for s in self.scene_ids]}
        elif name == "search_scenes":
            asked = [s for s in args.get("scene_ids", []) if isinstance(s, str)]
            unknown = [s for s in asked if s not in self.scene_ids]
            todo = [s for s in asked if s in self.scene_ids and s not in searched]
            # a progress event (the scenes about to be verified) for the web view's wait estimate
            self._log(trace, TraceEvent(step, "tool", "searching", {"scene_ids": todo}))
            result: PipelineResult = run(spec, todo, self.open_scene, self.track_source, self.run_root)
            searched += todo
            for f in result.findings():
                found[f.finding_id] = f
            out = {
                "scenes": [_scene_summary(s) for s in result.scenes],
                "already_searched": [s for s in asked if s in searched and s not in todo],
                "unknown_scene_ids": unknown,
                "verified_so_far": sum(f.verified for f in found.values()),
            }
        else:
            out = {"error": f"unknown tool {name!r}"}
        self._log(trace, TraceEvent(step, "tool", str(name), {"args": args, "result": out}))
        return out

    def _auto_finish(self, query, spec, found, searched, reason, budget, trace) -> SearchResult:
        ids = [f.finding_id for f in sorted(found.values(), key=lambda f: -f.score) if f.verified]
        summary = (
            f"Stopped ({reason}) after searching {len(searched)} scene(s); every verified finding is shown."
        )
        return self._done(query, spec, found, ids, summary, reason, budget, trace, searched)

    def _done(self, query, spec, found, ids, summary, reason, budget, trace, searched) -> SearchResult:
        chosen = [found[i] for i in dict.fromkeys(ids) if i in found and found[i].verified]
        dropped = [i for i in ids if i not in found or not found[i].verified]
        if dropped:
            self._log(trace, TraceEvent(len(trace), "stop", "finish_dropped", {"finding_ids": dropped}))
        rejected = sorted((f for f in found.values() if not f.verified), key=lambda f: -f.score)
        search_id = f"search-{spec_run_id(spec)}"
        run_dirs = {sid: scene_run_dir(spec, sid, Path(".")).as_posix() for sid in searched}
        result = SearchResult(
            query,
            spec,
            chosen,
            rejected,
            summary,
            reason,
            budget.spent_usd,
            trace,
            list(searched),
            search_id,
            run_dirs,
        )
        out = self.run_root / search_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "search.json").write_text(json.dumps(result.to_json(), indent=1, default=str))
        return result


def _task(query: str, spec: ScenarioSpec) -> str:
    return (
        f"Query: {query}\nScenario spec (already validated):\n{spec.model_dump_json(indent=1)}\n"
        "Find where this happens."
    )


def open_with(scenes: dict[str, Scene]) -> SceneOpener:
    """A SceneOpener over scenes already in memory (tests, the synthetic demo)."""
    return lambda scene_id: scenes[scene_id]
