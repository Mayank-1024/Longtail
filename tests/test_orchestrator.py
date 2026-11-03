"""M3: the linear pipeline and the search agent, end to end on the synthetic scene with a scripted model.

The model is scripted; the verifier is real. A test passes only if the real verifier produced the finding.
"""

import json
from pathlib import Path

import pytest
from synthetic import SyntheticConfig, SyntheticScene, track_source

from longtail.agent.orchestrator import SearchAgent, open_with
from longtail.contracts import ScenarioSpec
from longtail.llm.cache import ResponseCache
from longtail.llm.client import Budget, ModelClient
from longtail.llm.ledger import Ledger
from longtail.pipeline import run, uniform_windows

QUERY = "a pedestrian crossing in front of us while we move"
SPEC = {
    "query": QUERY,
    "actors": [{"id": "a1", "cls": "pedestrian", "sam_prompt": "person"}],
    "predicates": [
        {"name": "exists", "actors": ["a1"], "args": {}},
        {"name": "crossing_ego_path", "actors": ["a1"], "args": {}},
        {"name": "ego_moving", "actors": [], "args": {}},
    ],
    "assumptions": [],
}


def fc(name: str, args: dict, call_id: str) -> dict:
    return {"type": "function_call", "name": name, "arguments": json.dumps(args), "call_id": call_id}


def last_tool_output(request: dict) -> dict:
    outs = [i for i in request["input"] if i.get("type") == "function_call_output"]
    return json.loads(outs[-1]["output"]) if outs else {}


class Scripted:
    """Each step is a function of the request, so a step can read earlier tool results."""

    def __init__(self, steps, usage=(2000, 300)) -> None:
        self.steps, self.usage, self.requests = list(steps), usage, []

    def create(self, request: dict) -> dict:
        self.requests.append(request)
        step = self.steps.pop(0) if self.steps else (lambda r: [fc("list_scenes", {}, "again")])
        u = self.usage(len(self.requests)) if callable(self.usage) else self.usage
        return {"output": step(request), "usage": {"input_tokens": u[0], "output_tokens": u[1]}}

    def upload(self, path):  # pragma: no cover
        raise AssertionError

    def stream(self, request):  # pragma: no cover
        raise AssertionError


@pytest.fixture(scope="module")
def scenes():
    crossing = SyntheticScene(scene_id="crossing")
    empty = SyntheticScene(scene_id="empty")
    sources = {"crossing": track_source(crossing, {"person": ["crosser"]}), "empty": track_source(empty, {})}

    def source(scene, camera, concept):
        return sources[scene.scene_id](scene, camera, concept)

    return {"crossing": crossing, "empty": empty}, source


def agent_for(tmp_path: Path, transport, scenes, **kw) -> SearchAgent:
    by_id, source = scenes
    client = ModelClient(
        transport,
        cache=ResponseCache(tmp_path / "cache"),
        ledger=Ledger(tmp_path / "ledger.jsonl"),
        allow_paid=True,
        replay=False,
        sleep=lambda s: None,
    )
    return SearchAgent(client, list(by_id), open_with(by_id), source, tmp_path / "runs", **kw)


def planner_step(request):
    return [fc("emit_scenario_spec", SPEC, "p1")]


def test_uniform_windows_cover_an_8s_scene_and_a_short_one():
    long = SyntheticScene(SyntheticConfig(n_frames=80), scene_id="long")
    ws = uniform_windows(long)
    ts = long.lidar_timestamps()
    assert [round(w.t0 - ts[0], 2) for w in ws] == [0.0, 2.0, 3.9]
    assert ws[-1].t1 == pytest.approx(ts[-1]) and all(w.source == "uniform" for w in ws)
    short = uniform_windows(SyntheticScene(scene_id="short"))
    assert len(short) == 1


def test_linear_pipeline_verifies_the_crosser_and_isolates_a_bad_scene(tmp_path, scenes):
    by_id, source = scenes

    def opener(scene_id):
        if scene_id == "broken":
            raise FileNotFoundError("no such scene")
        return by_id[scene_id]

    result = run(
        ScenarioSpec.model_validate({**SPEC, "clarification": None}),
        ["crossing", "broken", "empty"],
        opener,
        source,
        tmp_path,
    )
    by_scene = {s.scene_id: s for s in result.scenes}
    assert by_scene["broken"].error.startswith("FileNotFoundError")
    assert any(f.verified for f in by_scene["crossing"].findings)
    assert not any(f.verified for f in by_scene["empty"].findings)
    assert result.verified() and all(f.scene_id == "crossing" for f in result.verified())


def test_agent_plans_searches_and_finishes_with_verified_findings_only(tmp_path, scenes):
    def finish(request):
        out = last_tool_output(request)
        tops = [f for s in out["scenes"] for f in s.get("top", [])]
        ids = [f["finding_id"] for f in tops]  # includes unverified ones on purpose
        return [fc("finish", {"finding_ids": ids, "summary": "one crosser"}, "f1")]

    t = Scripted(
        [
            planner_step,
            lambda r: [fc("list_scenes", {}, "l1")],
            lambda r: [fc("search_scenes", {"scene_ids": ["crossing", "empty", "nope"]}, "s1")],
            finish,
        ]
    )
    result = agent_for(tmp_path, t, scenes).search(QUERY, Budget(5.0))
    assert result.stop_reason == "finish"
    assert result.findings and all(f.verified for f in result.findings)
    assert result.rejected and not any(f.verified for f in result.rejected)
    assert sorted(result.scenes_searched) == ["crossing", "empty"]
    kinds = [(e.kind, e.name) for e in result.trace]
    assert kinds[0] == ("plan", "planner") and ("tool", "search_scenes") in kinds
    assert ("stop", "finish_dropped") in kinds  # the unverified ids it named were dropped
    search_out = next(e for e in result.trace if e.name == "search_scenes").detail["result"]
    assert search_out["unknown_scene_ids"] == ["nope"]
    saved = list((tmp_path / "runs").glob("search-*/search.json"))
    assert saved and json.loads(saved[0].read_text())["stop_reason"] == "finish"
    assert result.usd == pytest.approx(sum(e["usd"] for e in json.loads(saved[0].read_text())["trace"]))


def test_max_steps_finishes_with_every_verified_finding(tmp_path, scenes):
    t = Scripted([planner_step, lambda r: [fc("search_scenes", {"scene_ids": ["crossing"]}, "s1")]])
    result = agent_for(tmp_path, t, scenes, max_steps=3).search(QUERY, Budget(5.0))
    assert result.stop_reason == "max_steps"
    assert result.findings and all(f.verified for f in result.findings)
    assert len(t.requests) == 1 + 3


def test_budget_stops_the_loop_and_still_answers(tmp_path, scenes):
    t = Scripted([planner_step], usage=lambda n: (249_000, 300) if n == 1 else (2000, 300))
    result = agent_for(tmp_path, t, scenes).search(QUERY, Budget(1.0))
    assert result.stop_reason == "budget" and len(t.requests) == 1
    assert result.findings == [] and result.scenes_searched == []


def test_clarification_stops_before_any_search(tmp_path, scenes):
    t = Scripted([lambda r: [fc("ask_clarification", {"question": "Which light state?"}, "q1")]])
    result = agent_for(tmp_path, t, scenes).search("a car running a red light", Budget(5.0))
    assert result.stop_reason == "clarification" and result.spec.clarification == "Which light state?"
    assert result.scenes_searched == [] and len(t.requests) == 1
