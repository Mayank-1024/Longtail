"""Offline demo mode for building and testing the web view without keys, data or spend.

The scenes are Chethan's synthetic scenes and the verifier is real, but the MODEL IS SCRIPTED: the planner
answers one fixed query and the orchestrator always lists, searches everything, then finishes. Every response
in this mode carries mode "offline-demo", and the web view shows a banner. Never use it for footage, numbers
or claims about the system; that is what the live and replay modes are for.
"""

import json
import sys
from pathlib import Path
from typing import Any

from longtail.data.base import Scene
from longtail.llm.cache import ResponseCache
from longtail.llm.client import ModelClient
from longtail.llm.ledger import Ledger

# ASSUMPTION A-224: offline demo = synthetic scenes + a scripted model, bannered; never for footage or numbers
DEMO_QUERY = "a pedestrian crossing in front of us while we move"
DEMO_SPEC = {
    "query": DEMO_QUERY,
    "actors": [{"id": "a1", "cls": "pedestrian", "sam_prompt": "pedestrian"}],
    "predicates": [
        {"name": "exists", "actors": ["a1"], "args": {}},
        {"name": "crossing_ego_path", "actors": ["a1"], "args": {}},
        {"name": "ego_moving", "actors": [], "args": {}},
    ],
    "assumptions": [],
}


def _synthetic():
    tests = Path(__file__).resolve().parents[1] / "tests"
    if str(tests) not in sys.path:
        sys.path.insert(0, str(tests))
    import synthetic  # test helpers, on the path only in this mode

    return synthetic


def demo_scenes() -> tuple[dict[str, Scene], Any]:
    """Two synthetic scenes (one crosser, one empty) and their scripted track source."""
    syn = _synthetic()
    crossing = syn.SyntheticScene(scene_id="synthetic-crossing")
    empty = syn.SyntheticScene(scene_id="synthetic-empty")
    sources = {
        crossing.scene_id: syn.track_source(crossing, {"person": ["crosser"]}),
        empty.scene_id: syn.track_source(empty, {}),
    }

    def source(scene, camera, concept):
        return sources[scene.scene_id](scene, camera, concept)

    return {crossing.scene_id: crossing, empty.scene_id: empty}, source


def _call(name: str, args: dict, call_id: str) -> dict:
    return {"type": "function_call", "name": name, "arguments": json.dumps(args), "call_id": call_id}


class ScriptedModel:
    """Stands in for Wonder: plans the one demo query, then list, search all, finish."""

    def __init__(self, scene_ids: list[str]) -> None:
        self.scene_ids = scene_ids

    def create(self, request: dict) -> dict:
        tools = {t["name"] for t in request.get("tools", [])}
        user = next((m["content"] for m in request["input"] if m.get("role") == "user"), "")
        if "emit_scenario_spec" in tools:
            if user.strip().lower() != DEMO_QUERY:
                q = f"The offline demo has a scripted model that only knows the query: '{DEMO_QUERY}'."
                out = [_call("ask_clarification", {"question": q}, "p1")]
            else:
                out = [_call("emit_scenario_spec", DEMO_SPEC, "p1")]
        else:
            done = [i for i in request["input"] if i.get("type") == "function_call_output"]
            if not done:
                out = [_call("list_scenes", {}, "t1")]
            elif len(done) == 1:
                out = [_call("search_scenes", {"scene_ids": self.scene_ids}, "t2")]
            else:
                result = json.loads(done[-1]["output"])
                ids = [
                    f["finding_id"]
                    for s in result.get("scenes", [])
                    for f in s.get("top", [])
                    if f["verified"]
                ]
                out = [
                    _call("finish", {"finding_ids": ids, "summary": "Offline demo (scripted model)."}, "t3")
                ]
        return {"output": out, "usage": {"input_tokens": 0, "output_tokens": 0}}

    def upload(self, path):  # pragma: no cover
        raise NotImplementedError

    def stream(self, request):  # pragma: no cover
        raise NotImplementedError


def demo_client(run_root: Path, scene_ids: list[str]) -> ModelClient:
    # Its own cache and ledger, so demo calls never mix with real ones; it costs $0 and sends nothing.
    return ModelClient(
        ScriptedModel(scene_ids),
        cache=ResponseCache(run_root / "offline-demo-cache"),
        ledger=Ledger(run_root / "offline-demo-ledger.jsonl"),
        allow_paid=True,
        replay=False,
    )
