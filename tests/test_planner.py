"""M2 planner: parsing, validation, retry and clarification, with a scripted fake model.

The 25 M7 gold specs drive the plumbing test (the fake model returns each gold spec, so this checks the
planner's parsing and normalisation, not the model's accuracy). The live golden test runs the real model on
the 10 development queries only; held-out queries are for the final evaluation and are never sent here.
"""

import json
import os
from pathlib import Path

import pytest
import yaml

from longtail.agent.planner import PlannerError, plan, same_spec, semantic_errors
from longtail.contracts import ScenarioSpec
from longtail.llm.cache import ResponseCache
from longtail.llm.client import ModelClient
from longtail.llm.ledger import Ledger

QUERIES = Path(__file__).resolve().parents[1] / "longtail" / "eval" / "queries" / "queries.yaml"


def gold_queries(split: str | None = None) -> list[dict]:
    qs = yaml.safe_load(QUERIES.read_text())["queries"]
    return [q for q in qs if split is None or q["split"] == split]


def call(name: str, args: dict, call_id: str = "c1") -> dict:
    return {"type": "function_call", "name": name, "arguments": json.dumps(args), "call_id": call_id}


class ScriptedTransport:
    """Returns the scripted outputs in order and records each request."""

    def __init__(self, outputs: list[list[dict]]) -> None:
        self.outputs = list(outputs)
        self.requests: list[dict] = []

    def create(self, request: dict) -> dict:
        self.requests.append(request)
        return {"output": self.outputs.pop(0), "usage": {"input_tokens": 2000, "output_tokens": 300}}

    def upload(self, path):  # pragma: no cover - not used by the planner
        raise AssertionError

    def stream(self, request):  # pragma: no cover
        raise AssertionError


def client_for(tmp_path: Path, transport) -> ModelClient:
    return ModelClient(
        transport,
        cache=ResponseCache(tmp_path / "cache"),
        ledger=Ledger(tmp_path / "ledger.jsonl"),
        allow_paid=True,
        replay=False,
        sleep=lambda s: None,
    )


def model_args(gold: dict) -> dict:
    """What a model would send: the gold spec with the class name as sam_prompt and no clarification."""
    args = {k: v for k, v in gold.items() if k != "clarification"}
    args["actors"] = [{**a, "sam_prompt": a["cls"]} for a in gold["actors"]]
    return args


@pytest.mark.parametrize("q", gold_queries(), ids=lambda q: q["query_id"])
def test_planner_parses_every_gold_spec(tmp_path, q):
    t = ScriptedTransport([[call("emit_scenario_spec", model_args(q["gold"]))]])
    result = plan(q["text"], client_for(tmp_path, t))
    gold = ScenarioSpec.model_validate(q["gold"])
    assert same_spec(result.spec, gold)
    assert result.spec.query == q["text"] and result.attempts == 1
    assert [a.sam_prompt for a in result.spec.actors] == [a.sam_prompt for a in gold.actors]


def test_gold_specs_pass_the_semantic_checks():
    for q in gold_queries():
        assert semantic_errors(ScenarioSpec.model_validate(q["gold"])) == [], q["query_id"]


def test_invalid_spec_goes_back_with_the_error_and_is_retried(tmp_path):
    q = gold_queries()[0]
    bad = model_args(q["gold"])
    bad["predicates"] = [*bad["predicates"], {"name": "parked", "actors": ["a1"], "args": {}}]
    t = ScriptedTransport(
        [[call("emit_scenario_spec", bad, "c1")], [call("emit_scenario_spec", model_args(q["gold"]))]]
    )
    result = plan(q["text"], client_for(tmp_path, t))
    assert result.attempts == 2 and len(result.errors) == 1
    feedback = t.requests[1]["input"][-1]
    assert feedback["type"] == "function_call_output" and feedback["call_id"] == "c1"
    assert "predicates" in feedback["output"]


def test_missing_exists_is_caught_and_retried(tmp_path):
    q = next(q for q in gold_queries() if q["query_id"] == "ped_near_truck")
    bad = model_args(q["gold"])
    bad["predicates"] = [
        p for p in bad["predicates"] if not (p["name"] == "exists" and p["actors"] == ["t1"])
    ]
    t = ScriptedTransport(
        [[call("emit_scenario_spec", bad)], [call("emit_scenario_spec", model_args(q["gold"]))]]
    )
    result = plan(q["text"], client_for(tmp_path, t))
    assert result.attempts == 2 and "no exists predicate" in result.errors[0]


def test_integer_metres_are_accepted(tmp_path):
    q = next(q for q in gold_queries() if q["query_id"] == "cyclist_within_15")
    args = model_args(q["gold"])
    args["predicates"][1]["args"]["m"] = 15
    t = ScriptedTransport([[call("emit_scenario_spec", args)]])
    assert plan(q["text"], client_for(tmp_path, t)).spec.predicates[1].args["m"] == 15.0


def test_clarification_returns_a_spec_that_asks(tmp_path):
    t = ScriptedTransport(
        [[call("ask_clarification", {"question": "Which traffic light state do you mean?"})]]
    )
    result = plan("a car running a red light", client_for(tmp_path, t))
    assert result.spec.clarification == "Which traffic light state do you mean?"
    assert result.spec.actors == [] and result.spec.predicates == []


def test_gives_up_after_three_bad_attempts(tmp_path):
    t = ScriptedTransport([[{"type": "message", "content": []}]] * 3)
    with pytest.raises(PlannerError):
        plan("a bus", client_for(tmp_path, t))
    assert len(t.requests) == 3


def test_same_spec_ignores_actor_ids_and_order():
    gold = ScenarioSpec.model_validate(
        next(q for q in gold_queries() if q["query_id"] == "two_peds_within_2")["gold"]
    )
    renamed = gold.model_copy(
        update={
            "actors": [a.model_copy(update={"id": {"a1": "p2", "a2": "p1"}[a.id]}) for a in gold.actors],
            "predicates": [
                p.model_copy(update={"actors": [{"a1": "p2", "a2": "p1"}[x] for x in p.actors]})
                for p in reversed(gold.predicates)
            ],
        }
    )
    assert same_spec(renamed, gold)
    fewer = gold.model_copy(update={"predicates": gold.predicates[:-1]})
    assert not same_spec(fewer, gold)


def test_the_request_forces_a_tool_call(tmp_path):
    q = gold_queries()[0]
    t = ScriptedTransport([[call("emit_scenario_spec", model_args(q["gold"]))]])
    plan(q["text"], client_for(tmp_path, t))
    req = t.requests[0]
    assert req["tool_choice"] == "auto"
    assert {tool["name"] for tool in req["tools"]} == {"emit_scenario_spec", "ask_clarification"}


LIVE = os.environ.get("LONGTAIL_ALLOW_PAID") == "1" and bool(os.environ.get("MODEL_API_KEY"))


@pytest.mark.skipif(not LIVE, reason="live golden test: needs LONGTAIL_ALLOW_PAID=1 and MODEL_API_KEY (paid)")
def test_live_planner_on_development_queries():
    """Real model, dev split only. Reports agreement; held-out queries are never sent to the planner here."""
    client = ModelClient.from_env()
    dev = gold_queries("dev")
    hits = [same_spec(plan(q["text"], client).spec, ScenarioSpec.model_validate(q["gold"])) for q in dev]
    print(f"planner agreement on dev: {sum(hits)}/{len(dev)}")
    assert sum(hits) >= len(dev) // 2
