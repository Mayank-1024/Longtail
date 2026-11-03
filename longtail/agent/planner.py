"""Planner (plan M2): a plain-English query to a ScenarioSpec, through one forced function call on Wonder.

The model must call exactly one tool: emit_scenario_spec (the spec) or ask_clarification (one question when
the query cannot be pinned down). The spec is validated by the contract (pydantic) and by the semantic checks
below; on failure the error goes back to the model, up to max_attempts calls in all.
"""

import itertools
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from longtail.contracts import PREDICATE_SIGNATURES, SIDES, TAXONOMY, ScenarioSpec
from longtail.llm.client import Budget, ModelClient

# ASSUMPTION A-209: the planner runs on muse-wonder-1.0 (plan 3); LONGTAIL_PLANNER_MODEL can switch to Spark
DEFAULT_MODEL = "muse-wonder-1.0"
MAX_ATTEMPTS = 3

# ASSUMPTION A-210: SAM phrases come from this table, not from the model (C3's phrase test may update it)
SAM_PROMPTS = {"pedestrian": "person", "cyclist": "cyclist", "car": "car", "truck": "truck", "bus": "bus"}

# ASSUMPTION A-211: default meanings of vague words, the same ones the M7 gold specs use
DEFAULTS = """\
- "near" or "next to" another object: near(a, b, m=5.0), assumption "near = within 5 m".
- "close to us", "right by us": within_distance(a, m=10.0) unless the query gives a number.
- "walking", "riding", "driving" (for another object): moving. "standing still", "parked", "waiting": stopped.
- "while we move / drive": ego_moving. "while we wait / are stopped": ego_stopped.
- "partly hidden by X" or "behind X" (visually): occluded_by(a, cls=X).
- "in front of us" as motion across our lane: crossing_ego_path.
- "ahead of us" as a place: relative_position ahead.
- "truck" includes pickups. A "cyclist" is a person riding a bicycle."""

_SIGNATURES = "\n".join(
    f"- {n}: {k} actor(s), args {sorted(a) or 'none'}" for n, (k, a) in PREDICATE_SIGNATURES.items()
)

SYSTEM = f"""You turn a search query over driving logs into a scenario spec for a verifier that checks it
on LiDAR. Always answer by calling exactly one tool, never with plain text.

Classes (cls): {", ".join(TAXONOMY)}. Nothing else exists in this dataset: no traffic lights, lanes,
crosswalks, signs, weather or animals. If the query needs something outside the classes and predicates,
or is too vague to check, call ask_clarification with one short question instead of guessing.

Predicates (name: number of actors, args):
{_SIGNATURES}
- relative_position side is one of {", ".join(SIDES)}; m is metres (a number); occluded_by cls is a class.

Rules:
- One actor per object the query names, ids a1, a2, ... in order of mention.
- Every actor gets an exists predicate.
- Keep the spec minimal: only predicates the query states or clearly implies.
- Record every default you apply in assumptions, as a short phrase like "near = within 5 m".
- query is the user's query, verbatim. Leave sam_prompt as the class name; the system fills it in.

Defaults:
{DEFAULTS}"""

EMIT = "emit_scenario_spec"
ASK = "ask_clarification"


def _tools() -> list[dict[str, Any]]:
    schema = ScenarioSpec.model_json_schema()
    schema["properties"].pop("clarification", None)
    schema["required"] = [r for r in schema.get("required", []) if r != "clarification"]
    return [
        {
            "type": "function",
            "name": EMIT,
            "description": "The scenario spec for the query.",
            "parameters": schema,
        },
        {
            "type": "function",
            "name": ASK,
            "description": "Ask the user one question when the query cannot be checked as written.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    ]


class PlannerError(RuntimeError):
    """No valid spec after max_attempts calls."""


@dataclass
class PlanResult:
    spec: ScenarioSpec
    attempts: int
    usd: float
    errors: list[str] = field(default_factory=list)


def semantic_errors(spec: ScenarioSpec) -> list[str]:
    """Checks the contract cannot express on its own."""
    errors = []
    exists_for = {p.actors[0] for p in spec.predicates if p.name == "exists"}
    for a in spec.actors:
        if a.id not in exists_for:
            errors.append(f"actor {a.id} ({a.cls}) has no exists predicate")
        used = any(a.id in p.actors for p in spec.predicates if p.name != "exists")
        if not used and len(spec.actors) > 1:
            errors.append(f"actor {a.id} ({a.cls}) is declared but no predicate other than exists uses it")
    for p in spec.predicates:
        m = p.args.get("m")
        if isinstance(m, float) and m > 80:
            errors.append(f"{p.name}: m={m} is beyond what LiDAR can verify (at most 80 m)")
    if not spec.actors and not any(p.name.startswith("ego_") for p in spec.predicates):
        errors.append("the spec has no actors and no ego predicate, so it matches nothing")
    return errors


def _normalise(args: dict[str, Any], query: str) -> dict[str, Any]:
    args = {**args, "query": query, "clarification": None}
    args.setdefault("assumptions", [])
    for a in args.get("actors") or []:
        if isinstance(a, dict) and a.get("cls") in SAM_PROMPTS:
            a["sam_prompt"] = SAM_PROMPTS[a["cls"]]
    for p in args.get("predicates") or []:
        if isinstance(p, dict) and isinstance(p.get("args"), dict) and isinstance(p["args"].get("m"), int):
            p["args"]["m"] = float(p["args"]["m"])
    return args


def plan(
    query: str,
    client: ModelClient,
    *,
    model: str | None = None,
    budget: Budget | None = None,
    max_attempts: int = MAX_ATTEMPTS,
) -> PlanResult:
    model = model or os.environ.get("LONGTAIL_PLANNER_MODEL", DEFAULT_MODEL)
    history: list[dict[str, Any]] = [
        {"type": "message", "role": "system", "content": SYSTEM},
        {"type": "message", "role": "user", "content": query},
    ]
    errors: list[str] = []
    usd = 0.0
    for attempt in range(1, max_attempts + 1):
        request = {
            "model": model,
            "input": history,
            "tools": _tools(),
            # ASSUMPTION A-212: the Meta API takes only tool_choice "auto"; a text reply is re-prompted below
            "tool_choice": "auto",
            "max_output_tokens": 2000,
        }
        resp = client.respond(request, purpose="planner", budget=budget)
        usd += resp.usd
        calls = resp.function_calls()
        if len(calls) != 1:
            err = f"expected exactly one tool call, got {len(calls)}"
            errors.append(err)
            history += [
                *resp.output,
                {"type": "message", "role": "user", "content": f"{err}. Call one tool."},
            ]
            continue
        call = calls[0]
        try:
            args = json.loads(call.get("arguments") or "{}")
        except json.JSONDecodeError as e:
            args, err = None, f"arguments are not valid JSON: {e}"
        else:
            err = None
        if args is not None and call.get("name") == ASK:
            question = str(args.get("question", "")).strip()
            if question:
                spec = ScenarioSpec(
                    query=query, actors=[], predicates=[], assumptions=[], clarification=question
                )
                return PlanResult(spec, attempt, usd, errors)
            err = "ask_clarification needs a non-empty question"
        elif args is not None and call.get("name") != EMIT:
            err = f"unknown tool {call.get('name')!r}"
        elif args is not None:
            try:
                spec = ScenarioSpec.model_validate(_normalise(args, query))
            except ValidationError as e:
                err = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
            else:
                problems = semantic_errors(spec)
                if not problems:
                    return PlanResult(spec, attempt, usd, errors)
                err = "; ".join(problems)
        errors.append(err)
        history += [
            *resp.output,
            {"type": "function_call_output", "call_id": call.get("call_id", ""), "output": f"Invalid: {err}"},
        ]
    raise PlannerError(f"no valid spec for {query!r} after {max_attempts} attempts: {errors}")


def _canon(spec: ScenarioSpec, mapping: dict[str, str]) -> Counter:
    return Counter(
        (p.name, tuple(mapping[a] for a in p.actors), tuple(sorted(p.args.items()))) for p in spec.predicates
    )


def same_spec(a: ScenarioSpec, b: ScenarioSpec) -> bool:
    """True when two specs check the same thing: same actor classes and predicates under some renaming of
    actor ids (assumption wording and SAM phrases are ignored)."""
    if sorted(x.cls for x in a.actors) != sorted(x.cls for x in b.actors):
        return False
    if a.min_duration_s != b.min_duration_s or (a.clarification is None) != (b.clarification is None):
        return False
    target = _canon(b, {x.id: x.id for x in b.actors})
    ids_b = [x.id for x in b.actors]
    for perm in itertools.permutations(ids_b):
        mapping = dict(zip([x.id for x in a.actors], perm, strict=True))
        if all(next(y.cls for y in b.actors if y.id == mapping[x.id]) == x.cls for x in a.actors):
            if _canon(a, mapping) == target:
                return True
    return False
