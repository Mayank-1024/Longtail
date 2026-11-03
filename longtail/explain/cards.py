"""Scenario cards (plan 3.4, M5): Spark Open writes a short card for a finding; the validator checks it.

One model attempt, one retry with the validator's errors, then a template built from the results, so a card
can never contain a claim the verifier did not make.
"""

import base64
import json
import re
from pathlib import Path
from typing import Any

from longtail.contracts import Finding, PredicateResult, ScenarioSpec
from longtail.explain.validate import MAX_TITLE_WORDS, Card, Sentence, parse_card, validate_card
from longtail.llm.baseten import SPARK_OPEN
from longtail.llm.client import Budget, ModelClient

# ASSUMPTION A-220: a card is at most 4 sentences, built from at most 3 keyframe images
MAX_SENTENCES = 4
MAX_IMAGES = 3

SYSTEM = f"""You write a short evidence card for one search result from driving logs. A LiDAR verifier has
already judged each predicate as PASS, FAIL or UNVERIFIABLE; you report what it found, nothing more.

Return only JSON: {{"title": str, "sentences": [{{"text": str, "cites": [result_id, ...]}}]}}
- At most {MAX_SENTENCES} sentences, each citing the result ids it rests on.
- State a PASS result plainly. A FAIL or UNVERIFIABLE result may only be reported as not verified
  ("could not be verified", "failed", "did not"), never as fact.
- Use only numbers that appear in the cited results, rounded as you write them; give units (m, m/s, s).
- The title is a few words with no numbers unless they are in the query. No speculation about intent."""


def _image_part(path: Path) -> dict[str, Any]:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "input_image", "image_url": f"data:image/jpeg;base64,{data}"}


def _result_view(r: PredicateResult) -> dict[str, Any]:
    ev = r.evidence
    return {
        "result_id": r.result_id,
        "predicate": r.name,
        "actors": r.actors,
        "status": r.status,
        "reason": ev.get("reason"),
        "value": r.value,
        "threshold": r.threshold,
        "longest_pass_s": ev.get("longest_pass_s"),
        "frame_status": ev.get("frame_status"),
        "example": ev.get("example"),
    }


def card_request(
    finding: Finding, spec: ScenarioSpec, images: list[Path] | None = None, feedback: str | None = None
) -> dict[str, Any]:
    facts = {
        "query": spec.query,
        "assumptions": spec.assumptions,
        "scene_id": finding.scene_id,
        "t0": finding.t0,
        "t1": finding.t1,
        "duration_s": round(finding.t1 - finding.t0, 2),
        "verified": finding.verified,
        "results": [_result_view(r) for r in finding.results],
    }
    content: list[dict[str, Any]] = [{"type": "input_text", "text": json.dumps(facts, default=str)}]
    content += [_image_part(p) for p in (images or [])[:MAX_IMAGES]]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
    if feedback:
        messages.append({"role": "user", "content": f"Your card was rejected: {feedback}. Write it again."})
    return {"model": SPARK_OPEN, "input": messages, "max_output_tokens": 600, "temperature": 0}


def _parse_json(text: str) -> Any:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    body = fenced.group(1) if fenced else text
    start, end = body.find("{"), body.rfind("}")
    return json.loads(body[start : end + 1] if start >= 0 and end > start else body)


def _label(r: PredicateResult) -> str:
    who = ", ".join(r.actors) if r.actors else "the ego vehicle"
    return f"{r.name.replace('_', ' ')} ({who})"


def template_card(finding: Finding, spec: ScenarioSpec, errors: list[str] | None = None) -> Card:
    """A card built only from the results; it always passes the validator."""
    sentences = []
    for r in finding.results:
        reason = str(r.evidence.get("reason", "")).replace("_", " ")
        if r.status == "PASS":
            detail = f": {r.value:.1f} against {r.threshold}" if r.value is not None and r.threshold else ""
            sentences.append(Sentence(f"Verified {_label(r)}{detail}.", [r.result_id]))
        else:
            word = "failed" if r.status == "FAIL" else "could not be verified"
            sentences.append(Sentence(f"{_label(r).capitalize()} {word} ({reason}).", [r.result_id]))
    head = "Verified:" if finding.verified else "Rejected:"
    title = " ".join([head, *spec.query.split()][:MAX_TITLE_WORDS])
    return Card(title, sentences, "template", list(errors or []))


def _title_fits(card: Card) -> Card:
    words = card.title.split()
    if len(words) > MAX_TITLE_WORDS:
        card.title = " ".join(words[:MAX_TITLE_WORDS])
    return card


def make_card(
    finding: Finding,
    spec: ScenarioSpec,
    client: ModelClient | None,
    *,
    images: list[Path] | None = None,
    budget: Budget | None = None,
) -> Card:
    """The model's card if it passes the validator within two attempts, otherwise the template card.

    With no client (offline, or no Baseten key), the template card is returned without any call.
    """
    if client is None:
        return _title_fits(template_card(finding, spec))
    errors: list[str] = []
    feedback = None
    for _ in range(2):
        resp = client.respond(card_request(finding, spec, images, feedback), purpose="card", budget=budget)
        try:
            card = parse_card(_parse_json(resp.text))
        except (ValueError, json.JSONDecodeError) as e:
            problems = [f"not a valid card: {e}"]
        else:
            if len(card.sentences) > MAX_SENTENCES:
                problems = [f"more than {MAX_SENTENCES} sentences"]
            else:
                problems = validate_card(card, finding, spec)
            if not problems:
                card.errors = errors
                return card
        errors += problems
        feedback = "; ".join(problems)
    return _title_fits(template_card(finding, spec, errors))
