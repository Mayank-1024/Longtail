"""Card validator (plan 3.4): a card may only say what the verifier showed.

Rules, each a reason a card is rejected:
  1. every sentence cites at least one of this finding's result ids;
  2. a sentence citing a FAIL or UNVERIFIABLE result must say so (a hedge phrase), never state it as fact;
  3. every number in a sentence matches a number in its cited results (or the query and spec) at the
     precision it is written with;
  4. the title is short and its numbers come from the query.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any

from longtail.contracts import Finding, ScenarioSpec

NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
# ASSUMPTION A-219: these phrases mark a sentence as reporting a non-PASS result rather than asserting it
HEDGES = (
    "not verified",
    "could not",
    "couldn't",
    "unverifiable",
    "not verifiable",
    "failed",
    "did not",
    "didn't",
    "not confirmed",
    "no lidar",
    "rejected",
    "not enough",
    "too few",
    "not observed",
)
MAX_TITLE_WORDS = 12
# Evidence keys that hold long index lists; their entries would make almost any integer "supported".
SKIP_KEYS = {"timeline", "frames", "binding_search"}


@dataclass
class Sentence:
    text: str
    cites: list[str]


@dataclass
class Card:
    title: str
    sentences: list[Sentence]
    source: str = "model"  # model | template
    errors: list[str] = field(default_factory=list)  # validation errors from rejected model attempts

    def to_json(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "sentences": [{"text": s.text, "cites": s.cites} for s in self.sentences],
            "source": self.source,
        }


def parse_card(data: Any) -> Card:
    if not isinstance(data, dict):
        raise ValueError("the card must be a JSON object")
    title = data.get("title")
    sentences = data.get("sentences")
    if not isinstance(title, str) or not isinstance(sentences, list):
        raise ValueError("the card needs a string title and a list of sentences")
    out = []
    for i, s in enumerate(sentences):
        if (
            not isinstance(s, dict)
            or not isinstance(s.get("text"), str)
            or not isinstance(s.get("cites"), list)
        ):
            raise ValueError(f"sentence {i} needs a text string and a cites list")
        out.append(Sentence(s["text"].strip(), [str(c) for c in s["cites"]]))
    return Card(title.strip(), out)


def _numbers_in(obj: Any, out: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, int | float):
        if math.isfinite(obj):
            out.append(float(obj))
    elif isinstance(obj, str):
        out.extend(float(n) for n in NUMBER.findall(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k in SKIP_KEYS:
                if isinstance(v, list):
                    out.append(float(len(v)))
                continue
            _numbers_in(v, out)
    elif isinstance(obj, list | tuple):
        for v in obj:
            _numbers_in(v, out)


def _spec_numbers(spec: ScenarioSpec) -> list[float]:
    out: list[float] = []
    _numbers_in(spec.query, out)
    _numbers_in(spec.assumptions, out)
    for p in spec.predicates:
        _numbers_in(p.args, out)
    return out


def _supported(written: str, candidates: list[float]) -> bool:
    value = float(written)
    decimals = len(written.split(".")[1]) if "." in written else 0
    tol = 0.5 * 10**-decimals + 1e-9
    return any(abs(abs(c) - abs(value)) <= tol for c in candidates)


def validate_card(card: Card, finding: Finding, spec: ScenarioSpec) -> list[str]:
    """Every reason this card cannot be shown; empty means it passes."""
    errors: list[str] = []
    results = {r.result_id: r for r in finding.results}
    base = _spec_numbers(spec)
    base.append(round(finding.t1 - finding.t0, 6))
    if not card.title:
        errors.append("the title is empty")
    if len(card.title.split()) > MAX_TITLE_WORDS:
        errors.append(f"the title is longer than {MAX_TITLE_WORDS} words")
    for n in NUMBER.findall(card.title):
        if not _supported(n, _spec_numbers(spec)):
            errors.append(f"title: the number {n} is not in the query")
    if not card.sentences:
        errors.append("the card has no sentences")
    for i, s in enumerate(card.sentences, 1):
        label = f"sentence {i}"
        if not s.cites:
            errors.append(f"{label} cites nothing")
            continue
        unknown = [c for c in s.cites if c not in results]
        if unknown:
            errors.append(f"{label} cites unknown result ids {unknown}")
            continue
        cited = [results[c] for c in s.cites]
        not_pass = [r for r in cited if r.status != "PASS"]
        lower = s.text.lower()
        if not_pass and not any(h in lower for h in HEDGES):
            names = ", ".join(f"{r.name} ({r.status})" for r in not_pass)
            errors.append(f"{label} states {names} as fact; say it was not verified")
        numbers = list(base)
        for r in cited:
            _numbers_in({"value": r.value, "threshold": r.threshold, "evidence": r.evidence}, numbers)
        for n in NUMBER.findall(s.text):
            if not _supported(n, numbers):
                errors.append(f"{label}: the number {n} matches nothing in its cited results")
    return errors
