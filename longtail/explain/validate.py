"""Card validator (plan 3.4): a card may only say what the verifier showed.

Rules, each a reason a card is rejected:
  1. every sentence cites at least one of this finding's result ids, and a sentence that uses a predicate's
     words ("hidden", "crossing", "parked", ...) cites that predicate's result; words for a predicate the
     verifier did not check are rejected outright (A-252);
  2. a sentence citing a FAIL or UNVERIFIABLE result must say so (a hedge phrase), never state it as fact;
     the hedge must sit in the same clause as the words for that predicate, so a hedge about something
     else ("..., which did not move") does not cover it (A-252);
  3. every number in a sentence matches a number in its cited results (or the query and spec) at the
     precision it is written with, and in the same unit when a unit is written: a point count is never a
     distance (A-252); numbers written as words ("nine seconds") are checked like digits;
  4. the title is short and its numbers come from the query.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any

from longtail.contracts import Finding, PredicateResult, ScenarioSpec

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

# ASSUMPTION A-252: the words that claim each predicate; a sentence using them must cite that predicate's
# result. Ego phrases are matched first and consumed, so "while we were moving" is ego_moving, not moving.
CLAIM_WORDS: dict[str, str] = {
    "ego_moving": r"\bego[ _-]?(?:vehicle |car )?(?:is |was )?moving\b"
    r"|\bwe (?:are |were )?(?:moving|driving)\b|\bwe (?:move|drive|moved|drove)\b"
    r"|\bour (?:car|vehicle) (?:is |was )?(?:moving|driving)\b",
    "ego_stopped": r"\bego[ _-]?(?:vehicle |car )?(?:is |was )?stopped\b"
    r"|\bwe (?:are |were )?(?:stopped|waiting)\b|\bwe (?:stop|wait|stopped|waited)\b"
    r"|\bour (?:car|vehicle) (?:is |was )?stopped\b",
    "crossing_ego_path": r"\bcross\w*|\bego path\b|\bour (?:path|lane)\b|\bin front of us\b"
    r"|\bcut\w* (?:across|in)\b",
    "occluded_by": r"\bocclu\w*|\bhidden\b|\bhid(?:es|ing)?\b|\bobscur\w*|\bblocked\b"
    r"|\bbehind (?:a|an|the)\b",
    "relative_position": r"\brelative position\b|\bahead\b|\b(?:to|on) (?:the|our) (?:left|right)\b"
    r"|\bbehind us\b",
    "near": r"\bnear\b|\bnext to\b|\bbeside\b|\bclose to (?:a|an|the)\b",
    "within_distance": r"\bwithin\b|\bclose to us\b",
    "moving": r"\bmov(?:e|es|ed|ing)\b|\bwalk\w*|\bdriv(?:e|es|ing|en)\b|\brid(?:e|es|ing)\b|\bin motion\b",
    "stopped": r"\bstop\w*|\bparked\b|\bstanding still\b|\bstationary\b|\bwaiting\b",
    "exists": r"\bexists?\b|\bexistence\b",
}
_CLAIMS = {name: re.compile(rx, re.I) for name, rx in CLAIM_WORDS.items()}
# A clause ends at , ; : or a joining word; a hedge covers only its own clause. Parentheses do not end one,
# so "Exists (a1) failed (too few points)." is one clause.
CLAUSE_BREAK = re.compile(r"[,;:]|\b(?:which|while|and|but|whereas|although|though|who|that)\b", re.I)

# Units (A-252): what a written unit means, and the unit of each kind of evidence number.
_UNIT_AFTER = re.compile(
    r"\s*(?:-\s*)?(?:lidar\s+)?(m/s|mps|met(?:re|er)s? per second|km/h|kph|mph|%|per ?cent\b"
    r"|met(?:re|er)s?\b|m\b|sweeps?\b|seconds?\b|secs?\b|s\b|points?\b|frames?\b)",
    re.I,
)
VALUE_UNITS = {
    "exists": "count",  # LiDAR points on the object
    "within_distance": "m",
    "near": "m",
    "moving": "m/s",
    "stopped": "m/s",
    "ego_moving": "m/s",
    "ego_stopped": "m/s",
    "crossing_ego_path": "m/s",  # lateral speed toward the corridor
    "occluded_by": "frac",
    "relative_position": "frac",
}
_COUNT_KEYS = {"n_points", "points", "coverage", "samples", "example_frame", "frame_status", "frame_reasons"}

_SMALL = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen"
).split()
_WORD_VALUES = {w: i for i, w in enumerate(_SMALL)}
_WORD_VALUES.update(
    {
        w: 10 * i
        for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split())
        if i > 1
    }
)
_WORD_VALUES["dozen"] = 12
_NOT_A_NUMBER_BEFORE_ONE = {"the", "this", "that", "each", "no", "any", "every", "which", "some", "another"}


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


Candidate = tuple[float, str | None]  # a number and its unit (None: unknown, matches any written unit)


def _unit_after(text: str, end: int) -> str | None:
    m = _UNIT_AFTER.match(text, end)
    if m is None:
        return None
    u = m.group(1).lower()
    if u in ("m/s", "mps") or "per second" in u:
        return "m/s"
    if u in ("km/h", "kph", "mph"):
        return u  # no evidence is in these units, so such a number is never supported
    if u in ("%",) or u.startswith("per"):
        return "pct"
    if u == "m" or u.startswith("met"):
        return "m"
    if u == "s" or u.startswith("sec"):
        return "s"
    return "count"  # points, frames, sweeps


def written_numbers(text: str) -> list[tuple[str, str | None]]:
    """Every number written in text, as digits or as words ("nine", "forty-two"), with its written unit."""
    out = [(m.group(), _unit_after(text, m.end())) for m in NUMBER.finditer(text)]
    words = list(re.finditer(r"[A-Za-z]+", text))
    i = 0
    while i < len(words):
        w = words[i].group().lower()
        before = words[i - 1].group().lower() if i else ""
        if w not in _WORD_VALUES and w != "hundred":
            i += 1
            continue
        if w == "one" and before in _NOT_A_NUMBER_BEFORE_ONE:
            i += 1
            continue
        total, current, j, end = 0, 0, i, words[i].end()
        while j < len(words):
            t = words[j].group().lower()
            gap = text[words[j - 1].end() : words[j].start()] if j > i else ""
            if j > i and gap.strip(" -") != "":
                break
            if t in _WORD_VALUES:
                current += _WORD_VALUES[t]
            elif t == "hundred":
                current = (current or 1) * 100
            elif t == "thousand":
                total, current = total + (current or 1) * 1000, 0
            elif t == "and" and j > i and j + 1 < len(words) and words[j + 1].group().lower() in _WORD_VALUES:
                pass
            else:
                break
            end = words[j].end()
            j += 1
        unit = _unit_after(text, end)
        if not (j == i + 1 and w == "one" and unit is None):  # a bare "one" is too often not a number
            out.append((str(total + current), unit))
        i = j
    return out


def _numbers_in(obj: Any, out: list[Candidate], unit: str | None = None) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, int | float):
        if math.isfinite(obj):
            out.append((float(obj), unit))
    elif isinstance(obj, str):
        out.extend((float(n), u) for n, u in written_numbers(obj) if NUMBER.fullmatch(n))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k in SKIP_KEYS:
                if isinstance(v, list):
                    out.append((float(len(v)), "count"))
                continue
            _numbers_in(v, out, _key_unit(str(k)) or unit)
    elif isinstance(obj, list | tuple):
        for v in obj:
            _numbers_in(v, out, unit)


def _key_unit(key: str) -> str | None:
    k = key.lower()
    if k.endswith("_mps"):
        return "m/s"
    if k.endswith("_m") or k == "m":
        return "m"
    if k.endswith("_s"):
        return "s"
    if k.endswith("_frac") or k.startswith("fraction"):
        return "frac"
    if k in _COUNT_KEYS:
        return "count"
    return None


def _value_unit(r: PredicateResult) -> str | None:
    if r.evidence.get("reason") == "shorter_than_min_duration":
        return "s"  # the value is then the run's duration (verify engine)
    return VALUE_UNITS.get(r.name)


def _result_numbers(r: PredicateResult, out: list[Candidate]) -> None:
    vu = _value_unit(r)
    if r.value is not None:
        _numbers_in(r.value, out, vu)
    _numbers_in(r.threshold, out)
    for k, v in r.evidence.items():
        if k == "margin":  # same unit as the value (A-08)
            _numbers_in(v, out, vu)
        else:
            _numbers_in({k: v}, out)


def _spec_numbers(spec: ScenarioSpec) -> list[Candidate]:
    out: list[Candidate] = []
    _numbers_in(spec.query, out)
    _numbers_in(spec.assumptions, out)
    for p in spec.predicates:
        _numbers_in(p.args, out)
    return out


def _supported(written: str, candidates: list[Candidate], unit: str | None = None) -> bool:
    value = float(written)
    decimals = len(written.split(".")[1]) if "." in written else 0
    tol = 0.5 * 10**-decimals + 1e-9
    for c, cu in candidates:
        if unit is None or cu is None or cu == unit:
            if abs(abs(c) - abs(value)) <= tol:
                return True
        elif unit == "pct" and cu == "frac" and abs(abs(c) * 100 - abs(value)) <= tol:
            return True
    return False


def _claimed(text: str) -> list[str]:
    """The predicates whose words appear in text; each match is consumed so it counts once (A-252)."""
    found = []
    for name, rx in _CLAIMS.items():
        if rx.search(text):
            found.append(name)
            text = rx.sub(" ", text)
    return found


def _scrub(text: str, cited: list[PredicateResult]) -> str:
    """The sentence without the verifier's own words for its cited results (threshold text, reason codes)."""
    for r in cited:
        if r.threshold:
            text = text.replace(r.threshold, " ")
        reason = r.evidence.get("reason")
        if isinstance(reason, str):
            text = re.sub(re.escape(reason.replace("_", " ")), " ", text, flags=re.I)
    return text


def validate_card(card: Card, finding: Finding, spec: ScenarioSpec) -> list[str]:
    """Every reason this card cannot be shown; empty means it passes."""
    errors: list[str] = []
    results = {r.result_id: r for r in finding.results}
    base = _spec_numbers(spec)
    base.append((round(finding.t1 - finding.t0, 6), "s"))
    if not card.title:
        errors.append("the title is empty")
    if len(card.title.split()) > MAX_TITLE_WORDS:
        errors.append(f"the title is longer than {MAX_TITLE_WORDS} words")
    for n, unit in written_numbers(card.title):
        if not _supported(n, _spec_numbers(spec), unit):
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
        cited_names = {r.name for r in cited}
        checked = {r.name for r in finding.results}
        text = _scrub(s.text, cited)
        for name in _claimed(text):
            if name not in checked:
                errors.append(f"{label} claims {name}, which the verifier did not check")
            elif name not in cited_names:
                errors.append(f"{label} claims {name} but does not cite its result")
        clauses = [c for c in CLAUSE_BREAK.split(text) if c and c.strip()]
        hedged = [any(h in c.lower() for h in HEDGES) for c in clauses]
        claims = [set(_claimed(c)) for c in clauses]
        for r in cited:
            if r.status == "PASS":
                continue
            about = [h for c, h in zip(claims, hedged, strict=True) if r.name in c]
            # every clause using this predicate's words must hedge; if none uses them, some clause must hedge
            # without being about another predicate ("..., which did not move" is about moving)
            ok = all(about) if about else any(h and not c for c, h in zip(claims, hedged, strict=True))
            if not ok:
                errors.append(f"{label} states {r.name} ({r.status}) as fact; say it was not verified")
        numbers = list(base)
        for r in cited:
            _result_numbers(r, numbers)
        for n, unit in written_numbers(s.text):
            if not _supported(n, numbers, unit):
                shown = n if unit in (None, "count") else f"{n}%" if unit == "pct" else f"{n} {unit}"
                errors.append(f"{label}: the number {shown} matches nothing in its cited results")
    return errors
