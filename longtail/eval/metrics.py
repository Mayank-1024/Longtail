"""Metrics (plan C8): P@10, recall, cost and seconds per query, catch and false-rejection rates, with 95%
bootstrap intervals over queries.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from longtail.contracts import Status

K = 10
# ASSUMPTION A-133: 1,000 bootstrap resamples over queries with this fixed seed; percentile 95% intervals
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20261103


class Claim(BaseModel):
    """One model claim: an object of a class in a box in one camera frame (a VLM box or a SAM track frame)."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    scene_id: str
    camera: str
    frame_idx: int
    ts: float
    cls: str
    box_xyxy: tuple[float, float, float, float]
    source: Literal["vlm", "sam"]
    # The verifier's exists verdict for this box; None when the configuration runs no verifier.
    verdict: Status | None


@dataclass
class QueryOutcome:
    """What one configuration returned for one query, scored against the ground truth."""

    query_id: str
    ranked: list[str]  # finding ids, best first: the configuration's output list
    matched: set[str]  # finding ids matched one-to-one to a positive
    n_positives: int
    cost_usd: float
    seconds: float
    # Claims the labels show are false or true (A-10); of those the verifier checked, how many it rejected
    # (FAIL or UNVERIFIABLE). Unchecked claims (no verifier in the configuration) count only as claims.
    claims_false: int = 0
    claims_true: int = 0
    claims_false_checked: int = 0
    claims_false_caught: int = 0
    claims_true_checked: int = 0
    claims_true_rejected: int = 0
    extra: dict = field(default_factory=dict)


def precision_at_k(o: QueryOutcome, k: int = K) -> float | None:
    """Matched share of the top k; None when the configuration returned nothing for the query."""
    # ASSUMPTION A-134: P@10 divides by the number returned when under 10, and a query with nothing returned
    # is left out of the mean (the count of such queries is reported next to it, and recall carries the miss)
    top = o.ranked[:k]
    return sum(f in o.matched for f in top) / len(top) if top else None


def count_claims(claims: list[Claim], truth: dict[str, bool]) -> dict[str, int]:
    """Counts for the catch and false-rejection rates. truth: claim id to "the labels support it" (A-10)."""
    out = dict.fromkeys(
        ("claims_false", "claims_true", "claims_false_checked", "claims_false_caught", "claims_true_checked"),
        0,
    )
    out["claims_true_rejected"] = 0
    for c in claims:
        kind = "true" if truth[c.claim_id] else "false"
        out[f"claims_{kind}"] += 1
        if c.verdict is None:
            continue  # no verifier ran: nothing to catch or reject
        out[f"claims_{kind}_checked"] += 1
        out["claims_true_rejected" if kind == "true" else "claims_false_caught"] += c.verdict in (
            "FAIL",
            "UNVERIFIABLE",
        )
    return out


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def p_at_10(qs: list[QueryOutcome]) -> float | None:
    vals = [v for o in qs if (v := precision_at_k(o)) is not None]
    return float(np.mean(vals)) if vals else None


def recall(qs: list[QueryOutcome]) -> float | None:
    # ASSUMPTION A-134: recall pools over queries: matched positives over all ground-truth positives
    return _ratio(sum(len(o.matched) for o in qs), sum(o.n_positives for o in qs))


def cost_per_query(qs: list[QueryOutcome]) -> float | None:
    return float(np.mean([o.cost_usd for o in qs])) if qs else None


def seconds_per_query(qs: list[QueryOutcome]) -> float | None:
    return float(np.mean([o.seconds for o in qs])) if qs else None


def catch_rate(qs: list[QueryOutcome]) -> float | None:
    """Of the claims the labels show are false, the share the verifier marked FAIL or UNVERIFIABLE."""
    return _ratio(sum(o.claims_false_caught for o in qs), sum(o.claims_false_checked for o in qs))


def false_rejection_rate(qs: list[QueryOutcome]) -> float | None:
    """Of the claims the labels support, the share the verifier marked FAIL or UNVERIFIABLE."""
    return _ratio(sum(o.claims_true_rejected for o in qs), sum(o.claims_true_checked for o in qs))


def hallucination_rate(qs: list[QueryOutcome]) -> float | None:
    """Share of the model's claims that the labels show are false (A-10), checked by the verifier or not."""
    return _ratio(sum(o.claims_false for o in qs), sum(o.claims_false + o.claims_true for o in qs))


METRICS: dict[str, Callable[[list[QueryOutcome]], float | None]] = {
    "p_at_10": p_at_10,
    "recall": recall,
    "cost_per_query_usd": cost_per_query,
    "seconds_per_query": seconds_per_query,
    "catch_rate": catch_rate,
    "false_rejection_rate": false_rejection_rate,
    "hallucination_rate": hallucination_rate,
}


def bootstrap_ci(
    qs: list[QueryOutcome],
    stat: Callable[[list[QueryOutcome]], float | None],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float] | None:
    """95% percentile interval of stat over resamples of the queries (with replacement)."""
    if not qs:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(qs), size=(resamples, len(qs)))
    vals = [v for row in idx if (v := stat([qs[i] for i in row])) is not None]
    if not vals:
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def summarise(
    qs: list[QueryOutcome], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> dict:
    """Every metric with its 95% interval; the catch rate always next to the false-rejection rate."""
    out = {}
    for name, fn in METRICS.items():
        ci = bootstrap_ci(qs, fn, resamples, seed)
        out[name] = {"value": fn(qs), "ci95": list(ci) if ci is not None else None}
    out["queries"] = len(qs)
    out["queries_with_no_output"] = sum(not o.ranked for o in qs)
    out["positives"] = sum(o.n_positives for o in qs)
    out["claims_false"] = sum(o.claims_false for o in qs)
    out["claims_true"] = sum(o.claims_true for o in qs)
    return out
