"""File formats C8 reads from Mayank's side: baseline (a) claims and rankings (M5) and triaged windows (M4).

Documented with examples in docs/eval-formats.md. A missing file raises WaitingOnError naming the task.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from longtail.contracts import CandidateWindow, TaxonomyClass

BASELINE_DIR = Path("runs") / "baseline_a"
TRIAGE_DIR = Path("runs") / "triage_d"


class WaitingOnError(FileNotFoundError):
    """An ablation input that Mayank's pipeline produces does not exist yet."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Baseline (a): Spark Open, describe then match (plan 3.4, M5)


class VlmClaim(_Strict):
    """One box the VLM claims, cached per window and reused across queries."""

    claim_id: str
    scene_id: str
    camera: str
    frame_idx: int = Field(ge=0)
    ts: float  # the camera frame's timestamp
    cls: TaxonomyClass
    box_xyxy: tuple[float, float, float, float]  # inclusive pixels, full image (A-20)
    behaviour: str | None = None


class BaselineMatch(_Strict):
    """One window the baseline returns for a query: its rank score and the claims it rests on."""

    window_id: str
    scene_id: str
    t0: float
    t1: float
    score: float
    claim_ids: list[str]  # assigned to the gold spec's actors by class (A-135)


class BaselineQuery(_Strict):
    matches: list[BaselineMatch]
    cost_usd: float = Field(ge=0)  # at list price, cache hits included (A-13)
    seconds: float = Field(ge=0)


class BaselineRankings(_Strict):
    # ASSUMPTION A-135: baseline (a) files: runs/baseline_a/claims.jsonl (one VlmClaim per line) and
    # runs/baseline_a/rankings.json (format tag, model, per query its matches, cost and seconds)
    format: Literal["longtail.baseline_a/v1"]
    model: str
    queries: dict[str, BaselineQuery]


def load_baseline(path: Path = BASELINE_DIR) -> tuple[dict[str, VlmClaim], BaselineRankings]:
    claims_file, rankings_file = Path(path) / "claims.jsonl", Path(path) / "rankings.json"
    missing = [str(p) for p in (claims_file, rankings_file) if not p.is_file()]
    if missing:
        raise WaitingOnError(
            f"configuration (a) is waiting on M5 (Spark Open baseline, eval/baseline_vlm.py): missing "
            f"{', '.join(missing)}; the format is in docs/eval-formats.md"
        )
    claims = {}
    for n, line in enumerate(claims_file.read_text().splitlines(), 1):
        if line.strip():
            c = VlmClaim.model_validate_json(line)
            if c.claim_id in claims:
                raise ValueError(f"{claims_file}:{n}: duplicate claim id {c.claim_id}")
            claims[c.claim_id] = c
    rankings = BaselineRankings.model_validate_json(rankings_file.read_text())
    for qid, q in rankings.queries.items():
        for m in q.matches:
            unknown = set(m.claim_ids) - set(claims)
            if unknown:
                raise ValueError(
                    f"{rankings_file}: query {qid} window {m.window_id} cites unknown claims {unknown}"
                )
    return claims, rankings


# Configuration (d): Glimmer triage (plan 3.3, M4)


class TriageQuery(_Strict):
    windows: list[CandidateWindow]  # source "glimmer", 2 s windows padded by 1 s
    windows_total: int = Field(ge=0)  # windows considered before triage, for the share kept
    cost_usd: float = Field(ge=0)  # at list price, cache hits included (A-13)
    seconds: float = Field(ge=0)


class TriageWindows(_Strict):
    # ASSUMPTION A-135: configuration (d) file: runs/triage_d/windows.json (format tag, model, per query the
    # kept CandidateWindows, the number considered, cost and seconds)
    format: Literal["longtail.triage_d/v1"]
    model: str
    queries: dict[str, TriageQuery]


def load_triage(path: Path = TRIAGE_DIR) -> TriageWindows:
    f = Path(path) / "windows.json"
    if not f.is_file():
        raise WaitingOnError(
            f"configuration (d) is waiting on M4 (Glimmer triage): missing {f}; "
            "the format is in docs/eval-formats.md"
        )
    return TriageWindows.model_validate_json(f.read_text())
