"""Ablation runner (plan C8): `python -m longtail.eval.ablation --config a|b|c|d`.

Input is the gold specs, so it does not wait on the planner. (a) is Spark Open alone and (d) adds
Glimmer triage: both read Mayank's output files (formats.py) and fail clearly while those do not exist.
(b) needs SAM tracks of the named classes, ranked by SAM confidence, with no LiDAR check. (c) is the full
verifier on uniform windows.

Each configuration first runs outside the label guard (it never sees labels), then is scored: against
gt/<query>.json for the findings, and inside eval_ground_truth() for the model claims (A-10).
"""

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from longtail.contracts import CandidateWindow, Frame2D, ScenarioSpec, Track2D
from longtail.data.base import Scene
from longtail.data.guard import eval_ground_truth, labels_allowed
from longtail.eval import formats, gt
from longtail.eval.metrics import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    Claim,
    QueryOutcome,
    count_claims,
    precision_at_k,
    returned_top_k,
    summarise,
)
from longtail.geometry.associate import associate
from longtail.geometry.sync import nearest_sweeps
from longtail.geometry.tracks3d import track_motion
from longtail.verify.config import VerifyConfig, load_verify_config
from longtail.verify.engine import TrackPool, TrackSource, verify
from longtail.verify.predicates import exists
from longtail.verify.predicates.base import Context
from longtail.vision.sam_client import billed_usd

CONFIGS = ("a", "b", "c", "d")
DESCRIPTIONS = {
    "a": "Spark Open alone (describe then match)",
    "b": "+ SAM 3.1 tracks of the named classes",
    "c": "+ LiDAR verification",
    "d": "+ Glimmer triage",
}
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FORMAT = "longtail.ablation/v1"
# ASSUMPTION A-136: uniform windows are the triage grid unfiltered: 2 s cores every 1 s, padded by 1 s each
# side and clipped to the scene (plan 3.3), so (c) and (d) differ only in which windows are kept
WINDOW_S, STRIDE_S, PAD_S = 2.0, 1.0, 1.0


def uniform_windows(scene: Scene) -> list[CandidateWindow]:
    ts = scene.lidar_timestamps()
    first, last = float(ts[0]), float(ts[-1])
    out, seen = [], set()
    start = first
    while True:
        t0, t1 = max(first, start - PAD_S), min(last, start + WINDOW_S + PAD_S)
        if (round(t0, 3), round(t1, 3)) not in seen:
            seen.add((round(t0, 3), round(t1, 3)))
            out.append(
                CandidateWindow(
                    window_id=f"{scene.scene_id}/u{len(out):03d}",
                    scene_id=scene.scene_id,
                    t0=t0,
                    t1=t1,
                    cameras=[],
                    score=0.0,
                    source="uniform",
                )
            )
        if start + WINDOW_S >= last:
            return out
        start += STRIDE_S


class MeteredSource:
    """A TrackSource that remembers each (scene, camera, concept) served, for list-price cost and claims."""

    def __init__(self, source: TrackSource) -> None:
        self.source = source
        self.calls: dict[tuple[str, str, str], tuple[int, list[Track2D]]] = {}

    def __call__(self, scene: Scene, camera: str, concept: str) -> list[Track2D]:
        key = (scene.scene_id, camera, concept)
        if key not in self.calls:
            self.calls[key] = (len(scene.frames(camera)), self.source(scene, camera, concept))
        return self.calls[key][1]

    def list_price_usd(self) -> float:
        # ASSUMPTION A-13: cost at list price with a cold cache: every SAM request a query makes is billed,
        # whether or not our cache served it ($0.20 per 1,000 frames, 50-frame minimum, A-116)
        return sum(billed_usd(n) for n, _ in self.calls.values())


@dataclass
class Output:
    """One configuration's answer to one query, before scoring."""

    findings: list[gt.ScoredFinding]  # ranked, best first
    claims: list[Claim]
    cost_usd: float
    seconds: float
    extra: dict = field(default_factory=dict)


# Temporal non-maximum suppression


def nms(findings: list[gt.ScoredFinding], gap_s: float) -> list[gt.ScoredFinding]:
    """Merge findings with the same binding whose intervals overlap or touch: union interval, best score."""
    # ASSUMPTION A-137: overlapping windows give one finding each; findings of the same scene and binding
    # whose intervals overlap or are within one sweep merge into one (union interval, highest score, both
    # traces)
    groups: dict[tuple, list[gt.ScoredFinding]] = defaultdict(list)
    for f in findings:
        groups[(f.scene_id, f.key)].append(f)
    out = []
    for fs in groups.values():
        fs.sort(key=lambda f: f.t0)
        cur = fs[0]
        for f in fs[1:]:
            if f.t0 <= cur.t1 + gap_s + 1e-6:
                cur = _merge(cur, f)
            else:
                out.append(cur)
                cur = f
        out.append(cur)
    return sorted(out, key=lambda f: (-f.score, f.scene_id, f.t0))


def _merge(a: gt.ScoredFinding, b: gt.ScoredFinding) -> gt.ScoredFinding:
    actors = {}
    for k, ta in a.actors.items():
        tb = b.actors.get(k)
        if ta is None or tb is None:
            actors[k] = ta or tb
            continue
        pos = {round(p[0], 6): p for p in [*ta.positions, *tb.positions]}
        boxes = {(x[0], x[1]): x for x in [*ta.boxes, *tb.boxes]}
        actors[k] = gt.ActorTrace(positions=sorted(pos.values()), boxes=sorted(boxes.values()))
    best = a if a.score >= b.score else b
    return best.model_copy(update={"t0": min(a.t0, b.t0), "t1": max(a.t1, b.t1), "actors": actors})


def _period(scene: Scene) -> float:
    ts = scene.lidar_timestamps()
    return float(np.median(np.diff(ts))) if len(ts) > 1 else 0.1


# (b): SAM tracks of the named classes, ranked by SAM confidence


def _conf(frames: list[Frame2D]) -> float:
    # ASSUMPTION A-138: (b) ranks a track by its mean SAM confidence over the window's frames; 0.5 without one
    vals = [f.conf for f in frames if f.conf is not None]
    return float(np.mean(vals)) if vals else 0.5


def run_b(spec: ScenarioSpec, scene: Scene, windows: list[CandidateWindow], source: TrackSource) -> list:
    """Per window, each actor bound to its most confident SAM track of the class (distinct tracks), if the
    bound tracks are present together; the finding spans that co-presence. No LiDAR, no predicates."""
    findings = []
    for w in windows:
        cams = w.cameras or list(scene.cameras)
        chosen: dict[str, tuple[Track2D, list[Frame2D]]] = {}
        used: set[tuple[str, int]] = set()
        for a in spec.actors:
            cands = []
            for cam in cams:
                for t in source(scene, cam, a.sam_prompt):
                    fr = [f for f in t.frames if w.t0 - 0.05 <= f.ts <= w.t1 + 0.05]
                    if fr and (t.camera, t.obj_id) not in used:
                        cands.append((_conf(fr), t, fr))
            if not cands:
                break
            _, t, fr = max(cands, key=lambda c: c[0])
            chosen[a.id] = (t, fr)
            used.add((t.camera, t.obj_id))
        if len(chosen) < len(spec.actors):
            continue
        spans = [(fr[0].ts, fr[-1].ts) for _, fr in chosen.values()]
        t0 = max([w.t0, *[s[0] for s in spans]])
        t1 = min([w.t1, *[s[1] for s in spans]])
        if t1 < t0:
            continue  # the actors' tracks never appear together
        actors = {
            k: gt.ActorTrace(
                boxes=[(t.camera, f.frame_idx, f.ts, *f.box_xyxy) for f in fr if t0 <= f.ts <= t1]
            )
            for k, (t, fr) in chosen.items()
        }
        score = float(np.mean([_conf(fr) for _, fr in chosen.values()])) if chosen else 0.0
        bound = "|".join(f"{k}={t.camera}:{t.obj_id}" for k, (t, _) in sorted(chosen.items()))
        findings.append(
            gt.ScoredFinding(
                finding_id=hashlib.sha256(f"b|{w.window_id}|{bound}".encode()).hexdigest()[:16],
                scene_id=scene.scene_id,
                t0=t0,
                t1=t1,
                score=score,
                actors=actors,
                key=bound,
            )
        )
    return findings


# (c): the verifier


def run_c(
    spec: ScenarioSpec,
    scene: Scene,
    windows: list[CandidateWindow],
    source: TrackSource,
    cfg: VerifyConfig,
    verifier: bool = True,
) -> tuple[list[gt.ScoredFinding], TrackPool]:
    """verify() on the windows; only verified findings, unless the verifier is switched off (a break)."""
    pool = TrackPool(scene, source, cfg)
    found = verify(spec, windows, scene, track_source=source, cfg=cfg, pool=pool)
    kept = [f for f in found if f.verified or not verifier]
    return [gt.scored_finding(f, pool) for f in kept], pool


def pool_claims(pool: TrackPool, scene: Scene, cfg: VerifyConfig, verifier: bool = True) -> list[Claim]:
    """Every SAM box the engine used, with the verifier's exists verdict at the nearest sweep."""
    # ASSUMPTION A-139: a SAM claim is one Frame2D box of a track with the actor's class; its verdict is
    # exists on the C4/C5 track built from it, at the sweep nearest the camera frame; verifier off = every box
    # PASSes
    sweeps = scene.lidar_timestamps()
    out: dict[str, Claim] = {}
    for (_concept, cls, _cams), objs in pool.items():
        for o in objs:
            verdicts = exists.evaluate(Context(scene, cfg, sweeps), [o], {}) if verifier else None
            for t in o.association.inputs:
                if not t.frames:
                    continue
                idx = nearest_sweeps(np.array([f.ts for f in t.frames]), sweeps)
                for f, i in zip(t.frames, idx, strict=True):
                    cid = f"{scene.scene_id}/{t.camera}/{t.concept}/{t.obj_id}/{f.frame_idx}"
                    out[cid] = Claim(
                        claim_id=cid,
                        scene_id=scene.scene_id,
                        camera=t.camera,
                        frame_idx=f.frame_idx,
                        ts=f.ts,
                        cls=cls,
                        box_xyxy=f.box_xyxy,
                        source="sam",
                        verdict=verdicts[i].status if verdicts is not None else "PASS",
                    )
    return list(out.values())


def metered_claims(meter: MeteredSource, spec: ScenarioSpec) -> list[Claim]:
    """SAM boxes served for the spec's actors, with no verdict ((b) runs no verifier)."""
    cls_of = {a.sam_prompt: a.cls for a in spec.actors}
    out = []
    for (scene_id, camera, concept), (_, tracks) in meter.calls.items():
        if concept not in cls_of:
            continue
        for t in tracks:
            for f in t.frames:
                out.append(
                    Claim(
                        claim_id=f"{scene_id}/{camera}/{concept}/{t.obj_id}/{f.frame_idx}",
                        scene_id=scene_id,
                        camera=camera,
                        frame_idx=f.frame_idx,
                        ts=f.ts,
                        cls=cls_of[concept],
                        box_xyxy=f.box_xyxy,
                        source="sam",
                        verdict=None,
                    )
                )
    return out


# (a): Spark Open alone, from M5's files


def verify_box_claims(scene: Scene, claims: list[formats.VlmClaim], cfg: VerifyConfig) -> list[Claim]:
    """The verifier's exists verdict on each VLM box, each associated on its own (plan 3.4)."""
    # ASSUMPTION A-139: each VLM box is checked alone as a one-frame, box-only Track2D (C4 step 8)
    sweeps = scene.lidar_timestamps()
    out = []
    for c in claims:
        t = Track2D(
            scene_id=c.scene_id,
            camera=c.camera,
            concept=c.cls,
            obj_id=0,
            truncated=False,
            frames=[Frame2D(frame_idx=c.frame_idx, ts=c.ts, box_xyxy=c.box_xyxy)],
        )
        sweep = float(sweeps[nearest_sweeps(np.array([c.ts]), sweeps)[0]])
        status = "UNVERIFIABLE"
        for assoc in associate(scene, [(t, c.cls)], cfg.association):
            obj = track_motion(scene, assoc, cfg.tracks)
            status = exists.evaluate(Context(scene, cfg, np.array([sweep])), [obj], {})[0].status
        out.append(Claim(**c.model_dump(exclude={"behaviour"}), source="vlm", verdict=status))
    return out


def run_a(query: gt.GoldQuery, scenes: dict[str, Scene], cfg: VerifyConfig, baseline_dir: Path) -> Output:
    claims, rankings = formats.load_baseline(baseline_dir)
    q = rankings.queries.get(query.query_id)
    if q is None:
        raise formats.WaitingOnError(f"configuration (a): the baseline has no rankings for {query.query_id}")
    findings = []
    for m in q.matches:
        if m.scene_id not in scenes:
            continue
        cited = [claims[c] for c in m.claim_ids]
        actors: dict[str, gt.ActorTrace | None] = {}
        for a in query.gold.actors:
            mine = [c for c in cited if c.cls == a.cls]
            actors[a.id] = (
                gt.ActorTrace(boxes=[(c.camera, c.frame_idx, c.ts, *c.box_xyxy) for c in mine])
                if mine
                else None
            )
        findings.append(
            gt.ScoredFinding(
                finding_id=f"a|{m.window_id}",
                scene_id=m.scene_id,
                t0=m.t0,
                t1=m.t1,
                score=m.score,
                actors=actors,
            )
        )
    findings.sort(key=lambda f: -f.score)
    # The verifier checks every box the baseline cited for this query: those checks give the catch rate.
    cited_ids = sorted({c for m in q.matches for c in m.claim_ids})
    by_scene: dict[str, list[formats.VlmClaim]] = defaultdict(list)
    for cid in cited_ids:
        if claims[cid].scene_id in scenes:
            by_scene[claims[cid].scene_id].append(claims[cid])
    checked = [c for sid, cs in by_scene.items() for c in verify_box_claims(scenes[sid], cs, cfg)]
    return Output(findings, checked, q.cost_usd, q.seconds)


# Running and scoring


@dataclass
class Setup:
    queries: list[gt.GoldQuery]
    scenes: list[Scene]  # what the configuration runs on (possibly perturbed, breaks.py)
    truth_scenes: list[Scene]  # the unperturbed scenes, for claim truth from the labels
    gt_dir: Path
    source: TrackSource
    verify_cfg: VerifyConfig = field(default_factory=load_verify_config)
    gt_cfg: gt.GtConfig = field(default_factory=gt.load_gt_config)
    baseline_dir: Path = formats.BASELINE_DIR
    triage_dir: Path = formats.TRIAGE_DIR


def run_query(config: str, query: gt.GoldQuery, setup: Setup, verifier: bool = True) -> Output:
    """One configuration on one query over every scene. Never inside the label guard."""
    if labels_allowed():
        raise RuntimeError("configurations run outside eval_ground_truth(): the system must not see labels")
    scenes = {s.scene_id: s for s in setup.scenes}
    if config == "a":
        return run_a(query, scenes, setup.verify_cfg, setup.baseline_dir)
    triage = formats.load_triage(setup.triage_dir) if config == "d" else None
    if triage is not None and query.query_id not in triage.queries:
        raise formats.WaitingOnError(
            f"configuration (d): the triage file has no windows for {query.query_id}"
        )
    meter = MeteredSource(setup.source)
    start = time.perf_counter()
    findings: list[gt.ScoredFinding] = []
    claims: list[Claim] = []
    windows_used = 0
    for scene in setup.scenes:
        if triage is not None:
            windows = [w for w in triage.queries[query.query_id].windows if w.scene_id == scene.scene_id]
        else:
            windows = uniform_windows(scene)
        windows_used += len(windows)
        if not windows:
            continue
        if config == "b":
            found = nms(run_b(query.gold, scene, windows, meter), _period(scene))
        else:
            found, pool = run_c(query.gold, scene, windows, meter, setup.verify_cfg, verifier)
            found = nms(found, _period(scene))
            claims += pool_claims(pool, scene, setup.verify_cfg, verifier)
        findings += found
    if config == "b":
        claims = metered_claims(meter, query.gold)
    seconds = time.perf_counter() - start
    cost = meter.list_price_usd()
    extra = {"windows": windows_used}
    if triage is not None:
        tq = triage.queries[query.query_id]
        cost, seconds = cost + tq.cost_usd, seconds + tq.seconds
        extra["windows_total"] = tq.windows_total
        extra["windows_kept_frac"] = len(tq.windows) / tq.windows_total if tq.windows_total else None
    findings.sort(key=lambda f: (-f.score, f.scene_id, f.t0))
    return Output(findings, claims, cost, seconds, extra)


class _Truth:
    """Label-derived truth for claims, per scene, read once inside the guard."""

    def __init__(self, setup: Setup) -> None:
        self.scenes = {s.scene_id: s for s in setup.truth_scenes}
        self.cfg = setup.gt_cfg
        self._labels: dict[str, gt.LabelScene] = {}

    def claims(self, claims: list[Claim]) -> dict[str, bool]:
        with eval_ground_truth():
            out = {}
            for c in claims:
                if c.scene_id not in self._labels:
                    self._labels[c.scene_id] = gt.LabelScene(self.scenes[c.scene_id], self.cfg)
                out[c.claim_id] = gt.claim_is_true(
                    self._labels[c.scene_id], c.camera, c.frame_idx, c.cls, c.box_xyxy
                )
            return out


def score(query: gt.GoldQuery, out: Output, setup: Setup, truth: _Truth) -> QueryOutcome:
    doc = gt.load_gt(setup.gt_dir, query.query_id)
    if doc.spec_hash != gt.spec_hash(query.gold):
        raise ValueError(f"{query.query_id}: gt/ was built from another gold spec; rebuild it")
    scene_ids = {s.scene_id for s in setup.scenes}
    missing = scene_ids - set(doc.scenes)
    if missing:
        raise ValueError(f"{query.query_id}: gt/ does not cover scenes {sorted(missing)}; rebuild it")
    positives = [p for p in doc.positives if p.scene_id in scene_ids]
    matches = gt.match(out.findings, positives, setup.gt_cfg.matching)
    counts = count_claims(out.claims, truth.claims(out.claims))
    return QueryOutcome(
        query_id=query.query_id,
        ranked=[f.finding_id for f in out.findings],
        matched={m.finding_id for m in matches},
        n_positives=len(positives),
        cost_usd=out.cost_usd,
        seconds=out.seconds,
        **counts,
        extra={**out.extra, "claims": len(out.claims), "hallucinated_claims": counts["claims_false"]},
    )


def run_config(config: str, setup: Setup, verifier: bool = True) -> list[QueryOutcome]:
    truth = _Truth(setup)
    return [score(q, run_query(config, q, setup, verifier), setup, truth) for q in setup.queries]


# ASSUMPTION A-233: sharding by scene; each scene runs as its own job (--partial), one merge scores them all.
# Exact, because run_query() treats scenes independently: NMS is per scene, findings are re-sorted across
# scenes, claims and list-price costs are per (scene, camera, concept), seconds and windows add up.

PARTIAL_FORMAT = "longtail.ablation_partial/v1"


def partial_path(out_dir: Path, config: str, scene_id: str) -> Path:
    return Path(out_dir) / config / f"{scene_id}.pkl"


def write_partial(config: str, setup: Setup, out_dir: Path, verifier: bool = True) -> list[Path]:
    """run_query() per query on each of setup.scenes alone, saved unscored, one file per scene."""
    paths = []
    for scene in setup.scenes:
        one = Setup(**{**setup.__dict__, "scenes": [scene], "truth_scenes": [scene]})
        outputs = {q.query_id: run_query(config, q, one, verifier) for q in setup.queries}
        path = partial_path(out_dir, config, scene.scene_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(
            pickle.dumps({"format": PARTIAL_FORMAT, "scene_id": scene.scene_id, "outputs": outputs})
        )
        tmp.replace(path)
        paths.append(path)
    return paths


def merge_outputs(parts: list[Output]) -> Output:
    findings = sorted((f for p in parts for f in p.findings), key=lambda f: (-f.score, f.scene_id, f.t0))
    extra: dict = {}
    for p in parts:
        for k, v in p.extra.items():
            extra[k] = extra.get(k, 0) + v if isinstance(v, int | float) and v is not None else v
    return Output(
        findings,
        [c for p in parts for c in p.claims],
        sum(p.cost_usd for p in parts),
        sum(p.seconds for p in parts),
        extra,
    )


def read_partials(out_dir: Path, config: str, scene_ids: list[str]) -> dict[str, Output]:
    """The merged Output per query over scene_ids; every scene must have its partial file."""
    missing = [s for s in scene_ids if not partial_path(out_dir, config, s).is_file()]
    if missing:
        raise FileNotFoundError(f"configuration ({config}): no partial results for scenes {missing}")
    per_query: dict[str, list[Output]] = defaultdict(list)
    for sid in scene_ids:
        doc = pickle.loads(partial_path(out_dir, config, sid).read_bytes())
        if doc.get("format") != PARTIAL_FORMAT:
            raise ValueError(f"{sid}: not a {PARTIAL_FORMAT} file")
        for qid, out in doc["outputs"].items():
            per_query[qid].append(out)
    return {qid: merge_outputs(parts) for qid, parts in per_query.items()}


def merge_config(config: str, setup: Setup, out_dir: Path) -> list[QueryOutcome]:
    merged = read_partials(out_dir, config, [s.scene_id for s in setup.scenes])
    truth = _Truth(setup)
    outcomes = []
    for q in setup.queries:
        if q.query_id not in merged:
            raise FileNotFoundError(f"configuration ({config}): the partial results have no {q.query_id}")
        outcomes.append(score(q, merged[q.query_id], setup, truth))
    return outcomes


# Results


def git_commit() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head.strip() + ("-dirty" if dirty.strip() else "")


def thresholds_hash(verify_cfg: VerifyConfig, gt_cfg: gt.GtConfig) -> str:
    # ASSUMPTION A-140: results carry the commit (with -dirty for uncommitted changes) and a hash of the
    # verify.yaml and gt.yaml thresholds actually used
    raw = verify_cfg.model_dump_json() + gt_cfg.model_dump_json()
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def results_doc(name: str, outcomes: list[QueryOutcome], setup: Setup, split: str, description: str) -> dict:
    per_query = []
    for o in outcomes:
        per_query.append(
            {
                "query_id": o.query_id,
                "positives": o.n_positives,
                "returned": len(o.ranked),
                "matched": len(o.matched),
                "precision": len(o.matched) / len(o.ranked) if o.ranked else None,
                "p_at_10": precision_at_k(o),
                "returned_top10": returned_top_k(o),
                "cost_usd": round(o.cost_usd, 6),
                "seconds": round(o.seconds, 3),
                "claims_false": o.claims_false,
                "claims_true": o.claims_true,
                "claims_false_checked": o.claims_false_checked,
                "claims_false_caught": o.claims_false_caught,
                "claims_true_checked": o.claims_true_checked,
                "claims_true_rejected": o.claims_true_rejected,
                **o.extra,
            }
        )
    return {
        "format": RESULTS_FORMAT,
        "config": name,
        "description": description,
        "commit": git_commit(),
        "thresholds_hash": thresholds_hash(setup.verify_cfg, setup.gt_cfg),
        "gt_config_hash": gt.config_hash(setup.gt_cfg),
        "split": split,
        "scenes": [s.scene_id for s in setup.scenes],
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "over": "queries"},
        "metrics": summarise(outcomes),
        "queries": per_query,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def check_results_dir(out_dir: Path) -> Path:
    # ASSUMPTION A-140: ablation and break results go to a directory named results (eval/results/ in the plan)
    out_dir = Path(out_dir)
    if out_dir.name != "results":
        raise ValueError(f"results go to a directory named results, not {out_dir}")
    return out_dir


def write_results(doc: dict, out_dir: Path) -> Path:
    out_dir = check_results_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ablation_{doc['config']}.json"
    path.write_text(json.dumps(doc, indent=1) + "\n")
    chart(out_dir)
    return path


# Validated categorical slots 1-4 (dataviz reference palette, light surface); colour follows the
# configuration.
COLOURS = {"a": "#2a78d6", "b": "#eb6834", "c": "#1baf7a", "d": "#eda100"}
CHART_METRICS = (
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("p_at_10", "P@10"),
    ("catch_rate", "Catch rate"),
)


def chart(out_dir: Path) -> Path | None:
    """ablation.png from every ablation_<config>.json in out_dir: P@10, recall and catch rate with 95% CIs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    docs = {}
    for c in CONFIGS:
        p = out_dir / f"ablation_{c}.json"
        if p.is_file():
            docs[c] = json.loads(p.read_text())
    if not docs:
        return None
    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    n = len(docs)
    width = 0.8 / n
    for j, (c, doc) in enumerate(docs.items()):
        for i, (key, _) in enumerate(CHART_METRICS):
            m = doc["metrics"][key]
            x = i - 0.4 + width * (j + 0.5)
            if m["value"] is None:
                ax.text(x, 0.02, "n/a", ha="center", va="bottom", fontsize=7, color=muted, rotation=90)
                continue
            ax.bar(x, m["value"], width * 0.92, color=COLOURS[c], label=None, zorder=2)
            if m["ci95"] is not None:
                lo, hi = m["ci95"]
                ax.errorbar(
                    x, m["value"], yerr=[[m["value"] - lo], [hi - m["value"]]], fmt="none", ecolor=muted,
                    elinewidth=1, capsize=2, zorder=3,
                )  # fmt: skip
            # Above the interval, so the label never sits on the error bar.
            top = m["ci95"][1] if m["ci95"] is not None else m["value"]
            v = m["value"]
            label = f"{v:.3f}" if 0 < v < 0.1 else f"{v:.2f}"  # 0.025 must not read as 0.03
            if key == "p_at_10":
                # ASSUMPTION A-134: P@10 divides by 10, so the bar shows how many findings came back
                label += f"\n{doc['metrics']['returned_top10']['value']:.1f}/10"
            ax.text(x, top + 0.015, label, ha="center", va="bottom", fontsize=7, color=ink, linespacing=1.1)
    for c in docs:
        ax.bar(0, 0, color=COLOURS[c], label=f"({c}) {DESCRIPTIONS[c]}")
    names = [
        name if key != "p_at_10" else f"{name}\n(returned per query / 10)" for key, name in CHART_METRICS
    ]
    ax.set_xticks(range(len(CHART_METRICS)), names, color=ink)
    ax.set_ylim(0, 1.2)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("share, with 95% bootstrap interval", color=muted, fontsize=8)
    ax.tick_params(colors=muted, labelsize=8)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    split = next(iter(docs.values()))["split"]
    ax.set_title(f"Ablation ({split} queries)", color=ink, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0), labelcolor=ink)
    fig.tight_layout()
    path = out_dir / "ablation.png"
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
    return path


# CLI


def select(queries: list[gt.GoldQuery], split: str) -> list[gt.GoldQuery]:
    return queries if split == "all" else [q for q in queries if q.split == split]


def sam_source(replay: bool) -> TrackSource:
    from longtail.vision.sam_client import SamClient, transport_from_env
    from longtail.vision.track_source import sam_track_source

    return sam_track_source(SamClient(None if replay else transport_from_env(), replay=replay))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m longtail.eval.ablation", description=__doc__.splitlines()[0])
    ap.add_argument("--config", choices=CONFIGS, required=True)
    ap.add_argument("--queries", type=Path, required=True, help="queries file (A-128)")
    ap.add_argument("--split", choices=["dev", "heldout", "all"], default="heldout")
    ap.add_argument("--gt-dir", type=Path, default=Path("gt"))
    ap.add_argument("--scenes", nargs="*", help="scene ids (default: every scene in the index)")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    ap.add_argument("--replay", action="store_true", help="SAM responses from the cache only (no paid calls)")
    ap.add_argument("--partial", type=Path, help="run each scene alone and save it unscored here (A-233)")
    ap.add_argument("--merge", type=Path, help="score the partial results saved here instead of running")
    args = ap.parse_args(argv)
    if args.partial and args.merge:
        ap.error("--partial and --merge are separate steps")
    from longtail.eval.cli import _scenes

    scenes = _scenes(args.scenes)
    setup = Setup(
        queries=select(gt.load_queries(args.queries), args.split),
        scenes=scenes,
        truth_scenes=scenes,
        gt_dir=args.gt_dir,
        source=sam_source(args.replay),
    )
    try:
        if args.partial:
            if args.config in ("b", "c"):
                from longtail.pipeline import SAM_WORKERS, prefetch

                by_id = {s.scene_id: s for s in setup.scenes}
                for failed in prefetch(
                    [q.gold for q in setup.queries], list(by_id), by_id.__getitem__, setup.source, SAM_WORKERS
                ):
                    print(f"prefetch failed (run_query retries it): {failed}", file=sys.stderr)
            for path in write_partial(args.config, setup, args.partial):
                print(f"partial written to {path}")
            return 0
        outcomes = (
            merge_config(args.config, setup, args.merge) if args.merge else run_config(args.config, setup)
        )
    except formats.WaitingOnError as e:
        print(e, file=sys.stderr)
        return 3
    doc = results_doc(args.config, outcomes, setup, args.split, DESCRIPTIONS[args.config])
    path = write_results(doc, args.out)
    for k, v in doc["metrics"].items():
        print(f"{k}: {v}")
    print(f"written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
