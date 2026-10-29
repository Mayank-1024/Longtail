"""Verification engine (plan C6): bind actors to tracks, evaluate predicates per frame, emit Findings.

The one entry point is verify(spec, windows, scene). Mayank's pipeline and agent tools call it.
"""

import hashlib
import itertools
import math
from collections import Counter
from collections.abc import Callable
from types import ModuleType

import numpy as np

from longtail.contracts import CandidateWindow, Finding, PredicateCall, PredicateResult, ScenarioSpec, Track2D
from longtail.data.base import Scene
from longtail.geometry.associate import associate
from longtail.geometry.tracks3d import TrackedObject, track_motion
from longtail.verify.config import VerifyConfig, load_verify_config
from longtail.verify.predicates import (
    crossing_ego_path,
    ego_moving,
    ego_stopped,
    exists,
    moving,
    near,
    occluded_by,
    relative_position,
    stopped,
    within_distance,
)
from longtail.verify.predicates.base import Context, FrameResult, Reason

PREDICATES: dict[str, ModuleType] = {
    "exists": exists,
    "within_distance": within_distance,
    "moving": moving,
    "stopped": stopped,
    "crossing_ego_path": crossing_ego_path,
    "relative_position": relative_position,
    "near": near,
    "occluded_by": occluded_by,
    "ego_moving": ego_moving,
    "ego_stopped": ego_stopped,
}

# ASSUMPTION A-86: tracks come from a TrackSource(scene, camera, concept); C3's SAM client will be the default
TrackSource = Callable[[Scene, str, str], list[Track2D]]

UNBOUND = "unbound"
PRUNE_WITH = ("exists", "within_distance")


def verify(
    spec: ScenarioSpec,
    windows: list[CandidateWindow],
    scene: Scene,
    *,
    track_source: TrackSource | None = None,
    cfg: VerifyConfig | None = None,
) -> list[Finding]:
    """One Finding per window with frames: the best-scoring binding, verified or not.

    Findings are sorted verified first, then by score.
    """
    # ASSUMPTION A-17: unverified findings are returned too; only the agent's finish tool drops them
    if spec.clarification is not None:
        raise ValueError("the spec asks a clarifying question; answer it before verifying")
    if track_source is None:
        raise RuntimeError("no track source: the SAM 3.1 client (C3) is not built yet, so pass track_source")
    cfg = cfg or load_verify_config()
    pool = TrackPool(scene, track_source, cfg)
    spec_hash = hashlib.sha256(spec.model_dump_json().encode()).hexdigest()[:16]
    findings = []
    for window in windows:
        if window.scene_id != scene.scene_id:
            raise ValueError(
                f"window {window.window_id} is for scene {window.scene_id}, not {scene.scene_id}"
            )
        finding = _verify_window(spec, window, scene, pool, cfg, spec_hash)
        if finding is not None:
            findings.append(finding)
    return sorted(findings, key=lambda f: (not f.verified, -f.score))


class TrackPool:
    """Tracked objects per (concept, class, cameras), associated (C4) and tracked (C5) once.

    Deterministic for a given TrackSource, so C9 can rebuild the tracks a finding was bound to.
    """

    def __init__(self, scene: Scene, source: TrackSource, cfg: VerifyConfig) -> None:
        self.scene, self.source, self.cfg = scene, source, cfg
        self._cache: dict[tuple[str, str, tuple[str, ...]], list[TrackedObject]] = {}

    def get(self, concept: str, cls: str, cameras: tuple[str, ...]) -> list[TrackedObject]:
        key = (concept, cls, cameras)
        if key not in self._cache:
            tracks = [(t, cls) for cam in cameras for t in self.source(self.scene, cam, concept)]
            assocs = associate(self.scene, tracks, self.cfg.association)
            self._cache[key] = [track_motion(self.scene, a, self.cfg.tracks) for a in assocs]
        return self._cache[key]


def _concept_for(spec: ScenarioSpec, cls: str) -> str:
    # ASSUMPTION A-87: an occluder class uses the SAM phrase of a spec actor of that class, else its name
    return next((a.sam_prompt for a in spec.actors if a.cls == cls), cls)


def _in_window(obj: TrackedObject, t0: float, t1: float) -> bool:
    times = [s.ts for s in obj.track.samples] + [s.ts for s in obj.association.skipped]
    return any(t0 - 0.05 <= t <= t1 + 0.05 for t in times)


def _verify_window(
    spec: ScenarioSpec,
    window: CandidateWindow,
    scene: Scene,
    pool: TrackPool,
    cfg: VerifyConfig,
    spec_hash: str,
) -> Finding | None:
    sweeps = scene.lidar_timestamps()
    # ASSUMPTION A-88: window t0 and t1 are dataset timestamps in seconds, like Sample3D.ts (A-03)
    frames = sweeps[(sweeps >= window.t0 - 1e-6) & (sweeps <= window.t1 + 1e-6)]
    if len(frames) == 0:
        return None
    cameras = tuple(window.cameras) or tuple(scene.cameras)
    ctx = Context(scene, cfg, frames, pool=lambda cls: pool.get(_concept_for(spec, cls), cls, cameras))
    cache: dict[tuple, list[FrameResult]] = {}

    def run(call: PredicateCall, objs: list[TrackedObject]) -> list[FrameResult]:
        key = (call.name, tuple(o.track.track_id for o in objs), tuple(sorted(call.args.items())))
        if key not in cache:
            cache[key] = PREDICATES[call.name].evaluate(ctx, objs, call.args)
        return cache[key]

    # Candidates by class, pruned by the cheap unary checks.
    candidates: dict[str, list[TrackedObject]] = {}
    relaxed: list[str] = []
    for actor in spec.actors:
        found = [
            o for o in pool.get(actor.sam_prompt, actor.cls, cameras) if _in_window(o, window.t0, window.t1)
        ]
        checks = [
            p for p in spec.predicates if p.required and p.name in PRUNE_WITH and p.actors == [actor.id]
        ]
        kept = [o for o in found if all(any(r.status == "PASS" for r in run(p, [o])) for p in checks)]
        if found and not kept:
            # ASSUMPTION A-89: if pruning would leave an actor with no candidate, keep them all to report why
            kept, relaxed = found, [*relaxed, actor.id]
        ex = [p for p in spec.predicates if p.name == "exists" and p.actors == [actor.id]]
        kept.sort(
            key=lambda o: -sum((r.margin or 0.0) for p in ex for r in run(p, [o]) if r.status == "PASS")
        )
        candidates[actor.id] = kept

    ids = [a.id for a in spec.actors]
    total = math.prod(max(1, len(candidates[i])) for i in ids)
    combos = itertools.product(*[candidates[i] or [None] for i in ids])
    best: tuple[tuple[bool, float], Finding] | None = None
    tried = 0
    cap_hit = False
    for combo in combos:
        tracks = [o for o in combo if o is not None]
        if len({o.track.track_id for o in tracks}) < len(tracks):
            continue  # one track cannot be two actors
        if tried == cfg.engine.max_bindings:
            cap_hit = True
            break
        tried += 1
        finding, judged = _evaluate(spec, window, ctx, dict(zip(ids, combo, strict=True)), run, spec_hash)
        # ASSUMPTION A-85: verified bindings rank by score; unverified first by the number of judged frames
        # (PASS or FAIL), so the evidence shown comes from the track the sensors actually saw
        key = (True, 0, finding.score) if finding.verified else (False, judged, finding.score)
        if best is None or key > best[0]:
            best = (key, finding)
    if best is None:
        best = (None, _evaluate(spec, window, ctx, dict.fromkeys(ids), run, spec_hash)[0])
    finding = best[1]
    # ASSUMPTION A-90: the binding search is recorded in every result's evidence, including a hit cap
    search = {
        "binding_cap_hit": cap_hit,
        "bindings_tried": tried,
        "bindings_total_upper_bound": total,
        "pruning_relaxed_for": relaxed,
    }
    results = [
        r.model_copy(update={"evidence": {**r.evidence, "binding_search": search}}) for r in finding.results
    ]
    return finding.model_copy(update={"results": results})


def _evaluate(
    spec: ScenarioSpec,
    window: CandidateWindow,
    ctx: Context,
    binding: dict[str, TrackedObject | None],
    run: Callable[[PredicateCall, list[TrackedObject]], list[FrameResult]],
    spec_hash: str,
) -> tuple[Finding, int]:
    """The Finding for one binding, and how many required frames were judged (PASS or FAIL)."""
    frames = ctx.frames
    n = len(frames)
    per_call = []
    for call in spec.predicates:
        objs = [binding[a] for a in call.actors]
        if any(o is None for o in objs):
            per_call.append([FrameResult.unverifiable(Reason.NOT_OBSERVED, unbound=True)] * n)
        else:
            per_call.append(run(call, objs))
    required = [i for i, c in enumerate(spec.predicates) if c.required]
    joint = [bool(required) and all(per_call[i][f].status == "PASS" for i in required) for f in range(n)]
    sweeps = ctx.scene.lidar_timestamps()
    dt = float(np.median(np.diff(sweeps))) if len(sweeps) > 1 else 0.1
    a, b = _longest_run(joint)
    # ASSUMPTION A-91: a run of frames lasts (last - first) + one sweep period; it must reach min_duration_s
    is_joint = a is not None and frames[b] - frames[a] + dt >= spec.min_duration_s - 1e-9
    if not is_joint:
        a, b = 0, n - 1
    bound = {k: (o.track.track_id if o is not None else UNBOUND) for k, o in binding.items()}
    finding_id = hashlib.sha256(
        f"{spec_hash}|{window.window_id}|{sorted(bound.items())}".encode()
    ).hexdigest()[:16]
    results = []
    for i, call in enumerate(spec.predicates):
        objs = [binding[x] for x in call.actors]
        thr = (
            PREDICATES[call.name].threshold(ctx, objs, call.args) if all(o is not None for o in objs) else ""
        )
        results.append(
            _summarise(
                f"{finding_id}:{i}",
                call,
                {x: bound[x] for x in call.actors},
                thr,
                per_call[i],
                a,
                b,
                is_joint,
                frames,
                sweeps,
                dt,
                spec.min_duration_s,
            )
        )
    judged = sum(r.status != "UNVERIFIABLE" for i in required for r in per_call[i])
    finding = Finding(
        finding_id=finding_id,
        scene_id=window.scene_id,
        t0=float(frames[a]),
        t1=float(frames[b]),
        spec_hash=spec_hash,
        bindings=bound,
        results=results,
        score=_score(results),
        # ASSUMPTION A-92: the verifier spends no model money; the pipeline fills cost_usd
        cost_usd=0.0,
    )
    return finding, judged


def _longest_run(flags: list[bool]) -> tuple[int | None, int | None]:
    best, start = (None, None), None
    for i, f in enumerate([*flags, False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if best[0] is None or i - 1 - start > best[1] - best[0]:
                best = (start, i - 1)
            start = None
    return best


def _summarise(
    result_id: str,
    call: PredicateCall,
    actors: dict[str, str],
    threshold: str,
    frames_out: list[FrameResult],
    a: int,
    b: int,
    is_joint: bool,
    frames: np.ndarray,
    sweeps: np.ndarray,
    dt: float,
    min_duration_s: float,
) -> PredicateResult:
    """One result for the interval [a, b]: the joint PASS run, or the whole window when there is none."""
    rs = frames_out[a : b + 1]
    statuses = Counter(r.status for r in rs)
    duration = float(frames[b] - frames[a] + dt)
    if statuses["PASS"] == len(rs) and (is_joint or duration >= min_duration_s - 1e-9):
        status, reason, pick = "PASS", Reason.OK, min(rs, key=lambda r: r.margin)
        value, margin = float(np.median([r.value for r in rs])), float(pick.margin)
    elif statuses["PASS"] == len(rs):
        status, reason, pick = "FAIL", Reason.SHORTER_THAN_MIN_DURATION, rs[0]
        value, margin = duration, duration - min_duration_s
    elif statuses["FAIL"]:
        fails = [r for r in rs if r.status == "FAIL"]
        reason = Counter(r.reason for r in fails).most_common(1)[0][0]
        pick = max(
            (r for r in fails if r.reason == reason), key=lambda r: r.margin if r.margin is not None else -1e9
        )
        status = "FAIL"
        vals = [r.value for r in fails if r.reason == reason and r.value is not None]
        value = float(np.median(vals)) if vals else None
        margin = float(pick.margin) if value is not None and pick.margin is not None else None
    else:
        unv = [r for r in rs if r.status == "UNVERIFIABLE"]
        reason = Counter(r.reason for r in unv).most_common(1)[0][0]
        status, pick, value, margin = "UNVERIFIABLE", next(r for r in unv if r.reason == reason), None, None
    if value is not None and margin is None:
        margin = 0.0
    sweep_idx = [int(np.argmin(np.abs(sweeps - t))) for t in frames[a : b + 1]]
    pa, pb = _longest_run([r.status == "PASS" for r in frames_out])
    every = [int(np.argmin(np.abs(sweeps - t))) for t in frames]
    evidence = {
        "reason": reason.value,
        # This predicate's own longest PASS run in the window, even when the others did not hold with it.
        "longest_pass_s": round(float(frames[pb] - frames[pa] + dt), 2) if pa is not None else 0.0,
        "frames": sweep_idx,
        "frame_status": {k: statuses[k] for k in ("PASS", "FAIL", "UNVERIFIABLE")},
        "frame_reasons": {k.value: v for k, v in Counter(r.reason for r in rs).items()},
        "example_frame": sweep_idx[rs.index(pick)],
        "example": pick.detail,
        # ASSUMPTION A-102: every window frame's verdict, [sweep index, status, reason], for the C9 timeline
        "timeline": [[i, r.status, r.reason.value] for i, r in zip(every, frames_out, strict=True)],
    }
    if value is not None:
        # ASSUMPTION A-08: every result with a value carries a numeric margin
        evidence["margin"] = margin
    return PredicateResult(
        result_id=result_id,
        name=call.name,
        actors=actors,
        status=status,
        value=value,
        threshold=threshold,
        evidence=evidence,
        required=call.required,
    )


def _score(results: list[PredicateResult]) -> float:
    """Mean over required results: tanh(margin) for PASS, -0.5 for UNVERIFIABLE, -1 for FAIL."""
    # ASSUMPTION A-85: score from the margins; verified findings rank first regardless (verify() sorts)
    parts = []
    for r in results:
        if not r.required:
            continue
        if r.status == "PASS":
            parts.append(math.tanh(r.evidence.get("margin", 0.0)))
        else:
            parts.append(-1.0 if r.status == "FAIL" else -0.5)
    return float(np.mean(parts)) if parts else 0.0
