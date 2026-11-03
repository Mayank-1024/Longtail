"""Shared types for predicates: reason codes, per-frame results and the evaluation context.

A predicate is evaluated per frame (each LiDAR sweep in the window) and returns a FrameResult. The engine
turns the frames into one PredicateResult (contracts.py) for the window.
"""

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

import numpy as np

from longtail.contracts import Sample3D
from longtail.data.base import Scene
from longtail.geometry.ego_path import Corridor, EgoMotion, corridor, ego_motion
from longtail.geometry.sync import nearest_sweeps
from longtail.geometry.tracks3d import TrackedObject
from longtail.verify.config import VerifyConfig

Status = Literal["PASS", "FAIL", "UNVERIFIABLE"]


class Reason(StrEnum):
    """Why a frame got its status; shown on the card and in the web view."""

    # ASSUMPTION A-79: reason codes; each non-PASS status carries exactly one
    OK = "ok"
    # Not enough sensor evidence (UNVERIFIABLE)
    NO_COVERAGE = "no_coverage"  # fewer than min_coverage LiDAR points in the mask's box
    OUT_OF_RANGE = "out_of_range"  # beyond max_range_m
    UNSYNCED = "unsynced"  # camera frame over 50 ms from its sweep (C2)
    TRUNCATED = "truncated"  # object missing and SAM hit its 16-object cap
    NOT_OBSERVED = "not_observed"  # the bound track has no observation at this frame
    TRACK_TOO_SHORT = "track_too_short"  # under the 1 s needed, or ends before the look-ahead
    UNRELIABLE_SPEED = "unreliable_speed"  # speed series too short to smooth (A-71)
    IMPLAUSIBLE_JUMP = "implausible_jump"  # position jumps faster than max_accel_mps2 allows
    REQUIRES_EXISTS = "requires_exists"  # depends on exists, which did not PASS
    # Evidence against the claim (FAIL)
    WRONG_SIZE = "wrong_size"
    TOO_FEW_POINTS = "too_few_points"
    NO_OBJECT_POINTS = "no_object_points"  # coverage was fine but only ground or background remained
    TOO_FAR = "too_far"
    NOT_MOVING = "not_moving"
    NOT_STOPPED = "not_stopped"
    OUTSIDE_CORRIDOR = "outside_corridor"
    NOT_TOWARD_CORRIDOR = "not_toward_corridor"
    WRONG_SIDE = "wrong_side"
    NOT_NEAR = "not_near"
    NOT_OCCLUDED = "not_occluded"
    SHORTER_THAN_MIN_DURATION = "shorter_than_min_duration"


UNVERIFIABLE_REASONS = {
    Reason.NO_COVERAGE,
    Reason.OUT_OF_RANGE,
    Reason.UNSYNCED,
    Reason.TRUNCATED,
    Reason.NOT_OBSERVED,
    Reason.TRACK_TOO_SHORT,
    Reason.UNRELIABLE_SPEED,
    Reason.IMPLAUSIBLE_JUMP,
    Reason.REQUIRES_EXISTS,
}


@dataclass(frozen=True)
class FrameResult:
    status: Status
    reason: Reason
    value: float | None = None
    margin: float | None = None  # how far value clears the threshold; negative when it does not (A-08)
    detail: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def unverifiable(reason: Reason, **detail: Any) -> "FrameResult":
        assert reason in UNVERIFIABLE_REASONS, reason
        return FrameResult("UNVERIFIABLE", reason, detail=detail)


@dataclass(eq=False)
class Context:
    """Everything a predicate needs for one window. Built by the engine."""

    scene: Scene
    cfg: VerifyConfig
    frames: np.ndarray  # sweep timestamps in the window
    # Tracked objects of a class, for occluded_by; filled lazily by the engine.
    pool: Callable[[str], list[TrackedObject]] = field(default=lambda cls: [])
    # ASSUMPTION A-120: predicates replaced by name (C7 label geometry); empty in the verifier
    overrides: dict[str, Callable[["Context", list[TrackedObject], dict[str, Any]], list[FrameResult]]] = (
        field(default_factory=dict)
    )
    _ego: EgoMotion | None = None
    _corridors: dict[float, Corridor] = field(default_factory=dict)
    _skips: dict[int, dict[float, str]] = field(default_factory=dict)

    def evaluate(self, name: str, objs: list[TrackedObject], args: dict[str, Any]) -> list[FrameResult]:
        """One predicate's per-frame results, through an override when one is set."""
        if name in self.overrides:
            return self.overrides[name](self, objs, args)
        return importlib.import_module(f"longtail.verify.predicates.{name}").evaluate(self, objs, args)

    @property
    def ego(self) -> EgoMotion:
        if self._ego is None:
            self._ego = ego_motion(self.scene)
        return self._ego

    def corridor(self, ts: float) -> Corridor:
        if ts not in self._corridors:
            self._corridors[ts] = corridor(self.scene, ts, self.cfg.ego_path)
        return self._corridors[ts]

    def skip_reason(self, obj: TrackedObject, ts: float) -> str | None:
        """C4's reason for a frame without a sample (A-46), placed on the nearest sweep."""
        key = id(obj)
        if key not in self._skips:
            sweeps = self.scene.lidar_timestamps()
            # Coverage-was-fine reasons outrank no-coverage, which outranks unsynced.
            rank = {"no_points": 3, "no_cluster": 3, "low_coverage": 2, "unsynced": 1}
            by_sweep: dict[float, str] = {}
            skips = obj.association.skipped
            if skips:
                idx = nearest_sweeps(np.array([s.ts for s in skips]), sweeps)
                for s, i in zip(skips, idx, strict=True):
                    t = float(sweeps[i])
                    if rank[s.reason] > rank.get(by_sweep.get(t, ""), 0):
                        by_sweep[t] = s.reason
            self._skips[key] = by_sweep
        return self._skips[key].get(float(ts))


def sample_at(obj: TrackedObject, ts: float) -> Sample3D | None:
    for s in obj.track.samples:
        if abs(s.ts - ts) < 1e-6:
            return s
    return None


def missing(ctx: Context, obj: TrackedObject, ts: float) -> FrameResult:
    """The FrameResult for a frame where the object has no sample, from C4's skip reason (A-46)."""
    why = ctx.skip_reason(obj, ts)
    if why == "unsynced":
        return FrameResult.unverifiable(Reason.UNSYNCED)
    if why == "low_coverage":
        return FrameResult.unverifiable(Reason.NO_COVERAGE)
    if why in ("no_points", "no_cluster"):
        # Coverage was fine, but nothing survived ground removal and clustering: evidence against the object.
        return FrameResult("FAIL", Reason.NO_OBJECT_POINTS, value=0.0, margin=-1.0)
    if obj.association.truncated:
        return FrameResult.unverifiable(Reason.TRUNCATED)
    return FrameResult.unverifiable(Reason.NOT_OBSERVED)


def horizontal_range(sample: Sample3D) -> float:
    return float(np.hypot(sample.xyz_ego[0], sample.xyz_ego[1]))
