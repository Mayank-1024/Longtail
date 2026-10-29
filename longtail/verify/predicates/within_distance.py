"""within_distance(a, m): the actor's range from the ego is at most m, with a tolerance (plan C6)."""

from typing import Any

from longtail.geometry.tracks3d import TrackedObject
from longtail.verify.predicates import exists
from longtail.verify.predicates.base import Context, FrameResult, Reason, horizontal_range, sample_at


def threshold(ctx: Context, objs: list[TrackedObject], args: dict[str, Any]) -> str:
    return f"range <= {args['m']:g} m (+{ctx.cfg.predicates.within_distance.tolerance_m:g} m tolerance)"


def evaluate(ctx: Context, objs: list[TrackedObject], args: dict[str, Any]) -> list[FrameResult]:
    [obj] = objs
    limit = float(args["m"]) + ctx.cfg.predicates.within_distance.tolerance_m
    out = []
    for ts, ex in zip(ctx.frames, exists.evaluate(ctx, objs, {}), strict=True):
        if ex.status != "PASS":
            out.append(FrameResult.unverifiable(Reason.REQUIRES_EXISTS, exists=ex.reason.value))
            continue
        rng = horizontal_range(sample_at(obj, ts))
        status, reason = ("PASS", Reason.OK) if rng <= limit else ("FAIL", Reason.TOO_FAR)
        out.append(FrameResult(status, reason, rng, limit - rng, {"range_m": round(rng, 2)}))
    return out
