"""exists(a): a foreground cluster with enough points that fits the class size prior (plan C6)."""

from typing import Any

from longtail.geometry.tracks3d import TrackedObject
from longtail.verify.predicates.base import Context, FrameResult, Reason, horizontal_range, missing, sample_at


def threshold(ctx: Context, objs: list[TrackedObject], args: dict[str, Any]) -> str:
    [obj] = objs
    c = ctx.cfg.predicates.exists
    p = c.size_priors[obj.track.cls]
    return (
        f">= {c.min_points} points (>= {c.min_points_far} beyond {c.far_m:g} m), within {c.max_range_m:g} m; "
        f"{obj.track.cls}: height {p.height_m[0]:g}-{p.height_m[1]:g} m, footprint {p.footprint_m[0]:g}-"
        f"{p.footprint_m[1]:g} m"
    )


def evaluate(ctx: Context, objs: list[TrackedObject], args: dict[str, Any]) -> list[FrameResult]:
    [obj] = objs
    c = ctx.cfg.predicates.exists
    # ASSUMPTION A-76: per-class size priors
    prior = c.size_priors[obj.track.cls]
    out = []
    for ts in ctx.frames:
        s = sample_at(obj, ts)
        if s is None:
            out.append(missing(ctx, obj, ts))
            continue
        rng = horizontal_range(s)
        detail = {
            "n_points": s.n_points,
            "coverage": s.coverage,
            "range_m": round(rng, 2),
            "size_lwh_m": [round(v, 2) for v in s.extent_lwh],
        }
        if rng > c.max_range_m:
            out.append(FrameResult.unverifiable(Reason.OUT_OF_RANGE, **detail))
            continue
        need = c.min_points_far if rng > c.far_m else c.min_points
        points_margin = (s.n_points - need) / need
        if s.n_points < need:
            out.append(FrameResult("FAIL", Reason.TOO_FEW_POINTS, float(s.n_points), points_margin, detail))
            continue
        # A-53: height is measured from the local ground, so the 0.2 m ground cut does not shrink it.
        length, width, height = s.extent_lwh
        (h_lo, h_hi), (f_lo, f_hi) = prior.height_m, prior.footprint_m
        size_margin = min(height - h_lo, h_hi - height, length - f_lo, f_hi - length, f_hi - width)
        if size_margin < 0:
            out.append(FrameResult("FAIL", Reason.WRONG_SIZE, float(s.n_points), size_margin, detail))
            continue
        out.append(FrameResult("PASS", Reason.OK, float(s.n_points), min(size_margin, points_margin), detail))
    return out
