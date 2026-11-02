"""crossing_ego_path(a): the actor enters the ego's corridor ahead, moving toward it (plan C6)."""

from typing import Any

import numpy as np

from longtail.geometry.tracks3d import TrackedObject
from longtail.geometry.transforms import world_to_ego
from longtail.verify.predicates import _motion
from longtail.verify.predicates.base import Context, FrameResult, Reason, missing


def threshold(ctx: Context, objs: list[TrackedObject], args: dict[str, Any]) -> str:
    c, e = ctx.cfg.predicates.crossing, ctx.cfg.ego_path
    return (
        f"enters the ego corridor (half-width {e.half_width_m:g} m, >= {e.min_length_m:g} m long) "
        f"0-{c.max_ahead_m:g} m ahead within {e.horizon_s:g} s, "
        f"lateral speed toward it > {c.min_lateral_mps:g} m/s"
    )


def evaluate(ctx: Context, objs: list[TrackedObject], args: dict[str, Any]) -> list[FrameResult]:
    [obj] = objs
    c, horizon = ctx.cfg.predicates.crossing, ctx.cfg.ego_path.horizon_s
    m = obj.motion
    series = _motion.of_object(obj)
    jump = _motion.jumps(series, ctx.cfg.predicates.motion)
    out = []
    for ts in ctx.frames:
        hit = np.flatnonzero(np.abs(m.ts - ts) < 1e-6)
        if len(hit) == 0:
            out.append(missing(ctx, obj, ts))
            continue
        k = int(hit[0])
        if not m.reliable[k]:
            out.append(FrameResult.unverifiable(Reason.UNRELIABLE_SPEED))
            continue
        seg = np.flatnonzero(m.segment == m.segment[k])
        future = seg[(m.ts[seg] >= ts - 1e-6) & (m.ts[seg] <= ts + horizon + 1e-6)]
        if jump[future].any():
            out.append(FrameResult.unverifiable(Reason.IMPLAUSIBLE_JUMP))
            continue
        lane = ctx.corridor(ts)
        xy = world_to_ego(ctx.scene.ego_pose(ts), m.smoothed_world[future])[:, :2]
        where = lane.locate(xy)
        inside = where.inside & (where.along_m <= c.max_ahead_m)
        t0 = where.tangent[0]
        lateral_v = float(m.velocity_ego[k, :2] @ np.array([-t0[1], t0[0]]))  # positive = toward the left
        detail = {
            "lateral_m": round(float(where.lateral_m[0]), 2),
            "along_m": round(float(where.along_m[0]), 2),
            "lateral_speed_mps": round(lateral_v, 3),
            "corridor_extended": lane.extended,
        }
        if inside[0]:
            # ASSUMPTION A-81: an actor already inside the corridor that moves across it counts as crossing
            value = abs(lateral_v)
            ok = value > c.min_lateral_mps
            out.append(
                FrameResult(
                    "PASS" if ok else "FAIL",
                    Reason.OK if ok else Reason.NOT_TOWARD_CORRIDOR,
                    value,
                    value - c.min_lateral_mps,
                    {**detail, "already_inside": True},
                )
            )
            continue
        toward = -np.sign(where.lateral_m[0]) * lateral_v
        entered = np.flatnonzero(inside)
        if entered.size:
            ok = toward > c.min_lateral_mps
            detail["enters_after_s"] = round(float(m.ts[future[entered[0]]] - ts), 2)
            out.append(
                FrameResult(
                    "PASS" if ok else "FAIL",
                    Reason.OK if ok else Reason.NOT_TOWARD_CORRIDOR,
                    toward,
                    toward - c.min_lateral_mps,
                    detail,
                )
            )
        elif m.ts[future[-1]] - ts >= horizon - 1e-6:
            out.append(
                FrameResult("FAIL", Reason.OUTSIDE_CORRIDOR, toward, toward - c.min_lateral_mps, detail)
            )
        else:
            # The observed track ends before the look-ahead does, without having entered.
            out.append(FrameResult.unverifiable(Reason.TRACK_TOO_SHORT, **detail))
    return out
