"""relative_position(a, side): the actor is left, right, ahead or behind in the ego frame (plan C6)."""

from typing import Any

import numpy as np

from longtail.geometry.tracks3d import TrackedObject
from longtail.verify.predicates.base import Context, FrameResult, Reason, missing, sample_at

# ASSUMPTION A-82: each side is a half-plane of the ego frame (ahead-left is both ahead and left)
SIDES = {"left": (1, 1.0), "right": (1, -1.0), "ahead": (0, 1.0), "behind": (0, -1.0)}


def threshold(ctx: Context, objs: list[TrackedObject], args: dict[str, Any]) -> str:
    c = ctx.cfg.predicates.relative_position
    return f"{args['side']} of the ego in at least {c.min_frac:.0%} of a {c.window_s:g} s window"


def evaluate(ctx: Context, objs: list[TrackedObject], args: dict[str, Any]) -> list[FrameResult]:
    [obj] = objs
    c = ctx.cfg.predicates.relative_position
    axis, sign = SIDES[args["side"]]
    times = np.array([s.ts for s in obj.track.samples])
    coord = np.array([sign * s.xyz_ego[axis] for s in obj.track.samples])
    out = []
    for ts in ctx.frames:
        if sample_at(obj, ts) is None:
            out.append(missing(ctx, obj, ts))
            continue
        # ASSUMPTION A-78: "at least half of the window" over a window_s window centred on the frame
        near = np.abs(times - ts) <= c.window_s / 2 + 1e-6
        frac = float(np.mean(coord[near] > 0))
        value = float(np.median(coord[near]))
        ok = frac >= c.min_frac
        detail = {"fraction_on_side": round(frac, 2), "samples": int(near.sum())}
        out.append(
            FrameResult(
                "PASS" if ok else "FAIL", Reason.OK if ok else Reason.WRONG_SIDE, value, value, detail
            )
        )
    return out
