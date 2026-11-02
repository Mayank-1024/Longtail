"""near(a, b, m): the two actors are at most m apart (plan C6)."""

from typing import Any

import numpy as np

from longtail.geometry.tracks3d import TrackedObject
from longtail.verify.predicates.base import Context, FrameResult, Reason, missing, sample_at


def threshold(ctx: Context, objs: list[TrackedObject], args: dict[str, Any]) -> str:
    return f"horizontal distance between the actors <= {args['m']:g} m"


def evaluate(ctx: Context, objs: list[TrackedObject], args: dict[str, Any]) -> list[FrameResult]:
    a, b = objs
    m = float(args["m"])
    out = []
    for ts in ctx.frames:
        sa, sb = sample_at(a, ts), sample_at(b, ts)
        if sa is None or sb is None:
            miss = missing(ctx, a if sa is None else b, ts)
            # A missing actor is no evidence about the distance.
            out.append(
                miss if miss.status == "UNVERIFIABLE" else FrameResult.unverifiable(Reason.NOT_OBSERVED)
            )
            continue
        # ASSUMPTION A-83: distance between the two sample centroids (C4 medians), horizontally
        d = float(np.hypot(sa.xyz_world[0] - sb.xyz_world[0], sa.xyz_world[1] - sb.xyz_world[1]))
        ok = d <= m
        out.append(
            FrameResult(
                "PASS" if ok else "FAIL",
                Reason.OK if ok else Reason.NOT_NEAR,
                d,
                m - d,
                {"distance_m": round(d, 2)},
            )
        )
    return out
