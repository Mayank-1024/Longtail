"""ego_moving: the ego's net speed over 1 s, same thresholds and hysteresis as moving (A-06)."""

from typing import Any

from longtail.geometry.tracks3d import TrackedObject
from longtail.verify.predicates import _motion
from longtail.verify.predicates.base import Context, FrameResult


def threshold(ctx: Context, objs: list[TrackedObject], args: dict[str, Any]) -> str:
    return "ego " + _motion.threshold(ctx, "moving")


def evaluate(ctx: Context, objs: list[TrackedObject], args: dict[str, Any]) -> list[FrameResult]:
    # ASSUMPTION A-06: ego-state predicates take no actors
    assert not objs
    return _motion.frames(ctx, _motion.of_ego(ctx.ego), "moving", None)
