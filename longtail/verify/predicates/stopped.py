"""stopped(a): net speed over 1 s below 0.3 m/s, with hysteresis (plan C6)."""

from typing import Any

from longtail.geometry.tracks3d import TrackedObject
from longtail.verify.predicates import _motion
from longtail.verify.predicates.base import Context, FrameResult


def threshold(ctx: Context, objs: list[TrackedObject], args: dict[str, Any]) -> str:
    return _motion.threshold(ctx, "stopped")


def evaluate(ctx: Context, objs: list[TrackedObject], args: dict[str, Any]) -> list[FrameResult]:
    [obj] = objs
    return _motion.frames(ctx, _motion.of_object(obj), "stopped", obj)
