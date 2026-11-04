"""Shared by moving, stopped, ego_moving and ego_stopped: net speed over a window, with hysteresis."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from longtail.geometry.ego_path import EgoMotion
from longtail.geometry.tracks3d import TrackedObject
from longtail.verify.config import MotionConfig
from longtail.verify.predicates.base import Context, FrameResult, Reason, missing

State = Literal["moving", "stopped"]


@dataclass(frozen=True, eq=False)
class Series:
    ts: np.ndarray
    position: np.ndarray  # (G, 3) world
    velocity: np.ndarray  # (G, 3) world
    segment: np.ndarray
    reliable: np.ndarray


def of_object(obj: TrackedObject) -> Series:
    m = obj.motion
    return Series(m.ts, m.smoothed_world, m.velocity_world, m.segment, m.reliable)


def of_ego(ego: EgoMotion) -> Series:
    n = len(ego.ts)
    return Series(ego.ts, ego.position_world, ego.velocity_world, np.zeros(n, int), np.ones(n, bool))


@dataclass(frozen=True)
class Step:
    state: State | None
    net_mps: float | None
    reason: Reason | None  # set when the step cannot be judged
    window: tuple[float, float] | None = None  # the timestamps the verdict was judged on


def jumps(series: Series, cfg: MotionConfig) -> np.ndarray:
    """Steps at either end of an acceleration above max_accel_mps2, within each speed series."""
    # ASSUMPTION A-77: both ends of a step whose acceleration exceeds max_accel_mps2 are a position jump
    out = np.zeros(len(series.ts), bool)
    for seg in np.unique(series.segment):
        idx = np.flatnonzero(series.segment == seg)
        if len(idx) < 2:
            continue
        accel = np.linalg.norm(np.diff(series.velocity[idx, :2], axis=0), axis=1) / np.diff(series.ts[idx])
        out[idx[:-1]] |= accel > cfg.max_accel_mps2
        out[idx[1:]] |= accel > cfg.max_accel_mps2
    return out


def steps(series: Series, cfg: MotionConfig) -> list[Step]:
    """Moving or stopped per step, judged by net displacement over window_s, with hysteresis."""
    out: list[Step] = [Step(None, None, Reason.NOT_OBSERVED)] * len(series.ts)
    all_jumps = jumps(series, cfg)
    for seg in np.unique(series.segment):
        idx = np.flatnonzero(series.segment == seg)
        t = series.ts[idx]
        if t[-1] - t[0] < cfg.window_s - 1e-6:
            for k in idx:
                out[k] = Step(None, None, Reason.TRACK_TOO_SHORT)
            continue
        jump = all_jumps[idx]
        state: State | None = None
        for j, k in enumerate(idx):
            if not series.reliable[k]:
                out[k] = Step(None, None, Reason.UNRELIABLE_SPEED)
                continue
            # ASSUMPTION A-94: each step is judged on a full window_s of clean steps, centred on it but
            # shifted to stay inside its series, so steps near a series end share one window; the finding's
            # run of such steps need only reach min_duration_s
            start = min(max(t[j] - cfg.window_s / 2, t[0]), t[-1] - cfg.window_s)
            a = int(np.searchsorted(t, start - 1e-6))
            b = int(np.searchsorted(t, start + cfg.window_s + 1e-6)) - 1
            # ASSUMPTION A-247: (v2) drop_jump_windows false judges the window; outliers were dropped upstream
            if cfg.drop_jump_windows is not False and jump[a : b + 1].any():
                out[k] = Step(None, None, Reason.IMPLAUSIBLE_JUMP)
                continue
            p = series.position[idx]
            net = float(np.linalg.norm(p[b, :2] - p[a, :2]) / (t[b] - t[a]))
            if state is None:
                # ASSUMPTION A-80: the first judged step starts in the state nearer its net speed
                state = "moving" if net >= (cfg.moving_mps + cfg.stopped_mps) / 2 else "stopped"
            elif net > cfg.moving_mps:
                state = "moving"
            elif net < cfg.stopped_mps:
                state = "stopped"
            out[k] = Step(state, net, None, (float(t[a]), float(t[b])))
    return out


def frames(ctx: Context, series: Series, want: State, obj: TrackedObject | None) -> list[FrameResult]:
    """FrameResults for moving (want="moving") or stopped, for an object or (obj None) the ego."""
    cfg = ctx.cfg.predicates.motion
    judged = steps(series, cfg)
    out = []
    for ts in ctx.frames:
        hit = np.flatnonzero(np.abs(series.ts - ts) < 1e-6)
        if len(hit) == 0:
            out.append(
                missing(ctx, obj, ts) if obj is not None else FrameResult.unverifiable(Reason.NOT_OBSERVED)
            )
            continue
        s = judged[hit[0]]
        if s.state is None:
            out.append(FrameResult.unverifiable(s.reason))
            continue
        detail = {"net_speed_mps": round(s.net_mps, 3), "state": s.state, "judged_window": list(s.window)}
        if want == "moving":
            ok = s.state == "moving"
            margin = s.net_mps - (cfg.stopped_mps if ok else cfg.moving_mps)
            out.append(
                FrameResult(
                    "PASS" if ok else "FAIL",
                    Reason.OK if ok else Reason.NOT_MOVING,
                    s.net_mps,
                    margin,
                    detail,
                )
            )
        else:
            ok = s.state == "stopped"
            margin = (cfg.moving_mps if ok else cfg.stopped_mps) - s.net_mps
            out.append(
                FrameResult(
                    "PASS" if ok else "FAIL",
                    Reason.OK if ok else Reason.NOT_STOPPED,
                    s.net_mps,
                    margin,
                    detail,
                )
            )
    return out


def threshold(ctx: Context, want: State) -> str:
    c = ctx.cfg.predicates.motion
    if want == "moving":
        return (
            f"net speed over {c.window_s:g} s > {c.moving_mps:g} m/s (stays moving until < {c.stopped_mps:g})"
        )
    return f"net speed over {c.window_s:g} s < {c.stopped_mps:g} m/s (stays stopped until > {c.moving_mps:g})"
