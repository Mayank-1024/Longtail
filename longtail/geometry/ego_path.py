"""The ego's own motion and its future path corridor (plan C5).

Logs are offline, so the ego's actual future poses are known. For a time t, the path over the next
horizon_s, written in the ego frame at t, is the centre line of a corridor of half-width half_width_m.
"""

from dataclasses import dataclass

import numpy as np

from longtail.data.base import Scene
from longtail.geometry.transforms import world_to_ego
from longtail.verify.config import EgoPathConfig, load_verify_config


@dataclass(frozen=True, eq=False)
class EgoMotion:
    """Ego speed over the scene, on the sweep timestamps. ego_moving and ego_stopped (A-06) read this."""

    ts: np.ndarray
    position_world: np.ndarray  # (N, 3)
    velocity_world: np.ndarray  # (N, 3)
    speed_mps: np.ndarray  # (N,) horizontal


def ego_motion(scene: Scene) -> EgoMotion:
    ts = scene.lidar_timestamps()
    pos = np.array([scene.ego_pose(t)[:3, 3] for t in ts])
    vel = np.gradient(pos, ts, axis=0) if len(ts) >= 2 else np.zeros_like(pos)
    return EgoMotion(
        ts=ts, position_world=pos, velocity_world=vel, speed_mps=np.linalg.norm(vel[:, :2], axis=1)
    )


@dataclass(frozen=True)
class CorridorPosition:
    lateral_m: np.ndarray  # signed distance from the centre line, positive to its left
    along_m: np.ndarray  # distance along the centre line to the nearest point
    inside: np.ndarray  # within half-width, and not beyond either end
    tangent: np.ndarray  # (N, 2) unit direction of the centre line at the nearest point


@dataclass(frozen=True, eq=False)
class Corridor:
    ts: float
    centerline: np.ndarray  # (M, 2) in the ego frame at ts, starting at the ego origin
    half_width_m: float
    extended: bool  # the path was shorter than min_length_m and was extended straight ahead
    future_truncated: bool  # the log ends before ts + horizon_s

    @property
    def length_m(self) -> float:
        return float(np.sum(np.linalg.norm(np.diff(self.centerline, axis=0), axis=1)))

    def locate(self, xy: np.ndarray) -> CorridorPosition:
        """Where (N, 2) ego-frame points are relative to the corridor."""
        xy = np.atleast_2d(xy)
        a, b = self.centerline[:-1], self.centerline[1:]
        seg = b - a
        seg_len = np.linalg.norm(seg, axis=1)
        tangent = seg / seg_len[:, None]
        rel = xy[:, None, :] - a[None]  # (N, S, 2)
        u_raw = np.einsum("nsk,sk->ns", rel, tangent)
        u = np.clip(u_raw, 0, seg_len)
        foot = a[None] + u[..., None] * tangent[None]
        dist = np.linalg.norm(xy[:, None, :] - foot, axis=2)
        best = np.argmin(dist, axis=1)
        n = np.arange(len(xy))
        t = tangent[best]
        off = xy - foot[n, best]
        lateral = np.sign(t[:, 0] * off[:, 1] - t[:, 1] * off[:, 0]) * dist[n, best]
        along = np.concatenate([[0.0], np.cumsum(seg_len)])[best] + u[n, best]
        beyond = ((best == 0) & (u_raw[n, best] < 0)) | (
            (best == len(seg) - 1) & (u_raw[n, best] > seg_len[best])
        )
        return CorridorPosition(
            lateral_m=lateral, along_m=along, inside=(dist[n, best] <= self.half_width_m) & ~beyond, tangent=t
        )


def corridor(scene: Scene, ts: float, cfg: EgoPathConfig | None = None) -> Corridor:
    """The ego's corridor at a LiDAR timestamp (A-29)."""
    cfg = cfg or load_verify_config().ego_path
    sweeps = scene.lidar_timestamps()
    future = sweeps[(sweeps >= ts - 1e-6) & (sweeps <= ts + cfg.horizon_s + 1e-6)]
    here = scene.ego_pose(ts)
    path = world_to_ego(here, np.array([scene.ego_pose(t)[:3, 3] for t in future]))[:, :2]
    keep = np.concatenate([[True], np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-6])
    path = path[keep]
    length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    extended = length < cfg.min_length_m
    if extended:
        # ASSUMPTION A-69: extend straight ahead along the current heading (ego +x) to min_length_m
        path = np.vstack([path, path[-1] + [cfg.min_length_m - length, 0.0]])
    if len(path) == 1:  # stopped with min_length_m == 0
        path = np.vstack([path, path])
    return Corridor(
        ts=float(ts),
        centerline=path,
        half_width_m=cfg.half_width_m,
        extended=extended,
        # ASSUMPTION A-72: near the end of the log the corridor uses the future that exists, and says so
        future_truncated=bool(sweeps[-1] < ts + cfg.horizon_s - 1e-6),
    )
