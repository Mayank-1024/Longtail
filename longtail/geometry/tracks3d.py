"""3D tracks over time (plan C5): smoothed world positions, velocities and speeds.

Speeds come from finite differences of world positions, so the ego's own motion cancels out. Velocities are
also given in the ego frame (world velocity in ego axes), which C6's crossing_ego_path needs.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

from longtail.contracts import Track3D
from longtail.data.base import Scene
from longtail.geometry.associate import Association
from longtail.geometry.project import project_sweep
from longtail.verify.config import TracksConfig, load_verify_config


@dataclass(frozen=True, eq=False)
class Motion:
    """Per time step on the sweep grid. Interpolated steps fill gaps of up to max_interp_gap_s."""

    ts: np.ndarray  # (G,) sweep timestamps
    observed: np.ndarray  # (G,) False where the position was interpolated
    segment: np.ndarray  # (G,) speed series id; a long gap starts a new one
    position_world: np.ndarray  # (G, 3) before smoothing
    smoothed_world: np.ndarray  # (G, 3)
    velocity_world: np.ndarray  # (G, 3) m/s
    # ASSUMPTION A-70: ego-frame velocity is the world velocity in the ego's axes at that time, not relative
    velocity_ego: np.ndarray  # (G, 3) m/s
    speed_mps: np.ndarray  # (G,) horizontal
    # ASSUMPTION A-71: a step is unreliable when its series is too short to smooth (under the window)
    reliable: np.ndarray  # (G,)

    def index(self, ts: float) -> int:
        i = int(np.argmin(np.abs(self.ts - ts)))
        if abs(self.ts[i] - ts) > 1e-6:
            raise KeyError(f"no motion step at ts={ts}")
        return i


@dataclass(eq=False)
class TrackedObject:
    """C5 output. track is the contract Track3D with speed_mps filled, one entry per sample."""

    # ASSUMPTION A-70: velocity vectors and reliability ride on this wrapper; Track3D only has speed_mps
    association: Association
    track: Track3D
    motion: Motion


# Positions


def lshape_yaw(xy: np.ndarray, step_deg: float, min_distance_m: float) -> float:
    """Rectangle orientation in [0, pi/2), closeness criterion of L-shape fitting (Zhang et al. 2017)."""
    angles = np.radians(np.arange(0.0, 90.0, step_deg))
    c1 = xy @ np.stack([np.cos(angles), np.sin(angles)])
    c2 = xy @ np.stack([-np.sin(angles), np.cos(angles)])
    d1 = np.minimum(c1.max(axis=0) - c1, c1 - c1.min(axis=0))
    d2 = np.minimum(c2.max(axis=0) - c2, c2 - c2.min(axis=0))
    score = np.sum(1.0 / np.maximum(np.minimum(d1, d2), min_distance_m), axis=0)
    return float(angles[np.argmax(score)])


def _rot2(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


@dataclass(frozen=True)
class _View:
    clean: np.ndarray  # (2, 2) bool: is each end (axis, lo/hi) a real object boundary in some camera
    margin_m: float  # how far erosion and point spacing can move an observed end, in metres at this range


def _view(
    scene: Scene,
    ts: float,
    clusters: list,
    rows: np.ndarray,
    local: np.ndarray,
    cfg: TracksConfig,
    erode_px: int,
) -> _View:
    """Which ends of the cluster are real, and the erosion margin, from every camera that sees it.

    An end is truncated when most of its points touch the image border or sit next to nearer points
    (an occluder). Distances in pixels grow with the cluster's own point spacing in the image.
    """
    # ASSUMPTION A-73: truncation = image border, or a point over occluder_margin_m nearer, within a distance
    # that grows with the pixel spacing; an end is real when half its points are clean in one camera
    sweep = scene.lidar(ts)
    clean = np.zeros((2, 2), dtype=bool)
    margin = 0.0
    lo, hi = local.min(axis=0), local.max(axis=0)
    for c in clusters:
        frame = scene.frames(c.camera)[c.frame_idx]
        proj = project_sweep(scene, sweep, c.camera, frame)
        w, h = scene.image_size(c.camera)
        pix, depth = proj.pixels(), proj.depth
        row_to_proj = {r: i for i, r in enumerate(proj.index)}
        own = np.isin(rows, c.point_index)
        own_idx = [row_to_proj[r] for r in rows[own] if r in row_to_proj]
        spacing = 0.0
        if len(own_idx) >= 2:
            spacing = float(np.median(cKDTree(pix[own_idx]).query(pix[own_idx], k=2)[0][:, 1]))
        if own_idx:
            metres_per_px = float(np.median(depth[own_idx])) / scene.intrinsics(c.camera)[0, 0]
            margin = max(margin, (erode_px + cfg.spacing_factor * spacing) * metres_per_px)
        border = max(cfg.truncation_border_px, cfg.spacing_factor * spacing)
        radius = max(cfg.occluder_radius_px, cfg.spacing_factor * spacing)
        others = ~np.isin(proj.index, rows)
        tree = cKDTree(pix[others]) if others.any() else None
        other_depth = depth[others]
        for k in range(2):
            for side, edge in enumerate((lo[k], hi[k])):
                band = own & (np.abs(local[:, k] - edge) <= cfg.edge_band_m)
                idx = [row_to_proj[r] for r in rows[band] if r in row_to_proj]
                if not idx:
                    continue
                p, d = pix[idx], depth[idx]
                ok = (
                    (p[:, 0] >= border)
                    & (p[:, 0] < w - border)
                    & (p[:, 1] >= border)
                    & (p[:, 1] < h - border)
                )
                if tree is not None:
                    for j, near in enumerate(tree.query_ball_point(p, radius)):
                        if near and other_depth[near].min() < d[j] - cfg.occluder_margin_m:
                            ok[j] = False
                clean[k, side] |= ok.mean() >= 0.5
    return _View(clean=clean, margin_m=margin)


def box_centres(scene: Scene, assoc: Association, cfg: TracksConfig) -> np.ndarray:
    """Per-sample centre of an L-shape box fit; rows are NaN where the centre cannot be observed.

    Along each box axis: when the sensor is clearly outside the cluster's extent, the face toward it is
    visible and its plane does not move with erosion or truncation, so the centre is that face plus half the
    object's size. Otherwise the visible face spans the axis, and its real ends (not cut by the image border
    or an occluder) place the centre. The size per axis is the largest seen over the track (logs are offline).
    """
    # ASSUMPTION A-68: the position used for speed is this face-anchored, truncation-aware box centre
    erode_px = load_verify_config().association.erode_px
    samples = assoc.track.samples
    clouds, rowsets = [], []
    for s in samples:
        rows = np.unique(np.concatenate([c.point_index for c in assoc.clusters[s.ts]]))
        rowsets.append(rows)
        clouds.append(scene.lidar(s.ts).points[rows, :3])
    yaws = [lshape_yaw(c[:, :2] - c[:, :2].mean(axis=0), cfg.lshape_angle_step_deg, cfg.lshape_min_distance_m)
            for c in clouds]  # fmt: skip
    ref = yaws[int(np.argmax([len(c) for c in clouds]))]
    # Box axes are defined modulo 90 degrees; align every frame's axes to the reference frame's.
    yaws = [y + np.pi / 2 * np.round(((ref - y + np.pi) % (2 * np.pi) - np.pi) / (np.pi / 2)) for y in yaws]
    locals_ = [c[:, :2] @ _rot2(y) for c, y in zip(clouds, yaws, strict=True)]
    size = np.max([np.ptp(loc, axis=0) for loc in locals_], axis=0)
    centres = []
    for s, y, rows, loc in zip(samples, yaws, rowsets, locals_, strict=True):
        view = _view(scene, s.ts, assoc.clusters[s.ts], rows, loc, cfg, erode_px)
        lo, hi = loc.min(axis=0), loc.max(axis=0)
        sensor = scene.ego_pose(s.ts)[:2, 3] @ _rot2(y)
        centre = np.full(2, np.nan)
        for k in range(2):
            outside = sensor[k] < lo[k] - view.margin_m or sensor[k] > hi[k] + view.margin_m
            near = 0 if sensor[k] < (lo[k] + hi[k]) / 2 else 1
            if outside and view.clean[k, near]:
                # The face toward the sensor; a cut-off side face reaching toward it does not count.
                centre[k] = lo[k] + size[k] / 2 if near == 0 else hi[k] - size[k] / 2
            elif not outside and view.clean[k].all():
                centre[k] = (lo[k] + hi[k]) / 2
            elif view.clean[k, 0]:
                centre[k] = lo[k] + size[k] / 2
            elif view.clean[k, 1]:
                centre[k] = hi[k] - size[k] / 2
        centres.append([*(_rot2(y) @ centre), s.xyz_world[2]])
    return np.array(centres)


def _positions(scene: Scene, assoc: Association, cfg: TracksConfig) -> np.ndarray:
    if cfg.position == "box_centre":
        return box_centres(scene, assoc, cfg)
    return np.array([s.xyz_world for s in assoc.track.samples])


# Motion


def track_motion(scene: Scene, assoc: Association, cfg: TracksConfig | None = None) -> TrackedObject:
    cfg = cfg or load_verify_config().tracks
    samples = assoc.track.samples
    if not samples:
        empty = np.empty((0, 3))
        motion = Motion(
            np.empty(0), np.empty(0, bool), np.empty(0, int), empty, empty, empty, empty, np.empty(0),
            np.empty(0, bool),
        )  # fmt: skip
        return TrackedObject(assoc, assoc.track.model_copy(update={"speed_mps": []}), motion)

    obs_ts = np.array([s.ts for s in samples])
    obs_pos = _positions(scene, assoc, cfg)
    # ASSUMPTION A-73: a sample whose centre cannot be observed is treated like a missing sample
    known = ~np.isnan(obs_pos).any(axis=1)
    if not known.any():
        known[:] = True
        obs_pos = np.array([s.xyz_world for s in samples])
    obs_ts, obs_pos = obs_ts[known], obs_pos[known]
    sweeps = scene.lidar_timestamps()
    # ASSUMPTION A-67: a gap longer than max_interp_gap_s splits the series, whatever caused it
    breaks = np.flatnonzero(np.diff(obs_ts) > cfg.max_interp_gap_s + 1e-6) + 1
    parts = []
    for seg_id, idx in enumerate(np.split(np.arange(len(obs_ts)), breaks)):
        t_obs, p_obs = obs_ts[idx], obs_pos[idx]
        grid = sweeps[(sweeps >= t_obs[0] - 1e-6) & (sweeps <= t_obs[-1] + 1e-6)]
        observed = np.array([np.any(np.abs(t_obs - t) < 1e-6) for t in grid])
        pos = np.column_stack([np.interp(grid, t_obs, p_obs[:, k]) for k in range(3)])
        reliable = len(grid) >= cfg.savgol_window
        smooth = savgol_filter(pos, cfg.savgol_window, cfg.savgol_polyorder, axis=0) if reliable else pos
        vel = np.gradient(smooth, grid, axis=0) if len(grid) >= 2 else np.zeros_like(smooth)
        vel_ego = np.array([scene.ego_pose(t)[:3, :3].T @ v for t, v in zip(grid, vel, strict=True)])
        parts.append((grid, observed, np.full(len(grid), seg_id), pos, smooth, vel, vel_ego, reliable))

    def cat(k):
        return np.concatenate([p[k] for p in parts])

    vel = cat(5)
    motion = Motion(
        ts=cat(0),
        observed=cat(1),
        segment=cat(2),
        position_world=cat(3),
        smoothed_world=cat(4),
        velocity_world=vel,
        velocity_ego=cat(6),
        # ASSUMPTION A-66: speed is horizontal
        speed_mps=np.linalg.norm(vel[:, :2], axis=1),
        reliable=np.concatenate([np.full(len(p[0]), p[7]) for p in parts]),
    )
    speeds = [float(motion.speed_mps[motion.index(s.ts)]) if _has(motion, s.ts) else 0.0 for s in samples]
    return TrackedObject(assoc, assoc.track.model_copy(update={"speed_mps": speeds}), motion)


def _has(motion: Motion, ts: float) -> bool:
    return bool(np.any(np.abs(motion.ts - ts) < 1e-6))
