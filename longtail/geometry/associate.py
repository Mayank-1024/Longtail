"""Mask to LiDAR association (plan C4): Track2D masks in, Track3D world tracks out.

Per frame: erode the mask, count coverage in its box, select the projected points inside the mask, remove
the ground, split by depth, keep the largest DBSCAN component, and summarise it as a Sample3D. Then merge
fragments of one camera and tracks of one object seen by adjacent cameras.

Thresholds come from configs/verify.yaml (A-58).
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from longtail.contracts import Frame2D, Sample3D, Track2D, Track3D
from longtail.data.base import LidarSweep, Scene
from longtail.geometry.project import project_sweep
from longtail.geometry.sync import SyncedFrame, sync_camera
from longtail.geometry.transforms import world_to_ego
from longtail.verify.config import AssociationConfig, GroundConfig, MergeConfig, load_verify_config
from longtail.vision.rle import decode

# ASSUMPTION A-54: camera adjacency for cross-camera merging (PandaSet rig names)
ADJACENT_CAMERAS = {
    frozenset(p)
    for p in (
        ("front_camera", "front_left_camera"),
        ("front_camera", "front_right_camera"),
        ("front_left_camera", "left_camera"),
        ("front_right_camera", "right_camera"),
        ("left_camera", "back_camera"),
        ("right_camera", "back_camera"),
    )
}

# ASSUMPTION A-46: frames without a sample are recorded with a reason; C6 reads them as UNVERIFIABLE
SkipReason = Literal["unsynced", "low_coverage", "no_points", "no_cluster"]


@dataclass(frozen=True)
class FrameSkip:
    camera: str
    frame_idx: int
    ts: float  # camera frame timestamp
    reason: SkipReason
    coverage: int | None  # None when the frame was not projected (unsynced)


@dataclass(frozen=True, eq=False)
class FrameCluster:
    """The points kept for one camera frame, before they become a Sample3D."""

    camera: str
    frame_idx: int
    # ASSUMPTION A-52: samples are stamped with the matched sweep's timestamp, the time of the points
    ts: float
    lidar_idx: int
    point_index: np.ndarray  # rows of the sweep's points
    coverage: int
    ground: Literal["local", "global", "none"]
    plane: "Plane | None"  # the ground plane used, for measuring height


@dataclass(eq=False)
class Association:
    """C4 output: the contract Track3D plus what C6 needs that the contract does not carry."""

    track: Track3D
    # ASSUMPTION A-46: frames without a sample, with the reason, so C6 can return UNVERIFIABLE
    skipped: list[FrameSkip]
    # ASSUMPTION A-47: Track2D.truncated passes through here (any source truncated); Track3D has no such field
    truncated: bool
    # Points behind each sample, by sample ts, for merging and the C9 overlays.
    clusters: dict[float, list[FrameCluster]] = field(repr=False)
    # ASSUMPTION A-97: the Track2Ds merged into this track, so C9 can draw their masks and boxes
    inputs: list[Track2D] = field(default_factory=list, repr=False)


# Ground


@dataclass(frozen=True)
class Plane:
    normal: np.ndarray  # unit, pointing up
    d: float  # normal . p + d = signed height above the plane
    inliers: int

    def height(self, points: np.ndarray) -> np.ndarray:
        return points @ self.normal + self.d


def fit_ground_plane(points: np.ndarray, cfg: GroundConfig, rng: np.random.Generator) -> Plane | None:
    """RANSAC plane, near-horizontal, with almost nothing below it (so a roof never counts as ground)."""
    # ASSUMPTION A-49: tilt limit and the "few points below" rule pick the ground over roofs and walls
    n = len(points)
    if n < 3:
        return None
    min_nz = np.cos(np.radians(cfg.max_tilt_deg))
    # ASSUMPTION A-49: seeds come from the lowest points, so the ground is sampled even where it is a minority
    lowest = np.argsort(points[:, 2])[: max(3, int(cfg.ransac_seed_lowest_frac * n))]
    best: tuple[int, np.ndarray, float] | None = None
    for _ in range(cfg.ransac_iterations):
        p0, p1, p2 = points[rng.choice(lowest, 3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm * np.sign(normal[2] or 1.0)
        if normal[2] < min_nz:
            continue
        h = points @ normal - normal @ p0
        count = int(np.sum(np.abs(h) < cfg.ransac_inlier_m))
        if best is not None and count <= best[0]:
            continue
        if np.mean(h < -cfg.below_margin_m) > cfg.max_below_frac:
            continue
        best = (count, normal, -normal @ p0)
    if best is None:
        return None
    # Least-squares refinement on the inliers.
    count, normal, d = best
    inl = points[np.abs(points @ normal + d) < cfg.ransac_inlier_m]
    centre = inl.mean(axis=0)
    refined = np.linalg.svd(inl - centre, full_matrices=False)[2][-1]
    refined = refined * np.sign(refined[2] or 1.0)
    if refined[2] >= min_nz:
        normal, d = refined, -refined @ centre
    return Plane(normal=normal, d=float(d), inliers=count)


class SweepContext:
    """Per-sweep data shared by every track and camera: points, an xy tree and the global ground plane."""

    def __init__(self, sweep: LidarSweep, cfg: GroundConfig) -> None:
        self.sweep = sweep
        self.xyz = sweep.points[:, :3]
        self._cfg = cfg
        self._tree: cKDTree | None = None
        self._global: Plane | None | bool = False

    def near(self, xy: np.ndarray, radius: float) -> np.ndarray:
        if self._tree is None:
            self._tree = cKDTree(self.xyz[:, :2])
        if len(xy) == 0:
            return np.empty(0, dtype=np.int64)
        hits = self._tree.query_ball_point(xy[:: max(1, len(xy) // 256)], radius)
        return np.unique(np.concatenate([np.asarray(h, dtype=np.int64) for h in hits]))

    def global_plane(self) -> Plane | None:
        if self._global is False:
            rng = np.random.default_rng(self._cfg.seed)
            pts = self.xyz
            if len(pts) > self._cfg.global_sample_points:
                pts = pts[rng.choice(len(pts), self._cfg.global_sample_points, replace=False)]
            self._global = fit_ground_plane(pts, self._cfg, rng)
        return self._global


def remove_ground(
    points: np.ndarray, ctx: SweepContext, cfg: AssociationConfig
) -> tuple[np.ndarray, str, Plane | None]:
    """Mask of points at least ground_height_m above the ground, which plane was used, and the plane."""
    g = cfg.ground
    plane, mode = None, "none"
    if g.mode == "local":
        # ASSUMPTION A-49: a plane fitted to sweep points near the object's footprint, else the sweep's plane
        region = ctx.near(points[:, :2], g.local_radius_m)
        local = fit_ground_plane(ctx.xyz[region], g, np.random.default_rng(g.seed))
        if local is not None and local.inliers >= g.min_local_inliers:
            plane, mode = local, "local"
    if plane is None:
        plane = ctx.global_plane()
        mode = "global" if plane is not None else "none"
    if plane is None:
        return np.ones(len(points), dtype=bool), mode, None
    return plane.height(points) >= cfg.ground_height_m, mode, plane


# One frame


def _disk(radius: int) -> np.ndarray:
    r = np.arange(-radius, radius + 1)
    return (r[:, None] ** 2 + r[None, :] ** 2) <= radius**2


def visible_in_camera(pixels: np.ndarray, depth: np.ndarray, cfg: AssociationConfig) -> np.ndarray:
    """Mask of points no more than visibility_margin_m behind the nearest point in their image cell."""
    # ASSUMPTION A-63: depth-buffer visibility filter over square image cells
    if cfg.visibility_cell_px <= 0 or len(depth) == 0:
        return np.ones(len(depth), dtype=bool)
    cells = pixels // cfg.visibility_cell_px
    _, cell = np.unique(cells, axis=0, return_inverse=True)
    nearest = np.full(cell.max() + 1, np.inf)
    np.minimum.at(nearest, cell, depth)
    return depth <= nearest[cell] + cfg.visibility_margin_m


def nearest_depth_part(depth: np.ndarray, cfg: AssociationConfig) -> np.ndarray:
    """Indices of the nearest depth segment holding nearest_cluster_min_frac of the points."""
    order = np.argsort(depth, kind="stable")
    parts = np.split(order, np.flatnonzero(np.diff(depth[order]) > cfg.depth_gap_m) + 1)
    for part in parts:
        if len(part) >= cfg.nearest_cluster_min_frac * len(depth):
            return part
    # ASSUMPTION A-51: when no segment holds the share, take the largest
    return max(parts, key=len)


def _selected(frame: Frame2D, pixels: np.ndarray, cfg: AssociationConfig) -> np.ndarray:
    e = cfg.erode_px
    if cfg.selection == "mask" and frame.mask_rle is not None:
        mask = decode(frame.mask_rle)
        if e:
            # ASSUMPTION A-50: erosion by a disk of erode_px
            mask = binary_erosion(mask, structure=_disk(e))
        return mask[pixels[:, 1], pixels[:, 0]]
    # Frustum fallback or box-only claim (plan C4 steps 7 and 8).
    # ASSUMPTION A-50: boxes are inset by erode_px, like the mask
    x1, y1, x2, y2 = np.round(frame.box_xyxy).astype(int)
    u, v = pixels[:, 0], pixels[:, 1]
    return (u >= x1 + e) & (u <= x2 - e) & (v >= y1 + e) & (v <= y2 - e)


def _frame(
    scene: Scene,
    track: Track2D,
    frame: Frame2D,
    synced: SyncedFrame,
    sweeps: dict[int, SweepContext],
    cfg: AssociationConfig,
) -> FrameCluster | FrameSkip:
    camera = track.camera
    cam_frame = scene.frames(camera)[frame.frame_idx]
    if abs(cam_frame.ts - frame.ts) > 1e-3:
        raise ValueError(
            f"{camera} frame {frame.frame_idx}: Track2D ts {frame.ts} vs scene ts {cam_frame.ts}"
        )

    def skip(reason: SkipReason, coverage: int | None) -> FrameSkip:
        return FrameSkip(
            camera=camera, frame_idx=frame.frame_idx, ts=frame.ts, reason=reason, coverage=coverage
        )

    # ASSUMPTION A-46: unsynced frames give no sample
    if not synced.synced:
        return skip("unsynced", None)
    if synced.lidar_idx not in sweeps:
        sweeps[synced.lidar_idx] = SweepContext(scene.lidar(synced.lidar_ts), cfg.ground)
    ctx = sweeps[synced.lidar_idx]

    if frame.mask_rle is not None and tuple(frame.mask_rle.size) != scene.image_size(camera)[::-1]:
        raise ValueError(
            f"mask is {tuple(frame.mask_rle.size)[::-1]} but the image is {scene.image_size(camera)}"
        )
    proj = project_sweep(scene, ctx.sweep, camera, cam_frame)
    pixels = proj.pixels()
    x1, y1, x2, y2 = np.round(frame.box_xyxy).astype(int)
    in_box = (pixels[:, 0] >= x1) & (pixels[:, 0] <= x2) & (pixels[:, 1] >= y1) & (pixels[:, 1] <= y2)
    coverage = int(in_box.sum())
    if coverage < cfg.min_coverage:
        return skip("low_coverage", coverage)

    sel = _selected(frame, pixels, cfg)
    idx, depth, pix = proj.index[sel], proj.depth[sel], pixels[sel]
    if len(idx) == 0:
        return skip("no_points", coverage)
    above, ground, plane = remove_ground(ctx.xyz[idx], ctx, cfg)
    idx, depth, pix = idx[above], depth[above], pix[above]
    seen = visible_in_camera(pix, depth, cfg)
    idx, depth = idx[seen], depth[seen]
    if len(idx) < cfg.dbscan_min_points:
        return skip("no_cluster", coverage)
    part = nearest_depth_part(depth, cfg)
    idx = idx[part]
    labels = DBSCAN(eps=cfg.dbscan_eps_m, min_samples=cfg.dbscan_min_points).fit(ctx.xyz[idx]).labels_
    if np.all(labels < 0):
        return skip("no_cluster", coverage)
    largest = np.bincount(labels[labels >= 0]).argmax()
    return FrameCluster(
        camera=camera,
        frame_idx=frame.frame_idx,
        ts=ctx.sweep.ts,
        lidar_idx=synced.lidar_idx,
        point_index=np.sort(idx[labels == largest]),
        coverage=coverage,
        ground=ground,
        plane=plane,
    )


# Samples


def _height(points: np.ndarray, centre: np.ndarray, plane: Plane | None) -> float:
    """Top of the cluster minus the ground height under its footprint (the z range when there is no plane)."""
    # ASSUMPTION A-53: height = top z minus the A-49 plane's z at the median x, y (largest cluster's plane)
    if plane is None or abs(plane.normal[2]) < 1e-9:
        return float(np.ptp(points[:, 2]))
    n = plane.normal
    ground_z = -(n[0] * centre[0] + n[1] * centre[1] + plane.d) / n[2]
    return float(points[:, 2].max() - ground_z)


def _sample(scene: Scene, points: np.ndarray, ts: float, coverage: int, plane: Plane | None) -> Sample3D:
    world = np.median(points, axis=0)
    ego = np.median(world_to_ego(scene.ego_pose(ts), points), axis=0)
    # ASSUMPTION A-53: length and width by PCA over world x, y (world z is up, A-26); height from the ground
    xy = points[:, :2] - points[:, :2].mean(axis=0)
    if len(points) >= 2 and np.ptp(xy, axis=0).any():
        axes = np.linalg.eigh(np.cov(xy.T))[1]
        spans = np.ptp(xy @ axes, axis=0)
    else:
        spans = np.zeros(2)
    return Sample3D(
        ts=ts,
        xyz_ego=tuple(float(v) for v in ego),
        xyz_world=tuple(float(v) for v in world),
        n_points=len(points),
        extent_lwh=(float(spans.max()), float(spans.min()), _height(points, world, plane)),
        coverage=coverage,
    )


# Merging


@dataclass(eq=False)
class _Group:
    cls: str
    sources: list[tuple[str, int]]
    clusters: dict[float, list[FrameCluster]]
    skipped: list[FrameSkip]
    truncated: bool
    inputs: list[Track2D] = field(default_factory=list)

    @property
    def times(self) -> list[float]:
        return sorted(self.clusters)

    def camera_times(self, camera: str) -> set[float]:
        return {t for t, cs in self.clusters.items() if any(c.camera == camera for c in cs)}

    @property
    def cameras(self) -> set[str]:
        return {c for c, _ in self.sources}

    def absorb(self, other: "_Group") -> None:
        self.sources += other.sources
        for t, cs in other.clusters.items():
            self.clusters.setdefault(t, []).extend(cs)
        self.skipped += other.skipped
        self.truncated |= other.truncated
        self.inputs += other.inputs


class _Points:
    """Union of cluster points per timestamp, read from the cached sweeps."""

    def __init__(self, sweeps: dict[int, SweepContext]) -> None:
        self._sweeps = sweeps

    def at(self, clusters: list[FrameCluster]) -> np.ndarray:
        rows = np.unique(np.concatenate([c.point_index for c in clusters]))
        return self._sweeps[clusters[0].lidar_idx].xyz[rows]

    def centroid(self, group: _Group, t: float) -> np.ndarray:
        return np.median(self.at(group.clusters[t]), axis=0)


def _chain_fragments(groups: list[_Group], pts: _Points, m: MergeConfig) -> list[_Group]:
    """Join fragments of one camera that follow each other in time (A-12)."""
    # ASSUMPTION A-12: same-camera fragments merge with the same 1 m rule
    # ASSUMPTION A-56: fragments never share timestamps, so the earlier one is extrapolated across the gap
    chains: list[_Group] = []
    for g in sorted(groups, key=lambda g: (g.times[0], g.sources)):
        start, best = g.times[0], None
        for ch in chains:
            if ch.cls != g.cls or ch.cameras != g.cameras:
                continue
            gap = start - ch.times[-1]
            if gap <= 0 or gap > m.fragment_max_gap_s:
                continue
            recent = [t for t in ch.times if t >= ch.times[-1] - m.fragment_velocity_window_s]
            cents = np.array([pts.centroid(ch, t) for t in recent])
            vel = np.polyfit(recent, cents, 1)[0] if len(recent) >= 2 else np.zeros(3)
            dist = np.linalg.norm(cents[-1] + vel * gap - pts.centroid(g, start))
            if dist <= m.max_centroid_dist_m and (best is None or dist < best[0]):
                best = (dist, ch)
        if best is None:
            chains.append(g)
        else:
            best[1].absorb(g)
    return chains


def _rows(group: _Group, t: float) -> np.ndarray:
    return np.unique(np.concatenate([c.point_index for c in group.clusters[t]]))


def _shares_points(a: _Group, b: _Group, t: float, m: MergeConfig) -> bool:
    """Clusters that share enough LiDAR points are one object, even with centroids far apart."""
    # ASSUMPTION A-74: at least min_shared_points and min_shared_points_frac of the smaller cluster
    ra, rb = _rows(a, t), _rows(b, t)
    common = len(np.intersect1d(ra, rb))
    return common >= m.min_shared_points and common >= m.min_shared_points_frac * min(len(ra), len(rb))


def _merge_cameras(groups: list[_Group], pts: _Points, m: MergeConfig) -> list[_Group]:
    """Merge one object's tracks from adjacent cameras (plan C4 step 6)."""
    candidates = []
    for i, a in enumerate(groups):
        for j in range(i + 1, len(groups)):
            b = groups[j]
            if a.cls != b.cls or not any(
                frozenset((x, y)) in ADJACENT_CAMERAS for x in a.cameras for y in b.cameras
            ):
                continue
            shared = sorted(set(a.clusters) & set(b.clusters))
            if not shared:
                continue
            dists = np.array([np.linalg.norm(pts.centroid(a, t) - pts.centroid(b, t)) for t in shared])
            # ASSUMPTION A-74: long objects split by the cameras' fields of view merge on shared points
            same = (dists <= m.max_centroid_dist_m) | np.array([_shares_points(a, b, t, m) for t in shared])
            if np.mean(same) >= m.min_shared_frac:
                candidates.append((float(np.median(np.where(same, dists, np.inf))), i, j))
    # ASSUMPTION A-55: greedy by median distance; never two simultaneous tracks of one camera in a group
    parent = list(range(len(groups)))

    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    for _, i, j in sorted(candidates):
        ri, rj = root(i), root(j)
        if ri == rj:
            continue
        a, b = groups[ri], groups[rj]
        if any(a.camera_times(c) & b.camera_times(c) for c in a.cameras & b.cameras):
            continue
        a.absorb(b)
        parent[rj] = ri
    return [g for k, g in enumerate(groups) if root(k) == k]


# Entry points


def associate(
    scene: Scene, tracks: list[tuple[Track2D, str]], cfg: AssociationConfig | None = None
) -> list[Association]:
    """Associate Track2Ds (each with its taxonomy class) to LiDAR, then merge fragments and cameras."""
    cfg = cfg or load_verify_config().association
    sweeps: dict[int, SweepContext] = {}
    synced = {cam: sync_camera(scene, cam) for cam in {t.camera for t, _ in tracks}}
    groups: list[_Group] = []
    empty: list[_Group] = []  # tracks where no frame produced a cluster
    for track, cls in tracks:
        clusters: dict[float, list[FrameCluster]] = defaultdict(list)
        skipped: list[FrameSkip] = []
        for frame in track.frames:
            out = _frame(scene, track, frame, synced[track.camera][frame.frame_idx], sweeps, cfg)
            if isinstance(out, FrameSkip):
                skipped.append(out)
            else:
                clusters[out.ts].append(out)
        # ASSUMPTION A-57: Track3D.cls is the taxonomy class the caller pairs with each Track2D
        group = _Group(cls, [(track.camera, track.obj_id)], dict(clusters), skipped, track.truncated, [track])
        (groups if clusters else empty).append(group)
    pts = _Points(sweeps)
    groups = _merge_cameras(_chain_fragments(groups, pts, cfg.merge), pts, cfg.merge)
    groups.sort(key=lambda g: (g.times[0], sorted(g.sources)))
    out = []
    for n, g in enumerate(groups + empty):
        samples = [
            _sample(
                scene,
                pts.at(g.clusters[t]),
                t,
                max(c.coverage for c in g.clusters[t]),
                max(g.clusters[t], key=lambda c: len(c.point_index)).plane,
            )
            for t in g.times
        ]
        track3d = Track3D(
            track_id=f"{scene.scene_id}/{g.cls}/{n}",
            cls=g.cls,
            sources=sorted(g.sources),
            samples=samples,
            # ASSUMPTION A-48: C4 leaves speed_mps empty; C5 fills it
            speed_mps=[],
        )
        out.append(
            Association(
                track=track3d, skipped=g.skipped, truncated=g.truncated, clusters=g.clusters, inputs=g.inputs
            )
        )
    return out
