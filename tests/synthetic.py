"""A generated scene that implements the Scene interface, so real code paths run without the dataset.

Plan C2 spec: known cameras, a box of points starting 10 m ahead and crossing at 1.5 m/s, a ground plane
and a wall. Extensions, registered as assumptions: the ego drives forward at constant speed (so world
and ego frames differ and C5 can check that ego motion cancels out), and a stationary object sits ahead
(C5's "parked object reads under 0.3 m/s" check).

Everything is parameterised through SyntheticConfig so C4 to C6 tests can reuse it. Truth for tests
(object poses, which object each point came from) is available through objects() and point_ids(),
which are not part of the Scene API; the pipeline never calls them.
"""

# ASSUMPTION A-40: the synthetic ego drives straight ahead at constant speed and yaw
# ASSUMPTION A-41: the synthetic scene has one stationary (parked) object

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np

from longtail.data.base import CameraFrame, LidarSweep, Scene, SceneMeta, index_of
from longtail.geometry.transforms import apply, make_pose, rot_z

GROUND, WALL, CROSSER, PARKED, PED_A, PED_B, PASS_CAR, CAR_A, CAR_B = 0, 1, 2, 3, 4, 5, 6, 7, 8
OBJECT_IDS = {
    "crosser": CROSSER,
    "parked": PARKED,
    "ped_a": PED_A,
    "ped_b": PED_B,
    "pass_car": PASS_CAR,
    "car_a": CAR_A,
    "car_b": CAR_B,
}

# ASSUMPTION A-44: synthetic rig of six pinhole cameras with PandaSet names, yawed about the ego up axis
CAMERA_YAWS_DEG = {
    "front_camera": 0.0,
    "front_left_camera": 60.0,
    "front_right_camera": -60.0,
    "left_camera": 90.0,
    "right_camera": -90.0,
    "back_camera": 180.0,
}


@dataclass(frozen=True)
class SyntheticConfig:
    n_frames: int = 30
    lidar_hz: float = 10.0
    t0: float = 1000.0
    # Camera timestamps are sweep timestamps plus this offset; override per camera to test sync.
    camera_offset_s: float = 0.01
    camera_offsets_s: dict[str, float] = field(default_factory=dict)
    # Ego: world position of the ego origin at t0, heading, speed. The ego origin is the LiDAR (A-25).
    ego_start_world: tuple[float, float] = (100.0, 50.0)
    ego_yaw_rad: float = 0.25
    ego_speed_mps: float = 2.0
    # ASSUMPTION A-64: optional constant yaw rate, so the ego can drive an arc (corridor tests)
    ego_yaw_rate_rps: float = 0.0
    sensor_height_m: float = 1.8
    # Cameras sit at the ego origin, this far below the LiDAR.
    camera_drop_m: float = 0.3
    image_size: tuple[int, int] = (640, 480)
    focal_px: float = 500.0
    # Crosser: centre at t0 in the ego frame at t0 (ahead, left), moving toward +left.
    crosser_start: tuple[float, float] = (10.0, -2.25)
    crosser_speed_mps: float = 1.5
    crosser_lwh: tuple[float, float, float] = (0.6, 0.6, 1.7)
    parked_center: tuple[float, float] = (15.0, -4.0)
    parked_lwh: tuple[float, float, float] = (4.5, 1.8, 1.5)
    wall_left_m: float = 8.0
    wall_height_m: float = 3.0
    extent_ahead: tuple[float, float] = (-10.0, 50.0)
    ground_half_width_m: float = 12.0
    ground_spacing_m: float = 0.5
    wall_spacing_m: float = 0.25
    box_spacing_m: float = 0.1
    # ASSUMPTION A-43: LiDAR sees only box faces turned toward it, and boxes occlude points behind them
    occlusion: bool = True
    # ASSUMPTION A-60: optional slope: flat to slope_start_m ahead, then a constant grade; boxes upright
    ground_grade: float = 0.0
    slope_start_m: float = 8.0
    # ASSUMPTION A-61: optional pair of standing pedestrians, centres pair_gap_m apart across the ego's path
    pedestrian_pair: bool = False
    pair_center: tuple[float, float] = (12.0, 3.0)
    pair_gap_m: float = 0.6
    pair_lwh: tuple[float, float, float] = (0.5, 0.5, 1.75)
    # ASSUMPTION A-65: optional car-sized parked object close to the ego's path, for the close-pass test
    pass_car: bool = False
    pass_car_center: tuple[float, float] = (8.0, -2.9)
    pass_car_lwh: tuple[float, float, float] = (4.5, 1.8, 1.5)
    # ASSUMPTION A-75: optional two cars parked nose to tail across the road, nose_to_tail_gap_m apart
    nose_to_tail: bool = False
    nose_to_tail_center: tuple[float, float] = (4.0, 4.0)
    nose_to_tail_gap_m: float = 0.3
    nose_to_tail_lwh: tuple[float, float, float] = (4.5, 1.8, 1.5)


@dataclass(frozen=True)
class SyntheticObject:
    center_world: np.ndarray
    lwh: tuple[float, float, float]
    yaw: float
    velocity_world: np.ndarray


def _grid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.stack(np.meshgrid(a, b, indexing="ij"), axis=-1).reshape(-1, 2)


def _box_surface(obj: SyntheticObject, spacing: float, sensor: np.ndarray, visible_only: bool) -> np.ndarray:
    """Points on the box faces, optionally only the faces turned toward the sensor."""
    half = np.array(obj.lwh) / 2
    pts = []
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        ticks = [np.linspace(-half[i], half[i], max(2, round(2 * half[i] / spacing) + 1)) for i in others]
        uv = _grid(*ticks)
        for sign in (-1.0, 1.0):
            face = np.zeros((len(uv), 3))
            face[:, others[0]], face[:, others[1]] = uv[:, 0], uv[:, 1]
            face[:, axis] = sign * half[axis]
            normal = np.zeros(3)
            normal[axis] = sign
            pts.append((face, normal))
    r = rot_z(obj.yaw)
    out = []
    for face, normal in pts:
        center = obj.center_world + r @ (normal * half)
        if visible_only and (sensor - center) @ (r @ normal) <= 0:
            continue
        out.append(face @ r.T + obj.center_world)
    return np.unique(np.concatenate(out).round(6), axis=0)


def _occluded_by(obj: SyntheticObject, sensor: np.ndarray, points: np.ndarray) -> np.ndarray:
    """True where the segment from the sensor to the point passes through the box before the point."""
    r = rot_z(obj.yaw)
    s = r.T @ (sensor - obj.center_world)
    p = (points - obj.center_world) @ r
    d = p - s
    half = np.array(obj.lwh) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-half - s) / d
        t2 = (half - s) / d
    t_near = np.nanmax(np.where(np.isfinite(t1), np.minimum(t1, t2), -np.inf), axis=1)
    t_far = np.nanmin(np.where(np.isfinite(t1), np.maximum(t1, t2), np.inf), axis=1)
    # Rays parallel to a slab miss the box unless the sensor lies inside that slab.
    parallel_miss = np.any((d == 0) & (np.abs(s) > half), axis=1)
    return ~parallel_miss & (t_near <= t_far) & (t_far > 0) & (t_near < 1 - 1e-6)


class SyntheticScene(Scene):
    def __init__(self, config: SyntheticConfig | None = None, scene_id: str = "synthetic") -> None:
        self.cfg = config or SyntheticConfig()
        self.scene_id = scene_id
        c = self.cfg
        self._lidar_ts = c.t0 + np.arange(c.n_frames) / c.lidar_hz
        self._fwd = rot_z(c.ego_yaw_rad)[:, 0]
        self._left = rot_z(c.ego_yaw_rad)[:, 1]
        self._start = np.array([*c.ego_start_world, c.sensor_height_m])
        self._static = self._static_points()

    # Truth (not Scene API)

    def ego_pose_at(self, t: float) -> np.ndarray:
        """world_from_ego at any time."""
        c = self.cfg
        dt, w = t - c.t0, c.ego_yaw_rate_rps
        heading = c.ego_yaw_rad + w * dt
        if w == 0:
            offset = self._fwd * c.ego_speed_mps * dt
        else:
            r = c.ego_speed_mps / w
            h0 = c.ego_yaw_rad
            offset = np.array([r * (np.sin(heading) - np.sin(h0)), -r * (np.cos(heading) - np.cos(h0)), 0.0])
        return make_pose(rot_z(heading), self._start + offset)

    def ground_z(self, ahead: np.ndarray | float) -> np.ndarray | float:
        """Ground height at a distance ahead of the ego's start position."""
        c = self.cfg
        return c.ground_grade * np.maximum(0.0, np.asarray(ahead) - c.slope_start_m)

    def _on_ground(self, ahead: float, left: float, height: float) -> np.ndarray:
        """World point given ego-at-t0 ground coordinates and a height above the ground."""
        xy = self._start[:2] + ahead * self._fwd[:2] + left * self._left[:2]
        return np.array([*xy, float(self.ground_z(ahead)) + height])

    def objects(self, t: float) -> dict[str, SyntheticObject]:
        c = self.cfg
        crosser_left = c.crosser_start[1] + c.crosser_speed_mps * (t - c.t0)
        objs = {
            "crosser": SyntheticObject(
                center_world=self._on_ground(c.crosser_start[0], crosser_left, c.crosser_lwh[2] / 2),
                lwh=c.crosser_lwh,
                yaw=c.ego_yaw_rad,
                velocity_world=self._left * c.crosser_speed_mps,
            ),
            "parked": SyntheticObject(
                center_world=self._on_ground(*c.parked_center, c.parked_lwh[2] / 2),
                lwh=c.parked_lwh,
                yaw=c.ego_yaw_rad,
                velocity_world=np.zeros(3),
            ),
        }
        if c.pass_car:
            objs["pass_car"] = SyntheticObject(
                center_world=self._on_ground(*c.pass_car_center, c.pass_car_lwh[2] / 2),
                lwh=c.pass_car_lwh,
                yaw=c.ego_yaw_rad,
                velocity_world=np.zeros(3),
            )
        if c.nose_to_tail:
            ahead, left = c.nose_to_tail_center
            for name, offset in (("car_a", 0.0), ("car_b", c.nose_to_tail_lwh[0] + c.nose_to_tail_gap_m)):
                objs[name] = SyntheticObject(
                    center_world=self._on_ground(ahead, left + offset, c.nose_to_tail_lwh[2] / 2),
                    lwh=c.nose_to_tail_lwh,
                    yaw=c.ego_yaw_rad + np.pi / 2,
                    velocity_world=np.zeros(3),
                )
        if c.pedestrian_pair:
            ahead, left = c.pair_center
            for name, side in (("ped_a", -0.5), ("ped_b", 0.5)):
                objs[name] = SyntheticObject(
                    center_world=self._on_ground(ahead, left + side * c.pair_gap_m, c.pair_lwh[2] / 2),
                    lwh=c.pair_lwh,
                    yaw=c.ego_yaw_rad,
                    velocity_world=np.zeros(3),
                )
        return objs

    def point_ids(self, ts: float) -> np.ndarray:
        """Which object each point of lidar(ts) came from: GROUND, WALL or an OBJECT_IDS value."""
        return self._sweep(index_of(self._lidar_ts, ts, "LiDAR sweep"))[1]

    # Scene API

    @property
    def cameras(self) -> tuple[str, ...]:
        return tuple(CAMERA_YAWS_DEG)

    @property
    def meta(self) -> SceneMeta:
        return SceneMeta(
            scene_id=self.scene_id,
            dataset="synthetic",
            n_frames=self.cfg.n_frames,
            start_ts=float(self._lidar_ts[0]),
            lat=None,
            lon=None,
            time_of_day=None,
            weather=None,
        )

    def _camera_ts(self, camera: str) -> np.ndarray:
        return self._lidar_ts + self.cfg.camera_offsets_s.get(camera, self.cfg.camera_offset_s)

    def frames(self, camera: str) -> list[CameraFrame]:
        return [
            CameraFrame(frame_idx=i, ts=float(t), image_path=Path(f"/synthetic/{camera}/{i:02d}.jpg"))
            for i, t in enumerate(self._camera_ts(camera))
        ]

    def lidar_timestamps(self) -> np.ndarray:
        return self._lidar_ts.copy()

    def lidar(self, ts: float) -> LidarSweep:
        i = index_of(self._lidar_ts, ts, "LiDAR sweep")
        xyz = self._sweep(i)[0]
        n = len(xyz)
        points = np.column_stack([xyz, np.full(n, 50.0), np.full(n, self._lidar_ts[i])])
        return LidarSweep(
            frame_idx=i,
            ts=float(self._lidar_ts[i]),
            points=points,
            sensor_id=np.zeros(n, dtype=np.int8),
            point_index=np.arange(n, dtype=np.int64),
        )

    def ego_pose(self, ts: float) -> np.ndarray:
        return self.ego_pose_at(self._lidar_ts[index_of(self._lidar_ts, ts, "LiDAR sweep")])

    def intrinsics(self, camera: str) -> np.ndarray:
        w, h = self.cfg.image_size
        f = self.cfg.focal_px
        return np.array([[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]])

    def ego_from_camera(self, camera: str) -> np.ndarray:
        """Camera mount in the ego frame (OpenCV optical axes)."""
        yaw = np.radians(CAMERA_YAWS_DEG[camera])
        z = np.array([np.cos(yaw), np.sin(yaw), 0.0])  # optical axis
        x = np.array([np.sin(yaw), -np.cos(yaw), 0.0])  # image right
        y = np.array([0.0, 0.0, -1.0])  # image down
        return make_pose(np.column_stack([x, y, z]), np.array([0.0, 0.0, -self.cfg.camera_drop_m]))

    def extrinsics(self, camera: str, ts: float) -> np.ndarray:
        cam_ts = self._camera_ts(camera)
        t = cam_ts[index_of(cam_ts, ts, f"{camera} frame")]
        return self.ego_pose_at(t) @ self.ego_from_camera(camera)

    def image_size(self, camera: str) -> tuple[int, int]:
        return self.cfg.image_size

    def _labels(self, frame_idx: int) -> list[dict]:
        objs = self.objects(self._lidar_ts[frame_idx])
        return [
            {"name": k, "center_world": o.center_world, "lwh": o.lwh, "yaw": o.yaw} for k, o in objs.items()
        ]

    def _semseg(self, frame_idx: int) -> np.ndarray:
        return self._sweep(frame_idx)[1]

    # Point generation

    def _static_points(self) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        a0, a1 = c.extent_ahead
        ahead = np.arange(a0, a1 + 1e-9, c.ground_spacing_m)
        left = np.arange(-c.ground_half_width_m, c.ground_half_width_m + 1e-9, c.ground_spacing_m)
        g = _grid(ahead, left)
        ground = np.column_stack(
            [
                self._start[:2] + np.outer(g[:, 0], self._fwd[:2]) + np.outer(g[:, 1], self._left[:2]),
                self.ground_z(g[:, 0]),
            ]
        )
        wa = _grid(
            np.arange(a0, a1 + 1e-9, c.wall_spacing_m),
            np.arange(c.wall_spacing_m, c.wall_height_m + 1e-9, c.wall_spacing_m),
        )
        wall = np.column_stack(
            [
                self._start[:2] + np.outer(wa[:, 0], self._fwd[:2]) + c.wall_left_m * self._left[:2],
                self.ground_z(wa[:, 0]) + wa[:, 1],
            ]
        )
        ids = np.concatenate([np.full(len(ground), GROUND), np.full(len(wall), WALL)])
        return np.concatenate([ground, wall]), ids

    @cache  # noqa: B019 - the scene is immutable and sweeps are reused across tests
    def _sweep(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        t = self._lidar_ts[i]
        sensor = self.ego_pose_at(t)[:3, 3]
        objs = self.objects(t)
        parts = [self._static]
        for name, obj in objs.items():
            pts = _box_surface(obj, self.cfg.box_spacing_m, sensor, self.cfg.occlusion)
            parts.append((pts, np.full(len(pts), OBJECT_IDS[name])))
        xyz = np.concatenate([p for p, _ in parts])
        ids = np.concatenate([d for _, d in parts])
        if self.cfg.occlusion:
            keep = np.ones(len(xyz), dtype=bool)
            for name, obj in objs.items():
                keep &= (ids == OBJECT_IDS[name]) | ~_occluded_by(obj, sensor, xyz)
            xyz, ids = xyz[keep], ids[keep]
        return xyz, ids


def world_to_pixel(scene: SyntheticScene, camera: str, frame_idx: int, point_world: np.ndarray) -> np.ndarray:
    """Expected pixel of a world point, computed independently of geometry.project (pinhole by hand)."""
    frame = scene.frames(camera)[frame_idx]
    world_from_cam = scene.ego_pose_at(frame.ts) @ scene.ego_from_camera(camera)
    p = apply(np.linalg.inv(world_from_cam), point_world[None])[0]
    k = scene.intrinsics(camera)
    return np.array([k[0, 0] * p[0] / p[2] + k[0, 2], k[1, 1] * p[1] / p[2] + k[1, 2]])


# Track2D helpers: stand-ins for SAM 3.1 output on the synthetic scene.
# ASSUMPTION A-62: synthetic masks are the convex hull of the projected box minus nearer objects' hulls


def _cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """Counter-clockwise hull (monotone chain) of (N, 2) points."""
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return pts

    def half(points):
        out: list[np.ndarray] = []
        for p in points:
            while len(out) >= 2 and _cross2(out[-1] - out[-2], p - out[-2]) <= 0:
                out.pop()
            out.append(p)
        return out[:-1]

    return np.array(half(pts) + half(pts[::-1]))


def _fill_convex(hull: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Pixels whose centre (integer coordinates, A-37) lies inside the hull."""
    w, h = size
    mask = np.zeros((h, w), dtype=bool)
    lo = np.clip(np.floor(hull.min(axis=0)).astype(int), 0, [w - 1, h - 1])
    hi = np.clip(np.ceil(hull.max(axis=0)).astype(int), 0, [w - 1, h - 1])
    if (
        hull.max(axis=0)[0] < 0
        or hull.max(axis=0)[1] < 0
        or hull.min(axis=0)[0] > w - 1
        or hull.min(axis=0)[1] > h - 1
    ):
        return mask
    us, vs = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
    p = np.stack([us.ravel(), vs.ravel()], axis=1).astype(float)
    inside = np.ones(len(p), dtype=bool)
    for a, b in zip(hull, np.roll(hull, -1, axis=0), strict=True):
        inside &= _cross2(b - a, p - a) >= 0
    mask[p[inside, 1].astype(int), p[inside, 0].astype(int)] = True
    return mask


def _silhouette(scene: SyntheticScene, obj: SyntheticObject, camera: str, t: float):
    """(mask, centre distance) of a box in a camera at time t; None if part of it is behind the camera."""
    from longtail.geometry.project import MIN_DEPTH_M
    from longtail.geometry.transforms import world_to_camera

    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    corners = obj.center_world + (signs * np.array(obj.lwh) / 2) @ rot_z(obj.yaw).T
    cam_pose = scene.ego_pose_at(t) @ scene.ego_from_camera(camera)
    cam = world_to_camera(cam_pose, corners)
    if np.any(cam[:, 2] <= MIN_DEPTH_M):
        return None
    k = scene.intrinsics(camera)
    uv = np.column_stack(
        [k[0, 0] * cam[:, 0] / cam[:, 2] + k[0, 2], k[1, 1] * cam[:, 1] / cam[:, 2] + k[1, 2]]
    )
    # Order overlapping objects by straight-line distance; camera z ties for objects side by side.
    distance = float(np.linalg.norm(world_to_camera(cam_pose, obj.center_world[None])[0]))
    return _fill_convex(_convex_hull(uv), scene.image_size(camera)), distance


def disk(radius: int) -> np.ndarray:
    r = np.arange(-radius, radius + 1)
    return (r[:, None] ** 2 + r[None, :] ** 2) <= radius**2


def object_mask(scene: SyntheticScene, name: str, camera: str, frame_idx: int, dilate_px: int = 0):
    """Visible mask of one object in one camera frame; None unless it is wholly in front of the camera."""
    from scipy.ndimage import binary_dilation

    t = scene.frames(camera)[frame_idx].ts
    objs = scene.objects(t)
    own = _silhouette(scene, objs[name], camera, t)
    if own is None:
        return None
    mask, distance = own
    for other, obj in objs.items():
        s = None if other == name else _silhouette(scene, obj, camera, t)
        if s is not None and s[1] < distance:
            mask &= ~s[0]
    if dilate_px:
        mask = binary_dilation(mask, structure=disk(dilate_px))
    return mask if mask.any() else None


def rect_mask(scene: SyntheticScene, u0: int, v0: int, u1: int, v1: int) -> np.ndarray:
    """Inclusive pixel rectangle as a mask."""
    w, h = scene.cfg.image_size
    mask = np.zeros((h, w), dtype=bool)
    mask[v0 : v1 + 1, u0 : u1 + 1] = True
    return mask


def track2d_from_masks(scene, camera: str, concept: str, obj_id: int, masks: dict, truncated: bool = False):
    from longtail.contracts import Frame2D, Track2D
    from longtail.vision.rle import encode

    frames = []
    for i, m in sorted(masks.items()):
        if m is None or not m.any():
            continue
        vs, us = np.nonzero(m)
        box = (float(us.min()), float(vs.min()), float(us.max()), float(vs.max()))
        frames.append(
            Frame2D(frame_idx=i, ts=scene.frames(camera)[i].ts, box_xyxy=box, mask_rle=encode(m), conf=0.9)
        )
    return Track2D(
        scene_id=scene.scene_id,
        camera=camera,
        concept=concept,
        obj_id=obj_id,
        truncated=truncated,
        frames=frames,
    )


def object_track2d(
    scene: SyntheticScene,
    name: str,
    camera: str,
    concept: str = "person",
    obj_id: int = 1,
    frames=None,
    dilate_px: int = 0,
    truncated: bool = False,
):
    """Track2D of one object in one camera; frames where it is not visible are left out."""
    idx = range(scene.cfg.n_frames) if frames is None else frames
    masks = {i: object_mask(scene, name, camera, i, dilate_px) for i in idx}
    return track2d_from_masks(scene, camera, concept, obj_id, masks, truncated)


def without_masks(track):
    """The same track as boxes only, like a VLM claim (plan C4 step 8)."""
    return track.model_copy(
        update={"frames": [f.model_copy(update={"mask_rle": None}) for f in track.frames]}
    )


def close_pass_config(**overrides) -> SyntheticConfig:
    """The ego drives at 5 m/s fully past a 4.5 m parked car whose near side is 2.0 m from its path.

    The crosser starts 30 m ahead so the ego never drives through it.
    """
    base = dict(pass_car=True, ego_speed_mps=5.0, n_frames=40, crosser_start=(30.0, -2.25))
    return SyntheticConfig(**{**base, **overrides})


def stopped_ego_config(**overrides) -> SyntheticConfig:
    """The ego stands still (a red light); everything else as the default scene."""
    return SyntheticConfig(**{"ego_speed_mps": 0.0, **overrides})


def track_source(scene: SyntheticScene, objects: dict, extra: dict | None = None, dilate_px: int = 0):
    """A TrackSource for the verifier (stand-in for C3's SAM client).

    objects maps a concept to synthetic object names; extra maps a concept to ready-made Track2Ds.
    """

    def source(_scene, camera: str, concept: str):
        out = []
        for k, name in enumerate(objects.get(concept, [])):
            t = object_track2d(scene, name, camera, concept=concept, obj_id=k + 1, dilate_px=dilate_px)
            if t.frames:
                out.append(t)
        return out + [t for t in (extra or {}).get(concept, []) if t.camera == camera]

    return source


class ImageScene(SyntheticScene):
    """The synthetic scene with camera images rendered to disk, so clips can be encoded (C3 tests).

    Each image is a plain background, object silhouettes in fixed colours, and a stripe along the top whose
    grey level encodes the frame index (so a dropped, duplicated or reordered frame shows after decoding).
    """

    COLOURS = {
        "crosser": (0, 200, 255),
        "parked": (255, 140, 0),
        "ped_a": (120, 255, 80),
        "ped_b": (255, 80, 200),
        "pass_car": (200, 200, 0),
        "car_a": (90, 90, 255),
        "car_b": (255, 255, 255),
    }

    def __init__(
        self, image_dir: Path, config: SyntheticConfig | None = None, scene_id: str = "synthetic"
    ) -> None:
        super().__init__(config, scene_id)
        self.image_dir = Path(image_dir)

    @staticmethod
    def stripe_level(frame_idx: int) -> int:
        return (40 + 7 * frame_idx) % 256

    def frames(self, camera: str) -> list[CameraFrame]:
        from PIL import Image

        out = []
        for f in super().frames(camera):
            path = self.image_dir / camera / f"{f.frame_idx:03d}.png"
            if not path.is_file():
                w, h = self.cfg.image_size
                arr = np.full((h, w, 3), 40, np.uint8)
                objs = self.objects(f.ts)
                # Far objects first, so nearer silhouettes paint over them.
                drawn = [(name, _silhouette(self, obj, camera, f.ts)) for name, obj in objs.items()]
                for name, sil in sorted(((n, s) for n, s in drawn if s is not None), key=lambda x: -x[1][1]):
                    arr[sil[0]] = self.COLOURS.get(name, (200, 200, 200))
                arr[:8] = self.stripe_level(f.frame_idx)
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(arr).save(path)
            out.append(CameraFrame(frame_idx=f.frame_idx, ts=f.ts, image_path=path))
        return out
