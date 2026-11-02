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

GROUND, WALL, CROSSER, PARKED = 0, 1, 2, 3

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
        return make_pose(rot_z(c.ego_yaw_rad), self._start + self._fwd * c.ego_speed_mps * (t - c.t0))

    def _on_ground(self, ahead: float, left: float, height: float) -> np.ndarray:
        """World point given ego-at-t0 ground coordinates and a height above the ground."""
        return np.array([*(self._start[:2] + ahead * self._fwd[:2] + left * self._left[:2]), height])

    def objects(self, t: float) -> dict[str, SyntheticObject]:
        c = self.cfg
        crosser_left = c.crosser_start[1] + c.crosser_speed_mps * (t - c.t0)
        return {
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

    def point_ids(self, ts: float) -> np.ndarray:
        """Which object each point of lidar(ts) came from: GROUND, WALL, CROSSER or PARKED."""
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
                np.zeros(len(g)),
            ]
        )
        wa = _grid(
            np.arange(a0, a1 + 1e-9, c.wall_spacing_m),
            np.arange(c.wall_spacing_m, c.wall_height_m + 1e-9, c.wall_spacing_m),
        )
        wall = np.column_stack(
            [
                self._start[:2] + np.outer(wa[:, 0], self._fwd[:2]) + c.wall_left_m * self._left[:2],
                wa[:, 1],
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
        for oid, name in ((CROSSER, "crosser"), (PARKED, "parked")):
            pts = _box_surface(objs[name], self.cfg.box_spacing_m, sensor, self.cfg.occlusion)
            parts.append((pts, np.full(len(pts), oid)))
        xyz = np.concatenate([p for p, _ in parts])
        ids = np.concatenate([d for _, d in parts])
        if self.cfg.occlusion:
            keep = np.ones(len(xyz), dtype=bool)
            for oid, name in ((CROSSER, "crosser"), (PARKED, "parked")):
                keep &= (ids == oid) | ~_occluded_by(objs[name], sensor, xyz)
            xyz, ids = xyz[keep], ids[keep]
        return xyz, ids


def world_to_pixel(scene: SyntheticScene, camera: str, frame_idx: int, point_world: np.ndarray) -> np.ndarray:
    """Expected pixel of a world point, computed independently of geometry.project (pinhole by hand)."""
    frame = scene.frames(camera)[frame_idx]
    world_from_cam = scene.ego_pose_at(frame.ts) @ scene.ego_from_camera(camera)
    p = apply(np.linalg.inv(world_from_cam), point_world[None])[0]
    k = scene.intrinsics(camera)
    return np.array([k[0, 0] * p[0] / p[2] + k[0, 2], k[1, 1] * p[1] / p[2] + k[1, 2]])
