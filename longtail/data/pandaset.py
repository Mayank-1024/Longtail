"""PandaSet adapter (plan C1) on the official devkit, pinned in pyproject.toml.

Devkit conventions, read from its source at the pinned commit:
- LiDAR points are in the world frame, columns x, y, z, i (intensity), t (per-point timestamp),
  d (sensor id: 0 = Pandar64 mechanical 360, 1 = forward-facing PandarGT).
- lidar/poses.json holds the Pandar64 pose in the world; camera/<name>/poses.json the camera pose.
  Positions are metres; headings are quaternions (w, x, y, z).
- The devkit's projection inverts the camera pose and keeps camera z > 0, so camera frames are
  OpenCV optical frames (x right, y down, z forward).
- Timestamps are seconds since the Unix epoch (README sample: 1.5575e9, May 2019).
"""

# ASSUMPTION A-01: dataset is PandaSet

import json
import math
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from longtail.data.base import CameraFrame, LidarSweep, Scene, SceneMeta, index_of

DEVKIT_COMMIT = "59be180e2a3f3e37f6d66af9e67bf944ccbf6ec0"

# ASSUMPTION A-04: Pandar64 only (devkit sensor 0)
LIDAR_SENSOR_ID = 0

CAMERA_ORDER = (
    "front_camera",
    "front_left_camera",
    "front_right_camera",
    "left_camera",
    "right_camera",
    "back_camera",
)

# ASSUMPTION A-25: the Pandar64 frame has forward = -y, left = +x, up = +z, and the ego origin is the Pandar64
# origin. Columns are the ego axes (x forward, y left, z up) written in the Pandar64 frame.
_R_P64_FROM_EGO = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
P64_FROM_EGO = np.eye(4)
P64_FROM_EGO[:3, :3] = _R_P64_FROM_EGO

# ASSUMPTION A-30: night when the sun is more than 6 degrees below the horizon (civil twilight)
NIGHT_SUN_ELEVATION_DEG = -6.0


def pose_to_matrix(pose: dict[str, dict[str, float]]) -> np.ndarray:
    """4x4 matrix from a devkit pose {position: {x, y, z}, heading: {w, x, y, z}}."""
    h, p = pose["heading"], pose["position"]
    w, x, y, z = h["w"], h["x"], h["y"], h["z"]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    m = np.eye(4)
    m[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    m[:3, 3] = [p["x"], p["y"], p["z"]]
    return m


def sun_elevation_deg(ts: float, lat: float, lon: float) -> float:
    """Approximate solar elevation (declination and hour angle; no equation of time), good to ~1 degree."""
    t = datetime.fromtimestamp(ts, tz=UTC)
    doy = t.timetuple().tm_yday
    decl = math.radians(-23.44) * math.cos(2 * math.pi / 365 * (doy + 10))
    solar_hours = t.hour + t.minute / 60 + t.second / 3600 + lon / 15
    hour_angle = math.radians(15 * (solar_hours - 12))
    la = math.radians(lat)
    s = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(hour_angle)
    return math.degrees(math.asin(max(-1.0, min(1.0, s))))


def list_scenes(root: Path) -> list[str]:
    """Sequence ids under a PandaSet root (directories holding lidar/ and camera/)."""
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "lidar").is_dir() and (d / "camera").is_dir())


def _read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


class PandaSetScene(Scene):
    def __init__(self, root: Path, scene_id: str) -> None:
        from pandaset.sequence import Sequence

        self.scene_id = scene_id
        self._dir = Path(root) / scene_id
        if not (self._dir / "lidar").is_dir():
            raise FileNotFoundError(f"no PandaSet sequence at {self._dir}")
        # Builds the file structure only; nothing is loaded until a loader is called.
        self._seq = Sequence(str(self._dir))
        self._lidar_loaded = False
        self._cuboids_loaded = False
        self._semseg_loaded = False

    # Metadata
    # ASSUMPTION A-28: poses, timestamps, intrinsics and GPS come straight from the devkit's documented JSON
    # files, because the devkit loads them only together with every point cloud or decoded JPEG

    @cached_property
    def cameras(self) -> tuple[str, ...]:
        present = {d.name for d in (self._dir / "camera").iterdir() if d.is_dir()}
        known = [c for c in CAMERA_ORDER if c in present]
        return tuple(known + sorted(present - set(known)))

    @cached_property
    def _camera_meta(self) -> dict[str, dict[str, Any]]:
        meta = {}
        for cam in self.cameras:
            d = self._dir / "camera" / cam
            # ASSUMPTION A-28: image paths sorted by name are in frame order, as the devkit loads them
            images = sorted(d.glob("*.jpg"))
            ts = np.asarray(_read_json(d / "timestamps.json"), dtype=np.float64)
            poses = _read_json(d / "poses.json")
            if not len(images) == len(ts) == len(poses):
                raise ValueError(f"{d}: {len(images)} images, {len(ts)} timestamps, {len(poses)} poses")
            meta[cam] = {"images": images, "ts": ts, "poses": poses, "K": _read_json(d / "intrinsics.json")}
        return meta

    @cached_property
    def _lidar_ts(self) -> np.ndarray:
        return np.asarray(_read_json(self._dir / "lidar" / "timestamps.json"), dtype=np.float64)

    @cached_property
    def _lidar_poses(self) -> list[dict[str, dict[str, float]]]:
        return _read_json(self._dir / "lidar" / "poses.json")

    @cached_property
    def meta(self) -> SceneMeta:
        gps_file = self._dir / "meta" / "gps.json"
        lat = lon = None
        if gps_file.is_file():
            first = _read_json(gps_file)[0]
            lat, lon = float(first["lat"]), float(first["long"])
        start = float(self._lidar_ts[0])
        time_of_day = None
        if lat is not None and lon is not None:
            night = sun_elevation_deg(start, lat, lon) < NIGHT_SUN_ELEVATION_DEG
            time_of_day = "night" if night else "day"
        return SceneMeta(
            scene_id=self.scene_id,
            dataset="pandaset",
            n_frames=len(self._lidar_ts),
            start_ts=start,
            lat=lat,
            lon=lon,
            time_of_day=time_of_day,
            weather=None,  # PandaSet has no weather field
        )

    def frames(self, camera: str) -> list[CameraFrame]:
        m = self._camera_meta[camera]
        return [
            CameraFrame(frame_idx=i, ts=float(t), image_path=p)
            for i, (t, p) in enumerate(zip(m["ts"], m["images"], strict=True))
        ]

    def intrinsics(self, camera: str) -> np.ndarray:
        k = self._camera_meta[camera]["K"]
        return np.array([[k["fx"], 0.0, k["cx"]], [0.0, k["fy"], k["cy"]], [0.0, 0.0, 1.0]])

    def extrinsics(self, camera: str, ts: float) -> np.ndarray:
        # ASSUMPTION A-27: camera frames are OpenCV optical frames; this returns world_from_camera
        m = self._camera_meta[camera]
        return pose_to_matrix(m["poses"][index_of(m["ts"], ts, f"{camera} frame")])

    # LiDAR

    def lidar_timestamps(self) -> np.ndarray:
        return self._lidar_ts.copy()

    def ego_pose(self, ts: float) -> np.ndarray:
        # ASSUMPTION A-29: ego poses exist at LiDAR timestamps only
        i = index_of(self._lidar_ts, ts, "LiDAR sweep")
        # ASSUMPTION A-25: ego frame is the Pandar64 frame rotated to x forward, y left, z up
        return pose_to_matrix(self._lidar_poses[i]) @ P64_FROM_EGO

    def lidar(self, ts: float) -> LidarSweep:
        i = index_of(self._lidar_ts, ts, "LiDAR sweep")
        if not self._lidar_loaded:
            # ASSUMPTION A-15: per-modality loader only, never sequence.load()
            self._seq.load_lidar()
            # ASSUMPTION A-04: Pandar64 only (devkit sensor 0)
            self._seq.lidar.set_sensor(LIDAR_SENSOR_ID)
            self._lidar_loaded = True
        df = self._seq.lidar[i]
        return LidarSweep(
            frame_idx=i,
            ts=float(self._lidar_ts[i]),
            # ASSUMPTION A-18: keep the per-point timestamp t as a fifth column
            points=df[["x", "y", "z", "i", "t"]].to_numpy(dtype=np.float64),
            # ASSUMPTION A-04: the sensor id is kept with every point set
            sensor_id=df["d"].to_numpy(dtype=np.int8),
            point_index=df.index.to_numpy(dtype=np.int64),
        )

    # Labels: reached only through Scene.labels() and Scene.semseg(), which check the guard first.

    @property
    def has_cuboids(self) -> bool:
        return self._seq.cuboids is not None

    def _labels(self, frame_idx: int) -> Any:
        if self._seq.cuboids is None:
            raise FileNotFoundError(f"{self._dir} has no cuboid annotations")
        if not self._cuboids_loaded:
            # ASSUMPTION A-15: per-modality loader only, never sequence.load()
            self._seq.load_cuboids()
            self._cuboids_loaded = True
        return self._seq.cuboids[frame_idx]

    def _semseg(self, frame_idx: int) -> Any:
        if self._seq.semseg is None:
            return None
        if not self._semseg_loaded:
            # ASSUMPTION A-15: per-modality loader only, never sequence.load()
            self._seq.load_semseg()
            self._semseg_loaded = True
        return self._seq.semseg[frame_idx]
