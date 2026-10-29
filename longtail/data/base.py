"""Dataset-agnostic scene interface (plan C1), so a dataset switch never forces a rewrite."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from longtail.data.guard import require_label_access

# ASSUMPTION A-29: timestamp lookups are exact to within 1 ms; nearest-frame matching belongs to sync.py
TS_TOLERANCE_S = 1e-3


@dataclass(frozen=True)
class CameraFrame:
    # ASSUMPTION A-03: 0-based frame_idx; ts is the dataset timestamp in seconds
    frame_idx: int
    ts: float
    image_path: Path


@dataclass(frozen=True, eq=False)
class LidarSweep:
    """One LiDAR sweep in the world frame."""

    frame_idx: int
    ts: float
    # ASSUMPTION A-18: points are Nx5 (x, y, z, intensity, t) with the per-point timestamp kept
    points: np.ndarray
    # ASSUMPTION A-04: the sensor id is kept with every point set
    sensor_id: np.ndarray
    # Row index in the raw sweep, needed to join semantic segmentation in eval code.
    point_index: np.ndarray


@dataclass(frozen=True)
class SceneMeta:
    scene_id: str
    dataset: str
    n_frames: int
    start_ts: float
    lat: float | None
    lon: float | None
    # ASSUMPTION A-30: "day" or "night" from local solar time; None when there is no position
    time_of_day: str | None
    weather: str | None


class Scene(ABC):
    """One driving log. Poses are 4x4 world_from_X matrices; distances in metres, times in seconds."""

    scene_id: str

    @property
    @abstractmethod
    def cameras(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def meta(self) -> SceneMeta: ...

    @abstractmethod
    def frames(self, camera: str) -> list[CameraFrame]: ...

    @abstractmethod
    def lidar_timestamps(self) -> np.ndarray: ...

    @abstractmethod
    def lidar(self, ts: float) -> LidarSweep:
        """The sweep recorded at ts, points in the world frame."""

    @abstractmethod
    def ego_pose(self, ts: float) -> np.ndarray:
        """world_from_ego at a LiDAR timestamp."""

    @abstractmethod
    def intrinsics(self, camera: str) -> np.ndarray:
        """3x3 camera matrix K."""

    @abstractmethod
    def extrinsics(self, camera: str, ts: float) -> np.ndarray:
        """world_from_camera at a camera timestamp."""

    @abstractmethod
    def image_size(self, camera: str) -> tuple[int, int]:
        """(width, height) in pixels."""

    def labels(self, frame_idx: int) -> Any:
        """3D cuboid labels for one frame. Ground truth: only inside eval_ground_truth()."""
        require_label_access("cuboid labels")
        return self._labels(frame_idx)

    def semseg(self, frame_idx: int) -> Any:
        """Per-point semantic labels for one sweep, or None. Ground truth: only inside eval_ground_truth()."""
        # ASSUMPTION A-15: the guard covers cuboids and semantic segmentation
        require_label_access("semantic segmentation labels")
        return self._semseg(frame_idx)

    @abstractmethod
    def _labels(self, frame_idx: int) -> Any: ...

    @abstractmethod
    def _semseg(self, frame_idx: int) -> Any: ...


def index_of(timestamps: np.ndarray, ts: float, what: str) -> int:
    """Index of the timestamp equal to ts within TS_TOLERANCE_S, else KeyError."""
    i = int(np.argmin(np.abs(timestamps - ts)))
    if abs(timestamps[i] - ts) > TS_TOLERANCE_S:
        raise KeyError(f"no {what} at ts={ts:.6f} (nearest {timestamps[i]:.6f})")
    return i
