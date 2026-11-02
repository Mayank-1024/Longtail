"""Project world-frame LiDAR points into a camera image (plan C2)."""

# ASSUMPTION A-38: no lens distortion; PandaSet's intrinsics.json has only fx, fy, cx, cy

from dataclasses import dataclass

import numpy as np

from longtail.data.base import CameraFrame, LidarSweep, Scene

# ASSUMPTION A-05: keep points with camera depth > 0.5 m
MIN_DEPTH_M = 0.5


@dataclass(frozen=True, eq=False)
class Projection:
    """Points that land in the image. Row i of each array describes the same point."""

    uv: np.ndarray  # (M, 2) float pixel coordinates, u right and v down
    depth: np.ndarray  # (M,) camera z in metres
    index: np.ndarray  # (M,) row of the point in the array passed to project()

    def __len__(self) -> int:
        return len(self.index)

    def pixels(self) -> np.ndarray:
        """(M, 2) integer (column, row) of the pixel each point falls in."""
        # ASSUMPTION A-37: pixel centres sit at integer coordinates (OpenCV); a point takes the nearest one
        return np.floor(self.uv + 0.5).astype(np.int64)


def project(
    points_world: np.ndarray,
    world_from_camera: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    min_depth: float = MIN_DEPTH_M,
) -> Projection:
    """Project (N, >=3) world points; columns beyond x, y, z are ignored.

    World to camera through the inverse extrinsics, then the intrinsics. Keeps points deeper than
    min_depth whose pixel lies inside the image.
    """
    r, t = world_from_camera[:3, :3], world_from_camera[:3, 3]
    # camera_from_world = [R^T, -R^T t]; as row vectors that is (p - t) @ R.
    cam = (points_world[:, :3] - t) @ r
    front = np.flatnonzero(cam[:, 2] > min_depth)
    cam = cam[front]
    k = intrinsics
    u = k[0, 0] * cam[:, 0] / cam[:, 2] + k[0, 1] * cam[:, 1] / cam[:, 2] + k[0, 2]
    v = k[1, 1] * cam[:, 1] / cam[:, 2] + k[1, 2]
    w, h = image_size
    # ASSUMPTION A-37: inside the image means the nearest pixel centre is a real pixel
    inside = (u >= -0.5) & (u < w - 0.5) & (v >= -0.5) & (v < h - 0.5)
    return Projection(uv=np.column_stack([u[inside], v[inside]]), depth=cam[inside, 2], index=front[inside])


def project_sweep(scene: Scene, sweep: LidarSweep, camera: str, frame: CameraFrame) -> Projection:
    """Project one sweep into one camera frame. Projection.index refers to rows of sweep.points."""
    return project(
        sweep.points, scene.extrinsics(camera, frame.ts), scene.intrinsics(camera), scene.image_size(camera)
    )
