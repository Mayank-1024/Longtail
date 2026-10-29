"""Rigid transforms between the world, ego and camera frames.

Poses are 4x4 matrices named a_from_b: they map coordinates in frame b to frame a. Scene.ego_pose()
returns world_from_ego and Scene.extrinsics() returns world_from_camera. Points are (N, 3) arrays.
"""

# ASSUMPTION A-03: ego frame is x forward, y left, z up, in metres
# ASSUMPTION A-27: camera frames are OpenCV optical frames (x right, y down, z forward)

import numpy as np


def make_pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def rot_z(yaw: float) -> np.ndarray:
    """Rotation by yaw radians about +z (counter-clockwise seen from above)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def invert(pose: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform, without a general matrix inverse."""
    r, t = pose[:3, :3], pose[:3, 3]
    return make_pose(r.T, -r.T @ t)


def apply(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Transform (N, 3) points by a 4x4 pose."""
    return points @ pose[:3, :3].T + pose[:3, 3]


def world_to_ego(world_from_ego: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    return apply(invert(world_from_ego), points_world)


def ego_to_world(world_from_ego: np.ndarray, points_ego: np.ndarray) -> np.ndarray:
    return apply(world_from_ego, points_ego)


def world_to_camera(world_from_camera: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    return apply(invert(world_from_camera), points_world)


def camera_to_world(world_from_camera: np.ndarray, points_camera: np.ndarray) -> np.ndarray:
    return apply(world_from_camera, points_camera)


def camera_from_ego(world_from_camera: np.ndarray, world_from_ego: np.ndarray) -> np.ndarray:
    """Pose that maps ego coordinates into a camera, given both poses in the world."""
    return invert(world_from_camera) @ world_from_ego
