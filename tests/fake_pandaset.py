"""A tiny PandaSet sequence on disk, in the devkit's layout, so the adapter is tested without the dataset."""

import json
import math
from pathlib import Path

import pandas as pd
from PIL import Image

CAMERAS = (
    "front_camera",
    "front_left_camera",
    "front_right_camera",
    "left_camera",
    "right_camera",
    "back_camera",
)
T0 = 1557550800.0  # 2019-05-11 05:00 UTC = 22:00 PDT, night in San Francisco
SF = {"lat": 37.7761, "long": -122.3993}
IMAGE_SIZE = (64, 48)


def yaw_pose(x: float, y: float, z: float, yaw: float) -> dict:
    return {
        "position": {"x": x, "y": y, "z": z},
        "heading": {"w": math.cos(yaw / 2), "x": 0.0, "y": 0.0, "z": math.sin(yaw / 2)},
    }


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def write_sequence(
    root: Path, seq_id: str = "001", n_frames: int = 3, t0: float = T0, semseg: bool = True
) -> Path:
    """Write a sequence where the Pandar64 moves 1 m per frame along world -y with identity heading."""
    seq = root / seq_id
    lidar_ts = [t0 + 0.1 * k for k in range(n_frames)]
    for k, ts in enumerate(lidar_ts):
        # Points interleave the two sensors so filtering has to keep the original row index.
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "y": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
                "z": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "i": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "t": [ts + 0.001 * j for j in range(6)],
                "d": [0, 1, 0, 1, 0, 0],
            }
        )
        (seq / "lidar").mkdir(parents=True, exist_ok=True)
        df.to_pickle(seq / "lidar" / f"{k:02d}.pkl.gz")
    _dump(seq / "lidar" / "timestamps.json", lidar_ts)
    _dump(seq / "lidar" / "poses.json", [yaw_pose(0.0, -1.0 * k, 1.8, 0.0) for k in range(n_frames)])

    for c, cam in enumerate(CAMERAS):
        d = seq / "camera" / cam
        d.mkdir(parents=True, exist_ok=True)
        for k in range(n_frames):
            Image.new("RGB", IMAGE_SIZE).save(d / f"{k:02d}.jpg")
        _dump(d / "timestamps.json", [ts + 0.005 for ts in lidar_ts])
        _dump(d / "poses.json", [yaw_pose(0.1 * c, -1.0 * k, 1.5, 0.3 * c) for k in range(n_frames)])
        _dump(d / "intrinsics.json", {"fx": 900.0 + c, "fy": 901.0, "cx": 960.0, "cy": 540.0})

    _dump(seq / "meta" / "gps.json", [{**SF, "height": 3.0, "xvel": 0.0, "yvel": 10.0}] * n_frames)
    _dump(seq / "meta" / "timestamps.json", lidar_ts)

    for k in range(n_frames):
        cub = pd.DataFrame(
            {
                "uuid": ["p1", "b1", "c1"],
                "label": ["Pedestrian", "Bicycle", "Car"],
                "attributes.pedestrian_behavior": ["Walking", None, None],
                "attributes.rider_status": [None, "With Rider", None],
                "cuboids.sensor_id": [-1, -1, -1],
            }
        )
        (seq / "annotations" / "cuboids").mkdir(parents=True, exist_ok=True)
        cub.to_pickle(seq / "annotations" / "cuboids" / f"{k:02d}.pkl.gz")
        if semseg:
            (seq / "annotations" / "semseg").mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"class": ["1"] * 6}).to_pickle(seq / "annotations" / "semseg" / f"{k:02d}.pkl.gz")
    if semseg:
        _dump(seq / "annotations" / "semseg" / "classes.json", {"1": "Smoke"})
    return seq
