"""C2 checks on real PandaSet data. Skipped when the dataset is not under $LONGTAIL_DATA_DIR/pandaset.

Our projection must agree with the devkit's to within 1 px. Both use the same calibration, so this only
catches convention and arithmetic errors; scripts/overlay.py is the visual check.
"""

import json

import numpy as np
import pytest

from longtail.data.guard import eval_ground_truth
from longtail.data.index import pandaset_root
from longtail.data.pandaset import PandaSetScene, list_scenes
from longtail.geometry.project import project
from longtail.geometry.sync import format_gap_distribution, gap_distribution

REAL = list_scenes(pandaset_root())
pytestmark = pytest.mark.skipif(not REAL, reason="PandaSet not downloaded")

N_POINTS = 100


def _check_points(scene: PandaSetScene) -> np.ndarray:
    """Labelled cuboid corners of frame 0 (labels allowed here: test only), topped up with LiDAR points."""
    from pandaset.geometry import center_box_to_corners

    with eval_ground_truth():
        cuboids = scene.labels(0)
    cols = ["position.x", "position.y", "position.z", "dimensions.x", "dimensions.y", "dimensions.z", "yaw"]
    corners = np.concatenate([center_box_to_corners(row) for row in cuboids[cols].to_numpy()])
    sweep = scene.lidar(scene.lidar_timestamps()[0])
    rng = np.random.default_rng(0)
    extra = sweep.points[rng.choice(len(sweep.points), size=min(5000, len(sweep.points)), replace=False), :3]
    return np.concatenate([corners, extra])


@pytest.mark.parametrize("camera", ["front_camera", "front_left_camera", "front_right_camera", "left_camera",
                                    "right_camera", "back_camera"])  # fmt: skip
def test_projection_matches_devkit(camera):
    from pandaset.geometry import projection
    from pandaset.sensors import Intrinsics
    from PIL import Image

    scene = PandaSetScene(pandaset_root(), REAL[0])
    pts = _check_points(scene)
    frame = scene.frames(camera)[0]
    ours = project(
        pts, scene.extrinsics(camera, frame.ts), scene.intrinsics(camera), scene.image_size(camera)
    )

    cam_dir = pandaset_root() / REAL[0] / "camera" / camera
    pose = json.loads((cam_dir / "poses.json").read_text())[0]
    k = json.loads((cam_dir / "intrinsics.json").read_text())
    with Image.open(frame.image_path) as img:
        dev_uv, _, dev_idx = projection(pts, img, pose, Intrinsics(k["fx"], k["fy"], k["cx"], k["cy"]))

    common, ours_at, dev_at = np.intersect1d(ours.index, dev_idx, return_indices=True)
    assert len(common) >= 20, f"{camera}: only {len(common)} points visible to both"
    pick = np.random.default_rng(1).choice(len(common), size=min(N_POINTS, len(common)), replace=False)
    err = np.linalg.norm(ours.uv[ours_at[pick]] - dev_uv[dev_at[pick]], axis=1)
    assert err.max() < 1.0, f"{camera}: max {err.max():.3f} px over {len(pick)} points"


def test_print_gap_distribution(capsys):
    scene = PandaSetScene(pandaset_root(), REAL[0])
    report = gap_distribution(scene)
    with capsys.disabled():
        print(f"\ncamera-to-LiDAR gaps, scene {REAL[0]}\n{format_gap_distribution(report)}")
    assert report["all"].n == 6 * scene.meta.n_frames
