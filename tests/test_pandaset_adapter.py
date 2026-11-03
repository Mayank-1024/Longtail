"""PandaSet adapter against a fake sequence read through the real devkit; dataset checks skip without data."""

import math
import time

import numpy as np
import pytest
from fake_pandaset import CAMERAS, IMAGE_SIZE, T0, write_sequence, yaw_pose

from longtail.data.base import TS_TOLERANCE_S
from longtail.data.guard import LabelAccessError, eval_ground_truth, labels_allowed
from longtail.data.index import pandaset_root
from longtail.data.pandaset import PandaSetScene, list_scenes, pose_to_matrix

pytest.importorskip("pandaset")


@pytest.fixture
def scene(tmp_path):
    write_sequence(tmp_path)
    return PandaSetScene(tmp_path, "001")


def test_list_scenes(tmp_path):
    write_sequence(tmp_path, "002")
    write_sequence(tmp_path, "001")
    (tmp_path / "not_a_sequence").mkdir()
    assert list_scenes(tmp_path) == ["001", "002"]
    assert list_scenes(tmp_path / "missing") == []


def test_missing_sequence_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PandaSetScene(tmp_path, "999")


def test_cameras_frames_and_intrinsics(scene):
    assert scene.cameras == CAMERAS
    frames = scene.frames("front_camera")
    assert [f.frame_idx for f in frames] == [0, 1, 2]
    assert frames[1].ts == pytest.approx(T0 + 0.105)
    assert frames[2].image_path.name == "02.jpg" and frames[2].image_path.parent.name == "front_camera"
    np.testing.assert_allclose(scene.intrinsics("front_camera"), [[900, 0, 960], [0, 901, 540], [0, 0, 1]])
    # ASSUMPTION A-42: image size comes from the first image's header
    assert scene.image_size("back_camera") == IMAGE_SIZE


def test_lidar_keeps_pandar64_points_with_timestamps(scene):
    sweep = scene.lidar(T0 + 0.1)
    assert sweep.frame_idx == 1 and sweep.ts == pytest.approx(T0 + 0.1)
    # ASSUMPTION A-18: Nx5 (x, y, z, intensity, t)
    assert sweep.points.shape == (4, 5)
    np.testing.assert_allclose(sweep.points[:, 4], T0 + 0.1 + np.array([0.0, 0.002, 0.004, 0.005]))
    # ASSUMPTION A-04: Pandar64 only, sensor id kept per point
    assert sweep.sensor_id.tolist() == [0, 0, 0, 0]
    assert sweep.point_index.tolist() == [0, 2, 4, 5]
    np.testing.assert_allclose(sweep.points[:, 0], [1.0, 3.0, 5.0, 6.0])


def test_timestamp_lookup_is_exact(scene):
    assert scene.lidar(T0 + 0.1 + TS_TOLERANCE_S / 2).frame_idx == 1
    # ASSUMPTION A-29: no nearest-frame matching in the adapter
    with pytest.raises(KeyError):
        scene.lidar(T0 + 0.05)
    with pytest.raises(KeyError):
        scene.ego_pose(T0 + 0.005)  # a camera timestamp, not a LiDAR one
    np.testing.assert_allclose(scene.lidar_timestamps(), [T0, T0 + 0.1, T0 + 0.2])


def test_pose_to_matrix_yaw():
    m = pose_to_matrix(yaw_pose(1.0, 2.0, 3.0, math.pi / 2))
    np.testing.assert_allclose(m[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(m[:3, 3], [1, 2, 3])


def test_ego_frame_axes(scene):
    # ASSUMPTION A-25: Pandar64 forward is +y and left is -x; the fake ego drives along world +y
    pose = scene.ego_pose(T0)
    np.testing.assert_allclose(pose[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(pose[:3, :3] @ [0, 1, 0], [-1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(pose[:3, :3] @ [0, 0, 1], [0, 0, 1], atol=1e-12)
    step = np.linalg.inv(pose) @ np.append(scene.ego_pose(T0 + 0.1)[:3, 3], 1.0)
    np.testing.assert_allclose(step[:3], [1.0, 0.0, 0.0], atol=1e-12)


def test_extrinsics_are_camera_poses(scene):
    # ASSUMPTION A-27: world_from_camera straight from the camera poses
    np.testing.assert_allclose(
        scene.extrinsics("left_camera", T0 + 0.105), pose_to_matrix(yaw_pose(0.3, -1.0, 1.5, 0.9))
    )


def test_meta(scene):
    m = scene.meta
    assert (m.scene_id, m.dataset, m.n_frames, m.weather) == ("001", "pandaset", 3, None)
    assert m.lat == pytest.approx(37.7761)
    # ASSUMPTION A-30: 22:00 PDT in May; the sun is about 19 degrees below the horizon
    assert m.time_of_day == "night"


def test_meta_day(tmp_path):
    write_sequence(tmp_path, t0=T0 + 12 * 3600)  # 17:00 UTC = 10:00 PDT; sun well up
    assert PandaSetScene(tmp_path, "001").meta.time_of_day == "day"


def test_labels_and_semseg_behind_guard(scene):
    # ASSUMPTION A-15: the guard covers cuboids and semantic segmentation
    for read in (scene.labels, scene.semseg):
        with pytest.raises(LabelAccessError):
            read(0)
    with eval_ground_truth():
        assert scene.labels(0)["label"].tolist() == ["Pedestrian", "Bicycle", "Car"]
        assert scene.semseg(0)["class"].tolist() == ["1"] * 6
        assert scene.has_cuboids


def test_semseg_absent_returns_none(tmp_path):
    write_sequence(tmp_path, semseg=False)
    with eval_ground_truth():
        assert PandaSetScene(tmp_path, "001").semseg(0) is None


def test_pipeline_api_never_touches_labels(scene, monkeypatch):
    # ASSUMPTION A-15: per-modality loaders only; label loaders run only inside the guard
    from pandaset.sequence import Sequence

    def forbidden(*_a, **_k):
        raise AssertionError("Sequence.load() called")

    def guarded(original):
        def wrapper(self, *a, **k):
            assert labels_allowed(), f"{original.__name__} called outside eval_ground_truth()"
            return original(self, *a, **k)

        return wrapper

    monkeypatch.setattr(Sequence, "load", forbidden)
    monkeypatch.setattr(Sequence, "load_cuboids", guarded(Sequence.load_cuboids))
    monkeypatch.setattr(Sequence, "load_semseg", guarded(Sequence.load_semseg))
    for cam in scene.cameras:
        for f in scene.frames(cam):
            scene.extrinsics(cam, f.ts)
        scene.intrinsics(cam)
    for ts in scene.lidar_timestamps():
        scene.lidar(ts)
        scene.ego_pose(ts)
    assert scene.meta.n_frames == 3
    with eval_ground_truth():
        scene.labels(0)
        scene.semseg(0)


# Real-data checks. They skip when PandaSet is not under $LONGTAIL_DATA_DIR/pandaset.

REAL = list_scenes(pandaset_root())
needs_dataset = pytest.mark.skipif(not REAL, reason="PandaSet not downloaded")


@needs_dataset
def test_real_scene_loads_in_under_5s():
    start = time.perf_counter()
    scene = PandaSetScene(pandaset_root(), REAL[0])
    assert len(scene.cameras) == 6
    for cam in scene.cameras:
        frames = scene.frames(cam)
        assert len(frames) == scene.meta.n_frames
        for f in frames:
            scene.extrinsics(cam, f.ts)
    for ts in scene.lidar_timestamps():
        assert scene.lidar(ts).points.shape[1] == 5
    assert time.perf_counter() - start < 5.0


@needs_dataset
def test_real_timing_conventions():
    # ASSUMPTION A-03: ts in seconds; PandaSet is 10 Hz
    scene = PandaSetScene(pandaset_root(), REAL[0])
    ts = scene.lidar_timestamps()
    assert len(ts) == 80
    assert np.median(np.diff(ts)) == pytest.approx(0.1, abs=0.01)


@needs_dataset
def test_real_ego_axes_and_world_up():
    # ASSUMPTION A-25: forward is the direction of travel
    # ASSUMPTION A-26: world z is up
    checked = 0
    for sid in REAL[:10]:
        scene = PandaSetScene(pandaset_root(), sid)
        ts = scene.lidar_timestamps()
        for a, b in zip(ts[:-1], ts[1:], strict=True):
            pa = scene.ego_pose(a)
            assert pa[2, 2] > math.cos(math.radians(10)), f"{sid}: ego z is not near world z"
            step = np.linalg.inv(pa) @ np.append(scene.ego_pose(b)[:3, 3], 1.0)
            if np.linalg.norm(step[:2]) / (b - a) > 3.0:  # moving faster than 3 m/s
                assert step[0] > 0.9 * np.linalg.norm(step[:3]), f"{sid}: ego does not move along +x"
                checked += 1
    assert checked > 0, "no moving frames found in the first 10 scenes"


@needs_dataset
@pytest.mark.parametrize(
    "camera, look",
    [
        ("front_camera", [1, 0, 0]),
        ("back_camera", [-1, 0, 0]),
        ("left_camera", [0, 1, 0]),
        ("right_camera", [0, -1, 0]),
    ],
)
def test_real_cameras_look_along_the_ego_axes(camera, look):
    # ASSUMPTION A-25: checked on the camera poses themselves, not on the devkit's static calibration file
    # (read as Pandar64-to-camera it is 180 degrees off these poses in yaw)
    # ASSUMPTION A-27: extrinsics are world_from_camera
    scene = PandaSetScene(pandaset_root(), REAL[0])
    frame = scene.frames(camera)[0]
    ego_from_cam = np.linalg.inv(scene.ego_pose(scene.lidar_timestamps()[0])) @ scene.extrinsics(
        camera, frame.ts
    )
    assert ego_from_cam[:3, 2] @ look > 0.95, f"{camera} optical axis {ego_from_cam[:3, 2]} in the ego frame"
    assert ego_from_cam[:3, 1] @ [0, 0, -1] > 0.95, f"{camera} image-down axis is not ego -z"
