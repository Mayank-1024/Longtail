"""C2 on the synthetic scene: transforms, projection and sync."""

import time

import numpy as np
import pytest
from synthetic import CROSSER, PARKED, SyntheticConfig, SyntheticScene, world_to_pixel

from longtail.geometry import transforms as tf
from longtail.geometry.project import MIN_DEPTH_M, project, project_sweep
from longtail.geometry.sync import (
    MAX_GAP_S,
    format_gap_distribution,
    gap_distribution,
    nearest_sweeps,
    sync_camera,
    sync_scene,
)


@pytest.fixture(scope="module")
def scene():
    return SyntheticScene()


def _random_pose(rng):
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return tf.make_pose(r, rng.normal(scale=50, size=3))


# Transforms


def test_invert_is_exact_inverse():
    rng = np.random.default_rng(0)
    for _ in range(20):
        pose = _random_pose(rng)
        np.testing.assert_allclose(tf.invert(pose) @ pose, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(tf.invert(pose), np.linalg.inv(pose), atol=1e-9)


def test_world_ego_camera_round_trips(scene):
    rng = np.random.default_rng(1)
    pts = rng.normal(scale=30, size=(500, 3))
    ts = scene.lidar_timestamps()[7]
    ego = scene.ego_pose(ts)
    cam = scene.extrinsics("left_camera", scene.frames("left_camera")[7].ts)
    np.testing.assert_allclose(tf.ego_to_world(ego, tf.world_to_ego(ego, pts)), pts, atol=1e-9)
    np.testing.assert_allclose(tf.camera_to_world(cam, tf.world_to_camera(cam, pts)), pts, atol=1e-9)
    via_ego = tf.apply(tf.camera_from_ego(cam, ego), tf.world_to_ego(ego, pts))
    np.testing.assert_allclose(via_ego, tf.world_to_camera(cam, pts), atol=1e-9)


def test_ego_frame_axes(scene):
    # ASSUMPTION A-03: ego x forward, y left, z up
    t0 = scene.lidar_timestamps()[0]
    c = scene.cfg
    crosser = scene.objects(t0)["crosser"].center_world
    expected = [c.crosser_start[0], c.crosser_start[1], c.crosser_lwh[2] / 2 - c.sensor_height_m]
    np.testing.assert_allclose(tf.world_to_ego(scene.ego_pose(t0), crosser[None])[0], expected, atol=1e-9)
    np.testing.assert_allclose(tf.rot_z(np.pi / 2) @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_world_and_ego_differ_and_ego_moves_forward(scene):
    # ASSUMPTION A-40: the synthetic ego drives straight ahead at constant speed
    ts = scene.lidar_timestamps()
    a, b = scene.ego_pose(ts[0]), scene.ego_pose(ts[10])
    assert not np.allclose(a, np.eye(4))
    step = tf.world_to_ego(a, b[:3, 3][None])[0]
    np.testing.assert_allclose(step, [scene.cfg.ego_speed_mps * (ts[10] - ts[0]), 0, 0], atol=1e-9)


# Projection


def test_point_on_optical_axis_hits_principal_point(scene):
    frame = scene.frames("front_left_camera")[3]
    cam = scene.extrinsics("front_left_camera", frame.ts)
    point = cam[:3, 3] + 10.0 * cam[:3, 2]  # 10 m along the optical axis
    off = cam[:3, 3] + cam[:3, :3] @ [1.0, 0.5, 10.0]  # 1 m right, 0.5 m down, 10 m deep
    p = project(np.array([point, off]), cam, scene.intrinsics("front_left_camera"), scene.image_size("x"))
    np.testing.assert_allclose(p.uv, [[320, 240], [370, 265]], atol=1e-9)
    np.testing.assert_allclose(p.depth, [10, 10], atol=1e-9)


def test_objects_project_to_expected_pixels(scene):
    for i in (0, 15, 29):
        frame = scene.frames("front_camera")[i]
        objs = scene.objects(scene.lidar_timestamps()[i])
        centres = np.array([objs["crosser"].center_world, objs["parked"].center_world])
        p = project(
            centres, scene.extrinsics("front_camera", frame.ts), scene.intrinsics("x"), scene.image_size("x")
        )
        assert p.index.tolist() == [0, 1]
        for j in range(2):
            np.testing.assert_allclose(
                p.uv[j], world_to_pixel(scene, "front_camera", i, centres[j]), atol=1e-9
            )


def test_behind_near_and_outside_points_dropped(scene):
    frame = scene.frames("front_camera")[0]
    cam = scene.extrinsics("front_camera", frame.ts)
    k, size = scene.intrinsics("x"), scene.image_size("x")
    w, h = size

    def at(x, y, z):  # camera coordinates to world
        return cam[:3, 3] + cam[:3, :3] @ [x, y, z]

    f = scene.cfg.focal_px
    pts = np.array(
        [
            at(0, 0, -5),  # 0: behind
            at(0, 0, MIN_DEPTH_M - 0.01),  # 1: just inside the depth limit, dropped
            at(0, 0, MIN_DEPTH_M + 0.01),  # 2: just deeper, kept
            at(10 * (w / 2 + 5) / f, 0, 10),  # 3: right of the image
            at(0, -10 * (h / 2 + 5) / f, 10),  # 4: above the image
            at(10 * (-w / 2 - 0.45) / f, 0, 10),  # 5: u = -0.45, still nearest pixel 0, kept
            at(10 * (w / 2 - 0.45) / f, 0, 10),  # 6: u = w - 0.45, nearest "pixel" w, dropped
            at(1, 1, 8),  # 7: inside
        ]
    )
    p = project(pts, cam, k, size)
    # ASSUMPTION A-05: depth must exceed 0.5 m
    # ASSUMPTION A-37: a point is inside when its nearest pixel centre is a real pixel
    assert p.index.tolist() == [2, 5, 7]
    assert p.pixels()[1].tolist() == [0, h // 2]


def test_indices_round_trip_to_3d_points(scene):
    i = 12
    ts = scene.lidar_timestamps()[i]
    sweep = scene.lidar(ts)
    ids = scene.point_ids(ts)
    seen = set()
    for camera in scene.cameras:
        frame = scene.frames(camera)[i]
        p = project_sweep(scene, sweep, camera, frame)
        cam = scene.extrinsics(camera, frame.ts)
        back = tf.world_to_camera(cam, sweep.points[p.index, :3])
        np.testing.assert_allclose(back[:, 2], p.depth, atol=1e-9)
        np.testing.assert_allclose(
            back[:, :2] / back[:, 2:] * scene.cfg.focal_px + [320, 240], p.uv, atol=1e-6
        )
        seen.update(ids[p.index].tolist())
    assert {CROSSER, PARKED} <= seen


def test_extra_columns_ignored_and_empty_input(scene):
    frame = scene.frames("front_camera")[0]
    cam = scene.extrinsics("front_camera", frame.ts)
    sweep = scene.lidar(scene.lidar_timestamps()[0])
    full = project(sweep.points, cam, scene.intrinsics("x"), scene.image_size("x"))
    xyz = project(sweep.points[:, :3], cam, scene.intrinsics("x"), scene.image_size("x"))
    np.testing.assert_array_equal(full.index, xyz.index)
    assert len(project(np.empty((0, 5)), cam, scene.intrinsics("x"), scene.image_size("x"))) == 0


def test_full_frame_six_cameras_is_fast(scene):
    # PandaSet sweeps hold about 170k points across both LiDARs; the Pandar64 alone about 110k.
    rng = np.random.default_rng(2)
    t0 = scene.lidar_timestamps()[0]
    ego = scene.ego_pose(t0)
    pts = tf.ego_to_world(ego, rng.uniform([-60, -60, -2], [60, 60, 5], size=(170_000, 3)))
    frames = {c: scene.frames(c)[0] for c in scene.cameras}
    project(
        pts, scene.extrinsics("front_camera", frames["front_camera"].ts), scene.intrinsics("x"), (640, 480)
    )
    start = time.perf_counter()
    total = sum(
        len(project(pts, scene.extrinsics(c, f.ts), scene.intrinsics(c), scene.image_size(c)))
        for c, f in frames.items()
    )
    elapsed = time.perf_counter() - start
    assert total > 0
    assert elapsed < 0.5, f"six cameras took {elapsed:.3f} s"


# Sync


def test_sync_matches_same_sweep_by_default(scene):
    for frames in sync_scene(scene).values():
        assert [f.lidar_idx for f in frames] == list(range(scene.cfg.n_frames))
        assert all(f.synced and f.gap_s == pytest.approx(0.01) for f in frames)


def test_sync_picks_nearest_and_flags_large_gaps():
    s = SyntheticScene(
        SyntheticConfig(
            n_frames=5, camera_offsets_s={"front_camera": 0.07, "back_camera": -0.2, "left_camera": 0.05}
        )
    )
    front = sync_camera(s, "front_camera")
    # +70 ms is nearer the next sweep (30 ms before it), except for the last frame.
    assert [f.lidar_idx for f in front] == [1, 2, 3, 4, 4]
    assert [round(f.gap_s * 1000) for f in front] == [-30, -30, -30, -30, 70]
    assert [f.synced for f in front] == [True, True, True, True, False]
    back = sync_camera(s, "back_camera")
    assert [f.lidar_idx for f in back] == [0, 0, 0, 1, 2]
    assert [f.synced for f in back] == [False, False, True, True, True]
    # ASSUMPTION A-39: exactly 50 ms is synced; a tie goes to the earlier sweep
    left = sync_camera(s, "left_camera")
    assert all(f.synced for f in left)
    assert [f.lidar_idx for f in left][:4] == [0, 1, 2, 3]


def test_nearest_sweeps_edges():
    lidar = np.array([0.0, 0.25, 0.5])
    np.testing.assert_array_equal(
        nearest_sweeps(np.array([-1.0, 0.125, 0.126, 0.49, 9.0]), lidar), [0, 0, 1, 2, 2]
    )
    np.testing.assert_array_equal(nearest_sweeps(np.array([3.0, -3.0]), np.array([1.0])), [0, 0])


def test_gap_distribution_report():
    s = SyntheticScene(SyntheticConfig(n_frames=5, camera_offsets_s={"front_camera": 0.07}))
    report = gap_distribution(s)
    front = report["front_camera"]
    assert (front.n, front.n_unsynced, front.repeated_sweeps) == (5, 1, 1)
    assert front.max_ms == pytest.approx(70) and front.min_ms == pytest.approx(-30)
    assert report["all"].n == 30 and report["all"].n_unsynced == 1 and report["all"].repeated_sweeps == 1
    assert report["back_camera"].median_ms == pytest.approx(10)
    text = format_gap_distribution(report)
    assert "front_camera" in text and text.count("\n") == len(report) + 1
    assert MAX_GAP_S == 0.05
