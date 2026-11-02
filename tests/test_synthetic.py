"""The synthetic scene itself: it honours the Scene contract and has the properties C4 to C6 rely on."""

import numpy as np
import pytest
from synthetic import CROSSER, GROUND, PARKED, WALL, SyntheticConfig, SyntheticScene

from longtail.data.base import Scene
from longtail.data.guard import LabelAccessError, eval_ground_truth
from longtail.geometry.transforms import world_to_ego


@pytest.fixture(scope="module")
def scene():
    return SyntheticScene()


def test_implements_scene_contract(scene):
    assert isinstance(scene, Scene)
    ts = scene.lidar_timestamps()
    sweep = scene.lidar(ts[3])
    assert sweep.points.shape[1] == 5 and sweep.frame_idx == 3
    assert np.all(sweep.points[:, 4] == ts[3])
    assert len(scene.frames("back_camera")) == scene.meta.n_frames == 30
    # ASSUMPTION A-29: ego poses and sweeps exist at LiDAR timestamps only
    with pytest.raises(KeyError):
        scene.ego_pose(ts[0] + 0.05)
    with pytest.raises(KeyError):
        scene.extrinsics("front_camera", ts[0])  # a LiDAR time, not this camera's


def test_labels_behind_guard(scene):
    with pytest.raises(LabelAccessError):
        scene.labels(0)
    with eval_ground_truth():
        assert {o["name"] for o in scene.labels(0)} == {"crosser", "parked"}


def test_every_part_is_present(scene):
    ids = scene.point_ids(scene.lidar_timestamps()[0])
    assert {GROUND, WALL, CROSSER, PARKED} == set(ids.tolist())


def test_crosser_starts_10m_ahead_and_crosses_at_1_5_mps(scene):
    ts = scene.lidar_timestamps()
    a, b = scene.objects(ts[0])["crosser"], scene.objects(ts[20])["crosser"]
    ahead = world_to_ego(scene.ego_pose(ts[0]), a.center_world[None])[0]
    assert ahead[0] == pytest.approx(10.0)
    speed = np.linalg.norm(b.center_world - a.center_world) / (ts[20] - ts[0])
    assert speed == pytest.approx(1.5)
    # It moves across the ego's path: from the right side to the left side.
    left_end = world_to_ego(scene.ego_pose(ts[0]), b.center_world[None])[0][1]
    assert ahead[1] < 0 < left_end


def test_parked_object_is_stationary_while_ego_moves(scene):
    # ASSUMPTION A-41: one stationary object
    ts = scene.lidar_timestamps()
    first, last = scene.objects(ts[0])["parked"], scene.objects(ts[-1])["parked"]
    np.testing.assert_array_equal(first.center_world, last.center_world)
    in_ego = [world_to_ego(scene.ego_pose(t), first.center_world[None])[0][0] for t in (ts[0], ts[-1])]
    assert in_ego[0] - in_ego[1] == pytest.approx(scene.cfg.ego_speed_mps * (ts[-1] - ts[0]))


def test_crosser_points_lie_on_its_box(scene):
    ts = scene.lidar_timestamps()[10]
    obj = scene.objects(ts)["crosser"]
    pts = scene.lidar(ts).points[scene.point_ids(ts) == CROSSER, :3]
    half = np.array(obj.lwh) / 2
    local = np.abs(pts - obj.center_world)  # yaw is shared, but extents are checked loosely
    assert np.all(local.max(axis=1) <= np.linalg.norm(half) + 1e-9)
    np.testing.assert_allclose(pts[:, 2].min(), 0.0, atol=1e-9)


def test_occlusion_removes_hidden_points():
    # ASSUMPTION A-43: only faces toward the LiDAR, and boxes occlude what is behind them
    ts = 1000.0
    seen = SyntheticScene()
    full = SyntheticScene(SyntheticConfig(occlusion=False))
    ids_seen, ids_full = seen.point_ids(ts), full.point_ids(ts)
    for oid in (GROUND, CROSSER, PARKED):
        assert (ids_seen == oid).sum() < (ids_full == oid).sum()
    # No ground point survives directly under the crosser.
    obj = seen.objects(ts)["crosser"]
    ground = seen.lidar(ts).points[ids_seen == GROUND, :3]
    under = world_to_ego(seen.ego_pose(ts), ground) - world_to_ego(seen.ego_pose(ts), obj.center_world[None])
    assert not np.any((np.abs(under[:, 0]) < 0.25) & (np.abs(under[:, 1]) < 0.25))


def test_parameters_change_the_scene():
    s = SyntheticScene(SyntheticConfig(n_frames=5, crosser_speed_mps=0.0, ego_speed_mps=0.0))
    ts = s.lidar_timestamps()
    assert len(ts) == 5
    np.testing.assert_array_equal(s.ego_pose(ts[0]), s.ego_pose(ts[-1]))
    np.testing.assert_array_equal(
        s.objects(ts[0])["crosser"].center_world, s.objects(ts[-1])["crosser"].center_world
    )
