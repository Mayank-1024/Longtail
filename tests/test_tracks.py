"""C5 on the synthetic scene: 3D track motion, the ego's motion and its corridor."""

import numpy as np
import pytest
from synthetic import (
    SyntheticConfig,
    SyntheticScene,
    close_pass_config,
    object_track2d,
    stopped_ego_config,
)

from longtail.geometry.associate import associate
from longtail.geometry.ego_path import corridor, ego_motion
from longtail.geometry.tracks3d import track_motion
from longtail.verify.config import load_verify_config

CFG = load_verify_config().tracks
CAMERAS = ["front_camera", "front_right_camera", "right_camera", "back_camera"]
MOVING_MPS = 0.5  # plan C6: moving above 0.5 m/s for at least 1 s


def _tracked(scene, name, cameras, position=None, frames=None):
    tracks = [object_track2d(scene, name, c, obj_id=k, frames=frames) for k, c in enumerate(cameras)]
    cfg = CFG if position is None else CFG.model_copy(update={"position": position})
    return [track_motion(scene, a, cfg) for a in associate(scene, [(t, "car") for t in tracks if t.frames])]


def _longest_run_above(motion, threshold=MOVING_MPS) -> float:
    """Longest continuous time with reliable speed above threshold, within one speed series."""
    best = run = 0.0
    for k in range(len(motion.ts)):
        same = k > 0 and motion.segment[k] == motion.segment[k - 1]
        if motion.reliable[k] and motion.speed_mps[k] > threshold:
            run = run + (motion.ts[k] - motion.ts[k - 1] if same and run else 0.1)
            best = max(best, run)
        else:
            run = 0.0
    return best


@pytest.fixture(scope="module")
def scene():
    return SyntheticScene()


@pytest.fixture(scope="module")
def crosser(scene):
    [t] = _tracked(scene, "crosser", ["front_camera"])
    return t


# Crosser


def test_crosser_speed_is_1_5_mps(crosser):
    # Plan C5 done check, read as the track's speed (median over reliable steps).
    m = crosser.motion
    speeds = m.speed_mps[m.reliable]
    assert np.median(speeds) == pytest.approx(1.5, abs=0.1)
    # Per step it swings with the visible faces; C6's 1 s rule absorbs this. Bounded here; see eval-report.
    assert np.abs(speeds - 1.5).max() < 0.5
    assert crosser.track.speed_mps == pytest.approx(
        [m.speed_mps[m.index(s.ts)] for s in crosser.track.samples]
    )


def test_crosser_velocity_points_across_its_path(scene, crosser):
    # ASSUMPTION A-70: ego-frame velocity is the world velocity in the ego's axes
    m = crosser.motion
    v = np.median(m.velocity_ego[m.reliable], axis=0)
    np.testing.assert_allclose(v[:2], [0.0, 1.5], atol=0.1)
    np.testing.assert_allclose(
        np.median(m.velocity_world[m.reliable], axis=0)[:2], 1.5 * scene._left[:2], atol=0.1
    )


def test_crosser_moves_across_the_corridor(scene, crosser):
    m = crosser.motion
    sample = crosser.track.samples[0]
    c = corridor(scene, sample.ts)
    pos = c.locate(np.array([sample.xyz_ego[:2]]))
    assert pos.lateral_m[0] < 0 and not pos.inside[0]  # on the right, outside the corridor for now
    normal = np.array([-pos.tangent[0, 1], pos.tangent[0, 0]])  # points left
    toward = -np.sign(pos.lateral_m[0]) * (m.velocity_ego[m.index(sample.ts), :2] @ normal)
    assert toward > 1.0  # plan C6 needs lateral speed toward the corridor above 0.3 m/s


# Ego motion cancels


def test_parked_object_reads_still_while_the_ego_moves(scene):
    assert np.median(ego_motion(scene).speed_mps) == pytest.approx(2.0)
    for t in _tracked(scene, "parked", ["front_camera", "front_right_camera"]):
        m = t.motion
        assert np.median(m.speed_mps[m.reliable]) < 0.3
        # The crosser passes in front of it early on; the box centre keeps that under the moving threshold.
        assert m.speed_mps[m.reliable].max() < MOVING_MPS


def test_close_pass_median_drifts_box_centre_does_not():
    # ASSUMPTION A-65: a 4.5 m car whose near side is 2.0 m from the ego's path, passed at 5 m/s
    # ASSUMPTION A-68: the position used for speed is a face-anchored, truncation-aware box centre
    s = SyntheticScene(close_pass_config())
    before = _tracked(s, "pass_car", CAMERAS, position="median")
    after = _tracked(s, "pass_car", CAMERAS, position="box_centre")
    # The median follows the visible part of the car and would read as moving for over 1 s.
    assert max(_longest_run_above(t.motion) for t in before) >= 1.0
    for t in after:
        # Spikes remain where the car enters or leaves a camera's view; never for 1 s, so C6 reads it parked.
        assert _longest_run_above(t.motion) < 1.0
    # The merged front, front-right and right track reads near zero overall.
    [merged] = [t for t in after if len(t.track.sources) > 1]
    assert np.median(merged.motion.speed_mps[merged.motion.reliable]) < 0.3


@pytest.mark.xfail(
    strict=True,
    reason="a parked car partly hidden by another car during the close pass still reads as moving for 1.2 s "
    "with the box centre (2.1 s with the median); see docs/eval-report.md",
)
def test_occluded_parked_car_during_close_pass():
    s = SyntheticScene(close_pass_config())
    for t in _tracked(s, "parked", CAMERAS):
        assert _longest_run_above(t.motion) < 1.0


# Gaps and short tracks


def test_short_gaps_interpolated_long_gaps_split(scene):
    # ASSUMPTION A-67: gaps up to 0.3 s are interpolated, longer ones split the series
    frames = [i for i in range(30) if i not in (10, 11, 20, 21, 22)]
    [t] = _tracked(scene, "crosser", ["front_camera"], frames=frames)
    m = t.motion
    assert sorted(set(m.segment.tolist())) == [0, 1]
    first, second = m.ts[m.segment == 0], m.ts[m.segment == 1]
    ts = scene.lidar_timestamps()
    np.testing.assert_allclose(first, ts[0:20])
    np.testing.assert_allclose(second, ts[23:30])
    assert m.observed.tolist().count(False) == 2  # frames 10 and 11
    assert not m.observed[m.index(ts[10])] and not m.observed[m.index(ts[11])]
    for seg in (0, 1):
        assert np.median(m.speed_mps[m.segment == seg]) == pytest.approx(1.5, abs=0.15)
    assert len(t.track.speed_mps) == len(t.track.samples) == 25


def test_short_track_is_not_smoothed_and_unreliable(scene):
    # ASSUMPTION A-71: under the Savitzky-Golay window, no smoothing and speed marked unreliable
    [t] = _tracked(scene, "crosser", ["front_camera"], frames=range(4))
    m = t.motion
    assert len(m.ts) == 4 and not m.reliable.any()
    np.testing.assert_array_equal(m.smoothed_world, m.position_world)


def test_empty_track(scene):
    masks_nowhere = object_track2d(scene, "crosser", "back_camera")
    assert masks_nowhere.frames == []
    from longtail.contracts import Track3D
    from longtail.geometry.associate import Association

    a = Association(track=Track3D(track_id="x", cls="car", sources=[], samples=[], speed_mps=[]), skipped=[],
                    truncated=False, clusters={})  # fmt: skip
    t = track_motion(scene, a)
    assert t.track.speed_mps == [] and len(t.motion.ts) == 0


# Ego path


def test_ego_speed_over_time():
    assert np.allclose(ego_motion(SyntheticScene(close_pass_config())).speed_mps, 5.0)
    assert np.allclose(ego_motion(SyntheticScene(stopped_ego_config())).speed_mps, 0.0)


def test_corridor_straight_moving_ego():
    s = SyntheticScene(SyntheticConfig(ego_speed_mps=5.0, n_frames=40))
    c = corridor(s, s.lidar_timestamps()[0])
    assert not c.extended and not c.future_truncated
    assert c.length_m == pytest.approx(15.0)
    np.testing.assert_allclose(c.centerline[:, 1], 0.0, atol=1e-9)
    assert c.half_width_m == 1.5


def test_corridor_follows_a_turning_ego():
    # ASSUMPTION A-64: the synthetic ego can drive an arc
    w, v = 0.2, 5.0
    s = SyntheticScene(SyntheticConfig(ego_speed_mps=v, ego_yaw_rate_rps=w, n_frames=40))
    ts = s.lidar_timestamps()
    c = corridor(s, ts[5])
    tau = ts[5:36] - ts[5]
    r = v / w
    expected = np.column_stack([r * np.sin(w * tau), r * (1 - np.cos(w * tau))])  # left turn: +y
    np.testing.assert_allclose(c.centerline, expected, atol=1e-9)
    assert c.length_m == pytest.approx(v * 3.0, rel=1e-3)


def test_corridor_for_a_stopped_ego_has_the_minimum_length():
    # ASSUMPTION A-69: extended straight ahead to min_length_m
    s = SyntheticScene(stopped_ego_config())
    c = corridor(s, s.lidar_timestamps()[3])
    min_len = load_verify_config().ego_path.min_length_m
    assert c.extended and c.length_m == pytest.approx(min_len)
    np.testing.assert_allclose(c.centerline, [[0.0, 0.0], [min_len, 0.0]], atol=1e-9)
    pos = c.locate(np.array([[5.0, 1.0], [5.0, -2.0], [12.0, 0.0], [-1.0, 0.0]]))
    assert pos.inside.tolist() == [True, False, False, False]
    np.testing.assert_allclose(pos.lateral_m[:2], [1.0, -2.0])


def test_corridor_near_the_end_of_the_log():
    # ASSUMPTION A-72: uses the future that exists, and says so
    s = SyntheticScene(SyntheticConfig(ego_speed_mps=5.0))
    c = corridor(s, s.lidar_timestamps()[-5])
    assert c.future_truncated and c.length_m == pytest.approx(10.0)
