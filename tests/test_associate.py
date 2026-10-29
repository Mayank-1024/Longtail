"""C4 mask-to-LiDAR association on the synthetic scene."""

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.ndimage import binary_erosion
from synthetic import (
    CROSSER,
    GROUND,
    PARKED,
    PED_A,
    PED_B,
    WALL,
    SyntheticConfig,
    SyntheticScene,
    disk,
    object_track2d,
    rect_mask,
    track2d_from_masks,
    without_masks,
)

from longtail.geometry.associate import (
    SweepContext,
    associate,
    fit_ground_plane,
    nearest_depth_part,
    remove_ground,
    visible_in_camera,
)
from longtail.geometry.project import project_sweep
from longtail.verify.config import AssociationConfig, load_verify_config
from longtail.vision.rle import decode

CFG = load_verify_config().association


def _cfg(**update) -> AssociationConfig:
    ground = update.pop("ground", None)
    merge = update.pop("merge", None)
    cfg = CFG.model_copy(update=update)
    if ground is not None:
        cfg = cfg.model_copy(update={"ground": CFG.ground.model_copy(update=ground)})
    if merge is not None:
        cfg = cfg.model_copy(update={"merge": CFG.merge.model_copy(update=merge)})
    return cfg


@pytest.fixture(scope="module")
def scene():
    return SyntheticScene()


@pytest.fixture(scope="module")
def crosser(scene):
    """The clean crosser in the front camera, associated once for several tests."""
    [a] = associate(scene, [(object_track2d(scene, "crosser", "front_camera"), "pedestrian")])
    return a


def _purity(scene, assoc, oid) -> list[float]:
    out = []
    for t, clusters in sorted(assoc.clusters.items()):
        rows = np.unique(np.concatenate([c.point_index for c in clusters]))
        out.append(float(np.mean(scene.point_ids(t)[rows] == oid)))
    return out


def _xy_errors(scene, assoc, name) -> np.ndarray:
    return np.array(
        [
            np.linalg.norm(np.array(s.xyz_world[:2]) - scene.objects(s.ts)[name].center_world[:2])
            for s in assoc.track.samples
        ]
    )


def _raw_ids(scene, track, frame_idx, erode_px=0):
    """Object ids of every point inside the mask of one frame (uneroded unless erode_px is given)."""
    frame = next(f for f in track.frames if f.frame_idx == frame_idx)
    sweep = scene.lidar(scene.lidar_timestamps()[frame_idx])
    p = project_sweep(scene, sweep, track.camera, scene.frames(track.camera)[frame_idx])
    px = p.pixels()
    mask = decode(frame.mask_rle)
    if erode_px:
        mask = binary_erosion(mask, structure=disk(erode_px))
    inside = mask[px[:, 1], px[:, 0]]
    return scene.point_ids(sweep.ts)[p.index[inside]], sweep, sweep.points[p.index[inside], :3]


# The clean crosser


def test_crosser_cluster_is_the_crosser_in_every_frame(scene, crosser):
    assert len(crosser.track.samples) == scene.cfg.n_frames and not crosser.skipped
    assert min(_purity(scene, crosser, CROSSER)) == 1.0


def test_crosser_centroid_error_is_bounded_by_surface_bias(scene, crosser):
    # LiDAR returns lie on the faces toward the sensor (A-43), so the median sits up to half the box depth in
    # front of the true centre. Association error on top of that must stay within the plan's 0.15 m.
    half_depth = max(scene.cfg.crosser_lwh[:2]) / 2
    assert _xy_errors(scene, crosser, "crosser").max() < 0.15 + half_depth


@pytest.mark.xfail(
    strict=True,
    reason="plan C4 done check (< 0.15 m from the box centre) cannot hold with surface-only LiDAR returns: "
    "the median of the visible faces sits about half the box depth (0.3 m) in front of the centre",
)
def test_crosser_centroid_error_plan_criterion(scene, crosser):
    assert _xy_errors(scene, crosser, "crosser").max() < 0.15


def test_samples_carry_both_frames_and_counts(scene, crosser):
    s = crosser.track.samples[0]
    # ASSUMPTION A-52: samples are stamped with the sweep timestamp
    assert s.ts == scene.lidar_timestamps()[0]
    from longtail.geometry.transforms import world_to_ego

    ego_of_world = world_to_ego(scene.ego_pose(s.ts), np.array([s.xyz_world]))[0]
    # Medians are taken in each frame separately (plan C4 step 5), so they differ by a few centimetres.
    np.testing.assert_allclose(s.xyz_ego, ego_of_world, atol=0.15)
    assert s.n_points >= 15 and s.coverage >= s.n_points
    length, width, height = s.extent_lwh
    assert length >= width
    # ASSUMPTION A-53: height is the top above the local ground, so the 0.2 m ground cut does not shrink it
    assert height == pytest.approx(scene.cfg.crosser_lwh[2], abs=0.15)
    # ASSUMPTION A-48: C5 fills speeds
    assert crosser.track.speed_mps == []
    assert crosser.track.cls == "pedestrian" and crosser.track.sources == [("front_camera", 1)]


# Masks that include background


def test_dilated_mask_returns_the_object_not_wall_or_ground(scene):
    track = object_track2d(scene, "crosser", "front_camera", dilate_px=15)
    raw = {i: _raw_ids(scene, track, i)[0] for i in (15, 25)}
    assert (raw[15] == GROUND).sum() > 100 and (raw[25] == WALL).sum() > 100  # the spill is real
    [a] = associate(scene, [(track, "pedestrian")])
    assert len(a.track.samples) == scene.cfg.n_frames
    assert min(_purity(scene, a, CROSSER)) == 1.0


def test_wall_points_seen_past_the_object_are_dropped(scene):
    # Without the visibility filter (A-63), LiDAR points on the wall behind the crosser win the depth split.
    track = object_track2d(scene, "crosser", "front_camera", frames=range(22, 25))
    [without] = associate(scene, [(track, "pedestrian")], _cfg(visibility_cell_px=0))
    [with_filter] = associate(scene, [(track, "pedestrian")])
    assert min(_purity(scene, without, CROSSER)) < 0.5
    assert min(_purity(scene, with_filter, CROSSER)) == 1.0


def test_mask_over_empty_space_is_unverifiable(scene):
    masks = {i: rect_mask(scene, 420, 0, 600, 120) for i in range(5)}
    [a] = associate(scene, [(track2d_from_masks(scene, "front_camera", "person", 7, masks), "pedestrian")])
    assert a.track.samples == []
    assert {(s.reason, s.coverage < CFG.min_coverage) for s in a.skipped} == {("low_coverage", True)}


def test_mask_on_wall_only_returns_the_wall_at_its_real_size(scene):
    masks = {i: rect_mask(scene, 220, 160, 420, 320) for i in range(0, 30, 5)}
    track = track2d_from_masks(scene, "left_camera", "person", 3, masks)
    [a] = associate(scene, [(track, "pedestrian")])
    assert len(a.track.samples) == len(masks) and min(_purity(scene, a, WALL)) == 1.0
    for sample, i in zip(a.track.samples, sorted(masks), strict=True):
        ids, _, pts = _raw_ids(scene, track, i, erode_px=CFG.erode_px)
        wall = pts[ids == WALL]
        length, width, height = sample.extent_lwh
        assert length == pytest.approx(np.ptp(wall @ scene._fwd), abs=0.3)
        assert height == pytest.approx(wall[:, 2].max(), abs=0.3)  # ground is z = 0 here
        # Plan C6 pedestrian prior: footprint 0.3 to 1.2 m, so C6 can FAIL this on size.
        assert width < 0.05 and length > 1.2 and height > 1.5


# Ground


def test_sloped_ground_local_plane_removes_ground_and_keeps_object():
    # ASSUMPTION A-49: local plane; the global plane is fitted to the flat stretch and misses the slope
    # ASSUMPTION A-60: flat, then a 15% grade from 8 m ahead
    s = SyntheticScene(SyntheticConfig(ground_grade=0.15, slope_start_m=8.0, extent_ahead=(-10.0, 20.0)))
    track = object_track2d(s, "crosser", "front_camera", dilate_px=10)
    kept = {}
    for mode in ("local", "global"):
        cfg = _cfg(ground={"mode": mode})
        ground_kept = crosser_kept = crosser_total = 0
        for i in (0, 10, 20):
            ids, sweep, pts = _raw_ids(s, track, i)
            keep, used, _ = remove_ground(pts, SweepContext(sweep, cfg.ground), cfg)
            assert used == mode
            ground_kept += int(((ids == GROUND) & keep).sum())
            crosser_kept += int(((ids == CROSSER) & keep).sum())
            crosser_total += int((ids == CROSSER).sum())
        kept[mode] = (ground_kept, crosser_kept / crosser_total)
    assert kept["local"][0] == 0 and kept["local"][1] > 0.8
    assert kept["global"][0] > 0
    [a] = associate(s, [(track, "pedestrian")])
    assert len(a.track.samples) == s.cfg.n_frames and min(_purity(s, a, CROSSER)) == 1.0
    # Height is measured from the sloped ground under the object, not from z = 0.
    heights = [x.extent_lwh[2] for x in a.track.samples]
    assert max(abs(h - s.cfg.crosser_lwh[2]) for h in heights) < 0.15


def test_ground_plane_is_not_a_roof():
    rng = np.random.default_rng(0)
    ground = np.column_stack([rng.uniform(-5, 5, (300, 2)), np.zeros(300)])
    roof = np.column_stack([rng.uniform(-2, 2, (800, 2)), np.full(800, 1.5)])
    sides = np.column_stack([rng.uniform(-2, 2, 400), np.full(400, 2.0), rng.uniform(0, 1.5, 400)])
    plane = fit_ground_plane(np.concatenate([ground, roof, sides]), CFG.ground, rng)
    assert plane is not None and abs(plane.d) < 0.05 and plane.normal[2] > 0.99


# Clusters and selection helpers


def test_nearest_depth_part():
    depth = np.array([5.0, 5.2, 5.4, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6])
    assert sorted(nearest_depth_part(depth, CFG)) == [0, 1, 2]  # 3 of 10 is 30%
    # ASSUMPTION A-51: when no part holds 30%, the largest is used
    depth = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 9.1, 11.0, 13.0, 15.0, 17.0])
    assert sorted(nearest_depth_part(depth, CFG)) == [4, 5]


def test_visibility_filter():
    pixels = np.array([[0, 0], [1, 1], [2, 2], [20, 20]])
    depth = np.array([5.0, 5.9, 7.0, 30.0])
    # ASSUMPTION A-63: per 8 px cell, drop points more than 1 m behind the nearest
    assert visible_in_camera(pixels, depth, CFG).tolist() == [True, True, False, True]


def test_two_close_pedestrians_stay_apart():
    # ASSUMPTION A-61: two pedestrians 0.6 m apart, each with its own mask
    s = SyntheticScene(SyntheticConfig(pedestrian_pair=True))
    tracks = [
        (object_track2d(s, n, "front_camera", obj_id=k), "pedestrian")
        for k, n in ((1, "ped_a"), (2, "ped_b"))
    ]
    a, b = sorted(associate(s, tracks), key=lambda r: r.track.sources)
    assert a.track.sources == [("front_camera", 1)] and b.track.sources == [("front_camera", 2)]
    assert min(_purity(s, a, PED_A)) == 1.0 and min(_purity(s, b, PED_B)) == 1.0
    shared = {x.ts for x in a.track.samples} & {x.ts for x in b.track.samples}
    gaps = [
        np.linalg.norm(np.subtract(sa.xyz_world[:2], sb.xyz_world[:2]))
        for sa in a.track.samples
        for sb in b.track.samples
        if sa.ts == sb.ts
    ]
    assert len(shared) > 20 and min(gaps) > 0.4


# Merging


def test_crosser_entering_a_second_camera_merges():
    s = SyntheticScene(SyntheticConfig(n_frames=40))
    front = object_track2d(s, "crosser", "front_camera", obj_id=4)
    left = object_track2d(s, "crosser", "front_left_camera", obj_id=9)
    shared = {f.frame_idx for f in front.frames} & {f.frame_idx for f in left.frames}
    assert len(shared) >= 3
    [merged] = associate(s, [(front, "pedestrian"), (left, "pedestrian")])
    assert merged.track.sources == [("front_camera", 4), ("front_left_camera", 9)]
    assert len(merged.track.samples) == s.cfg.n_frames
    assert min(_purity(s, merged, CROSSER)) == 1.0


def test_cars_parked_nose_to_tail_stay_apart():
    # ASSUMPTION A-74: a few shared points from a mask spilling onto the neighbour are not enough to merge
    # ASSUMPTION A-75: cars nose to tail; front-left sees car_a (mask spills onto car_b), left sees car_b
    s = SyntheticScene(SyntheticConfig(nose_to_tail=True, n_frames=8, ego_speed_mps=0.0, wall_left_m=14.0))
    a = object_track2d(s, "car_a", "front_left_camera", concept="car", obj_id=1, dilate_px=8)
    b = object_track2d(s, "car_b", "left_camera", concept="car", obj_id=2)
    tracks = [(a, "car"), (b, "car")]
    loose = _cfg(merge={"min_shared_points_frac": 0.0})
    assert len(associate(s, tracks, loose)) == 1  # points alone would wrongly join the two cars
    apart = associate(s, tracks)
    assert sorted(r.track.sources for r in apart) == [[("front_left_camera", 1)], [("left_camera", 2)]]


def test_different_classes_or_far_tracks_do_not_merge():
    s = SyntheticScene(SyntheticConfig(n_frames=40))
    front = object_track2d(s, "crosser", "front_camera", obj_id=4)
    left = object_track2d(s, "crosser", "front_left_camera", obj_id=9)
    assert len(associate(s, [(front, "pedestrian"), (left, "cyclist")])) == 2
    parked = object_track2d(s, "parked", "front_right_camera", obj_id=5)
    assert len(associate(s, [(front, "pedestrian"), (parked, "pedestrian")])) == 2


def test_split_track_in_one_camera_merges_back():
    # ASSUMPTION A-56: fragments are joined by extrapolating across the gap
    s = SyntheticScene()
    first = object_track2d(s, "crosser", "front_camera", obj_id=1, frames=range(0, 12))
    second = object_track2d(s, "crosser", "front_camera", obj_id=2, frames=range(16, 30))
    [merged] = associate(s, [(second, "pedestrian"), (first, "pedestrian")])
    assert merged.track.sources == [("front_camera", 1), ("front_camera", 2)]
    assert len(merged.track.samples) == 26


def test_fragments_of_different_objects_stay_apart():
    s = SyntheticScene()
    first = object_track2d(s, "crosser", "front_camera", obj_id=1, frames=range(0, 12))
    other = object_track2d(s, "parked", "front_camera", obj_id=2, frames=range(16, 30))
    assert len(associate(s, [(first, "pedestrian"), (other, "pedestrian")])) == 2


# Frustum and box-only


@pytest.mark.parametrize("mode", ["frustum", "box-only"])
def test_frustum_and_box_only_agree_with_mask_mode(scene, crosser, mode):
    track = object_track2d(scene, "crosser", "front_camera")
    if mode == "frustum":
        [a] = associate(scene, [(track, "pedestrian")], _cfg(selection="frustum"))
    else:
        [a] = associate(scene, [(without_masks(track), "pedestrian")])
    mask_xy = {s.ts: np.array(s.xyz_world[:2]) for s in crosser.track.samples}
    diffs = [
        np.linalg.norm(np.array(s.xyz_world[:2]) - mask_xy[s.ts]) for s in a.track.samples if s.ts in mask_xy
    ]
    assert len(diffs) >= 25 and max(diffs) < 0.5


# Sync, truncation, inputs


def test_unsynced_frames_are_skipped_with_a_reason():
    s = SyntheticScene(SyntheticConfig(n_frames=6, camera_offsets_s={"front_camera": -0.2}))
    track = object_track2d(s, "crosser", "front_camera", truncated=True)
    [a] = associate(s, [(track, "pedestrian")])
    # ASSUMPTION A-46: skipped frames carry their reason
    assert [(k.frame_idx, k.reason) for k in a.skipped] == [(0, "unsynced"), (1, "unsynced")]
    assert len(a.track.samples) == 4
    # ASSUMPTION A-47: truncated passes through
    assert a.truncated is True


def test_mask_size_must_match_image(scene):
    track = object_track2d(scene, "crosser", "front_camera", frames=[0])
    small = SyntheticScene(SyntheticConfig(image_size=(320, 240)))
    with pytest.raises(ValueError, match="mask is"):
        associate(small, [(track.model_copy(update={"scene_id": "x"}), "pedestrian")])


def test_config_is_strict(tmp_path):
    bad = tmp_path / "verify.yaml"
    bad.write_text("association:\n  erode_px: 3\n")
    load_verify_config.cache_clear()
    with pytest.raises(ValidationError):
        load_verify_config(bad)
    assert load_verify_config().association.erode_px == 3
    assert PARKED != CROSSER  # ids are distinct
