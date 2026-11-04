"""The post-freeze verifier (v2, A-244 to A-248): v1 is unchanged; each fix does what was pre-registered."""

from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from synthetic import SyntheticScene, object_mask, track2d_from_masks

from longtail.contracts import ScenarioSpec, Track2D
from longtail.eval import ablation, gt
from longtail.geometry.associate import associate
from longtail.geometry.tracks3d import _inliers
from longtail.verify.config import CONFIG_PATHS, load_verify_config
from longtail.verify.predicates import _motion
from longtail.verify.predicates.base import Reason, missing
from longtail.vision import track_source

V1 = load_verify_config(CONFIG_PATHS["v1"])
V2 = load_verify_config(CONFIG_PATHS["v2"])

# The pre-registered changes (docs/eval-report.md, "v2 (post-freeze, pre-registered)"); nothing else differs.
PREREGISTERED = {
    "association.min_eroded_px": 50,
    "association.thin_mask_unverifiable": True,
    "association.truncated_min_objects": 16,
    "tracks.savgol_window": 11,
    "tracks.outlier_window": 7,
    "tracks.outlier_m": 1.0,
    "predicates.motion.drop_jump_windows": False,
}


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        out.update(_flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def test_v2_differs_from_v1_only_in_the_preregistered_fields():
    v1 = _flat(yaml.safe_load(CONFIG_PATHS["v1"].read_text()))
    v2 = _flat(yaml.safe_load(CONFIG_PATHS["v2"].read_text()))
    changed = {k: v2.get(k) for k in v1.keys() | v2.keys() if v1.get(k) != v2.get(k)}
    assert changed == PREREGISTERED


def test_v1_is_the_default_and_keeps_the_frozen_thresholds_hash(monkeypatch):
    monkeypatch.delenv("LONGTAIL_VERIFIER", raising=False)
    assert load_verify_config() == V1
    # the hash recorded in every frozen v1 result (longtail/eval/runs/*/results/*.json)
    assert ablation.thresholds_hash(V1, gt.load_gt_config()) == "aedd697d2cbb2e1d"
    assert ablation.thresholds_hash(V2, gt.load_gt_config()) != "aedd697d2cbb2e1d"


def test_the_switch_selects_v2_and_rejects_anything_else(monkeypatch):
    monkeypatch.setenv("LONGTAIL_VERIFIER", "v2")
    assert load_verify_config() == V2
    monkeypatch.setenv("LONGTAIL_VERIFIER", "v3")
    with pytest.raises(ValueError, match="LONGTAIL_VERIFIER"):
        load_verify_config()


# A-245, A-246: thin masks


@pytest.fixture(scope="module")
def scene():
    return SyntheticScene()


def _sliver_track(scene, frames=range(10, 16), width=4):
    """A vertical strip a few pixels wide through the crosser's mask, like a partly hidden object."""
    masks = {}
    for i in frames:
        full = object_mask(scene, "crosser", "front_camera", i)
        cols = np.flatnonzero(full.any(axis=0))
        mid = int(cols.mean())
        strip = np.zeros_like(full)
        strip[:, mid : mid + width] = full[:, mid : mid + width]
        masks[i] = strip
    return track2d_from_masks(scene, "front_camera", "person", 7, masks)


def test_v1_reads_an_eroded_away_sliver_as_evidence_against(scene):
    [a] = associate(scene, [(_sliver_track(scene), "pedestrian")], V1.association)
    assert a.track.samples == []
    assert {s.reason for s in a.skipped} <= {"no_points", "no_cluster"}


def test_thin_mask_alone_makes_the_empty_sliver_unverifiable(scene):
    cfg = V1.association.model_copy(update={"thin_mask_unverifiable": True})
    [a] = associate(scene, [(_sliver_track(scene), "pedestrian")], cfg)
    assert a.skipped and {s.reason for s in a.skipped} == {"thin_mask"}


def test_v2_keeps_the_unerodded_sliver_and_finds_the_object(scene):
    [a] = associate(scene, [(_sliver_track(scene), "pedestrian")], V2.association)
    assert len(a.track.samples) > 0
    assert not any(s.reason in ("no_points", "no_cluster") for s in a.skipped)


def test_v2_still_fails_a_full_mask_with_no_object_in_it(scene):
    """A normal-sized mask over a region with LiDAR coverage but no object stays evidence against."""
    masks = {}
    for i in range(10, 13):
        full = object_mask(scene, "crosser", "front_camera", i)
        rows, cols = np.flatnonzero(full.any(axis=1)), np.flatnonzero(full.any(axis=0))
        shifted = np.zeros_like(full)
        w = cols[-1] - cols[0] + 1
        shifted[rows[0] : rows[-1] + 1, cols[0] - 3 * w : cols[0] - 2 * w] = True  # beside the crosser
        masks[i] = shifted
    v1 = associate(scene, [(track2d_from_masks(scene, "front_camera", "person", 8, masks), "pedestrian")])
    v2 = associate(
        scene, [(track2d_from_masks(scene, "front_camera", "person", 8, masks), "pedestrian")], V2.association
    )
    assert [s.reason for s in v1[0].skipped] == [s.reason for s in v2[0].skipped]


def test_a_thin_mask_skip_is_unverifiable():
    ctx = SimpleNamespace(skip_reason=lambda obj, ts: "thin_mask")
    obj = SimpleNamespace(association=SimpleNamespace(truncated=False))
    r = missing(ctx, obj, 0.0)
    assert (r.status, r.reason) == ("UNVERIFIABLE", Reason.THIN_MASK)
    ctx = SimpleNamespace(skip_reason=lambda obj, ts: "no_points")
    assert missing(ctx, obj, 0.0).status == "FAIL"  # v1's rule is unchanged


# A-247: outliers, jump windows, smoothing


T = np.arange(30) * 0.1
ALL = np.ones(30, bool)


def _line(speed=15.0):
    return np.column_stack([speed * T, 0.5 * speed * T, np.zeros(len(T))])


@pytest.mark.parametrize("at", [3, 12, 26])
def test_inliers_drop_only_a_single_sample_from_another_object_at_speed(at):
    pos = _line()
    pos[at] += [9.0, 0.0, 0.0]
    assert np.flatnonzero(~_inliers(T, pos, ALL, V2.tracks)).tolist() == [at]


def test_inliers_keep_samples_too_near_an_end_to_test():
    pos = _line()
    pos[[2, 27]] += [9.0, 0.0, 0.0]
    assert _inliers(T, pos, ALL, V2.tracks).all()


def test_inliers_drop_two_bad_samples_in_a_row():
    pos = _line()
    pos[12:14] += [9.0, 0.0, 0.0]
    assert np.flatnonzero(~_inliers(T, pos, ALL, V2.tracks)).tolist() == [12, 13]


def test_inliers_keep_fast_steady_motion_braking_and_jitter():
    pos = _line(speed=20.0) + np.random.default_rng(0).normal(0, 0.15, (30, 3))
    assert _inliers(T, pos, ALL, V2.tracks).all()
    braking = np.column_stack([20 * T - 4 * T**2, np.zeros(30), np.zeros(30)])  # 8 m/s^2
    assert _inliers(T, braking, ALL, V2.tracks).all()


def test_inliers_are_off_in_v1():
    pos = _line()
    pos[12] += [9.0, 0.0, 0.0]
    assert _inliers(T, pos, ALL, V1.tracks).all()


def _parked_with_a_velocity_spike():
    n = 30
    ts = np.arange(n) * 0.1
    velocity = np.zeros((n, 3))
    velocity[15] = [3.0, 0.0, 0.0]  # 30 m/s^2 either side: a jitter spike, not motion
    return _motion.Series(ts, np.zeros((n, 3)), velocity, np.zeros(n, int), np.ones(n, bool))


def test_v1_drops_windows_with_a_jump_and_v2_judges_them():
    series = _parked_with_a_velocity_spike()
    v1 = _motion.steps(series, V1.predicates.motion)
    v2 = _motion.steps(series, V2.predicates.motion)
    assert any(s.reason == Reason.IMPLAUSIBLE_JUMP for s in v1)
    assert all(s.state == "stopped" for s in v2)


# A-248: SAM's per-response cap


def _tracks(n: int) -> list[Track2D]:
    return [
        Track2D(scene_id="001", camera="front_camera", concept="car", obj_id=i, truncated=False, frames=[])
        for i in range(n)
    ]


CAPPED = [("v1", 16, False), ("v2", 15, False), ("v2", 16, True)]


@pytest.mark.parametrize(("version", "n", "truncated"), CAPPED)
def test_a_response_with_16_objects_is_truncated_in_v2(monkeypatch, version, n, truncated):
    monkeypatch.setenv("LONGTAIL_VERIFIER", version)
    assert {t.truncated for t in track_source._capped(_tracks(n))} == {truncated}


def test_a_v2_search_writes_to_its_own_run_directory(monkeypatch, tmp_path):
    from longtail.pipeline import scene_run_dir, spec_run_id

    spec = ScenarioSpec.model_validate(
        {
            "query": "q",
            "assumptions": [],
            "clarification": None,
            "actors": [{"id": "a1", "cls": "pedestrian", "sam_prompt": "person"}],
            "predicates": [{"name": "exists", "actors": ["a1"], "args": {}}],
        }
    )
    monkeypatch.delenv("LONGTAIL_VERIFIER", raising=False)
    assert scene_run_dir(spec, "001", tmp_path).name == f"{spec_run_id(spec)}-001"
    monkeypatch.setenv("LONGTAIL_VERIFIER", "v2")
    assert scene_run_dir(spec, "001", tmp_path).name == f"{spec_run_id(spec)}-001-v2"


def test_a_resumed_v2_search_never_loads_a_v1_run(monkeypatch, tmp_path):
    from longtail import pipeline

    spec = ScenarioSpec.model_validate(
        {
            "query": "q",
            "assumptions": [],
            "clarification": None,
            "actors": [{"id": "a1", "cls": "pedestrian", "sam_prompt": "person"}],
            "predicates": [{"name": "exists", "actors": ["a1"], "args": {}}],
        }
    )
    scene = SimpleNamespace(scene_id="001")
    saved = SimpleNamespace(spec=spec, windows=[], scene_id="001")
    monkeypatch.setattr(pipeline, "uniform_windows", lambda s: [])
    monkeypatch.setattr(pipeline, "load_run", lambda d: saved)
    monkeypatch.setattr(pipeline, "verify_and_record", lambda *a: "verified again")
    monkeypatch.setenv("LONGTAIL_RESUME", "1")
    monkeypatch.delenv("LONGTAIL_VERIFIER", raising=False)
    v1_dir = pipeline.scene_run_dir(spec, "001", tmp_path)
    v1_dir.mkdir()
    (v1_dir / pipeline.RUN_FILE).write_text("{}")
    assert pipeline.search_scene(spec, scene, None, tmp_path) is saved  # v1 resumes its own run
    monkeypatch.setenv("LONGTAIL_VERIFIER", "v2")
    assert pipeline.search_scene(spec, scene, None, tmp_path) == "verified again"  # v2 ignores it
