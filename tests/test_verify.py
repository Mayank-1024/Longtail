"""C6 on the synthetic scene: every predicate's PASS, FAIL and UNVERIFIABLE, and the engine."""

import numpy as np
import pytest
from pydantic import ValidationError
from synthetic import (
    SyntheticConfig,
    SyntheticScene,
    close_pass_config,
    object_track2d,
    rect_mask,
    stopped_ego_config,
    track2d_from_masks,
    track_source,
)

from longtail.contracts import CandidateWindow, Finding, ScenarioSpec
from longtail.verify.config import load_verify_config
from longtail.verify.engine import _Pool, verify
from longtail.verify.predicates.base import Reason

CFG = load_verify_config()
PED = ("a1", "pedestrian", "person")
CAR = ("c1", "car", "car")


def _cfg(section: str, **update):
    p = CFG.predicates
    if section == "engine":
        return CFG.model_copy(update={"engine": CFG.engine.model_copy(update=update)})
    sub = getattr(p, section).model_copy(update=update)
    return CFG.model_copy(update={"predicates": p.model_copy(update={section: sub})})


def _verify(scene, source, actors, preds, i0=0, i1=None, min_duration_s=0.5, cfg=None) -> Finding:
    spec = ScenarioSpec(
        query="test",
        actors=[{"id": i, "cls": c, "sam_prompt": p} for i, c, p in actors],
        predicates=[{"name": n, "actors": a, "args": g} for n, a, g in preds],
        min_duration_s=min_duration_s,
        assumptions=[],
        clarification=None,
    )
    ts = scene.lidar_timestamps()
    i1 = len(ts) - 1 if i1 is None else i1
    w = CandidateWindow(
        window_id="w",
        scene_id=scene.scene_id,
        t0=float(ts[i0]),
        t1=float(ts[i1]),
        cameras=[],
        score=1.0,
        source="uniform",
    )
    [finding] = verify(spec, [w], scene, track_source=source, cfg=cfg)
    return finding


def _one(scene, source, actors, name, actor_ids, args=None, **kw):
    """Verify a single predicate and return its readable result."""
    f = _verify(scene, source, actors, [(name, actor_ids, args or {})], **kw)
    [r] = f.results
    e = r.evidence
    # Readable evidence (plan C6 done check): a reason code, the frames, counts, an example, the threshold.
    assert Reason(e["reason"]) and e["frames"] and set(e["frame_status"]) == {"PASS", "FAIL", "UNVERIFIABLE"}
    assert "example" in e and "longest_pass_s" in e and "binding_search" in e
    assert r.threshold or r.status == "UNVERIFIABLE"
    assert ("margin" in e) == (r.value is not None)
    return r


@pytest.fixture(scope="module")
def scene():
    return SyntheticScene()


@pytest.fixture(scope="module")
def wall_and_empty(scene):
    wall = track2d_from_masks(
        scene, "left_camera", "person", 3, {i: rect_mask(scene, 220, 160, 420, 320) for i in range(30)}
    )
    empty = track2d_from_masks(
        scene, "front_camera", "person", 7, {i: rect_mask(scene, 420, 0, 600, 120) for i in range(30)}
    )
    return wall, empty


# exists


def test_exists(scene, wall_and_empty):
    wall, empty = wall_and_empty
    r = _one(scene, track_source(scene, {"person": ["crosser"]}), [PED], "exists", ["a1"])
    assert (r.status, r.evidence["reason"]) == ("PASS", "ok") and r.value >= 15
    # ASSUMPTION A-76: the pedestrian prior caps the footprint at 1.2 m; the wall patch is 2.75 m long
    r = _one(scene, track_source(scene, {}, {"person": [wall]}), [PED], "exists", ["a1"])
    assert (r.status, r.evidence["reason"]) == ("FAIL", "wrong_size") and r.evidence["margin"] < 0
    r = _one(scene, track_source(scene, {}, {"person": [empty]}), [PED], "exists", ["a1"])
    assert (r.status, r.evidence["reason"]) == ("UNVERIFIABLE", "no_coverage") and r.value is None


def test_exists_reasons_for_rejection(scene):
    src = track_source(scene, {"person": ["crosser"]})
    # Early frames only: the ego drives toward the crosser and later gets within 5 m.
    r = _one(scene, src, [PED], "exists", ["a1"], i0=0, i1=5, cfg=_cfg("exists", max_range_m=5.0))
    assert (r.status, r.evidence["reason"]) == ("UNVERIFIABLE", "out_of_range")
    r = _one(scene, src, [PED], "exists", ["a1"], cfg=_cfg("exists", min_points=10**6))
    assert (r.status, r.evidence["reason"]) == ("FAIL", "too_few_points")
    # At 10 Hz every camera frame is within 50 ms of some sweep, so unsynced frames occur only at the
    # log's ends or beside a dropped sweep. Here sweeps are 200 ms apart and each frame is 70 ms off.
    unsynced = SyntheticScene(SyntheticConfig(n_frames=10, lidar_hz=5.0, camera_offset_s=0.07))
    r = _one(unsynced, track_source(unsynced, {"person": ["crosser"]}), [PED], "exists", ["a1"])
    assert (r.status, r.evidence["reason"]) == ("UNVERIFIABLE", "unsynced")
    # SAM hit its object cap (truncated) and the object is missing from these frames.
    gap = object_track2d(scene, "crosser", "front_camera", concept="person", frames=range(10), truncated=True)
    r = _one(
        scene,
        track_source(scene, {}, {"person": [gap]}),
        [PED],
        "exists",
        ["a1"],
        i0=5,
        i1=15,
        min_duration_s=1.0,
    )
    assert (r.status, r.evidence["reason"]) == ("UNVERIFIABLE", "truncated")
    assert r.evidence["frame_reasons"] == {"ok": 5, "truncated": 6}


# within_distance


def test_within_distance(scene, wall_and_empty):
    src = track_source(scene, {"person": ["crosser"]})
    r = _one(scene, src, [PED], "within_distance", ["a1"], {"m": 12.0})
    assert r.status == "PASS" and r.value <= 12.5
    r = _one(scene, src, [PED], "within_distance", ["a1"], {"m": 3.0})
    assert (r.status, r.evidence["reason"]) == ("FAIL", "too_far")
    r = _one(
        scene,
        track_source(scene, {}, {"person": [wall_and_empty[1]]}),
        [PED],
        "within_distance",
        ["a1"],
        {"m": 5.0},
    )
    assert (r.status, r.evidence["reason"]) == ("UNVERIFIABLE", "requires_exists")


# moving and stopped


def test_moving_and_stopped(scene):
    src = track_source(scene, {"person": ["crosser"], "car": ["parked"]})
    r = _one(scene, src, [PED], "moving", ["a1"])
    assert r.status == "PASS" and r.value == pytest.approx(1.5, abs=0.3)
    r = _one(scene, src, [CAR], "moving", ["c1"])
    assert (r.status, r.evidence["reason"]) == ("FAIL", "not_moving")
    r = _one(scene, src, [CAR], "stopped", ["c1"])
    assert r.status == "PASS" and r.value < 0.3
    r = _one(scene, src, [PED], "stopped", ["a1"])
    assert (r.status, r.evidence["reason"]) == ("FAIL", "not_stopped")
    short = track_source(
        scene,
        {},
        {"person": [object_track2d(scene, "crosser", "front_camera", concept="person", frames=range(7))]},
    )
    for name in ("moving", "stopped"):
        r = _one(scene, short, [PED], name, ["a1"], i0=0, i1=6)
        assert (r.status, r.evidence["reason"]) == ("UNVERIFIABLE", "track_too_short")


def test_occluded_parked_car_never_passes_moving():
    # The C5 strict expected failure: with the box centre it shows a 1.2 s run above 0.5 m/s.
    s = SyntheticScene(close_pass_config())
    r = _one(s, track_source(s, {"car": ["parked"]}), [CAR], "moving", ["c1"])
    assert r.status != "PASS"
    assert r.evidence["frame_status"]["PASS"] == 0
    # ASSUMPTION A-77: the position jumps behind its speed spikes are caught
    assert r.evidence["frame_reasons"].get("implausible_jump", 0) > 0


def test_close_pass_car_reads_stopped():
    s = SyntheticScene(close_pass_config())
    f = _verify(s, track_source(s, {"car": ["pass_car"]}), [CAR], [("stopped", ["c1"], {})])
    assert f.verified
    # ASSUMPTION A-94: the PASS run is 0.6 s, but every verdict in it was judged on a full 1 s window of clean
    # steps (the series' last second, shared by the frames near its end)
    a, b = f.results[0].evidence["example"]["judged_window"]
    assert b - a == pytest.approx(1.0) and f.t1 - f.t0 < 1.0
    r = _one(s, track_source(s, {"car": ["pass_car"]}), [CAR], "moving", ["c1"])
    assert r.status != "PASS" and r.evidence["frame_status"]["PASS"] == 0


# crossing_ego_path


def test_crossing_ego_path(scene):
    src = track_source(scene, {"person": ["crosser"], "car": ["parked"]})
    r = _one(scene, src, [PED], "crossing_ego_path", ["a1"])
    assert r.status == "PASS" and r.value > 0.3
    # FAIL needs the full 3 s look-ahead, so a longer scene.
    long = SyntheticScene(SyntheticConfig(n_frames=60))
    r = _one(long, track_source(long, {"car": ["parked"]}), [CAR], "crossing_ego_path", ["c1"], i0=0, i1=20)
    assert (r.status, r.evidence["reason"]) == ("FAIL", "outside_corridor")
    few = track_source(
        scene,
        {},
        {"person": [object_track2d(scene, "crosser", "front_camera", concept="person", frames=range(4))]},
    )
    r = _one(scene, few, [PED], "crossing_ego_path", ["a1"], i0=0, i1=3)
    assert (r.status, r.evidence["reason"]) == ("UNVERIFIABLE", "unreliable_speed")


def test_crossing_ego_path_with_the_ego_stopped():
    # ASSUMPTION A-69: the stopped ego's corridor still reaches 10 m ahead
    s = SyntheticScene(stopped_ego_config(crosser_start=(8.0, -2.25), n_frames=40))
    f = _verify(
        s,
        track_source(s, {"person": ["crosser"]}),
        [PED],
        [("crossing_ego_path", ["a1"], {}), ("ego_stopped", [], {})],
    )
    assert f.verified
    assert f.results[0].evidence["example"]["corridor_extended"] is True


def test_crossing_ego_path_already_inside_and_moving_across():
    # ASSUMPTION A-81: inside the corridor at the start and moving across it counts
    s = SyntheticScene(SyntheticConfig(crosser_start=(8.0, -0.5), n_frames=40))
    r = _one(
        s,
        track_source(s, {"person": ["crosser"]}),
        [PED],
        "crossing_ego_path",
        ["a1"],
        i0=0,
        i1=5,
        min_duration_s=0.3,
    )
    assert r.status == "PASS" and r.evidence["example"].get("already_inside") is True


# relative_position


def test_relative_position(scene, wall_and_empty):
    src = track_source(scene, {"person": ["crosser"]})
    # ASSUMPTION A-82: sides are half-planes; the crosser starts on the right and walks left
    r = _one(scene, src, [PED], "relative_position", ["a1"], {"side": "right"}, i0=0, i1=8)
    assert r.status == "PASS" and r.evidence["example"]["fraction_on_side"] >= 0.5
    r = _one(scene, src, [PED], "relative_position", ["a1"], {"side": "left"}, i0=0, i1=8)
    assert (r.status, r.evidence["reason"]) == ("FAIL", "wrong_side")
    r = _one(
        scene,
        track_source(scene, {}, {"person": [wall_and_empty[1]]}),
        [PED],
        "relative_position",
        ["a1"],
        {"side": "ahead"},
    )
    assert (r.status, r.evidence["reason"]) == ("UNVERIFIABLE", "no_coverage")


# near


def test_near(scene, wall_and_empty):
    pair = SyntheticScene(SyntheticConfig(pedestrian_pair=True))
    two = [PED, ("a2", "pedestrian", "person")]
    # ASSUMPTION A-83: distance between the two centroids, horizontally
    r = _one(pair, track_source(pair, {"person": ["ped_a", "ped_b"]}), two, "near", ["a1", "a2"], {"m": 1.0})
    assert r.status == "PASS" and r.value == pytest.approx(0.6, abs=0.15)
    r = _one(
        scene,
        track_source(scene, {"person": ["crosser"], "car": ["parked"]}),
        [PED, CAR],
        "near",
        ["a1", "c1"],
        {"m": 1.0},
    )
    assert (r.status, r.evidence["reason"]) == ("FAIL", "not_near")
    r = _one(
        scene,
        track_source(
            scene, {"person": ["crosser"]}, {"car": [wall_and_empty[1].model_copy(update={"concept": "car"})]}
        ),
        [PED, CAR],
        "near",
        ["a1", "c1"],
        {"m": 5.0},
    )
    assert (r.status, r.evidence["reason"]) == ("UNVERIFIABLE", "no_coverage")


# occluded_by


def test_occluded_by(scene, wall_and_empty):
    # The crosser walks in front of the parked car during the first frames.
    src = track_source(scene, {"car": ["parked"], "pedestrian": ["crosser"]})
    # ASSUMPTION A-93: the amodal box includes the car's track-wide size, so a part hidden at the side counts
    r = _one(scene, src, [CAR], "occluded_by", ["c1"], {"cls": "pedestrian"}, i0=0, i1=2, min_duration_s=0.3)
    assert r.status == "PASS" and r.value >= 0.2
    r = _one(scene, src, [CAR], "occluded_by", ["c1"], {"cls": "pedestrian"}, i0=10, i1=20)
    assert (r.status, r.evidence["reason"]) == ("FAIL", "not_occluded")
    r = _one(
        scene,
        track_source(scene, {"pedestrian": ["crosser"]}, {"person": [wall_and_empty[1]]}),
        [PED],
        "occluded_by",
        ["a1"],
        {"cls": "pedestrian"},
    )
    assert (r.status, r.evidence["reason"]) == ("UNVERIFIABLE", "no_coverage")


# ego_moving and ego_stopped


def test_ego_moving_and_stopped(scene):
    stopped = SyntheticScene(stopped_ego_config())
    tiny = SyntheticScene(SyntheticConfig(n_frames=5))
    none = track_source(scene, {})
    assert _one(scene, none, [], "ego_moving", []).status == "PASS"
    assert _one(stopped, none, [], "ego_stopped", []).status == "PASS"
    r = _one(stopped, none, [], "ego_moving", [])
    assert (r.status, r.evidence["reason"]) == ("FAIL", "not_moving")
    r = _one(scene, none, [], "ego_stopped", [])
    assert (r.status, r.evidence["reason"]) == ("FAIL", "not_stopped")
    for name in ("ego_moving", "ego_stopped"):
        r = _one(tiny, none, [], name, [])
        assert (r.status, r.evidence["reason"]) == ("UNVERIFIABLE", "track_too_short")


# Engine


def test_pedestrian_crossing_while_we_move_is_verified_among_distractors():
    s = SyntheticScene(SyntheticConfig(pedestrian_pair=True))
    src = track_source(s, {"person": ["ped_a", "ped_b", "crosser"]})
    f = _verify(
        s, src, [PED], [("exists", ["a1"], {}), ("crossing_ego_path", ["a1"], {}), ("ego_moving", [], {})]
    )
    assert f.verified and all(r.status == "PASS" for r in f.results)
    by_id = {o.track.track_id: o for o in _Pool(s, src, CFG).get("person", "pedestrian", tuple(s.cameras))}
    assert ("front_camera", 3) in by_id[f.bindings["a1"]].track.sources  # the crosser (third object)
    assert f.cost_usd == 0.0 and f.score > 0


def test_pruning_keeps_the_true_binding():
    s = SyntheticScene(SyntheticConfig(pedestrian_pair=True))
    src = track_source(s, {"person": ["ped_a", "ped_b", "crosser"]})
    # In the first 0.4 s the standing pair is over 11 m away and fails within_distance(10.5); the crosser,
    # about 10 m away, is the only candidate left after pruning.
    f = _verify(
        s,
        src,
        [PED],
        [("exists", ["a1"], {}), ("within_distance", ["a1"], {"m": 10.5}), ("moving", ["a1"], {})],
        i0=0,
        i1=3,
        min_duration_s=0.3,
    )
    assert f.verified
    search = f.results[0].evidence["binding_search"]
    assert search["bindings_tried"] == 1 and search["bindings_total_upper_bound"] == 1


def test_binding_cap_is_recorded():
    s = SyntheticScene(SyntheticConfig(pedestrian_pair=True))
    src = track_source(s, {"person": ["ped_a", "ped_b", "crosser"]})
    actors = [PED, ("a2", "pedestrian", "person"), ("a3", "pedestrian", "person")]
    preds = [("exists", [a], {}) for a in ("a1", "a2", "a3")]
    f = _verify(s, src, actors, preds, cfg=_cfg("engine", max_bindings=2))
    search = f.results[0].evidence["binding_search"]
    assert search["binding_cap_hit"] is True and search["bindings_tried"] == 2
    assert search["bindings_total_upper_bound"] == 27  # 3 x 3 x 3 before removing shared tracks
    uncapped = _verify(s, src, actors, preds)
    assert uncapped.results[0].evidence["binding_search"]["binding_cap_hit"] is False
    assert uncapped.verified


def test_unverified_findings_are_returned_and_verified_is_computed(scene, wall_and_empty):
    # ASSUMPTION A-17: the engine returns unverified findings too
    src = track_source(scene, {"person": ["crosser"]})
    f = _verify(scene, src, [PED], [("exists", ["a1"], {}), ("stopped", ["a1"], {})])
    assert not f.verified and {r.status for r in f.results} == {"PASS", "FAIL"}
    with pytest.raises((AttributeError, ValidationError)):
        f.verified = True
    with pytest.raises(ValidationError, match="verified is computed"):
        Finding.model_validate({**f.model_dump(), "verified": True})
    good = _verify(scene, src, [PED], [("exists", ["a1"], {}), ("moving", ["a1"], {})])
    # ASSUMPTION A-85: score from the margins; a verified finding outranks an unverified one
    assert good.verified and good.score > f.score


def test_verify_rejects_bad_calls(scene):
    spec = ScenarioSpec(query="q", actors=[], predicates=[], assumptions=[], clarification="which side?")
    with pytest.raises(ValueError, match="clarifying question"):
        verify(spec, [], scene, track_source=track_source(scene, {}))
    ok = spec.model_copy(update={"clarification": None})
    # ASSUMPTION A-86: there is no default track source until C3 exists
    with pytest.raises(RuntimeError, match="track source"):
        verify(ok, [], scene)
    other = CandidateWindow(
        window_id="w", scene_id="other", t0=0, t1=1, cameras=[], score=0, source="uniform"
    )
    with pytest.raises(ValueError, match="is for scene"):
        verify(ok, [other], scene, track_source=track_source(scene, {}))
    # ASSUMPTION A-88: windows are in dataset seconds; one outside the log has no frames and no finding
    late = other.model_copy(update={"scene_id": scene.scene_id, "t0": 5000.0, "t1": 5001.0})
    assert verify(ok, [late], scene, track_source=track_source(scene, {})) == []
    assert np.isfinite(scene.lidar_timestamps()).all()
