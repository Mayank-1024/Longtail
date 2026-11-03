"""C7: ground truth from labels, on the synthetic scene with its true boxes as cuboids."""

import json
from pathlib import Path

import numpy as np
import pytest
from synthetic import SyntheticConfig, SyntheticScene, track_source

from longtail import cli
from longtail.contracts import CandidateWindow, ScenarioSpec
from longtail.data.guard import LabelAccessError, eval_ground_truth
from longtail.eval import cli as gt_cli
from longtail.eval import gt
from longtail.verify.config import load_verify_config
from longtail.verify.engine import TrackPool, verify

FIXTURE = Path(__file__).parent / "fixtures" / "gold_synthetic.yaml"
QUERIES = {q.query_id: q for q in gt.load_queries(FIXTURE)}
CFG = gt.load_gt_config()


@pytest.fixture(scope="module")
def scene():
    # The default scene plus a pedestrian standing behind the parked car (A-130).
    return SyntheticScene(SyntheticConfig(hidden_pedestrian=True))


@pytest.fixture(scope="module")
def labels(scene):
    with eval_ground_truth():
        return gt.LabelScene(scene)


def _positives(labels, query_id):
    with eval_ground_truth():
        return gt.scene_positives(query_id, QUERIES[query_id].gold, labels)[0]


def _rel(p):
    return round(p.t0 - p.scene_start_ts, 2), round(p.t1 - p.scene_start_ts, 2)


# Positives per gold spec


def test_crossing_positive(labels):
    [p] = _positives(labels, "syn_crossing")
    # The crosser enters the corridor (half-width 1.5 m) at 0.5 s and leaves it at 2.5 s; from 2.6 s its
    # future is shorter than the 3 s look-ahead and it never re-enters, so those frames are not judged (A-96).
    assert _rel(p) == (0.0, 2.5)
    assert p.actors == {"a1": ["crosser"]} and p.label_ids == ["crosser"]
    [track] = p.labels
    assert track.label == "Pedestrian" and len(track.positions) == 26
    assert {b[0] for b in track.boxes} == {"front_camera"}


def test_parked_car_positive(labels):
    [p] = _positives(labels, "syn_parked")
    assert _rel(p) == (0.0, 2.9) and p.actors == {"c1": ["parked"]}


def test_occluded_pedestrian_positive(labels):
    [p] = _positives(labels, "syn_occluded")
    # Only the pedestrian behind the parked car; the crosser walks in front of it, so it is not occluded.
    assert _rel(p) == (0.0, 2.9) and p.actors == {"a1": ["hidden_ped"]}
    ctx = gt.gt_context(labels)
    r = ctx.evaluate("occluded_by", [labels.tracked("hidden_ped")], {"cls": "car"})
    assert all(x.status == "PASS" for x in r) and min(x.value for x in r) > 0.8
    r = ctx.evaluate("occluded_by", [labels.tracked("crosser")], {"cls": "car"})
    assert {(x.status, x.reason.value) for x in r} == {("FAIL", "not_occluded")}


def test_cyclist_has_zero_positives_and_is_flagged(scene, tmp_path):
    assert _positives(_labels(scene), "syn_cyclist") == []
    with eval_ground_truth():
        summary = gt.build(list(QUERIES.values()), [scene], tmp_path / "gt")
    assert summary["zero_positive_queries"] == ["syn_cyclist"]
    rows = {r["query_id"]: r for r in summary["queries"]}
    assert rows["syn_crossing"]["positives"] == 1 and not rows["syn_crossing"]["zero_positives"]
    doc = gt.load_gt(tmp_path / "gt", "syn_crossing")
    assert doc.format == "longtail.gt/v1" and doc.scenes == ["synthetic"] and doc.motion_source == "geometry"
    assert doc.positives[0].scene_id == "synthetic"


def _labels(scene):
    with eval_ground_truth():
        return gt.LabelScene(scene)


def test_class_mapping_and_visibility(labels):
    assert {o.uuid: o.cls for o in labels.objects.values()} == {
        "crosser": "pedestrian",
        "parked": "car",
        "hidden_ped": "pedestrian",
    }
    # ASSUMPTION A-123: a bicycle without a rider is not a cyclist; unmapped labels are ignored
    rows = [
        {"label": "Bicycle", "attributes.rider_status": "Without Rider"},
        {"label": "Bicycle", "attributes.rider_status": "With Rider"},
        {"label": "Signs"},
    ]
    base = {"uuid": "u", "yaw": 0.0, "position.x": 0.0, "position.y": 0.0, "position.z": 0.0}
    dims = {"dimensions.x": 0.6, "dimensions.y": 1.8, "dimensions.z": 1.7}
    got = [gt.to_cuboid({**base, **dims, **r}, CFG) for r in rows]
    assert got[0] is None and got[1].cls == "cyclist" and got[2] is None
    # Devkit convention: yaw 0 puts the length (dimensions.y) along +y.
    assert got[1].lwh == (1.8, 0.6, 1.7) and np.isclose(got[1].heading, np.pi / 2)


def test_visibility_needs_a_camera_and_60_m():
    # ASSUMPTION A-122: beyond 60 m an object is not counted even though it projects into a camera
    far = SyntheticScene(SyntheticConfig(parked_center=(62.0, -4.0), extent_ahead=(-10.0, 80.0)))
    with eval_ground_truth():
        ls = gt.LabelScene(far)
    parked = ls.objects["parked"]
    seen = [t for t, v in parked.visible.items() if v]
    assert all(parked.boxes[t] for t in parked.boxes)  # it projects into the front camera throughout
    # The ego drives 2 m/s toward it: the car comes within 60 m (centre range) only near the end.
    assert seen and min(seen) > far.lidar_timestamps()[0] + 0.5
    assert _positives(ls, "syn_parked")[0].t0 == pytest.approx(min(seen))


def test_horizon_rule():
    # The crosser starts 8.3 m to the right, so it enters the corridor (1.5 m) after 4.53 s.
    s = SyntheticScene(SyntheticConfig(n_frames=60, crosser_start=(10.0, -8.3)))
    with eval_ground_truth():
        ls = gt.LabelScene(s)
        r = gt.gt_context(ls).evaluate("crossing_ego_path", [ls.tracked("crosser")], {})
        [p] = _positives(ls, "syn_crossing")
    t = ls.sweeps - ls.sweeps[0]
    status = np.array([x.status for x in r])
    # ASSUMPTION A-96: a frame counts from 3 s before entering; earlier frames FAIL (outside the corridor)
    assert set(status[t < 1.55]) == {"FAIL"} and set(status[(t > 1.55) & (t < 4.5)]) == {"PASS"}
    assert _rel(p)[0] == 1.6
    # In the default scene the crosser leaves the corridor at 2.5 s; with under 3 s of scene left, the frames
    # after that are UNVERIFIABLE (it might re-enter after the scene ends), not FAIL.
    with eval_ground_truth():
        ls = gt.LabelScene(SyntheticScene())
        r = gt.gt_context(ls).evaluate("crossing_ego_path", [ls.tracked("crosser")], {})
    tail = [x for x, tt in zip(r, ls.sweeps - ls.sweeps[0], strict=True) if tt > 2.55]
    assert tail and {(x.status, x.reason.value) for x in tail} == {("UNVERIFIABLE", "track_too_short")}


# Siblings


def test_sibling_cuboids_are_deduplicated():
    # ASSUMPTION A-16: the sensor-0 copy of a sibling pair is kept, the sensor-1 copy dropped
    s = SyntheticScene(SyntheticConfig(hidden_pedestrian=True, label_siblings=True))
    with eval_ground_truth():
        ls = gt.LabelScene(s)
    assert ls.stats["sibling_copies_dropped"] == 30  # the crosser, the only moving object, in every sweep
    assert "crosser~front" not in ls.objects
    [p] = _positives(ls, "syn_crossing")
    assert p.label_ids == ["crosser"] and _rel(p) == (0.0, 2.5)


def test_dedup_keeps_unpaired_copies():
    rows = [
        {"uuid": "a", "cuboids.sensor_id": 0, "cuboids.sibling_id": "b"},
        {"uuid": "b", "cuboids.sensor_id": 1, "cuboids.sibling_id": "a"},
        {"uuid": "c", "cuboids.sensor_id": 1, "cuboids.sibling_id": "gone"},
        {"uuid": "d", "cuboids.sensor_id": -1, "cuboids.sibling_id": float("nan")},
    ]
    kept, dropped = gt.dedup_siblings(rows)
    assert [r["uuid"] for r in kept] == ["a", "c", "d"] and dropped == 1


# Events


def test_events_merge_bindings_and_respect_min_duration():
    ts = 1000.0 + np.arange(20) * 0.1
    flags = np.zeros((3, 20), bool)
    flags[0, 2:6] = True  # 0.4 s: too short for 0.5 s on its own
    flags[1, 4:10] = True  # 0.6 s, overlapping binding 0
    flags[2, 10:13] = True  # 0.3 s, adjacent to binding 1 but too short
    flags[2, 15:20] = True  # 0.5 s
    intervals, held = gt.events(ts, flags, 0.1, 0.5)
    # Binding 0's run is too short on its own, so it adds nothing; binding 1 gives 4-9; binding 2's 0.3 s
    # run is dropped, so 10-14 is a gap; 15-19 is a second event.
    assert intervals == [(4, 9), (15, 19)]
    assert held[0].sum() == 0 and held[1, 4:10].all()
    intervals, _ = gt.events(ts, flags, 0.1, 0.3)
    assert intervals == [(2, 12), (15, 19)]  # with 0.3 s every run counts, and touching runs merge


def test_two_objects_in_one_event():
    s = SyntheticScene(SyntheticConfig(pedestrian_pair=True))
    spec = ScenarioSpec(
        query="a pedestrian standing still",
        actors=[{"id": "a1", "cls": "pedestrian", "sam_prompt": "person"}],
        predicates=[
            {"name": "exists", "actors": ["a1"], "args": {}},
            {"name": "stopped", "actors": ["a1"], "args": {}},
        ],
        assumptions=[],
        clarification=None,
    )
    with eval_ground_truth():
        ls = gt.LabelScene(s)
        [p] = gt.scene_positives("standing", spec, ls)[0]
    # ASSUMPTION A-126: both standing pedestrians hold over the same frames, so they are one positive
    assert p.actors == {"a1": ["ped_a", "ped_b"]} and len(p.labels) == 2


# Matching


def _positive(pid, t0, t1, xy=(0.0, 0.0), scene_id="s"):
    ts = np.round(np.arange(t0, t1 + 1e-9, 0.1), 3)
    return gt.Positive(
        positive_id=pid,
        scene_id=scene_id,
        t0=t0,
        t1=t1,
        period_s=0.1,
        scene_start_ts=0.0,
        label_ids=["L"],
        actors={"a1": ["L"]},
        labels=[
            gt.LabelTrack(
                label_id="L",
                label="Pedestrian",
                cls="pedestrian",
                positions=[(float(t), xy[0], xy[1], 0.9) for t in ts],
                boxes=[("front_camera", round(t * 10), float(t), 100.0, 100.0, 140.0, 200.0) for t in ts],
            )
        ],
    )


def _finding(fid, t0, t1, xy=(0.0, 0.0), score=1.0, scene_id="s", boxes_only=False, unbound=False):
    ts = np.round(np.arange(t0, t1 + 1e-9, 0.1), 3)
    if unbound:
        trace = None
    elif boxes_only:
        trace = gt.ActorTrace(
            boxes=[("front_camera", round(t * 10), float(t), 105.0, 110.0, 140.0, 200.0) for t in ts]
        )
    else:
        trace = gt.ActorTrace(positions=[(float(t), xy[0], xy[1], 0.5) for t in ts])
    return gt.ScoredFinding(
        finding_id=fid, scene_id=scene_id, t0=t0, t1=t1, score=score, actors={"a1": trace}
    )


def test_temporal_iou_threshold():
    m = CFG.matching
    p = _positive("p", 0.0, 0.9)  # lasts 1.0 s
    # [0.0, 1.0) against [0.69, 1.69): 0.31 / 1.69 = 0.18; against [0.5, 1.5): 0.5 / 1.5 = 0.333
    assert gt.temporal_iou(0.0, 0.9, 0.5, 1.4, 0.1) == pytest.approx(1 / 3)
    above = _finding("above", 0.0, 1.9)  # [0, 2.0): IoU 1.0 / 2.0 = 0.5
    assert gt.eligible(above, p, m) == pytest.approx(0.5)
    just_above = _finding("ja", 0.0, 3.2)  # [0, 3.3): 1 / 3.3 = 0.303
    just_below = _finding("jb", 0.0, 3.4)  # [0, 3.5): 1 / 3.5 = 0.286
    assert gt.eligible(just_above, p, m) == pytest.approx(1 / 3.3)
    assert gt.eligible(just_below, p, m) is None


def test_actor_distance_rule():
    m = CFG.matching
    p = _positive("p", 0.0, 0.9)
    assert gt.eligible(_finding("near", 0.0, 0.9, xy=(1.9, 0.0)), p, m) is not None
    assert gt.eligible(_finding("far", 0.0, 0.9, xy=(2.1, 0.0)), p, m) is None
    assert gt.eligible(_finding("unbound", 0.0, 0.9, unbound=True), p, m) is None
    assert gt.eligible(_finding("other scene", 0.0, 0.9, scene_id="t"), p, m) is None
    # ASSUMPTION A-127: boxes only (no LiDAR): box IoU >= 0.3 with the label's projected box
    assert gt.eligible(_finding("boxes", 0.0, 0.9, boxes_only=True), p, m) is not None


def test_matching_is_one_to_one_and_greedy_by_iou():
    m = CFG.matching
    p1, p2 = _positive("p1", 0.0, 0.9), _positive("p2", 1.5, 2.4)
    f_best = _finding("best", 0.0, 0.9)  # IoU 1 with p1
    f_dup = _finding("dup", 0.2, 1.1, score=5.0)  # IoU 0.67 with p1, higher score, loses on IoU
    f_both = _finding("both", 0.5, 2.4)  # p1 0.25 (no), p2 1.0 / 2.0 = 0.5
    got = gt.match([f_dup, f_both, f_best], [p1, p2], m)
    assert {(x.finding_id, x.positive_id) for x in got} == {("best", "p1"), ("both", "p2")}
    # The duplicate is left unmatched: a false positive in precision.
    assert "dup" not in {x.finding_id for x in got}


# The guard


def test_ground_truth_fails_outside_the_guard(scene, labels, tmp_path):
    with pytest.raises(LabelAccessError):
        gt.LabelScene(scene)
    with pytest.raises(LabelAccessError):
        gt.build(list(QUERIES.values()), [scene], tmp_path / "gt")
    with pytest.raises(LabelAccessError):
        gt.scene_positives("syn_crossing", QUERIES["syn_crossing"].gold, labels)
    with pytest.raises(LabelAccessError):
        gt.claim_is_true(labels, "front_camera", 0, "pedestrian", (0, 0, 1, 1))
    with pytest.raises(LabelAccessError):
        gt.motion_agreement(labels)


def test_output_directory_must_be_gt(scene, tmp_path):
    with eval_ground_truth(), pytest.raises(ValueError, match="named gt"):
        gt.build(list(QUERIES.values()), [scene], tmp_path / "results")


# The verifier against the ground truth


def test_verifier_crossing_finding_matches_ground_truth(scene, labels):
    spec = QUERIES["syn_crossing"].gold
    src = track_source(scene, {"person": ["crosser", "hidden_ped"]})
    cfg = load_verify_config()
    pool = TrackPool(scene, src, cfg)
    ts = scene.lidar_timestamps()
    window = CandidateWindow(
        window_id="w0",
        scene_id=scene.scene_id,
        t0=float(ts[0]),
        t1=float(ts[-1]),
        cameras=[],
        score=1.0,
        source="uniform",
    )
    [finding] = verify(spec, [window], scene, track_source=src, cfg=cfg, pool=pool)
    assert finding.verified
    scored = gt.scored_finding(finding, pool)
    [positive] = _positives(labels, "syn_crossing")
    [m] = gt.match([scored], [positive], CFG.matching)
    assert m.positive_id == positive.positive_id and m.iou > 0.5


# Claims (A-10)


def test_claim_truth_is_intersection_over_the_claimed_box(scene, labels):
    from synthetic import object_track2d

    # ASSUMPTION A-10: the hidden pedestrian's SAM box covers only its head above the car roof: IoU with its
    # projected cuboid is tiny, but the box lies inside it, so the claim is true
    head = object_track2d(scene, "hidden_ped", "front_camera").frames[0]
    [(_, label_box)] = [(c, b) for c, b in labels.boxes_at("front_camera", 0) if c.uuid == "hidden_ped"]
    assert gt.box_iou(head.box_xyxy, label_box) < 0.1 and gt.inside_frac(head.box_xyxy, label_box) == 1.0
    with eval_ground_truth():
        assert gt.claim_is_true(labels, "front_camera", 0, "pedestrian", head.box_xyxy)
        assert not gt.claim_is_true(labels, "front_camera", 0, "car", head.box_xyxy)  # wrong class
        assert not gt.claim_is_true(labels, "front_camera", 0, "cyclist", (420, 0, 600, 120))  # empty sky
        # Widened to the right (only the car is there): half of it inside passes, less does not.
        x1, y1, x2, y2 = label_box
        w = x2 - x1
        assert gt.claim_is_true(labels, "front_camera", 0, "pedestrian", (x1, y1, x2 + w, y2))
        assert not gt.claim_is_true(labels, "front_camera", 0, "pedestrian", (x1, y1, x2 + 1.2 * w, y2))


# Motion source


def test_motion_source_attribute_agrees_on_the_synthetic_scene(scene):
    attr = CFG.model_copy(update={"motion_source": "attribute"})
    with eval_ground_truth():
        ls = gt.LabelScene(scene, attr)
        [p] = gt.scene_positives("syn_parked", QUERIES["syn_parked"].gold, ls)[0]
        report = gt.motion_agreement(ls)
    assert _rel(p) == (0.0, 2.9)
    # ASSUMPTION A-124: the report counts geometry against the attribute per visible frame
    assert report["pedestrian"]["agree"] == 60 and "disagree" not in report["pedestrian"]
    assert report["car"]["geometry_stopped_attribute_stopped"] == 30


# Queries file and CLI


def test_queries_file_is_checked(tmp_path):
    raw = FIXTURE.read_text()
    dup = tmp_path / "dup.yaml"
    dup.write_text(raw.replace("query_id: syn_parked", "query_id: syn_crossing"))
    with pytest.raises(ValueError, match="duplicate"):
        gt.load_queries(dup)
    ask = tmp_path / "ask.yaml"
    ask.write_text(raw.replace("clarification: null", "clarification: which car?", 1))
    with pytest.raises(ValueError, match="clarifying"):
        gt.load_queries(ask)


def test_cli_build_spotcheck_and_motion_report(scene, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gt_cli, "load_scene", lambda sid: scene)
    out = tmp_path / "gt"
    assert cli.main(["gt", "build", str(FIXTURE), "--scenes", "synthetic", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "syn_cyclist" in text and "ZERO" in text
    assert sorted(p.name for p in out.iterdir()) == sorted([f"{q}.json" for q in QUERIES] + ["summary.json"])
    assert cli.main(["gt", "spotcheck", "syn_parked", "--n", "5", "--gt-dir", str(out)]) == 0
    text = capsys.readouterr().out
    assert "scene synthetic" in text and "parked (Car)" in text and "0.00-  2.90 s" in text
    assert cli.main(["gt", "spotcheck", "syn_cyclist", "--gt-dir", str(out)]) == 1
    capsys.readouterr()
    assert cli.main(["gt", "motion-report", "--scenes", "synthetic", "--out", str(out)]) == 0
    report = json.loads((out / "motion_agreement.json").read_text())
    assert report["total"]["car"]["agree"] == 30
