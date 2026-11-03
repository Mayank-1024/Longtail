"""C8: metrics, the ablation runner and the deliberate breaks, on the synthetic scene."""

import json
from pathlib import Path

import numpy as np
import pytest
from synthetic import SyntheticConfig, SyntheticScene, rect_mask, track2d_from_masks, track_source

from longtail.data.guard import eval_ground_truth
from longtail.eval import ablation, breaks, formats, gt
from longtail.eval import metrics as M
from longtail.eval.metrics import Claim, QueryOutcome

FIXTURE = Path(__file__).parent / "fixtures" / "gold_synthetic.yaml"
QUERIES = gt.load_queries(FIXTURE)


# Metrics on hand-computed examples


def _o(qid, ranked, matched, n_pos, cost=0.0, seconds=0.0, **claims):
    return QueryOutcome(qid, ranked, set(matched), n_pos, cost, seconds, **claims)


def test_precision_recall_cost_and_time():
    a = _o("a", [f"f{i}" for i in range(12)], ["f0", "f2", "f11"], 4, cost=0.10, seconds=2.0)
    b = _o("b", ["g0", "g1"], ["g1"], 1, cost=0.30, seconds=4.0)
    c = _o("c", [], [], 3, cost=0.20, seconds=6.0)
    assert M.precision_at_k(a) == 0.2  # f11 is outside the top 10
    # ASSUMPTION A-134: P@10 divides by 10 whatever was returned; the returned count goes next to it
    assert M.precision_at_k(b) == 0.1
    assert M.precision_at_k(c) == 0.0  # nothing returned
    assert M.p_at_10([a, b, c]) == pytest.approx(0.1)
    assert M.returned_top10([a, b, c]) == pytest.approx(4.0)  # (10 + 2 + 0) / 3
    # ASSUMPTION A-134: precision pools every returned finding: (3 + 1 + 0) / (12 + 2 + 0)
    assert M.precision([a, b, c]) == pytest.approx(4 / 14)
    assert M.precision([c]) is None
    assert M.recall([a, b, c]) == pytest.approx(4 / 8)  # pooled: (3 + 1 + 0) / (4 + 1 + 3)
    assert M.cost_per_query([a, b, c]) == pytest.approx(0.2)
    assert M.seconds_per_query([a, b, c]) == pytest.approx(4.0)
    assert M.recall([_o("z", ["x"], [], 0)]) is None


def _claim(cid, verdict):
    return Claim(
        claim_id=cid,
        scene_id="s",
        camera="front_camera",
        frame_idx=0,
        ts=0.0,
        cls="pedestrian",
        box_xyxy=(0, 0, 1, 1),
        source="sam",
        verdict=verdict,
    )


def test_catch_and_false_rejection_rates_on_constructed_claims():
    false = [_claim(f"x{i}", v) for i, v in enumerate(["FAIL", "UNVERIFIABLE", "PASS", "PASS", None])]
    true = [_claim(f"t{i}", v) for i, v in enumerate(["PASS", "PASS", "PASS", "PASS", "FAIL", None])]
    truth = {c.claim_id: c.claim_id.startswith("t") for c in false + true}
    counts = M.count_claims(false + true, truth)
    assert counts == {
        "claims_false": 5,
        "claims_true": 6,
        "claims_false_checked": 4,
        "claims_false_caught": 2,
        "claims_true_checked": 5,
        "claims_true_rejected": 1,
    }
    o = _o("q", [], [], 0, **counts)
    assert M.catch_rate([o]) == 0.5  # 2 of the 4 checked false claims were FAIL or UNVERIFIABLE
    assert M.false_rejection_rate([o]) == 0.2  # 1 of the 5 checked true claims was rejected
    assert M.hallucination_rate([o]) == pytest.approx(5 / 11)  # unchecked claims count here too
    unchecked = _o("u", [], [], 0, **M.count_claims([_claim("x", None)], {"x": False}))
    assert M.catch_rate([unchecked]) is None and M.hallucination_rate([unchecked]) == 1.0


def test_bootstrap_is_deterministic():
    qs = [_o(f"q{i}", ["f"], ["f"] if i % 3 else [], 1, cost=0.01 * i) for i in range(15)]
    a = M.bootstrap_ci(qs, M.p_at_10)
    assert a == M.bootstrap_ci(qs, M.p_at_10)  # fixed seed (A-133)
    cost = M.bootstrap_ci(qs, M.cost_per_query)
    assert cost == M.bootstrap_ci(qs, M.cost_per_query) != M.bootstrap_ci(qs, M.cost_per_query, seed=1)
    lo, hi = a
    assert lo <= M.p_at_10(qs) <= hi and 0.0 <= lo < hi <= 1.0
    same = [_o(f"q{i}", ["f"], ["f"], 1) for i in range(5)]
    assert M.bootstrap_ci(same, M.p_at_10) == (0.1, 0.1)
    s = M.summarise(qs)
    assert set(M.METRICS) <= set(s) and s["queries"] == 15
    assert s["p_at_10"]["ci95"] == list(a)


# Windows and duplicate findings


def test_uniform_windows_cover_the_scene():
    s = SyntheticScene(SyntheticConfig(n_frames=60))  # 5.9 s
    ws = ablation.uniform_windows(s)
    rel = [(round(w.t0 - 1000, 1), round(w.t1 - 1000, 1)) for w in ws]
    # 2 s cores every 1 s, padded by 1 s and clipped (A-136)
    assert rel == [(0.0, 3.0), (0.0, 4.0), (1.0, 5.0), (2.0, 5.9), (3.0, 5.9)]
    assert all(w.source == "uniform" and w.cameras == [] for w in ws)


def test_nms_merges_the_same_binding_over_overlapping_windows():
    def f(fid, t0, t1, score, key):
        trace = gt.ActorTrace(positions=[(t0, 0.0, 0.0, 0.0), (t1, 1.0, 0.0, 0.0)])
        return gt.ScoredFinding(
            finding_id=fid, scene_id="s", t0=t0, t1=t1, score=score, actors={"a1": trace}, key=key
        )

    out = ablation.nms(
        [f("w1", 0.0, 2.0, 0.5, "a1=t1"), f("w2", 1.0, 3.0, 0.9, "a1=t1"), f("w3", 1.0, 3.0, 0.7, "a1=t2")],
        0.1,
    )
    # ASSUMPTION A-137: w1 and w2 bind the same track and overlap: one finding, union interval, best score
    assert [(x.finding_id, x.t0, x.t1, x.score) for x in out] == [
        ("w2", 0.0, 3.0, 0.9),
        ("w3", 1.0, 3.0, 0.7),
    ]
    assert [p[0] for p in out[0].actors["a1"].positions] == [0.0, 1.0, 2.0, 3.0]


# Configurations on the synthetic scene


def _ghost(scene, camera, concept, obj_id, box):
    """A hallucinated SAM track: a confident box where no object of the class is."""
    t = track2d_from_masks(scene, camera, concept, obj_id, {i: rect_mask(scene, *box) for i in range(30)})
    return t.model_copy(update={"frames": [f.model_copy(update={"conf": 0.97}) for f in t.frames]})


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    scene = SyntheticScene(SyntheticConfig(hidden_pedestrian=True))
    # Stand-in SAM output: the true objects (conf 0.9) plus two confident hallucinations, a "person" on the
    # wall and a "cyclist" on the empty sky.
    ghosts = {
        "person": [_ghost(scene, "left_camera", "person", 90, (220, 160, 420, 320))],
        "cyclist": [_ghost(scene, "front_camera", "cyclist", 91, (420, 0, 600, 120))],
    }
    source = track_source(scene, {"person": ["crosser", "hidden_ped"], "car": ["parked"]}, ghosts)
    gt_dir = tmp_path_factory.mktemp("eval") / "gt"
    with eval_ground_truth():
        gt.build(QUERIES, [scene], gt_dir)
    return ablation.Setup(queries=QUERIES, scenes=[scene], truth_scenes=[scene], gt_dir=gt_dir, source=source)


@pytest.fixture(scope="module")
def results(setup):
    return {c: ablation.run_config(c, setup) for c in ("b", "c")}


def test_config_b_and_c_run_and_c_beats_b_on_precision(results):
    b, c = M.summarise(results["b"]), M.summarise(results["c"])
    # (b) binds the most confident SAM track: the ghosts outrank the true pedestrians; nothing checks them.
    # P@10 divides by 10: (b) has one hit (the parked car) among 4 queries, (c) two (crossing and parked car).
    assert b["p_at_10"]["value"] == pytest.approx(0.025) and b["returned_top10"]["value"] == 1.0
    assert c["p_at_10"]["value"] == pytest.approx(0.05) and c["returned_top10"]["value"] == 0.5
    assert b["recall"]["value"] == pytest.approx(1 / 3) and c["recall"]["value"] == pytest.approx(2 / 3)
    # Precision: (b) 1 of its 4 findings is correct, (c) both of its 2.
    assert b["precision"]["value"] == 0.25 and c["precision"]["value"] == 1.0
    assert c["precision"]["value"] > b["precision"]["value"] and c["p_at_10"]["value"] > b["p_at_10"]["value"]
    assert c["p_at_10"]["value"] > b["p_at_10"]["value"]
    # (b) runs no verifier, so it has a hallucination rate but no catch rate.
    assert b["catch_rate"]["value"] is None and b["hallucination_rate"]["value"] > 0
    # A-10: the hidden pedestrian's head boxes are true claims; the verifier rejects them (too few points).
    assert c["catch_rate"]["value"] == 1.0 and c["false_rejection_rate"]["value"] == pytest.approx(60 / 210)
    # Cost at list price with a cold cache (A-13): 6 cameras x $0.01 (a 30-frame clip bills the 50-frame
    # minimum); the occluded query also asks for cars, the occluder class.
    cost = {o.query_id: round(o.cost_usd, 4) for o in results["c"]}
    assert cost == {"syn_crossing": 0.06, "syn_parked": 0.06, "syn_occluded": 0.12, "syn_cyclist": 0.06}


def test_configs_run_outside_the_label_guard(setup):
    with eval_ground_truth(), pytest.raises(RuntimeError, match="outside eval_ground_truth"):
        ablation.run_query("c", QUERIES[0], setup)


def test_a_and_d_wait_on_mayanks_files(setup, tmp_path):
    s = ablation.Setup(
        **{**setup.__dict__, "baseline_dir": tmp_path / "none", "triage_dir": tmp_path / "none"}
    )
    with pytest.raises(formats.WaitingOnError, match="waiting on M5"):
        ablation.run_query("a", QUERIES[0], s)
    with pytest.raises(formats.WaitingOnError, match="waiting on M4"):
        ablation.run_query("d", QUERIES[0], s)


def test_a_and_d_adapters_read_the_documented_formats(setup, tmp_path):
    scene = setup.scenes[0]
    ts = scene.frames("front_camera")[5].ts
    with eval_ground_truth():
        ls = gt.LabelScene(scene)
    crosser_box = next(tuple(map(float, b)) for c, b in ls.boxes_at("front_camera", 5) if c.uuid == "crosser")
    claims = [
        formats.VlmClaim(
            claim_id="c1", scene_id="synthetic", camera="front_camera", frame_idx=5, ts=ts, cls="pedestrian",
            box_xyxy=crosser_box,
        ),
        formats.VlmClaim(
            claim_id="c2", scene_id="synthetic", camera="front_camera", frame_idx=5, ts=ts, cls="pedestrian",
            box_xyxy=(420.0, 0.0, 600.0, 120.0),
        ),
    ]  # fmt: skip
    base = tmp_path / "baseline_a"
    base.mkdir()
    (base / "claims.jsonl").write_text("\n".join(c.model_dump_json() for c in claims) + "\n")
    t = scene.lidar_timestamps()
    rankings = {
        "format": "longtail.baseline_a/v1",
        "model": "spark-open",
        "queries": {
            "syn_crossing": {
                "matches": [
                    {"window_id": "v0", "scene_id": "synthetic", "t0": t[0], "t1": t[20], "score": 0.9,
                     "claim_ids": ["c1"]},
                    {"window_id": "v1", "scene_id": "synthetic", "t0": t[0], "t1": t[20], "score": 0.8,
                     "claim_ids": ["c2"]},
                ],
                "cost_usd": 0.02,
                "seconds": 3.0,
            }
        },
    }  # fmt: skip
    (base / "rankings.json").write_text(json.dumps(rankings))
    tri = tmp_path / "triage_d"
    tri.mkdir()
    windows = [w.model_dump() for w in ablation.uniform_windows(scene)]
    triage = {
        "format": "longtail.triage_d/v1",
        "model": "glimmer",
        "queries": {
            "syn_crossing": {"windows": windows, "windows_total": 4, "cost_usd": 0.5, "seconds": 2.0}
        },
    }
    (tri / "windows.json").write_text(json.dumps(triage))
    s = ablation.Setup(**{**setup.__dict__, "queries": QUERIES[:1], "baseline_dir": base, "triage_dir": tri})
    [a] = ablation.run_config("a", s)
    # The first window cites the crosser's box (a 2D match, A-127); the second a box on the empty sky.
    assert a.ranked == ["a|v0", "a|v1"] and a.matched == {"a|v0"} and a.cost_usd == 0.02
    # The verifier checks both boxes: the sky box is false and rejected, the crosser's true and accepted.
    assert (a.claims_false, a.claims_false_caught, a.claims_true, a.claims_true_rejected) == (1, 1, 1, 0)
    [d] = ablation.run_config("d", s)
    assert len(d.matched) == 1 and d.cost_usd == pytest.approx(0.06 + 0.5)
    assert d.extra["windows_kept_frac"] == 0.25


def test_results_json_and_chart(setup, results, tmp_path):
    out = tmp_path / "results"
    for c in ("b", "c"):
        doc = ablation.results_doc(c, results[c], setup, "all", ablation.DESCRIPTIONS[c])
        ablation.write_results(doc, out)
    doc = json.loads((out / "ablation_c.json").read_text())
    # ASSUMPTION A-140: the commit and a hash of the thresholds used travel with the numbers
    assert doc["format"] == "longtail.ablation/v1" and len(doc["commit"].split("-")[0]) == 40
    assert doc["thresholds_hash"] == ablation.thresholds_hash(setup.verify_cfg, setup.gt_cfg)
    assert doc["bootstrap"] == {"resamples": 1000, "seed": M.BOOTSTRAP_SEED, "over": "queries"}
    assert doc["metrics"]["p_at_10"]["value"] == pytest.approx(0.05) and len(doc["queries"]) == 4
    assert doc["metrics"]["returned_top10"]["value"] == 0.5 and doc["metrics"]["precision"]["value"] == 1.0
    assert [q["returned_top10"] for q in doc["queries"]] == [1, 1, 0, 0]
    png = out / "ablation.png"
    assert png.is_file() and png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(ValueError, match="named results"):
        ablation.write_results(doc, tmp_path / "elsewhere")


# Breaks


@pytest.fixture(scope="module")
def break_report(setup):
    return breaks.run_breaks(setup, "all")


def test_each_break_lowers_a_metric(break_report):
    # ASSUMPTION A-141: a break passes when it lowers precision, the catch rate or recall
    for name in breaks.BREAKS:
        b = break_report["breaks"][name]
        assert b["passed"] and b["moved"], f"{name} lowered nothing: {b['lowered']}"
    assert break_report["holes"] == []


def test_break_results(break_report):
    moved = {k: v["moved"] for k, v in break_report["breaks"].items()}
    m = {k: v["metrics"] for k, v in break_report["breaks"].items()}
    # Verifier off: the unverified ghost findings come back (precision 1.0 to 0.5) and nothing is caught.
    assert moved["verifier_off"] == ["precision", "catch_rate"]
    assert m["verifier_off"]["precision"]["value"] == 0.5 and m["verifier_off"]["catch_rate"]["value"] == 0.0
    # Calibration shifted: the crossing finding is lost and a wrong one verified.
    assert moved["calibration_2deg_0.3m"] == ["precision", "recall"]
    assert m["calibration_2deg_0.3m"]["precision"]["value"] == 0.0
    # LiDAR 200 ms late: fails safe, shows up as lost recall; what it returns is still right.
    assert moved["lidar_offset_200ms"] == ["recall"]
    assert m["lidar_offset_200ms"]["precision"]["value"] == 1.0


def test_scene_wrappers():
    s = SyntheticScene()
    shifted = breaks.CalibrationShift(s)
    ts = s.frames("front_camera")[0].ts
    a, b = s.extrinsics("front_camera", ts), shifted.extrinsics("front_camera", ts)
    assert np.linalg.norm(a[:3, 3] - b[:3, 3]) == pytest.approx(0.3)
    cos = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2
    assert np.degrees(np.arccos(np.clip(cos, -1, 1))) == pytest.approx(2.0)
    late = breaks.LidarTimeOffset(s)
    t0 = s.lidar_timestamps()[0]
    assert late.lidar_timestamps()[0] == pytest.approx(t0 + 0.2)
    sweep = late.lidar(t0 + 0.2)
    assert sweep.ts == pytest.approx(t0 + 0.2) and len(sweep.points) == len(s.lidar(t0).points)
    assert np.allclose(late.ego_pose(t0 + 0.2), s.ego_pose(t0))
    with eval_ground_truth():
        assert list(late.labels(0)["uuid"]) == list(s.labels(0)["uuid"])


# Sharding by scene (A-233)


def test_partial_runs_per_scene_merge_to_the_same_scores_as_one_run(tmp_path):
    a = SyntheticScene(SyntheticConfig(hidden_pedestrian=True), scene_id="syn-a")
    b = SyntheticScene(SyntheticConfig(), scene_id="syn-b")
    ghosts = {"person": [_ghost(a, "left_camera", "person", 90, (220, 160, 420, 320))]}
    sources = {
        "syn-a": track_source(a, {"person": ["crosser", "hidden_ped"], "car": ["parked"]}, ghosts),
        "syn-b": track_source(b, {"person": ["crosser"], "car": ["parked"]}),
    }

    def source(scene, camera, concept):
        return sources[scene.scene_id](scene, camera, concept)

    gt_dir = tmp_path / "gt"
    with eval_ground_truth():
        gt.build(QUERIES, [a, b], gt_dir)
    setup = ablation.Setup(queries=QUERIES, scenes=[a, b], truth_scenes=[a, b], gt_dir=gt_dir, source=source)
    for config in ("b", "c"):
        whole = ablation.run_config(config, setup)
        paths = ablation.write_partial(config, setup, tmp_path / "partial")
        assert [p.name for p in paths] == ["syn-a.pkl", "syn-b.pkl"]
        merged = ablation.merge_config(config, setup, tmp_path / "partial")
        # wall time differs between runs; the cost only in float summation order
        strip = lambda o: {**o.__dict__, "seconds": None, "cost_usd": round(o.cost_usd, 9)}  # noqa: E731
        for m, w in zip(merged, whole, strict=True):
            diff = {k: (v, strip(w)[k]) for k, v in strip(m).items() if v != strip(w)[k]}
            assert not diff, (m.query_id, diff)
        assert M.summarise(merged)["p_at_10"] == M.summarise(whole)["p_at_10"]
    with pytest.raises(FileNotFoundError, match="syn-c"):
        ablation.read_partials(tmp_path / "partial", "c", ["syn-a", "syn-c"])
