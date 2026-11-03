"""scripts/report_results.py on results JSON from the real ablation and breaks on the synthetic scene."""

import importlib.util
import json
from pathlib import Path

import pytest
from synthetic import SyntheticConfig, SyntheticScene, rect_mask, track2d_from_masks, track_source

from longtail.data.guard import eval_ground_truth
from longtail.eval import ablation, breaks, gt

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("report_results", ROOT / "scripts" / "report_results.py")
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)

QUERIES = gt.load_queries(Path(__file__).parent / "fixtures" / "gold_synthetic.yaml")


def _ghost(scene, camera, concept, obj_id, box):
    t = track2d_from_masks(scene, camera, concept, obj_id, {i: rect_mask(scene, *box) for i in range(30)})
    return t.model_copy(update={"frames": [f.model_copy(update={"conf": 0.97}) for f in t.frames]})


@pytest.fixture(scope="module")
def results_dir(tmp_path_factory):
    """ablation_b.json, ablation_c.json and breaks.json as the CLIs write them (test_ablation's setup)."""
    scene = SyntheticScene(SyntheticConfig(hidden_pedestrian=True))
    ghosts = {
        "person": [_ghost(scene, "left_camera", "person", 90, (220, 160, 420, 320))],
        "cyclist": [_ghost(scene, "front_camera", "cyclist", 91, (420, 0, 600, 120))],
    }
    source = track_source(scene, {"person": ["crosser", "hidden_ped"], "car": ["parked"]}, ghosts)
    root = tmp_path_factory.mktemp("eval")
    with eval_ground_truth():
        gt.build(QUERIES, [scene], root / "gt")
    setup = ablation.Setup(
        queries=QUERIES, scenes=[scene], truth_scenes=[scene], gt_dir=root / "gt", source=source
    )
    out = root / "results"
    for c in ("b", "c"):
        doc = ablation.results_doc(c, ablation.run_config(c, setup), setup, "all", ablation.DESCRIPTIONS[c])
        ablation.write_results(doc, out)
    (out / "breaks.json").write_text(json.dumps(breaks.run_breaks(setup, "all")))
    return out


def test_tables_from_the_synthetic_results(results_dir):
    docs, br = rr.load(results_dir)
    assert set(docs) == {"b", "c"} and br is not None
    assert rr.check_consistent(docs, br) == []
    text = rr.render(docs, br, "results")
    rows = {line.split(" | ")[0].lstrip("| "): line for line in text.splitlines() if line.startswith("| (")}
    # (a) waits on the baseline files; (b) and (c) carry their numbers with intervals (see test_ablation)
    assert "not run" in rows["(a) Spark Open alone"]
    assert "| 0.25 [" in rows["(b) + SAM 3.1 tracks"]  # precision
    c = rows["(c) + LiDAR verification"]
    assert (
        c.startswith("| (c) + LiDAR verification | 1.00 [")
        and "0.050 [0.000, 0.100]" in c
        and "returned 0.5" in c
    )
    assert "$0.075 [" in c  # cost per query: (0.06 + 0.06 + 0.12 + 0.06) / 4
    assert c.count("[") == 8  # every number has an interval: 7 metrics plus the returned count
    # every break, with what it moved
    for name in breaks.BREAKS:
        line = next(x for x in text.splitlines() if x.startswith(f"| {rr.BREAK_NAMES[name]} "))
        assert line.rstrip().endswith("| passes |"), line
    assert "Split: **all**, 4 queries, 1 scene." in text and "(d)" in text and "Holes" not in text


def test_d_is_never_reported(results_dir, tmp_path):
    for p in results_dir.glob("*.json"):
        (tmp_path / p.name).write_text(p.read_text())
    (tmp_path / "ablation_d.json").write_text((results_dir / "ablation_c.json").read_text())
    docs, _ = rr.load(tmp_path)
    assert set(docs) == {"b", "c"}
    assert "+ Glimmer" not in rr.render(*rr.load(tmp_path), "x")


def test_mismatched_runs_are_flagged(results_dir, tmp_path):
    for p in results_dir.glob("*.json"):
        doc = json.loads(p.read_text())
        if p.name == "ablation_b.json":
            doc["split"] = "heldout"
        (tmp_path / p.name).write_text(json.dumps(doc))
    docs, br = rr.load(tmp_path)
    assert any("split differs" in p for p in rr.check_consistent(docs, br))
    assert "not comparable" in rr.render(docs, br, "x")
    assert rr.main([str(tmp_path)]) == 1


def test_write_replaces_only_the_results_section(results_dir, tmp_path):
    report = tmp_path / "eval-report.md"
    report.write_text("# Report\n\nBefore.\n")
    assert rr.main([str(results_dir), "--write", str(report)]) == 0
    once = report.read_text()
    assert once.startswith("# Report\n\nBefore.\n\n" + rr.START) and once.rstrip().endswith(rr.END)
    report.write_text(once + "\n## After\n")
    rr.main([str(results_dir), "--write", str(report)])
    twice = report.read_text()
    assert twice.count(rr.START) == 1 and twice.endswith("## After\n")
    assert twice.replace("\n## After\n", "").rstrip() == once.rstrip()


def test_missing_results_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        rr.load(tmp_path)
