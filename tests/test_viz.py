"""C9 export on the synthetic scene: overlays, bird's-eye view, timeline, manifest, schema and CLI."""

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from PIL import Image
from synthetic import SyntheticScene, object_track2d, track_source, without_masks

from longtail.contracts import CandidateWindow, ScenarioSpec
from longtail.geometry.project import project_sweep
from longtail.verify.run import load_run, verify_and_record
from longtail.vision.rle import decode
from longtail.viz import cli as viz_cli
from longtail.viz import export as viz
from longtail.viz.schema import json_schema

ROOT = Path(__file__).resolve().parents[1]
CROSSING = [("exists", ["a1"]), ("crossing_ego_path", ["a1"]), ("ego_moving", [])]


def _spec(preds) -> ScenarioSpec:
    return ScenarioSpec(
        query="test",
        actors=[{"id": "a1", "cls": "pedestrian", "sam_prompt": "person"}],
        predicates=[{"name": n, "actors": a, "args": {}} for n, a in preds],
        assumptions=[],
        clarification=None,
    )


def _windows(scene, *ranges):
    ts = scene.lidar_timestamps()
    return [
        CandidateWindow(
            window_id=f"w{k}",
            scene_id=scene.scene_id,
            t0=float(ts[a]),
            t1=float(ts[b]),
            cameras=[],
            score=1.0,
            source="uniform",
        )  # fmt: skip
        for k, (a, b) in enumerate(ranges)
    ]


@pytest.fixture(scope="module")
def scene():
    return SyntheticScene()


@pytest.fixture(scope="module")
def exported(scene, tmp_path_factory):
    """Three runs under runs/: a verified crossing, a rejected claim and a box-only (VLM) claim."""
    base = tmp_path_factory.mktemp("viz")
    before = {p for p in base.rglob("*")}
    runs = base / "runs"
    out = {}
    src = track_source(scene, {"person": ["crosser"]})
    vlm = without_masks(object_track2d(scene, "crosser", "front_camera", concept="person"))
    cases = {
        "crossing": (_spec(CROSSING), src),
        "rejected": (_spec([("exists", ["a1"]), ("stopped", ["a1"])]), src),
        "vlm": (_spec([("exists", ["a1"])]), track_source(scene, {}, {"person": [vlm]})),
        # A mask dilated past the crosser onto empty sky, where the tint is not covered by LiDAR dots.
        "dilated": (_spec(CROSSING), track_source(scene, {"person": ["crosser"]}, dilate_px=12)),
    }
    for name, (spec, source) in cases.items():
        run = verify_and_record(spec, _windows(scene, (0, 29)), scene, source, runs / name)
        manifest = viz.export_run(run, scene, runs / name / "viz")
        out[name] = (run, manifest, runs / name / "viz")
    out["_new_files"] = {p for p in base.rglob("*") if p.is_file()} - before
    out["_base"] = base
    return out


def _px(path: Path, uv) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))[int(uv[1]), int(uv[0])].astype(int)


# Overlays


def test_verified_crossing_has_an_overlay_for_every_camera_and_frame(scene, exported):
    run, manifest, out = exported["crossing"]
    [f] = manifest.findings
    assert f.verified and run.findings[0].verified
    ts = scene.lidar_timestamps()
    frames = [i for i, t in enumerate(ts) if f.t0 - 1e-6 <= t <= f.t1 + 1e-6]
    expected = {(c, i) for c in scene.cameras for i in frames}
    assert {(o.camera, o.frame_idx) for o in f.overlays} == expected and not f.missing_overlays
    assert all((out / o.path).is_file() for o in f.overlays)


def test_mask_and_cluster_points_are_where_expected(scene, exported):
    run, manifest, out = exported["dilated"]
    [f] = manifest.findings
    i = 10
    overlay = out / next(o.path for o in f.overlays if o.camera == "front_camera" and o.frame_idx == i)
    pool = viz.TrackPool(scene, run.replay_source(), viz.load_verify_config())
    [obj] = pool.get("person", "pedestrian", tuple(scene.cameras))
    f2d = next(
        fr
        for t in obj.association.inputs
        if t.camera == "front_camera"
        for fr in t.frames
        if fr.frame_idx == i
    )
    mask = decode(f2d.mask_rle)
    sweep = scene.lidar(scene.lidar_timestamps()[i])
    proj = project_sweep(scene, sweep, "front_camera", scene.frames("front_camera")[i])
    pix = proj.pixels()
    near_point = np.zeros(mask.shape, bool)
    for dy in range(-5, 6):
        for dx in range(-5, 6):
            u, v = (
                np.clip(pix[:, 0] + dx, 0, mask.shape[1] - 1),
                np.clip(pix[:, 1] + dy, 0, mask.shape[0] - 1),
            )
            near_point[v, u] = True
    near_point[:30] = True  # the header label
    # Mask pixels with no point near them: the background tinted 35% toward the first actor's colour (cyan).
    arr = np.asarray(Image.open(overlay).convert("RGB")).astype(int)
    tint = (1 - viz.MASK_ALPHA) * np.array(viz.BACKGROUND) + viz.MASK_ALPHA * np.array(viz.ACTOR_COLOURS[0])
    tinted = arr[mask & ~near_point]
    assert len(tinted) > 50 and np.median(np.abs(tinted - tint).max(axis=1)) < 20
    # Pixels away from the mask and every point: plain background.
    plain = arr[~mask & ~near_point]
    assert np.median(np.abs(plain - np.array(viz.BACKGROUND)).max(axis=1)) < 10
    # Cluster points: drawn in the actor's colour.
    rows = np.concatenate([c.point_index for c in obj.association.clusters[float(sweep.ts)]])
    hits = pix[np.isin(proj.index, rows)]
    colours = np.array([_px(overlay, uv) for uv in hits[:20]])
    assert np.median(np.abs(colours - np.array(viz.ACTOR_COLOURS[0])).max(axis=1)) < 40


def test_box_only_claims_render_dashed_yellow(scene, exported):
    _, manifest, out = exported["vlm"]
    [f] = manifest.findings
    overlay = out / next(o.path for o in f.overlays if o.camera == "front_camera" and o.frame_idx == 5)
    claim = object_track2d(scene, "crosser", "front_camera", concept="person", frames=[5]).frames[0]
    x1, y1, x2, y2 = (int(v) for v in claim.box_xyxy)
    arr = np.asarray(Image.open(overlay).convert("RGB")).astype(int)

    def yellow(row: np.ndarray) -> np.ndarray:
        return np.abs(row - np.array(viz.CLAIM_COLOUR)).max(axis=1) < 70

    # ASSUMPTION A-108: a box-only claim is a dashed yellow box: about half of its bottom edge is yellow
    assert 0.3 < yellow(arr[y2, x1 : x2 + 1]).mean() < 0.8
    # A mask-backed box (the crossing run) is solid and in the actor's colour, never yellow.
    _, crossing, cdir = exported["crossing"]
    solid = cdir / next(
        o.path for o in crossing.findings[0].overlays if o.camera == "front_camera" and o.frame_idx == 5
    )
    carr = np.asarray(Image.open(solid).convert("RGB")).astype(int)
    assert yellow(carr[y2, x1 : x2 + 1]).mean() < 0.05


# Timeline


def test_rejected_finding_exports_with_its_reason_codes(exported):
    run, manifest, out = exported["rejected"]
    [f] = manifest.findings
    assert not f.verified and f.overlays
    timeline = json.loads((out / f.timeline_path).read_text())
    stopped = next(p for p in timeline["predicates"] if p["name"] == "stopped")
    assert stopped["status"] == "FAIL" and stopped["reason"] == "not_stopped"
    assert {e["reason"] for e in stopped["entries"]} >= {"not_stopped"}
    assert len(stopped["entries"]) == len(timeline["frames"]) == 30
    assert all(e["status"] in ("PASS", "FAIL", "UNVERIFIABLE", "NOT_JUDGED") for e in stopped["entries"])
    # ASSUMPTION A-102: the timeline is the engine's per-frame verdicts, the same ones the Finding summarises
    finding = run.findings[0]
    result = next(r for r in finding.results if r.name == "stopped")
    assert [e["status"] for e in stopped["entries"]] == [s for _, s, _ in result.evidence["timeline"]]


def test_timeline_marks_the_interval(exported):
    _, manifest, out = exported["crossing"]
    [f] = manifest.findings
    timeline = json.loads((out / f.timeline_path).read_text())
    # ASSUMPTION A-106: the timeline covers the window; in_interval marks the frames with overlays
    inside = {fr["frame_idx"] for fr in timeline["frames"] if fr["in_interval"]}
    assert inside == {o.frame_idx for o in f.overlays}


# Bird's-eye view


def test_bev_validates_and_has_corridor_and_ego_box(exported):
    schema = json_schema()
    for name in ("crossing", "rejected", "vlm"):
        _, manifest, out = exported[name]
        for f in manifest.findings:
            bev = json.loads((out / f.bev_path).read_text())
            jsonschema.validate(bev, {"$ref": "#/$defs/BevExport", "$defs": schema["$defs"]})
            assert len(bev["frames"]) == len(f.overlays) // 6
            for frame in bev["frames"]:
                assert len(frame["ego_box"]) == 4 and len(frame["corridor"]) >= 4 and frame["points_xyz"]
                assert frame["tracks"] and frame["tracks"][0]["actors"] == ["a1"]
            # ASSUMPTION A-99: under the 5 MB budget
            assert f.bev_bytes <= viz.BEV_BUDGET_BYTES


def test_bev_track_is_where_the_crosser_is(scene, exported):
    _, manifest, out = exported["crossing"]
    bev = json.loads((out / manifest.findings[0].bev_path).read_text())
    frame = bev["frames"][0]
    track = frame["tracks"][0]
    truth = scene.objects(frame["ts"])["crosser"].center_world
    from longtail.geometry.transforms import world_to_ego

    centre = world_to_ego(scene.ego_pose(frame["ts"]), truth[None])[0]
    assert np.hypot(*(np.array(track["box"]["center_xy"]) - centre[:2])) < 0.3
    assert track["velocity_xy_mps"][1] > 1.0  # crossing toward the left


def test_bev_decimation_fits_a_smaller_budget(scene, exported, monkeypatch):
    run, _, out = exported["crossing"]
    monkeypatch.setattr(viz, "BEV_BUDGET_BYTES", 400_000)
    manifest = viz.export_run(run, scene, out.parent / "viz_small")
    [f] = manifest.findings
    bev = json.loads((out.parent / "viz_small" / f.bev_path).read_text())
    assert f.bev_bytes <= 400_000 and bev["decimation"]["max_points_per_frame"] < 1000


# Manifest and output location


def test_manifest_lists_every_asset_and_every_file_exists(exported):
    for name in ("crossing", "rejected", "vlm", "dilated"):
        _, manifest, out = exported[name]
        on_disk = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()} - {"manifest.json"}
        listed = set()
        for f in manifest.findings:
            listed |= {o.path for o in f.overlays} | {f.bev_path, f.timeline_path}
        assert listed == on_disk
        saved = json.loads((out / "manifest.json").read_text())
        jsonschema.validate(saved, {"$ref": "#/$defs/VizManifest", "$defs": json_schema()["$defs"]})


def test_nothing_is_written_outside_runs(scene, exported):
    base = exported["_base"]
    assert all("runs" in p.relative_to(base).parts[:1] for p in exported["_new_files"])
    run = exported["crossing"][0]
    # ASSUMPTION A-104: exports outside a runs/ directory are refused
    with pytest.raises(ValueError, match="runs/"):
        viz.export_run(run, scene, base / "elsewhere")
    assert not (base / "elsewhere").exists()


def test_export_never_reads_labels(scene, exported, monkeypatch):
    def forbidden(*_a, **_k):
        raise AssertionError("C9 read labels")

    monkeypatch.setattr(SyntheticScene, "_labels", forbidden)
    monkeypatch.setattr(SyntheticScene, "_semseg", forbidden)
    run, _, out = exported["rejected"]
    viz.export_run(run, scene, out.parent / "viz_guarded")


# Schema and CLI


def test_schema_file_is_current():
    committed = json.loads((ROOT / "docs" / "viz-schema.json").read_text())
    assert committed == json.loads(json.dumps(json_schema())), (
        "run: longtail viz --write-schema docs/viz-schema.json"
    )
    for name in ("VizManifest", "BevExport", "Timeline", "BevFrame", "TimelinePredicate", "OverlayAsset"):
        assert name in committed["$defs"]


def test_cli_exports_a_run_or_one_finding(scene, exported, monkeypatch, capsys):
    from longtail.cli import main

    run_dir = exported["crossing"][2].parent
    monkeypatch.setattr(viz_cli, "load_scene", lambda run: scene)
    assert main(["viz", str(run_dir), "--out", str(run_dir / "viz_cli")]) == 0
    assert (run_dir / "viz_cli" / "manifest.json").is_file()
    fid = load_run(run_dir).findings[0].finding_id
    assert main(["viz", str(run_dir / fid), "--out", str(run_dir / "viz_one")]) == 0
    assert "verified" in capsys.readouterr().out
    assert main(["viz", str(run_dir / "nope" / "nothing")]) == 1
