"""C9 export: camera overlays, bird's-eye view JSON, predicate timeline and a manifest, per finding.

Verified and unverified findings alike (the web view has a "rejected by verifier" tab). Built from a saved
verification run (A-101), never from labels. Everything is written under a runs/ directory.
"""

import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from longtail.contracts import Finding, Frame2D
from longtail.data.base import Scene
from longtail.geometry.ego_path import Corridor, corridor
from longtail.geometry.project import project_sweep
from longtail.geometry.sync import sync_camera
from longtail.geometry.tracks3d import TrackedObject
from longtail.geometry.transforms import world_to_ego
from longtail.verify.config import VerifyConfig, load_verify_config
from longtail.verify.engine import UNBOUND, TrackPool
from longtail.verify.run import VerifyRun
from longtail.vision.rle import decode
from longtail.viz.schema import (
    SCHEMA_VERSION,
    BevBox,
    BevDecimation,
    BevExport,
    BevFrame,
    BevTrack,
    FindingAssets,
    MissingOverlay,
    OverlayAsset,
    Timeline,
    TimelineEntry,
    TimelineFrame,
    TimelinePredicate,
    VizManifest,
)

# ASSUMPTION A-99: bev.json stays under 5 MB per finding: points within 50 m, one per 0.2 m voxel,
# then an even stride to the cap the budget allows, coordinates rounded to 1 cm
BEV_BUDGET_BYTES = 5_000_000
BEV_MAX_RANGE_M = 50.0
BEV_VOXEL_M = 0.2
BEV_BYTES_PER_POINT = 22  # "[-12.34,5.67,-0.89]," with room to spare
# ASSUMPTION A-100: ego footprint 4.8 x 1.9 m centred on the ego origin (the Pandar64); a placeholder
EGO_LENGTH_M, EGO_WIDTH_M = 4.8, 1.9

# ASSUMPTION A-108: overlay style: actor colours below, masks at 35% opacity, clusters as 3 px dots in
# the actor colour, mask-backed boxes solid, box-only claims dashed yellow and labelled "VLM claim"
ACTOR_COLOURS = [(0, 200, 255), (255, 80, 200), (120, 255, 80), (255, 150, 40), (160, 120, 255)]
CLAIM_COLOUR = (255, 214, 0)
MASK_ALPHA = 0.35
BACKGROUND = (28, 30, 34)


def _require_runs(path: Path) -> Path:
    # ASSUMPTION A-104: exports are written only inside a directory named runs (git-ignored)
    path = Path(path).resolve()
    if "runs" not in path.parts:
        raise ValueError(f"exports go under a runs/ directory, not {path}")
    return path


# Drawing


def depth_colours(depth: np.ndarray, far_m: float = 50.0) -> np.ndarray:
    """Red (near) through yellow and green to blue (far), as uint8 RGB."""
    x = np.clip(depth / far_m, 0, 1)
    rgb = [np.clip(1.5 - np.abs(4 * x - c), 0, 1) for c in (0.5, 2.0, 3.5)]
    return (np.column_stack(rgb) * 255).astype(np.uint8)


def _dots(arr: np.ndarray, pixels: np.ndarray, colours: np.ndarray, radius: int) -> None:
    h, w = arr.shape[:2]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            u, v = pixels[:, 0] + dx, pixels[:, 1] + dy
            ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            arr[v[ok], u[ok]] = colours[ok]


def _dashed_rect(draw: ImageDraw.ImageDraw, box, colour, width: int = 3, dash: int = 10) -> None:
    x1, y1, x2, y2 = box
    for (a, b), horizontal in (((x1, x2), True), ((y1, y2), False)):
        for start in np.arange(a, b, 2 * dash):
            end = min(start + dash, b)
            if horizontal:
                draw.line([(start, y1), (end, y1)], fill=colour, width=width)
                draw.line([(start, y2), (end, y2)], fill=colour, width=width)
            else:
                draw.line([(x1, start), (x1, end)], fill=colour, width=width)
                draw.line([(x2, start), (x2, end)], fill=colour, width=width)


def _label(draw: ImageDraw.ImageDraw, xy, text: str, colour) -> None:
    x, y = xy
    left, top, right, bottom = draw.textbbox((x, y), text)
    draw.rectangle([left - 2, top - 2, right + 2, bottom + 2], fill=(0, 0, 0))
    draw.text((x, y), text, fill=colour)


# Per finding


def _bound_objects(
    run: VerifyRun, scene: Scene, finding: Finding, pool: TrackPool
) -> dict[str, TrackedObject | None]:
    """The tracked object behind each actor, rebuilt exactly as the engine built it."""
    window = run.window_of(finding.finding_id)
    cameras = tuple(window.cameras) or tuple(scene.cameras)
    out: dict[str, TrackedObject | None] = {}
    for actor in run.spec.actors:
        tid = finding.bindings.get(actor.id, UNBOUND)
        found = [o for o in pool.get(actor.sam_prompt, actor.cls, cameras) if o.track.track_id == tid]
        out[actor.id] = found[0] if found else None
    return out


def _camera_frame_for(sync: list, sweep_idx: int):
    """The camera frame C2 sync matched to this sweep (nearest synced one), or None."""
    # ASSUMPTION A-105: overlay frames are the synced camera frames matched to each sweep; gaps are listed
    matches = [s for s in sync if s.synced and s.lidar_idx == sweep_idx]
    return min(matches, key=lambda s: abs(s.gap_s)) if matches else None


def _overlay(
    scene: Scene,
    finding: Finding,
    camera: str,
    sweep_idx: int,
    cam_sync,
    objects: dict[str, TrackedObject | None],
) -> Image.Image:
    frame = scene.frames(camera)[cam_sync.frame_idx]
    size = scene.image_size(camera)
    if frame.image_path.is_file():
        image = Image.open(frame.image_path).convert("RGB")
    else:
        image = Image.new("RGB", size, BACKGROUND)  # no camera images (the synthetic scene)
    arr = np.asarray(image).astype(np.float32)
    sweep = scene.lidar(cam_sync.lidar_ts)
    proj = project_sweep(scene, sweep, camera, frame)
    pixels = proj.pixels()

    boxes: list[tuple[str, str, Frame2D, tuple[int, int, int]]] = []
    for k, (actor_id, obj) in enumerate(objects.items()):
        if obj is None:
            continue
        colour = ACTOR_COLOURS[k % len(ACTOR_COLOURS)]
        for t2d in obj.association.inputs:
            if t2d.camera != camera:
                continue
            for f2d in t2d.frames:
                if f2d.frame_idx != cam_sync.frame_idx:
                    continue
                if f2d.mask_rle is not None:
                    m = decode(f2d.mask_rle)
                    arr[m] = (1 - MASK_ALPHA) * arr[m] + MASK_ALPHA * np.array(colour, np.float32)
                boxes.append((actor_id, obj.track.track_id, f2d, colour))
    arr = arr.astype(np.uint8)
    _dots(arr, pixels, depth_colours(proj.depth), radius=1)
    for k, (_actor_id, obj) in enumerate(objects.items()):
        if obj is None:
            continue
        rows = [c.point_index for c in obj.association.clusters.get(float(sweep.ts), [])]
        if rows:
            hit = np.isin(proj.index, np.concatenate(rows))
            colour = np.array(ACTOR_COLOURS[k % len(ACTOR_COLOURS)], np.uint8)
            _dots(arr, pixels[hit], np.tile(np.array([0, 0, 0], np.uint8), (hit.sum(), 1)), radius=4)
            _dots(arr, pixels[hit], np.tile(colour, (hit.sum(), 1)), radius=3)
    image = Image.fromarray(arr)
    draw = ImageDraw.Draw(image)
    for actor_id, track_id, f2d, colour in boxes:
        box = [float(v) for v in f2d.box_xyxy]
        if f2d.mask_rle is None:
            _dashed_rect(draw, box, CLAIM_COLOUR)
            _label(draw, (box[0], max(box[1] - 14, 0)), f"VLM claim: {actor_id} = {track_id}", CLAIM_COLOUR)
        else:
            draw.rectangle(box, outline=colour, width=2)
            _label(draw, (box[0], max(box[1] - 14, 0)), f"{actor_id} = {track_id}", colour)
    verdict = "VERIFIED" if finding.verified else "REJECTED"
    failing = [
        f"{r.name}: {r.evidence.get('reason')}" for r in finding.results if r.required and r.status != "PASS"
    ]
    header = f"{verdict}  {finding.finding_id[:8]}  {camera}  sweep {sweep_idx}  " + "; ".join(failing[:2])
    _label(draw, (6, 6), header, (90, 255, 120) if finding.verified else (255, 90, 90))
    return image


def _corridor_polygon(c: Corridor) -> list[list[float]]:
    line = c.centerline
    seg = np.diff(line, axis=0)
    seg = seg / np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-9)
    normals = np.column_stack([-seg[:, 1], seg[:, 0]])
    vertex_normals = np.vstack([normals[:1], (normals[:-1] + normals[1:]) / 2, normals[-1:]])
    vertex_normals /= np.maximum(np.linalg.norm(vertex_normals, axis=1, keepdims=True), 1e-9)
    left = line + c.half_width_m * vertex_normals
    right = line - c.half_width_m * vertex_normals
    return np.round(np.vstack([left, right[::-1]]), 3).tolist()


def _decimate(points_ego: np.ndarray, cap: int) -> np.ndarray:
    near = points_ego[np.hypot(points_ego[:, 0], points_ego[:, 1]) <= BEV_MAX_RANGE_M]
    _, first = np.unique(np.floor(near / BEV_VOXEL_M).astype(np.int64), axis=0, return_index=True)
    kept = near[np.sort(first)]
    if len(kept) > cap:
        kept = kept[np.linspace(0, len(kept) - 1, cap).astype(int)]
    return np.round(kept, 2)


def _bev_tracks(scene: Scene, ts: float, ego: np.ndarray, objects: dict[str, TrackedObject | None]) -> list:
    yaw_ego = float(np.arctan2(ego[1, 0], ego[0, 0]))
    by_track: dict[str, tuple[TrackedObject, list[str]]] = {}
    for actor_id, obj in objects.items():
        if obj is not None:
            by_track.setdefault(obj.track.track_id, (obj, []))[1].append(actor_id)
    tracks = []
    for tid, (obj, actors) in by_track.items():
        sample = next((s for s in obj.track.samples if abs(s.ts - ts) < 1e-6), None)
        box = None
        fit = obj.box
        j = fit.at(ts) if fit is not None else None
        if j is not None and not np.isnan(fit.centres[j]).any():
            centre = world_to_ego(ego, fit.centres[j][None])[0]
            box = BevBox(
                center_xy=[round(float(centre[0]), 3), round(float(centre[1]), 3)],
                yaw_rad=round(float(fit.yaws[j] - yaw_ego), 4),
                length_m=round(float(fit.size[0]), 3),
                width_m=round(float(fit.size[1]), 3),
            )
        m = obj.motion
        hit = np.flatnonzero(np.abs(m.ts - ts) < 1e-6)
        k = int(hit[0]) if len(hit) else None
        # ASSUMPTION A-107: BEV tracks are the finding's bound tracks; position is C4's median, box the C5 fit
        tracks.append(
            BevTrack(
                track_id=tid,
                cls=obj.track.cls,
                actors=actors,
                position_xyz=[round(float(v), 3) for v in sample.xyz_ego] if sample else None,
                box=box,
                speed_mps=round(float(m.speed_mps[k]), 3) if k is not None else None,
                velocity_xy_mps=[round(float(v), 3) for v in m.velocity_ego[k, :2]]
                if k is not None
                else None,
                speed_reliable=bool(m.reliable[k]) if k is not None else False,
            )
        )
    return tracks


def _bev(scene: Scene, finding: Finding, frames: np.ndarray, objects, cfg: VerifyConfig) -> BevExport:
    sweeps = scene.lidar_timestamps()
    l2, w2 = EGO_LENGTH_M / 2, EGO_WIDTH_M / 2
    ego_box = [[l2, w2], [l2, -w2], [-l2, -w2], [-l2, w2]]
    static = []
    for ts in frames:
        ego = scene.ego_pose(ts)
        lane = corridor(scene, ts, cfg.ego_path)
        static.append(
            (
                int(np.argmin(np.abs(sweeps - ts))),
                float(ts),
                world_to_ego(ego, scene.lidar(ts).points[:, :3]),
                lane,
                _bev_tracks(scene, float(ts), ego, objects),
            )
        )
    overhead = 4000 * len(frames) + 2000
    cap = max(100, (BEV_BUDGET_BYTES - overhead) // max(1, len(frames) * BEV_BYTES_PER_POINT))
    while True:
        bev = BevExport(
            schema_version=SCHEMA_VERSION,
            finding_id=finding.finding_id,
            scene_id=finding.scene_id,
            frame="ego",
            decimation=BevDecimation(
                max_range_m=BEV_MAX_RANGE_M,
                voxel_m=BEV_VOXEL_M,
                max_points_per_frame=int(cap),
                budget_bytes=BEV_BUDGET_BYTES,
            ),
            frames=[
                BevFrame(
                    frame_idx=i,
                    ts=ts,
                    points_xyz=_decimate(pts, cap).tolist(),
                    ego_box=ego_box,
                    corridor=_corridor_polygon(lane),
                    corridor_centerline=np.round(lane.centerline, 3).tolist(),
                    corridor_extended=lane.extended,
                    tracks=tracks,
                )
                for i, ts, pts, lane, tracks in static
            ],
        )
        if len(bev.model_dump_json()) <= BEV_BUDGET_BYTES or cap <= 100:
            return bev
        cap = int(cap * 0.8)


def _timeline(finding: Finding, frames: np.ndarray, sweeps: np.ndarray) -> Timeline:
    idx = [int(np.argmin(np.abs(sweeps - t))) for t in frames]
    predicates = []
    for r in finding.results:
        per_frame = {i: (status, reason) for i, status, reason in r.evidence.get("timeline", [])}
        entries = []
        for i, ts in zip(idx, frames, strict=True):
            status, reason = per_frame.get(i, ("NOT_JUDGED", None))
            entries.append(TimelineEntry(frame_idx=i, ts=float(ts), status=status, reason=reason))
        predicates.append(
            TimelinePredicate(
                result_id=r.result_id,
                name=r.name,
                actors=r.actors,
                required=r.required,
                status=r.status,
                reason=r.evidence.get("reason", ""),
                entries=entries,
            )
        )
    return Timeline(
        schema_version=SCHEMA_VERSION,
        finding_id=finding.finding_id,
        verified=finding.verified,
        t0=finding.t0,
        t1=finding.t1,
        # ASSUMPTION A-106: the timeline covers the whole candidate window; overlays cover [t0, t1]
        frames=[
            TimelineFrame(
                frame_idx=i, ts=float(t), in_interval=bool(finding.t0 - 1e-6 <= t <= finding.t1 + 1e-6)
            )
            for i, t in zip(idx, frames, strict=True)
        ],
        predicates=predicates,
    )


def export_finding(
    run: VerifyRun, scene: Scene, finding: Finding, out_dir: Path, pool: TrackPool, cfg: VerifyConfig
) -> FindingAssets:
    start = time.perf_counter()
    out_dir = _require_runs(out_dir)
    fdir = out_dir / finding.finding_id
    (fdir / "overlays").mkdir(parents=True, exist_ok=True)
    sweeps = scene.lidar_timestamps()
    window = run.window_of(finding.finding_id)
    window_frames = sweeps[(sweeps >= window.t0 - 1e-6) & (sweeps <= window.t1 + 1e-6)]
    frames = sweeps[(sweeps >= finding.t0 - 1e-6) & (sweeps <= finding.t1 + 1e-6)]
    objects = _bound_objects(run, scene, finding, pool)

    overlays, missing = [], []
    for camera in scene.cameras:
        cam_sync = sync_camera(scene, camera)
        for ts in frames:
            i = int(np.argmin(np.abs(sweeps - ts)))
            s = _camera_frame_for(cam_sync, i)
            if s is None:
                missing.append(
                    MissingOverlay(camera=camera, frame_idx=i, reason="no camera frame synced to sweep")
                )
                continue
            path = fdir / "overlays" / f"{i:03d}_{camera}.jpg"
            _overlay(scene, finding, camera, i, s, objects).save(path, quality=85)
            overlays.append(
                OverlayAsset(
                    camera=camera,
                    frame_idx=i,
                    camera_frame_idx=s.frame_idx,
                    ts=s.ts,
                    path=str(path.relative_to(out_dir)),
                )
            )

    bev = _bev(scene, finding, frames, objects, cfg)
    bev_path = fdir / "bev.json"
    bev_path.write_text(bev.model_dump_json())
    timeline_path = fdir / "timeline.json"
    timeline_path.write_text(_timeline(finding, window_frames, sweeps).model_dump_json())
    return FindingAssets(
        finding_id=finding.finding_id,
        window_id=window.window_id,
        verified=finding.verified,
        score=finding.score,
        t0=finding.t0,
        t1=finding.t1,
        bindings=finding.bindings,
        overlays=overlays,
        missing_overlays=missing,
        bev_path=str(bev_path.relative_to(out_dir)),
        bev_bytes=bev_path.stat().st_size,
        timeline_path=str(timeline_path.relative_to(out_dir)),
        export_seconds=round(time.perf_counter() - start, 3),
    )


def export_run(
    run: VerifyRun,
    scene: Scene,
    out_dir: Path,
    finding_ids: list[str] | None = None,
    cfg: VerifyConfig | None = None,
) -> VizManifest:
    """Export every finding of a run (or the listed ones) to out_dir, which must be inside runs/."""
    if scene.scene_id != run.scene_id:
        raise ValueError(f"run is for scene {run.scene_id}, not {scene.scene_id}")
    cfg = cfg or load_verify_config()
    out_dir = _require_runs(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = TrackPool(scene, run.replay_source(), cfg)
    chosen = [f for f in run.findings if finding_ids is None or f.finding_id in finding_ids]
    if finding_ids is not None and len(chosen) != len(set(finding_ids)):
        raise KeyError(f"unknown finding ids: {sorted(set(finding_ids) - {f.finding_id for f in chosen})}")
    manifest = VizManifest(
        schema_version=SCHEMA_VERSION,
        run_id=run.run_id,
        scene_id=run.scene_id,
        dataset=run.dataset,
        findings=[export_finding(run, scene, f, out_dir, pool, cfg) for f in chosen],
    )
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=1))
    return manifest
