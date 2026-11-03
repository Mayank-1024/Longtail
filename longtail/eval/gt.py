"""Ground truth from labels (plan C7): each query's gold spec evaluated on the labelled cuboids.

The C6 predicates run unchanged on cuboid tracks (centre, size, heading, velocity from the cuboids over time),
with configs/gt.yaml thresholds. Only exists and occluded_by, which read LiDAR clusters in the verifier, have
label-geometry versions here; with motion_source "attribute", so do moving and stopped.

Every public function reads labels and fails outside eval_ground_truth(). Outputs go to gt/ (git-ignored).
"""

import hashlib
import itertools
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import cache, partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from longtail.contracts import Finding, Sample3D, ScenarioSpec, TaxonomyClass, Track3D
from longtail.data.base import Scene
from longtail.data.guard import require_label_access
from longtail.geometry.associate import Association
from longtail.geometry.project import MIN_DEPTH_M
from longtail.geometry.sync import sync_camera
from longtail.geometry.tracks3d import BoxFit, TrackedObject, motion_from_positions
from longtail.geometry.transforms import rot_z, world_to_camera, world_to_ego
from longtail.verify.config import (
    CrossingConfig,
    EgoPathConfig,
    MotionConfig,
    OccludedByConfig,
    RelativePositionConfig,
    WithinDistanceConfig,
)
from longtail.verify.engine import UNBOUND, TrackPool
from longtail.verify.predicates import _motion
from longtail.verify.predicates.base import Context, FrameResult, Reason

GT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "gt.yaml"
GT_FORMAT = "longtail.gt/v1"
QUERIES_FORMAT = "longtail.queries/v1"

# Config


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VisibilityConfig(_Strict):
    max_range_m: float = Field(gt=0)


class GtTracksConfig(_Strict):
    savgol_window: int = Field(ge=3)
    savgol_polyorder: int = Field(ge=0)
    max_interp_gap_s: float = Field(ge=0)


class GtPredicatesConfig(_Strict):
    within_distance: WithinDistanceConfig
    motion: MotionConfig
    crossing: CrossingConfig
    relative_position: RelativePositionConfig
    occluded_by: OccludedByConfig


class MatchingConfig(_Strict):
    min_temporal_iou: float = Field(gt=0, le=1)
    max_actor_distance_m: float = Field(gt=0)
    min_box_iou: float = Field(gt=0, le=1)


class ClaimsConfig(_Strict):
    min_inside_frac: float = Field(gt=0, le=1)


class GtConfig(_Strict):
    """C6 predicates read ctx.cfg.predicates.<name> and ctx.cfg.ego_path; those sections match verify.yaml."""

    label_map: dict[str, TaxonomyClass]
    cyclist_requires_rider: bool
    visibility: VisibilityConfig
    motion_source: Literal["geometry", "attribute"]
    tracks: GtTracksConfig
    ego_path: EgoPathConfig
    predicates: GtPredicatesConfig
    matching: MatchingConfig
    claims: ClaimsConfig


@cache
def load_gt_config(path: Path = GT_CONFIG_PATH) -> GtConfig:
    # ASSUMPTION A-09: ground truth has its own thresholds in configs/gt.yaml
    with open(path) as f:
        return GtConfig.model_validate(yaml.safe_load(f))


def config_hash(cfg: GtConfig) -> str:
    return hashlib.sha256(cfg.model_dump_json().encode()).hexdigest()[:16]


# Queries file (M7 writes it; the format is ours until Mayank's lands)


class GoldQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(min_length=1)
    text: str
    split: Literal["dev", "heldout"]
    tier: Literal["single", "behaviour", "relational"] | None = None
    gold: ScenarioSpec


class QueriesFile(BaseModel):
    # ASSUMPTION A-128: queries file: format tag, then query_id, text, split, tier and the gold ScenarioSpec
    model_config = ConfigDict(extra="forbid")

    format: Literal["longtail.queries/v1"]
    queries: list[GoldQuery]


def load_queries(path: Path) -> list[GoldQuery]:
    """A queries file in YAML or JSON."""
    raw = yaml.safe_load(Path(path).read_text())
    queries = QueriesFile.model_validate(raw).queries
    ids = [q.query_id for q in queries]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate query ids {sorted(k for k, n in Counter(ids).items() if n > 1)}")
    for q in queries:
        if q.gold.clarification is not None:
            raise ValueError(f"{path}: gold spec {q.query_id} asks a clarifying question")
    return queries


def spec_hash(spec: ScenarioSpec) -> str:
    return hashlib.sha256(spec.model_dump_json().encode()).hexdigest()[:16]


# Cuboids


@dataclass(frozen=True)
class Cuboid:
    uuid: str
    label: str  # dataset label string
    cls: str  # taxonomy class
    centre: np.ndarray  # (3,) world
    lwh: tuple[float, float, float]  # length along heading, width, height
    heading: float  # radians, direction of the length axis in the world x-y plane
    motion_attr: Literal["moving", "stopped"] | None  # from the human-labelled attributes


def _rows(raw: Any) -> list[dict[str, Any]]:
    if hasattr(raw, "to_dict"):  # a pandas DataFrame from the devkit
        return raw.to_dict("records")
    return list(raw)


def _str(v: Any) -> str | None:
    return v if isinstance(v, str) and v else None


def dedup_siblings(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop the front-LiDAR copy of a cuboid labelled once per LiDAR; returns (rows kept, rows dropped)."""
    # ASSUMPTION A-16: of a sibling pair (cuboids.sensor_id 0 and 1, each naming the other in sibling_id) the
    # Pandar64 copy (sensor 0) is kept, the sensor the pipeline uses (A-25); an unpaired copy is kept
    uuids = {r["uuid"] for r in rows}
    kept = [
        r
        for r in rows
        if not (int(r.get("cuboids.sensor_id", -1)) == 1 and _str(r.get("cuboids.sibling_id")) in uuids)
    ]
    return kept, len(rows) - len(kept)


def _motion_attr(row: dict[str, Any]) -> Literal["moving", "stopped"] | None:
    # ASSUMPTION A-124: attributes.object_motion Moving = moving, Parked or Stopped = stopped (vehicles,
    # bicycles); attributes.pedestrian_behavior Walking = moving, Standing, Sitting or Lying = stopped
    om = _str(row.get("attributes.object_motion"))
    if om is not None:
        return "moving" if om == "Moving" else "stopped" if om in ("Parked", "Stopped") else None
    pb = _str(row.get("attributes.pedestrian_behavior"))
    if pb is not None:
        return "moving" if pb == "Walking" else "stopped" if pb in ("Standing", "Sitting", "Lying") else None
    return None


def to_cuboid(row: dict[str, Any], cfg: GtConfig) -> Cuboid | None:
    """A taxonomy-class cuboid from one PandaSet label row, or None for a label outside the taxonomy."""
    # ASSUMPTION A-123: PandaSet labels map onto the taxonomy through gt.yaml's label_map
    cls = cfg.label_map.get(row["label"])
    if cls is None:
        return None
    rider = _str(row.get("attributes.rider_status"))
    if cls == "cyclist" and cfg.cyclist_requires_rider and rider != "With Rider":
        return None
    # ASSUMPTION A-121: devkit convention: yaw 0 points the length (dimensions.y) along +y, so the heading is
    # yaw + 90 degrees; dimensions.x is the width
    return Cuboid(
        uuid=str(row["uuid"]),
        label=str(row["label"]),
        cls=cls,
        centre=np.array([row["position.x"], row["position.y"], row["position.z"]], dtype=float),
        lwh=(float(row["dimensions.y"]), float(row["dimensions.x"]), float(row["dimensions.z"])),
        heading=float(row["yaw"]) + np.pi / 2,
        motion_attr=_motion_attr(row),
    )


def corners(c: Cuboid) -> np.ndarray:
    """(8, 3) world corners."""
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    return c.centre + (signs * np.array(c.lwh) / 2) @ rot_z(c.heading).T


_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]


def projected_box(
    k: np.ndarray, world_from_camera: np.ndarray, size: tuple[int, int], pts: np.ndarray
) -> tuple[np.ndarray, float] | None:
    """Image box (x1, y1, x2, y2) of a cuboid's corners, clipped at the near plane and to the image, and the
    camera depth of its centre; None when no part lands in the image."""
    cam = world_to_camera(world_from_camera, pts)
    front = cam[:, 2] > MIN_DEPTH_M
    if not front.any():
        return None
    keep = [cam[front]]
    for a, b in _EDGES:  # where an edge crosses the near plane
        za, zb = cam[a, 2] - MIN_DEPTH_M, cam[b, 2] - MIN_DEPTH_M
        if za * zb < 0:
            keep.append((cam[a] + za / (za - zb) * (cam[b] - cam[a]))[None])
    p = np.concatenate(keep)
    uv = np.column_stack([k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2], k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]])
    w, h = size
    x1, y1 = max(uv[:, 0].min(), 0.0), max(uv[:, 1].min(), 0.0)
    x2, y2 = min(uv[:, 0].max(), w - 1.0), min(uv[:, 1].max(), h - 1.0)
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2]), float(cam.mean(axis=0)[2])


def box_iou(a: tuple[float, ...] | np.ndarray, b: tuple[float, ...] | np.ndarray) -> float:
    """IoU of two (x1, y1, x2, y2) boxes."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


# Labels of one scene


@dataclass(frozen=True)
class CameraBox:
    camera: str
    frame_idx: int
    ts: float  # camera frame timestamp
    box: np.ndarray
    depth: float  # camera depth of the cuboid centre


@dataclass
class LabelObject:
    uuid: str
    cls: str
    label: str
    cuboids: dict[float, Cuboid] = field(default_factory=dict)  # by sweep ts
    boxes: dict[float, list[CameraBox]] = field(default_factory=dict)  # synced camera frames, by sweep ts
    visible: dict[float, bool] = field(default_factory=dict)


class LabelScene:
    """One scene's labels read once, deduplicated, mapped to the taxonomy and projected into every camera."""

    def __init__(self, scene: Scene, cfg: GtConfig | None = None) -> None:
        require_label_access("ground truth")
        self.scene, self.cfg = scene, cfg or load_gt_config()
        self.sweeps = scene.lidar_timestamps()
        self.period_s = float(np.median(np.diff(self.sweeps))) if len(self.sweeps) > 1 else 0.1
        self.stats: Counter[str] = Counter()
        self.objects: dict[str, LabelObject] = {}
        # Camera frames per sweep (C2's sync); each camera frame also maps to its nearest sweep for claims.
        self._frames: dict[int, list[tuple[str, int, float]]] = defaultdict(list)
        self._frame_sweep: dict[tuple[str, int], int] = {}
        for cam in scene.cameras:
            for f in sync_camera(scene, cam):
                self._frame_sweep[(cam, f.frame_idx)] = f.lidar_idx
                if f.synced:
                    self._frames[f.lidar_idx].append((cam, f.frame_idx, f.ts))
        self._by_sweep: dict[int, list[Cuboid]] = {}
        for i, ts in enumerate(self.sweeps):
            rows, dropped = dedup_siblings(_rows(scene.labels(i)))
            self.stats["cuboids"] += len(rows) + dropped
            self.stats["sibling_copies_dropped"] += dropped
            cuboids = []
            for r in rows:
                c = to_cuboid(r, self.cfg)
                if c is None:
                    self.stats[f"ignored:{r['label']}"] += 1
                    continue
                cuboids.append(c)
            self._by_sweep[i] = cuboids
            ego = scene.ego_pose(ts)
            for c in cuboids:
                obj = self.objects.setdefault(c.uuid, LabelObject(c.uuid, c.cls, c.label))
                obj.cuboids[float(ts)] = c
                boxes = []
                for cam, fidx, fts in self._frames[i]:
                    pb = self._project(c, cam, fidx, fts)
                    if pb is not None:
                        boxes.append(CameraBox(cam, fidx, fts, *pb))
                obj.boxes[float(ts)] = boxes
                rng = float(np.hypot(*world_to_ego(ego, c.centre[None])[0, :2]))
                # ASSUMPTION A-122: visible = projects into a synced camera frame and within max_range_m
                obj.visible[float(ts)] = bool(boxes) and rng <= self.cfg.visibility.max_range_m
        self._tracked: dict[str, TrackedObject] = {}
        self._boxes_at: dict[tuple[str, int], list[tuple[Cuboid, np.ndarray]]] = {}

    def _project(self, c: Cuboid, camera: str, frame_idx: int, ts: float) -> tuple[np.ndarray, float] | None:
        return projected_box(
            self.scene.intrinsics(camera),
            self.scene.extrinsics(camera, ts),
            self.scene.image_size(camera),
            corners(c),
        )

    def boxes_at(self, camera: str, frame_idx: int) -> list[tuple[Cuboid, np.ndarray]]:
        """Every taxonomy cuboid of the nearest sweep projected into one camera frame (A-10 claim checks)."""
        # Memoised: scoring checks many claims (every query's, every window's) against the same frames
        key = (camera, frame_idx)
        if key not in self._boxes_at:
            frame = self.scene.frames(camera)[frame_idx]
            out = []
            for c in self._by_sweep[self._frame_sweep[key]]:
                pb = self._project(c, camera, frame_idx, frame.ts)
                if pb is not None:
                    out.append((c, pb[0]))
            self._boxes_at[key] = out
        return self._boxes_at[key]

    def of_class(self, cls: str) -> list[LabelObject]:
        return [o for o in self.objects.values() if o.cls == cls and any(o.visible.values())]

    def tracked(self, uuid: str) -> TrackedObject:
        """The label track as C5 output, so the C6 predicates run on it unchanged."""
        if uuid not in self._tracked:
            self._tracked[uuid] = self._build_tracked(self.objects[uuid])
        return self._tracked[uuid]

    def _build_tracked(self, obj: LabelObject) -> TrackedObject:
        # ASSUMPTION A-121: a label track: cuboid centre as position, cuboid size as extent; velocity from the
        # object's cuboids in every labelled sweep, smoothed like C5; samples only where it is visible (A-122)
        ts = np.array(sorted(obj.cuboids))
        centres = np.array([obj.cuboids[t].centre for t in ts])
        samples = []
        for t in ts:
            if not obj.visible[t]:
                continue
            c = obj.cuboids[t]
            ego_xyz = world_to_ego(self.scene.ego_pose(t), c.centre[None])[0]
            samples.append(
                Sample3D(
                    ts=float(t),
                    xyz_ego=tuple(float(v) for v in ego_xyz),
                    xyz_world=tuple(float(v) for v in c.centre),
                    n_points=0,
                    extent_lwh=c.lwh,
                    coverage=0,
                )
            )
        motion = motion_from_positions(self.scene, ts, centres, self.cfg.tracks)
        speeds = [float(motion.speed_mps[motion.index(s.ts)]) for s in samples]
        track = Track3D(track_id=obj.uuid, cls=obj.cls, sources=[], samples=samples, speed_mps=speeds)
        size = np.max([obj.cuboids[t].lwh[:2] for t in ts], axis=0)
        box = BoxFit(ts, centres, np.array([obj.cuboids[t].heading for t in ts]), size)
        return TrackedObject(Association(track, [], False, {}), track, motion, box)


# Label-geometry predicates


def _exists(
    labels: LabelScene, ctx: Context, objs: list[TrackedObject], args: dict[str, Any]
) -> list[FrameResult]:
    """A labelled cuboid that is visible (A-122): no point count or size prior, the label is the truth."""
    # ASSUMPTION A-125: exists on labels = the cuboid is visible; value is its range, margin to max_range_m
    [obj] = objs
    limit = labels.cfg.visibility.max_range_m
    by_ts = {s.ts: s for s in obj.track.samples}
    out = []
    for ts in ctx.frames:
        s = by_ts.get(float(ts))
        if s is None:
            out.append(FrameResult.unverifiable(Reason.NOT_OBSERVED))
            continue
        rng = float(np.hypot(s.xyz_ego[0], s.xyz_ego[1]))
        out.append(FrameResult("PASS", Reason.OK, rng, limit - rng, {"range_m": round(rng, 2)}))
    return out


def _occluded_by(
    labels: LabelScene, ctx: Context, objs: list[TrackedObject], args: dict[str, Any]
) -> list[FrameResult]:
    """A nearer cuboid of class cls covers enough of the target's projected cuboid in some camera."""
    # ASSUMPTION A-125: occluded_by on labels: projected cuboid boxes (amodal by construction); an occluder is
    # nearer when its centre's camera depth is smaller; coverage as A-84 (occluder box over target box)
    [target] = objs
    need = ctx.cfg.predicates.occluded_by.min_cover_frac
    tobj = labels.objects[target.track.track_id]
    others = [o for o in labels.of_class(str(args["cls"])) if o.uuid != tobj.uuid]
    out = []
    for ts in ctx.frames:
        ts = float(ts)
        if not tobj.visible.get(ts, False):
            out.append(FrameResult.unverifiable(Reason.NOT_OBSERVED))
            continue
        best, detail = 0.0, {}
        for tb in tobj.boxes[ts]:
            area = (tb.box[2] - tb.box[0]) * (tb.box[3] - tb.box[1])
            for o in others:
                for ob in o.boxes.get(ts, []):
                    if ob.camera != tb.camera or ob.depth >= tb.depth:
                        continue
                    iw = max(0.0, min(tb.box[2], ob.box[2]) - max(tb.box[0], ob.box[0]))
                    ih = max(0.0, min(tb.box[3], ob.box[3]) - max(tb.box[1], ob.box[1]))
                    cover = float(iw * ih / area)
                    if cover > best:
                        best, detail = cover, {"camera": tb.camera, "occluder": o.uuid}
        ok = best >= need
        out.append(
            FrameResult(
                "PASS" if ok else "FAIL",
                Reason.OK if ok else Reason.NOT_OCCLUDED,
                best,
                best - need,
                {"covered_frac": round(best, 3), **detail},
            )
        )
    return out


def _attribute_motion(
    labels: LabelScene, want: str, ctx: Context, objs: list[TrackedObject], args: dict[str, Any]
) -> list[FrameResult]:
    """moving or stopped from each frame's human-labelled attribute (motion_source: attribute)."""
    # ASSUMPTION A-124: with motion_source "attribute", each frame's verdict is that cuboid's attribute;
    # frames without the attribute are not judged
    [obj] = objs
    lobj = labels.objects[obj.track.track_id]
    out = []
    for ts in ctx.frames:
        c = lobj.cuboids.get(float(ts))
        if c is None or c.motion_attr is None:
            out.append(FrameResult.unverifiable(Reason.NOT_OBSERVED))
            continue
        ok = c.motion_attr == want
        reason = Reason.OK if ok else (Reason.NOT_MOVING if want == "moving" else Reason.NOT_STOPPED)
        out.append(FrameResult("PASS" if ok else "FAIL", reason, 1.0 if ok else 0.0, 1.0 if ok else -1.0))
    return out


def gt_context(labels: LabelScene) -> Context:
    """A C6 Context whose predicates read label geometry: the label tracks as pool, gt.yaml as thresholds."""
    overrides = {
        "exists": partial(_exists, labels),
        "occluded_by": partial(_occluded_by, labels),
    }
    if labels.cfg.motion_source == "attribute":
        overrides["moving"] = partial(_attribute_motion, labels, "moving")
        overrides["stopped"] = partial(_attribute_motion, labels, "stopped")
    # ASSUMPTION A-120: C7 swaps label-geometry predicates in by name; the rest are the C6 code
    return Context(
        labels.scene,
        labels.cfg,  # the sections the C6 predicates read have verify.yaml's shapes
        labels.sweeps,
        pool=lambda cls: [labels.tracked(o.uuid) for o in labels.of_class(cls)],
        overrides=overrides,
    )


# Positives


class LabelTrack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label_id: str
    label: str
    cls: str
    positions: list[tuple[float, float, float, float]]  # (sweep ts, x, y, z) world, within the positive
    boxes: list[tuple[str, int, float, float, float, float, float]]  # (camera, frame_idx, ts, x1, y1, x2, y2)


class Positive(BaseModel):
    model_config = ConfigDict(extra="forbid")

    positive_id: str
    scene_id: str
    t0: float  # first sweep in which the spec holds
    t1: float  # last sweep
    period_s: float  # sweep period; the event lasts t1 - t0 + period_s
    scene_start_ts: float
    label_ids: list[str]
    actors: dict[str, list[str]]  # spec actor id to the label ids bound to it during the event
    labels: list[LabelTrack]


class QueryGT(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["longtail.gt/v1"] = GT_FORMAT
    query_id: str
    query: str
    split: str
    spec_hash: str
    gt_config_hash: str
    motion_source: str
    scenes: list[str]
    positives: list[Positive]
    stats: dict[str, int]


def frame_results(spec: ScenarioSpec, labels: LabelScene) -> tuple[list[dict[str, str]], np.ndarray, dict]:
    """Every binding of label objects to the spec's actors, and per binding and sweep whether every required
    predicate PASSes there (with every bound actor visible). Returns (bindings, (B, N) flags, stats)."""
    ctx = gt_context(labels)
    n = len(labels.sweeps)
    memo: dict[tuple, list[FrameResult]] = {}

    def run(i: int, uuids: tuple[str, ...]) -> np.ndarray:
        call = spec.predicates[i]
        key = (call.name, uuids, tuple(sorted(call.args.items())))
        if key not in memo:
            memo[key] = ctx.evaluate(call.name, [labels.tracked(u) for u in uuids], call.args)
        return np.array([r.status == "PASS" for r in memo[key]])

    def visible(u: str) -> np.ndarray:
        return np.array([labels.objects[u].visible.get(float(t), False) for t in labels.sweeps])

    required = [i for i, p in enumerate(spec.predicates) if p.required]
    # Candidates per actor: visible objects of the class whose required one-actor predicates hold together
    # somewhere (an exact prune: the joint needs them in the same frames).
    candidates: dict[str, list[str]] = {}
    for a in spec.actors:
        unary = [i for i in required if spec.predicates[i].actors == [a.id]]
        keep = []
        for o in labels.of_class(a.cls):
            ok = visible(o.uuid)
            for i in unary:
                ok = ok & run(i, (o.uuid,))
            if ok.any():
                keep.append(o.uuid)
        candidates[a.id] = keep
    ids = [a.id for a in spec.actors]
    bindings, rows = [], []
    for combo in itertools.product(*[candidates[i] for i in ids]):
        if len(set(combo)) < len(combo):
            continue  # one object cannot be two actors
        binding = dict(zip(ids, combo, strict=True))
        flags = np.ones(n, bool) if required else np.zeros(n, bool)
        for u in combo:
            flags &= visible(u)
        for i in required:
            flags &= run(i, tuple(binding[x] for x in spec.predicates[i].actors))
        bindings.append(binding)
        rows.append(flags)
    stats = {"bindings": len(bindings), "candidates": sum(len(v) for v in candidates.values())}
    return bindings, (np.array(rows) if rows else np.zeros((0, n), bool)), stats


def runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True as inclusive (first, last) index pairs."""
    out, start = [], None
    for i, f in enumerate([*flags.tolist(), False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - 1))
            start = None
    return out


def events(
    sweeps: np.ndarray, flags: np.ndarray, period_s: float, min_duration_s: float
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Events from per-binding flags: each binding's runs lasting min_duration_s, merged over bindings into
    maximal intervals. Returns the intervals and which bindings hold in each sweep."""
    # ASSUMPTION A-126: a positive is a maximal interval of sweeps in which some binding satisfies the spec
    # within a run lasting (last - first) + one period >= min_duration_s (A-91); overlapping or adjacent runs
    # of any bindings merge, so an event with two pedestrians is one positive
    held = np.zeros_like(flags)
    for b, row in enumerate(flags):
        for a, z in runs(row):
            if sweeps[z] - sweeps[a] + period_s >= min_duration_s - 1e-9:
                held[b, a : z + 1] = True
    return runs(held.any(axis=0)) if len(held) else [], held


def scene_positives(query_id: str, spec: ScenarioSpec, labels: LabelScene) -> tuple[list[Positive], dict]:
    """The positives (events) of one gold spec in one scene."""
    require_label_access("ground truth")
    bindings, flags, stats = frame_results(spec, labels)
    # ASSUMPTION A-96: crossing_ego_path's 3 s look-ahead comes with the C6 code: a frame whose future ends
    # before the horizon without entering the corridor is UNVERIFIABLE, never a FAIL or a positive
    intervals, held = events(labels.sweeps, flags, labels.period_s, spec.min_duration_s)
    out = []
    for a, z in intervals:
        active = [b for b in range(len(bindings)) if held[b, a : z + 1].any()]
        actors: dict[str, set[str]] = defaultdict(set)
        for b in active:
            for actor, u in bindings[b].items():
                actors[actor].add(u)
        uuids = sorted({u for v in actors.values() for u in v})
        t0, t1 = float(labels.sweeps[a]), float(labels.sweeps[z])
        tracks = []
        for u in uuids:
            obj = labels.objects[u]
            inside = [t for t in sorted(obj.cuboids) if t0 - 1e-6 <= t <= t1 + 1e-6]
            tracks.append(
                LabelTrack(
                    label_id=u,
                    label=obj.label,
                    cls=obj.cls,
                    positions=[(t, *(float(v) for v in obj.cuboids[t].centre)) for t in inside],
                    boxes=[
                        (b.camera, b.frame_idx, b.ts, *(round(float(v), 1) for v in b.box))
                        for t in inside
                        for b in obj.boxes[t]
                    ],
                )
            )
        pid = hashlib.sha256(f"{query_id}|{labels.scene.scene_id}|{t0:.3f}".encode()).hexdigest()[:12]
        out.append(
            Positive(
                positive_id=pid,
                scene_id=labels.scene.scene_id,
                t0=t0,
                t1=t1,
                period_s=labels.period_s,
                scene_start_ts=float(labels.sweeps[0]),
                label_ids=uuids,
                actors={k: sorted(v) for k, v in actors.items()},
                labels=tracks,
            )
        )
    return out, stats


def check_out_dir(out_dir: Path) -> Path:
    # ASSUMPTION A-129: ground truth is written only to a directory named gt (git-ignored); others are refused
    out_dir = Path(out_dir)
    if out_dir.name != "gt":
        raise ValueError(f"ground truth goes to a directory named gt (git-ignored), not {out_dir}")
    return out_dir


def build(
    queries: list[GoldQuery], scenes: list[Scene], out_dir: Path, cfg: GtConfig | None = None
) -> dict[str, Any]:
    """Ground truth for every query over every scene: gt/<query_id>.json and gt/summary.json."""
    require_label_access("ground truth")
    out_dir = check_out_dir(out_dir)
    cfg = cfg or load_gt_config()
    positives: dict[str, list[Positive]] = {q.query_id: [] for q in queries}
    stats: dict[str, Counter] = {q.query_id: Counter() for q in queries}
    label_stats: Counter[str] = Counter()
    for scene in scenes:
        labels = LabelScene(scene, cfg)
        label_stats.update(labels.stats)
        for q in queries:
            found, st = scene_positives(q.query_id, q.gold, labels)
            positives[q.query_id] += found
            stats[q.query_id].update(st)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_ids = [s.scene_id for s in scenes]
    rows = []
    for q in queries:
        doc = QueryGT(
            query_id=q.query_id,
            query=q.text,
            split=q.split,
            spec_hash=spec_hash(q.gold),
            gt_config_hash=config_hash(cfg),
            motion_source=cfg.motion_source,
            scenes=scene_ids,
            positives=positives[q.query_id],
            stats=dict(stats[q.query_id]),
        )
        (out_dir / f"{q.query_id}.json").write_text(doc.model_dump_json(indent=1))
        n = len(doc.positives)
        rows.append(
            {
                "query_id": q.query_id,
                "split": q.split,
                "positives": n,
                "scenes_with_positives": len({p.scene_id for p in doc.positives}),
                # Plan C7: zero-positive queries go back to Mayank before the freeze.
                "zero_positives": n == 0,
            }
        )
    summary = {
        "gt_config_hash": config_hash(cfg),
        "motion_source": cfg.motion_source,
        "scenes": scene_ids,
        "labels": dict(label_stats),
        "queries": rows,
        "zero_positive_queries": [r["query_id"] for r in rows if r["zero_positives"]],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def load_gt(gt_dir: Path, query_id: str) -> QueryGT:
    return QueryGT.model_validate_json((Path(gt_dir) / f"{query_id}.json").read_text())


# Matching (plan C7)


class ActorTrace(BaseModel):
    """Where a finding's bound actor was: LiDAR positions (C4/C5) and/or image boxes (SAM or VLM)."""

    model_config = ConfigDict(extra="forbid")

    positions: list[tuple[float, float, float, float]] = []  # (sweep ts, x, y, z) world
    boxes: list[tuple[str, int, float, float, float, float, float]] = []  # (camera, frame_idx, ts, box)


class ScoredFinding(BaseModel):
    """A finding as scoring sees it, from any configuration (a) to (d)."""

    model_config = ConfigDict(extra="forbid")

    finding_id: str
    scene_id: str
    t0: float
    t1: float
    score: float
    actors: dict[str, ActorTrace | None]  # None: the actor is unbound
    key: str = ""  # identity of the bound objects, so duplicates from overlapping windows can merge


def scored_finding(finding: Finding, pool: TrackPool) -> ScoredFinding:
    """A verifier Finding with its bound tracks' LiDAR positions and SAM boxes, from the engine's pool."""
    actors: dict[str, ActorTrace | None] = {}
    for actor, track_id in finding.bindings.items():
        if track_id == UNBOUND:
            actors[actor] = None
            continue
        o = pool.find(track_id)
        actors[actor] = ActorTrace(
            positions=[(s.ts, *s.xyz_world) for s in o.track.samples],
            boxes=[
                (t.camera, f.frame_idx, f.ts, *f.box_xyxy) for t in o.association.inputs for f in t.frames
            ],
        )
    return ScoredFinding(
        finding_id=finding.finding_id,
        scene_id=finding.scene_id,
        t0=finding.t0,
        t1=finding.t1,
        score=finding.score,
        actors=actors,
        key="|".join(f"{k}={v}" for k, v in sorted(finding.bindings.items())),
    )


@dataclass(frozen=True)
class Match:
    finding_id: str
    positive_id: str
    iou: float


def temporal_iou(a0: float, a1: float, b0: float, b1: float, period_s: float) -> float:
    """IoU of two runs of sweeps, each lasting (last - first) + one period (A-91)."""
    # ASSUMPTION A-127: temporal IoU over [t0, t1 + period], so a one-sweep interval has a length
    a1, b1 = a1 + period_s, b1 + period_s
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def _actor_matches(trace: ActorTrace, p: Positive, lo: float, hi: float, m: MatchingConfig) -> bool:
    # ASSUMPTION A-127: an actor matches when, over the sweeps both cover, the median horizontal distance from
    # its positions to one of the positive's labels is <= max_actor_distance_m; with boxes only, when one of
    # its boxes has IoU >= min_box_iou with a label's projected box in the same camera frame
    pos = [q for q in trace.positions if lo - 1e-6 <= q[0] <= hi + 1e-6]
    if pos:
        for lt in p.labels:
            at = {round(q[0], 3): q for q in lt.positions}
            d = [np.hypot(q[1] - at[k][1], q[2] - at[k][2]) for q in pos if (k := round(q[0], 3)) in at]
            if d and float(np.median(d)) <= m.max_actor_distance_m:
                return True
        return False
    boxes = [b for b in trace.boxes if lo - 1e-6 <= b[2] <= hi + 1e-6]
    for lt in p.labels:
        by_frame = defaultdict(list)
        for lb in lt.boxes:
            by_frame[(lb[0], lb[1])].append(lb[3:])
        for b in boxes:
            if any(box_iou(b[3:], lb) >= m.min_box_iou for lb in by_frame.get((b[0], b[1]), [])):
                return True
    return False


def eligible(f: ScoredFinding, p: Positive, m: MatchingConfig) -> float | None:
    """The temporal IoU if the finding may match the positive, else None."""
    if f.scene_id != p.scene_id:
        return None
    iou = temporal_iou(f.t0, f.t1, p.t0, p.t1, p.period_s)
    if iou < m.min_temporal_iou:
        return None
    lo, hi = max(f.t0, p.t0), min(f.t1, p.t1) + p.period_s - 1e-6
    for trace in f.actors.values():
        if trace is None or not _actor_matches(trace, p, lo, hi, m):
            return None
    return iou


def match(findings: list[ScoredFinding], positives: list[Positive], m: MatchingConfig) -> list[Match]:
    """One-to-one matching, greedy by temporal IoU (ties: the higher-scoring finding)."""
    # ASSUMPTION A-127: one-to-one, greedy by temporal IoU
    pairs = []
    for f in findings:
        for p in positives:
            iou = eligible(f, p, m)
            if iou is not None:
                pairs.append((-iou, -f.score, f.finding_id, p.positive_id))
    used_f, used_p, out = set(), set(), []
    for neg_iou, _, fid, pid in sorted(pairs):
        if fid in used_f or pid in used_p:
            continue
        used_f.add(fid)
        used_p.add(pid)
        out.append(Match(fid, pid, -neg_iou))
    return out


# Model claims (A-10), for C8's hallucination-catch rate


def inside_frac(claim: tuple[float, ...] | np.ndarray, label: tuple[float, ...] | np.ndarray) -> float:
    """Share of the claimed box's area that lies inside the label box (intersection over the claimed box)."""
    iw = max(0.0, min(claim[2], label[2]) - max(claim[0], label[0]))
    ih = max(0.0, min(claim[3], label[3]) - max(claim[1], label[1]))
    area = (claim[2] - claim[0]) * (claim[3] - claim[1])
    return float(iw * ih / area) if area > 0 else 0.0


def claim_is_true(labels: LabelScene, camera: str, frame_idx: int, cls: str, box: tuple[float, ...]) -> bool:
    """A box claim is true when min_inside_frac of it lies inside a projected label box of the class."""
    require_label_access("ground truth")
    # ASSUMPTION A-10: true = >= 50% of the claimed box inside a projected cuboid box of the claimed class, so
    # a box on the visible part of an occluded object counts; IoU would call it a hallucination
    need = labels.cfg.claims.min_inside_frac
    return any(c.cls == cls and inside_frac(box, b) >= need for c, b in labels.boxes_at(camera, frame_idx))


# Motion source agreement (geometry against the human-labelled attributes)


def motion_agreement(labels: LabelScene) -> dict[str, Any]:
    """Per class, how often the geometric moving/stopped state agrees with the labelled attribute, over every
    visible sweep where both exist."""
    require_label_access("ground truth")
    # ASSUMPTION A-124: the report compares C6's hysteresis state on the cuboid track with the attribute
    cfg = labels.cfg.predicates.motion
    out: dict[str, Counter] = defaultdict(Counter)
    for obj in labels.objects.values():
        tracked = labels.tracked(obj.uuid)
        series = _motion.of_object(tracked)
        steps = _motion.steps(series, cfg)
        by_ts = {round(float(t), 6): s for t, s in zip(series.ts, steps, strict=True)}
        for ts, c in obj.cuboids.items():
            if not obj.visible[ts]:
                continue
            geo = by_ts.get(round(ts, 6))
            g = geo.state if geo is not None else None
            a = c.motion_attr
            n = out[obj.cls]
            if a is None:
                n["no_attribute"] += 1
            elif g is None:
                n["geometry_not_judged"] += 1
            else:
                n["agree" if g == a else "disagree"] += 1
                n[f"geometry_{g}_attribute_{a}"] += 1
    return {cls: dict(v) for cls, v in sorted(out.items())}
