"""JSON outputs of the C9 export, as pydantic models. docs/viz-schema.json is generated from these.

Coordinates in the bird's-eye view are in the ego frame at that frame's sweep (A-03): metres, x forward,
y left, z up. Times are dataset timestamps in seconds. frame_idx is the LiDAR sweep index in the scene.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import models_json_schema

SCHEMA_VERSION = "1.0"

# Fixed-length number arrays rather than tuples: tuples become "prefixItems", which not every
# JSON-Schema-to-TypeScript generator understands.
Vec2 = Annotated[list[float], Field(min_length=2, max_length=2)]
Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]
Verdict = Literal["PASS", "FAIL", "UNVERIFIABLE"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Bird's-eye view


class BevBox(_Model):
    """C5 box fit (A-68) in the ego frame."""

    center_xy: Vec2 = Field(description="Box centre, ego frame, metres")
    yaw_rad: float = Field(
        description="Direction of the box's length axis, ego frame, radians, counter-clockwise from +x"
    )
    length_m: float = Field(
        description="Extent along yaw_rad (the track-wide size; not always the longer side)"
    )
    width_m: float = Field(description="Extent perpendicular to yaw_rad")


class BevTrack(_Model):
    track_id: str
    cls: str = Field(description="Taxonomy class")
    actors: list[str] = Field(description="Spec actor ids bound to this track in the finding")
    position_xyz: Vec3 | None = Field(
        description="C4 median of the cluster, ego frame, metres; null if unobserved"
    )
    box: BevBox | None = Field(description="Box fit, or null where it could not be placed")
    speed_mps: float | None = Field(
        description="Horizontal speed over ground (C5), m/s; null if no motion step"
    )
    velocity_xy_mps: Vec2 | None = Field(description="World velocity in ego axes (A-70), m/s")
    speed_reliable: bool = Field(description="False when the speed series was too short to smooth (A-71)")


class BevFrame(_Model):
    frame_idx: int = Field(description="LiDAR sweep index in the scene")
    ts: float = Field(description="Sweep timestamp, seconds")
    points_xyz: list[Vec3] = Field(
        description="Decimated LiDAR points, ego frame, metres (centimetre precision)"
    )
    ego_box: list[Vec2] = Field(description="Ego vehicle footprint polygon, ego frame (A-100)")
    corridor: list[Vec2] = Field(description="Ego corridor polygon (C5), ego frame; closed implicitly")
    corridor_centerline: list[Vec2] = Field(
        description="The ego's future path, ego frame, starting at the origin"
    )
    corridor_extended: bool = Field(
        description="True when the path was extended to the minimum length (A-69)"
    )
    tracks: list[BevTrack]


class BevDecimation(_Model):
    max_range_m: float = Field(description="Points beyond this horizontal range are dropped")
    voxel_m: float = Field(description="One point kept per voxel of this size")
    max_points_per_frame: int = Field(description="Cap after the voxel grid, from the size budget")
    budget_bytes: int = Field(description="Size budget for this file")


class BevExport(_Model):
    schema_version: Literal["1.0"]
    finding_id: str
    scene_id: str
    frame: Literal["ego"] = Field(description="Every coordinate is in the ego frame of its own frame_idx")
    decimation: BevDecimation
    frames: list[BevFrame]


# Predicate timeline


class TimelineEntry(_Model):
    frame_idx: int
    ts: float
    status: Literal["PASS", "FAIL", "UNVERIFIABLE", "NOT_JUDGED"] = Field(
        description="The predicate's verdict at this frame; NOT_JUDGED if the engine did not evaluate it"
    )
    reason: str | None = Field(description="Reason code (A-79); null when NOT_JUDGED")


class TimelinePredicate(_Model):
    result_id: str = Field(description="PredicateResult.result_id in the finding")
    name: str
    actors: dict[str, str] = Field(description="Actor id to track id")
    required: bool
    status: Verdict = Field(description="The predicate's summary status in the finding")
    reason: str = Field(description="The summary's reason code")
    entries: list[TimelineEntry] = Field(description="One per timeline frame, in frame order")


class TimelineFrame(_Model):
    frame_idx: int
    ts: float
    in_interval: bool = Field(description="Inside the finding's [t0, t1], the frames that have overlays")


class Timeline(_Model):
    schema_version: Literal["1.0"]
    finding_id: str
    verified: bool
    t0: float
    t1: float
    frames: list[TimelineFrame] = Field(description="Every frame of the candidate window, in order")
    predicates: list[TimelinePredicate]


# Manifest


class OverlayAsset(_Model):
    camera: str
    frame_idx: int = Field(description="LiDAR sweep index")
    camera_frame_idx: int = Field(description="The camera frame drawn, matched to the sweep by C2 sync")
    ts: float = Field(description="Camera frame timestamp, seconds")
    path: str = Field(description="JPEG path relative to the manifest's directory")


class MissingOverlay(_Model):
    camera: str
    frame_idx: int
    reason: str = Field(description="Why there is no overlay, e.g. no camera frame synced to this sweep")


class FindingAssets(_Model):
    finding_id: str
    window_id: str
    verified: bool
    score: float
    t0: float
    t1: float
    bindings: dict[str, str]
    overlays: list[OverlayAsset]
    missing_overlays: list[MissingOverlay]
    bev_path: str
    bev_bytes: int
    timeline_path: str
    export_seconds: float


class VizManifest(_Model):
    schema_version: Literal["1.0"]
    run_id: str
    scene_id: str
    dataset: str
    findings: list[FindingAssets]


def json_schema() -> dict[str, Any]:
    """One JSON Schema with VizManifest, BevExport and Timeline under $defs (for TypeScript generation)."""
    _, schema = models_json_schema(
        [(VizManifest, "validation"), (BevExport, "validation"), (Timeline, "validation")],
        title="LongTail C9 visualization export",
    )
    schema["description"] = "Generated from longtail/viz/schema.py; see docs/viz-schema.md."
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema
