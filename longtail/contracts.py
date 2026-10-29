"""Shared types between the two halves of LongTail (plan section 4).

This module is the only coupling between perception/verification (Chethan) and the agent and
product side (Mayank). Change it only through a pull request the other person approves.

DRAFT: written before the joint contracts session. Every choice the plan does not fix is tagged
with an assumption ID from docs/assumptions.md.

Units and frames:
- Distances in metres, times in seconds, speeds in metres per second.
- Ego frame: x forward, y left, z up. World frame: the dataset's world frame.
- frame_idx is the 0-based index of a camera frame within its scene. ts is that frame's dataset
  timestamp in seconds.
- Image coordinates are pixels in the source image, x right and y down.
"""

# ASSUMPTION A-02: types follow plan section 4 except where another assumption changes them
# ASSUMPTION A-03: metres and seconds; ego x forward, y left, z up; 0-based frame_idx; ts in seconds

from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class _Contract(BaseModel):
    # ASSUMPTION A-21: contract models reject unknown fields
    model_config = ConfigDict(extra="forbid")


# ASSUMPTION A-19: the LongTail taxonomy is this fixed tuple; dataset labels map onto it elsewhere
TaxonomyClass = Literal["pedestrian", "cyclist", "car", "truck", "bus"]
TAXONOMY: tuple[str, ...] = get_args(TaxonomyClass)

PredicateName = Literal[
    "exists",
    "within_distance",
    "moving",
    "stopped",
    "crossing_ego_path",
    "relative_position",
    "near",
    "occluded_by",
    "ego_moving",  # ASSUMPTION A-06: ego-state predicates with no actors
    "ego_stopped",  # ASSUMPTION A-06: ego-state predicates with no actors
]
PREDICATE_NAMES: tuple[str, ...] = get_args(PredicateName)

Status = Literal["PASS", "FAIL", "UNVERIFIABLE"]

SIDES: tuple[str, ...] = ("left", "right", "ahead", "behind")

# ASSUMPTION A-23: predicate signatures (actor count, argument names) follow the plan C6 table
PREDICATE_SIGNATURES: dict[str, tuple[int, frozenset[str]]] = {
    "exists": (1, frozenset()),
    "within_distance": (1, frozenset({"m"})),
    "moving": (1, frozenset()),
    "stopped": (1, frozenset()),
    "crossing_ego_path": (1, frozenset()),
    "relative_position": (1, frozenset({"side"})),
    "near": (2, frozenset({"m"})),
    "occluded_by": (1, frozenset({"cls"})),
    "ego_moving": (0, frozenset()),  # ASSUMPTION A-06: ego-state predicates with no actors
    "ego_stopped": (0, frozenset()),  # ASSUMPTION A-06: ego-state predicates with no actors
}


class ActorSpec(_Contract):
    id: str = Field(min_length=1)  # "a1"
    cls: TaxonomyClass
    sam_prompt: str = Field(min_length=1)  # from configs/taxonomy.yaml, e.g. "person"


class PredicateCall(_Contract):
    name: PredicateName
    actors: list[str]
    args: dict[str, float | str]
    required: bool = True

    @model_validator(mode="after")
    def _check_signature(self) -> "PredicateCall":
        # ASSUMPTION A-23: arity and argument names are checked; unknown arguments are rejected
        n_actors, arg_names = PREDICATE_SIGNATURES[self.name]
        if len(self.actors) != n_actors:
            raise ValueError(f"{self.name} takes {n_actors} actor(s), got {len(self.actors)}")
        if len(set(self.actors)) != len(self.actors):
            raise ValueError(f"{self.name} repeats an actor: {self.actors}")
        if set(self.args) != arg_names:
            raise ValueError(f"{self.name} takes args {sorted(arg_names)}, got {sorted(self.args)}")
        if "m" in self.args:
            m = self.args["m"]
            if not isinstance(m, float) or m <= 0:
                raise ValueError(f"{self.name}: m must be a positive number of metres, got {m!r}")
        if "side" in self.args and self.args["side"] not in SIDES:
            raise ValueError(f"{self.name}: side must be one of {SIDES}, got {self.args['side']!r}")
        if "cls" in self.args and self.args["cls"] not in TAXONOMY:
            raise ValueError(f"{self.name}: cls must be one of {TAXONOMY}, got {self.args['cls']!r}")
        return self


class ScenarioSpec(_Contract):
    query: str
    actors: list[ActorSpec]
    predicates: list[PredicateCall]
    min_duration_s: float = Field(default=0.5, ge=0)
    scene_filters: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str]  # "near = within 15 m", shown on the card
    clarification: str | None  # set = ask the user before running

    @model_validator(mode="after")
    def _check_actor_refs(self) -> "ScenarioSpec":
        ids = [a.id for a in self.actors]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate actor ids: {ids}")
        for p in self.predicates:
            unknown = set(p.actors) - set(ids)
            if unknown:
                raise ValueError(f"{p.name} references undeclared actors {sorted(unknown)}")
        return self


class CandidateWindow(_Contract):
    window_id: str
    scene_id: str
    t0: float
    t1: float
    cameras: list[str]
    score: float
    source: Literal["glimmer", "uniform", "vlm"]

    @model_validator(mode="after")
    def _check_times(self) -> "CandidateWindow":
        if self.t1 < self.t0:
            raise ValueError(f"t1 ({self.t1}) is before t0 ({self.t0})")
        return self


class MaskRLE(_Contract):
    # ASSUMPTION A-20: masks are COCO-style compressed RLE with size (height, width)
    size: tuple[int, int]
    counts: str


class Frame2D(_Contract):
    frame_idx: int = Field(ge=0)
    ts: float
    # ASSUMPTION A-20: inclusive pixel box in the source image; mask and conf may be absent
    box_xyxy: tuple[float, float, float, float]
    mask_rle: MaskRLE | None = None
    conf: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _check_box(self) -> "Frame2D":
        x1, y1, x2, y2 = self.box_xyxy
        if x2 < x1 or y2 < y1:
            raise ValueError(f"box_xyxy must have x1 <= x2 and y1 <= y2, got {self.box_xyxy}")
        return self


class Track2D(_Contract):
    scene_id: str
    camera: str
    concept: str
    obj_id: int
    truncated: bool
    frames: list[Frame2D]


class Sample3D(_Contract):
    ts: float
    xyz_ego: tuple[float, float, float]
    xyz_world: tuple[float, float, float]
    n_points: int = Field(ge=0)
    extent_lwh: tuple[float, float, float]
    coverage: int = Field(ge=0)  # points inside the mask's box at any depth (plan C4 step 2)


class Track3D(_Contract):
    track_id: str
    cls: str
    sources: list[tuple[str, int]]  # (camera, obj_id)
    samples: list[Sample3D]
    speed_mps: list[float]


class PredicateResult(_Contract):
    result_id: str
    name: str
    actors: dict[str, str]  # actor id to track id
    status: Status
    value: float | None
    threshold: str
    evidence: dict[str, Any]  # counts, frames, reason, margin
    # ASSUMPTION A-22: each result carries its call's required flag so Finding.verified is computable
    required: bool = True

    @model_validator(mode="after")
    def _check_margin(self) -> "PredicateResult":
        # ASSUMPTION A-08: evidence holds a numeric margin for ranking
        # ASSUMPTION A-24: margin is required exactly when value is set
        margin = self.evidence.get("margin")
        if self.value is None:
            if margin is not None:
                raise ValueError("evidence.margin must be absent when value is None")
        elif isinstance(margin, bool) or not isinstance(margin, int | float):
            raise ValueError("evidence.margin must be a number when value is set")
        return self


class Finding(_Contract):
    finding_id: str
    scene_id: str
    t0: float
    t1: float
    spec_hash: str
    bindings: dict[str, str]
    results: list[PredicateResult]
    score: float
    cost_usd: float

    # ASSUMPTION A-17: unverified findings are valid objects; only the agent's finish tool drops them
    @computed_field
    @property
    def verified(self) -> bool:
        # ASSUMPTION A-22: a finding with no required results is not verified
        required = [r for r in self.results if r.required]
        return bool(required) and all(r.status == "PASS" for r in required)

    @model_validator(mode="wrap")
    @classmethod
    def _verified_is_computed(cls, data: Any, handler: Any) -> "Finding":
        # A serialised Finding carries "verified"; accept it only if it matches the computed value.
        claimed = None
        if isinstance(data, dict) and "verified" in data:
            data = dict(data)
            claimed = data.pop("verified")
        finding = handler(data)
        if claimed is not None and claimed != finding.verified:
            raise ValueError(f"verified is computed from results ({finding.verified}); got {claimed!r}")
        return finding

    @model_validator(mode="after")
    def _check_times(self) -> "Finding":
        if self.t1 < self.t0:
            raise ValueError(f"t1 ({self.t1}) is before t0 ({self.t0})")
        return self
