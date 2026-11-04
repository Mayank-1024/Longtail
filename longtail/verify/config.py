"""Load configs/verify.yaml. The file is the only source of thresholds; models have no defaults."""

# ASSUMPTION A-58: every threshold lives in configs/verify.yaml; the loader has no defaults in code

import os
from functools import cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "verify.yaml"
# ASSUMPTION A-244: the post-freeze verifier (v2) is configs/verify_v2.yaml, chosen by LONGTAIL_VERIFIER=v2.
# Its fields are optional and absent from verify.yaml, where None keeps v1's behaviour, so v1 results, its
# thresholds hash (dumped without None fields) and the recorded demo reproduce exactly
CONFIG_PATHS = {"v1": CONFIG_PATH, "v2": CONFIG_PATH.with_name("verify_v2.yaml")}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GroundConfig(_Strict):
    mode: Literal["local", "global"]
    local_radius_m: float = Field(gt=0)
    min_local_inliers: int = Field(ge=3)
    ransac_iterations: int = Field(ge=1)
    ransac_seed_lowest_frac: float = Field(gt=0, le=1)
    ransac_inlier_m: float = Field(gt=0)
    max_tilt_deg: float = Field(gt=0, lt=90)
    max_below_frac: float = Field(ge=0, le=1)
    below_margin_m: float = Field(ge=0)
    global_sample_points: int = Field(ge=3)
    seed: int


class MergeConfig(_Strict):
    max_centroid_dist_m: float = Field(gt=0)
    min_shared_frac: float = Field(gt=0, le=1)
    min_shared_points: int = Field(ge=1)
    min_shared_points_frac: float = Field(ge=0, le=1)
    fragment_max_gap_s: float = Field(ge=0)
    fragment_velocity_window_s: float = Field(gt=0)


class AssociationConfig(_Strict):
    erode_px: int = Field(ge=0)
    # v2 (A-245): an eroded mask under this many pixels falls back to the uneroded mask
    min_eroded_px: int | None = Field(default=None, ge=0)
    # v2 (A-246): no points or cluster in a mask expected to hold under dbscan_min_points is UNVERIFIABLE
    thin_mask_unverifiable: bool | None = None
    # v2 (A-248): a SAM response with this many distinct objects marks all its tracks truncated
    truncated_min_objects: int | None = Field(default=None, ge=1)
    min_coverage: int = Field(ge=0)
    ground_height_m: float
    visibility_cell_px: int = Field(ge=0)
    visibility_margin_m: float = Field(ge=0)
    depth_gap_m: float = Field(gt=0)
    nearest_cluster_min_frac: float = Field(gt=0, le=1)
    dbscan_eps_m: float = Field(gt=0)
    dbscan_min_points: int = Field(ge=1)
    selection: Literal["mask", "frustum"]
    ground: GroundConfig
    merge: MergeConfig


class TracksConfig(_Strict):
    savgol_window: int = Field(ge=3)
    savgol_polyorder: int = Field(ge=0)
    max_interp_gap_s: float = Field(ge=0)
    position: Literal["median", "box_centre"]
    lshape_angle_step_deg: float = Field(gt=0, le=45)
    lshape_min_distance_m: float = Field(gt=0)
    edge_band_m: float = Field(gt=0)
    truncation_border_px: int = Field(ge=0)
    occluder_radius_px: float = Field(gt=0)
    occluder_margin_m: float = Field(ge=0)
    spacing_factor: float = Field(ge=0)
    # v2 (A-247): drop a sample further than outlier_m from every neighbour interpolation of it
    outlier_window: int | None = Field(default=None, ge=3)
    outlier_m: float | None = Field(default=None, gt=0)


class EgoPathConfig(_Strict):
    horizon_s: float = Field(gt=0)
    half_width_m: float = Field(gt=0)
    min_length_m: float = Field(ge=0)


class SizePrior(_Strict):
    height_m: tuple[float, float]
    footprint_m: tuple[float, float]
    typical_height_m: float = Field(gt=0)


class ExistsConfig(_Strict):
    min_points: int = Field(ge=1)
    min_points_far: int = Field(ge=1)
    far_m: float = Field(gt=0)
    max_range_m: float = Field(gt=0)
    size_priors: dict[str, SizePrior]


class WithinDistanceConfig(_Strict):
    tolerance_m: float = Field(ge=0)


class MotionConfig(_Strict):
    moving_mps: float = Field(gt=0)
    stopped_mps: float = Field(ge=0)
    window_s: float = Field(gt=0)
    max_accel_mps2: float = Field(gt=0)
    # v2 (A-247): False judges windows that contain a jump (outlying samples are dropped upstream instead)
    drop_jump_windows: bool | None = None


class CrossingConfig(_Strict):
    max_ahead_m: float = Field(gt=0)
    min_lateral_mps: float = Field(ge=0)


class RelativePositionConfig(_Strict):
    window_s: float = Field(gt=0)
    min_frac: float = Field(gt=0, le=1)


class OccludedByConfig(_Strict):
    min_cover_frac: float = Field(gt=0, le=1)


class PredicatesConfig(_Strict):
    exists: ExistsConfig
    within_distance: WithinDistanceConfig
    motion: MotionConfig
    crossing: CrossingConfig
    relative_position: RelativePositionConfig
    occluded_by: OccludedByConfig


class EngineConfig(_Strict):
    max_bindings: int = Field(ge=1)


class VerifyConfig(_Strict):
    association: AssociationConfig
    tracks: TracksConfig
    ego_path: EgoPathConfig
    predicates: PredicatesConfig
    engine: EngineConfig


def verifier_version() -> str:
    v = os.environ.get("LONGTAIL_VERIFIER", "v1")
    if v not in CONFIG_PATHS:
        raise ValueError(f"LONGTAIL_VERIFIER={v!r}: expected one of {sorted(CONFIG_PATHS)}")
    return v


def load_verify_config(path: Path | None = None) -> VerifyConfig:
    return _load(path or CONFIG_PATHS[verifier_version()])


@cache
def _load(path: Path) -> VerifyConfig:
    with open(path) as f:
        return VerifyConfig.model_validate(yaml.safe_load(f))


load_verify_config.cache_clear = _load.cache_clear  # type: ignore[attr-defined]
