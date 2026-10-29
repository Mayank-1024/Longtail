"""Load configs/verify.yaml. The file is the only source of thresholds; models have no defaults."""

# ASSUMPTION A-58: every threshold lives in configs/verify.yaml; the loader has no defaults in code

from functools import cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "verify.yaml"


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


class VerifyConfig(_Strict):
    association: AssociationConfig


@cache
def load_verify_config(path: Path = CONFIG_PATH) -> VerifyConfig:
    with open(path) as f:
        return VerifyConfig.model_validate(yaml.safe_load(f))
