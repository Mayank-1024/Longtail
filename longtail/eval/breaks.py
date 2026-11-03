"""Deliberate breaks (plan C8): `python -m longtail.eval.breaks`.

Configuration (c) is run again with one thing broken: the verifier off, the camera calibration shifted by
2 degrees of yaw and 0.3 m, or the LiDAR timestamps offset by 200 ms. Each must lower precision, the catch
rate or recall; one that lowers none is reported as a hole in the evaluation. The calibration and timing
breaks are Scene wrappers, so no production code changes; claim truth still comes from the unperturbed scenes.
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from longtail.data.base import CameraFrame, LidarSweep, Scene, SceneMeta
from longtail.eval import ablation, gt
from longtail.eval.metrics import summarise
from longtail.geometry.transforms import make_pose

YAW_DEG, SHIFT_M, LIDAR_OFFSET_S = 2.0, 0.3, 0.2
# A break passes when it lowers any of these; the false-rejection rate is reported alongside.
JUDGED = ("precision", "catch_rate", "recall")
COMPARED = ("precision", "catch_rate", "recall", "p_at_10", "false_rejection_rate")


class SceneWrapper(Scene):
    """Delegates everything to the wrapped scene; subclasses override what they break."""

    def __init__(self, inner: Scene) -> None:
        self.inner = inner
        self.scene_id = inner.scene_id

    @property
    def cameras(self) -> tuple[str, ...]:
        return self.inner.cameras

    @property
    def meta(self) -> SceneMeta:
        return self.inner.meta

    def frames(self, camera: str) -> list[CameraFrame]:
        return self.inner.frames(camera)

    def lidar_timestamps(self) -> np.ndarray:
        return self.inner.lidar_timestamps()

    def lidar(self, ts: float) -> LidarSweep:
        return self.inner.lidar(ts)

    def ego_pose(self, ts: float) -> np.ndarray:
        return self.inner.ego_pose(ts)

    def intrinsics(self, camera: str) -> np.ndarray:
        return self.inner.intrinsics(camera)

    def extrinsics(self, camera: str, ts: float) -> np.ndarray:
        return self.inner.extrinsics(camera, ts)

    def image_size(self, camera: str) -> tuple[int, int]:
        return self.inner.image_size(camera)

    def _labels(self, frame_idx: int) -> Any:
        return self.inner._labels(frame_idx)

    def _semseg(self, frame_idx: int) -> Any:
        return self.inner._semseg(frame_idx)


class CalibrationShift(SceneWrapper):
    """Every camera turned by yaw_deg about its vertical axis and moved shift_m sideways."""

    # ASSUMPTION A-141: the calibration break turns each camera 2 degrees about its own y axis (image down,
    # so a yaw for a level camera) and moves it 0.3 m along its x axis (image right)
    def __init__(self, inner: Scene, yaw_deg: float = YAW_DEG, shift_m: float = SHIFT_M) -> None:
        super().__init__(inner)
        a = np.radians(yaw_deg)
        r = np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])
        self.delta = make_pose(r, np.array([shift_m, 0.0, 0.0]))

    def extrinsics(self, camera: str, ts: float) -> np.ndarray:
        return self.inner.extrinsics(camera, ts) @ self.delta


class LidarTimeOffset(SceneWrapper):
    """LiDAR sweeps (and the poses that come with them) stamped offset_s late; cameras unchanged."""

    # ASSUMPTION A-141: the timing break adds 200 ms to every sweep timestamp, its per-point times and its
    # ego pose lookup, so C2 pairs each camera frame with the sweep from 200 ms earlier
    def __init__(self, inner: Scene, offset_s: float = LIDAR_OFFSET_S) -> None:
        super().__init__(inner)
        self.offset_s = offset_s

    def lidar_timestamps(self) -> np.ndarray:
        return self.inner.lidar_timestamps() + self.offset_s

    def lidar(self, ts: float) -> LidarSweep:
        sweep = self.inner.lidar(ts - self.offset_s)
        points = sweep.points.copy()
        points[:, 4] += self.offset_s
        return replace(sweep, ts=sweep.ts + self.offset_s, points=points)

    def ego_pose(self, ts: float) -> np.ndarray:
        return self.inner.ego_pose(ts - self.offset_s)


BREAKS = ("verifier_off", "calibration_2deg_0.3m", "lidar_offset_200ms")


def broken_setup(name: str, setup: ablation.Setup) -> tuple[ablation.Setup, bool]:
    """The setup for one break, and whether the verifier runs."""
    if name == "verifier_off":
        # ASSUMPTION A-141: verifier off = every finding of (c) counts as verified and every claim as accepted
        return setup, False
    wrap = CalibrationShift if name.startswith("calibration") else LidarTimeOffset
    return replace(setup, scenes=[wrap(s) for s in setup.scenes]), True


def _lower(base: float | None, broken: float | None) -> bool | None:
    if base is None or broken is None:
        return None  # not comparable (e.g. nothing returned)
    return broken < base - 1e-9


def run_breaks(setup: ablation.Setup, split: str) -> dict:
    """Configuration (c) clean and under each break, with which metrics dropped and any hole."""
    base = summarise(ablation.run_config("c", setup))
    out: dict[str, Any] = {
        "format": "longtail.breaks/v1",
        "commit": ablation.git_commit(),
        "thresholds_hash": ablation.thresholds_hash(setup.verify_cfg, setup.gt_cfg),
        "gt_config_hash": gt.config_hash(setup.gt_cfg),
        "split": split,
        "scenes": [s.scene_id for s in setup.scenes],
        "baseline_c": base,
        "breaks": {},
        "holes": [],
    }
    for name in BREAKS:
        s, verifier = broken_setup(name, setup)
        m = summarise(ablation.run_config("c", s, verifier=verifier))
        lowered = {k: _lower(base[k]["value"], m[k]["value"]) for k in COMPARED}
        # ASSUMPTION A-141: a break passes when precision, the catch rate or recall drops, and records which;
        # one that lowers none is a hole in the evaluation
        moved = [k for k in JUDGED if lowered[k]]
        hole = not moved
        out["breaks"][name] = {"metrics": m, "lowered": lowered, "moved": moved, "passed": not hole}
        if hole:
            out["holes"].append(name)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m longtail.eval.breaks", description=__doc__.splitlines()[0])
    ap.add_argument("--queries", type=Path, required=True, help="queries file (A-128)")
    ap.add_argument("--split", choices=["dev", "heldout", "all"], default="heldout")
    ap.add_argument("--gt-dir", type=Path, default=Path("gt"))
    ap.add_argument("--scenes", nargs="*", help="scene ids (default: every scene in the index)")
    ap.add_argument("--out", type=Path, default=ablation.RESULTS_DIR)
    ap.add_argument("--replay", action="store_true", help="SAM responses from the cache only (no paid calls)")
    args = ap.parse_args(argv)
    from longtail.eval.cli import _scenes

    scenes = _scenes(args.scenes)
    setup = ablation.Setup(
        queries=ablation.select(gt.load_queries(args.queries), args.split),
        scenes=scenes,
        truth_scenes=scenes,
        gt_dir=args.gt_dir,
        source=ablation.sam_source(args.replay),
    )
    report = run_breaks(setup, args.split)
    out = ablation.check_results_dir(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "breaks.json").write_text(json.dumps(report, indent=1) + "\n")
    for name, b in report["breaks"].items():
        print(f"{name}: {'moved ' + ', '.join(b['moved']) if b['passed'] else 'HOLE: lowered nothing'}")
    print(f"written to {out / 'breaks.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
