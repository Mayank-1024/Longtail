"""Match camera frames to LiDAR sweeps (plan C2).

Matching uses sweep timestamps only. Per-point timestamps (A-18) are left for motion compensation.
"""

from dataclasses import dataclass, replace

import numpy as np

from longtail.data.base import Scene

# Plan C2: a frame more than 50 ms from its nearest sweep is unsynced and goes to UNVERIFIABLE downstream.
MAX_GAP_S = 0.05
# Timestamps are ~1.5e9 s floats (resolution ~0.2 us), so an exact 50 ms gap can come out slightly over.
_FLOAT_SLACK_S = 1e-6


@dataclass(frozen=True)
class SyncedFrame:
    camera: str
    frame_idx: int
    ts: float
    lidar_idx: int
    lidar_ts: float
    gap_s: float  # camera ts minus sweep ts; negative when the camera fired first
    synced: bool


def nearest_sweeps(camera_ts: np.ndarray, lidar_ts: np.ndarray) -> np.ndarray:
    """Index of the nearest sweep for each camera timestamp. lidar_ts must be sorted."""
    # ASSUMPTION A-39: on an exact tie between two sweeps the earlier sweep wins
    right = np.clip(np.searchsorted(lidar_ts, camera_ts), 1, len(lidar_ts) - 1)
    left = right - 1
    if len(lidar_ts) == 1:
        return np.zeros(len(camera_ts), dtype=np.int64)
    take_right = np.abs(lidar_ts[right] - camera_ts) < np.abs(camera_ts - lidar_ts[left])
    return np.where(take_right, right, left)


def sync_camera(scene: Scene, camera: str, max_gap_s: float = MAX_GAP_S) -> list[SyncedFrame]:
    lidar_ts = scene.lidar_timestamps()
    frames = scene.frames(camera)
    cam_ts = np.array([f.ts for f in frames])
    idx = nearest_sweeps(cam_ts, lidar_ts)
    gaps = cam_ts - lidar_ts[idx]
    return [
        SyncedFrame(
            camera=camera,
            frame_idx=f.frame_idx,
            ts=f.ts,
            lidar_idx=int(i),
            lidar_ts=float(lidar_ts[i]),
            gap_s=float(g),
            # ASSUMPTION A-39: a gap of exactly 50 ms is still synced; only a larger one is not
            synced=bool(abs(g) <= max_gap_s + _FLOAT_SLACK_S),
        )
        for f, i, g in zip(frames, idx, gaps, strict=True)
    ]


def sync_scene(scene: Scene, max_gap_s: float = MAX_GAP_S) -> dict[str, list[SyncedFrame]]:
    return {cam: sync_camera(scene, cam, max_gap_s) for cam in scene.cameras}


@dataclass(frozen=True)
class GapStats:
    n: int
    n_unsynced: int
    min_ms: float
    median_ms: float
    p95_abs_ms: float
    max_ms: float
    repeated_sweeps: int  # frames that share their sweep with an earlier frame of the same camera


def _stats(frames: list[SyncedFrame]) -> GapStats:
    g = np.array([f.gap_s for f in frames]) * 1000.0
    sweeps = [f.lidar_idx for f in frames]
    return GapStats(
        n=len(frames),
        n_unsynced=sum(not f.synced for f in frames),
        min_ms=float(g.min()),
        median_ms=float(np.median(g)),
        p95_abs_ms=float(np.percentile(np.abs(g), 95)),
        max_ms=float(g.max()),
        repeated_sweeps=len(sweeps) - len(set(sweeps)),
    )


def gap_distribution(scene: Scene, max_gap_s: float = MAX_GAP_S) -> dict[str, GapStats]:
    """Camera-to-LiDAR gap statistics per camera, plus "all" across cameras."""
    synced = sync_scene(scene, max_gap_s)
    report = {cam: _stats(frames) for cam, frames in synced.items()}
    combined = _stats([f for frames in synced.values() for f in frames])
    # Cameras share sweeps with each other by design; only reuse within a camera counts.
    report["all"] = replace(combined, repeated_sweeps=sum(s.repeated_sweeps for s in report.values()))
    return report


def format_gap_distribution(report: dict[str, GapStats]) -> str:
    lines = [
        f"{'camera':<20}{'n':>4}{'unsynced':>10}{'min':>9}{'median':>9}{'p95|gap|':>10}{'max':>9}{'reused':>8}"
    ]
    for cam, s in report.items():
        lines.append(
            f"{cam:<20}{s.n:>4}{s.n_unsynced:>10}{s.min_ms:>9.1f}{s.median_ms:>9.1f}"
            f"{s.p95_abs_ms:>10.1f}{s.max_ms:>9.1f}{s.repeated_sweeps:>8}"
        )
    lines.append("gaps in ms, camera minus LiDAR; reused = frames sharing a sweep with an earlier frame")
    return "\n".join(lines)
