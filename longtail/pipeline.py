"""The linear pipeline (plan M3, first half): spec -> candidate windows -> verify, per scene, in parallel.

No model is called here. The agent (agent/orchestrator.py) decides which scenes to search and when to stop;
this module does the searching, so a run is the same whether a model or a script drives it.
"""

import hashlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from longtail.contracts import CandidateWindow, Finding, ScenarioSpec
from longtail.data.base import Scene
from longtail.verify.engine import TrackSource
from longtail.verify.run import VerifyRun, verify_and_record

# ASSUMPTION A-213: Glimmer triage is cut (plan cut list item 7): uniform 4 s windows with a 2 s stride
WINDOW_S = 4.0
STRIDE_S = 2.0
# ASSUMPTION A-214: scenes are searched by up to 4 worker threads, one scene per worker
WORKERS = 4

SceneOpener = Callable[[str], Scene]


def uniform_windows(
    scene: Scene, window_s: float = WINDOW_S, stride_s: float = STRIDE_S
) -> list[CandidateWindow]:
    """Overlapping windows over the whole scene; a scene shorter than one window gets a single window."""
    ts = scene.lidar_timestamps()
    t_first, t_last = float(ts[0]), float(ts[-1])
    starts = list(np.arange(t_first, max(t_first, t_last - window_s) + 1e-9, stride_s)) or [t_first]
    if t_last - starts[-1] > window_s + 1e-9:
        starts.append(t_last - window_s)
    windows = []
    for i, t0 in enumerate(starts):
        t1 = min(float(t0) + window_s, t_last)
        windows.append(
            CandidateWindow(
                window_id=f"{scene.scene_id}:w{i:02d}",
                scene_id=scene.scene_id,
                t0=float(t0),
                t1=t1,
                cameras=[],
                score=1.0,
                source="uniform",
            )
        )
    return windows


@dataclass
class SceneResult:
    scene_id: str
    run: VerifyRun | None
    error: str | None = None

    @property
    def findings(self) -> list[Finding]:
        return self.run.findings if self.run is not None else []


@dataclass
class PipelineResult:
    spec: ScenarioSpec
    scenes: list[SceneResult] = field(default_factory=list)

    def findings(self) -> list[Finding]:
        """Every finding over every searched scene: verified first, then by score."""
        out = [f for s in self.scenes for f in s.findings]
        return sorted(out, key=lambda f: (not f.verified, -f.score))

    def verified(self) -> list[Finding]:
        return [f for f in self.findings() if f.verified]


def spec_run_id(spec: ScenarioSpec) -> str:
    return hashlib.sha256(spec.model_dump_json().encode()).hexdigest()[:12]


def search_scene(spec: ScenarioSpec, scene: Scene, track_source: TrackSource, run_root: Path) -> VerifyRun:
    run_dir = Path(run_root) / f"{spec_run_id(spec)}-{scene.scene_id}"
    return verify_and_record(spec, uniform_windows(scene), scene, track_source, run_dir)


def run(
    spec: ScenarioSpec,
    scene_ids: list[str],
    open_scene: SceneOpener,
    track_source: TrackSource,
    run_root: Path,
    workers: int = WORKERS,
) -> PipelineResult:
    """Search the scenes. A scene that fails is reported with its error; it never stops the others."""
    if spec.clarification is not None:
        raise ValueError("the spec asks a clarifying question; answer it before searching")

    def one(scene_id: str) -> SceneResult:
        try:
            return SceneResult(scene_id, search_scene(spec, open_scene(scene_id), track_source, run_root))
        except Exception as e:  # one bad scene must not lose the others' results
            return SceneResult(scene_id, None, f"{type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        scenes = list(pool.map(one, scene_ids))
    return PipelineResult(spec, scenes)
