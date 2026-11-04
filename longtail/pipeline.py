"""The linear pipeline (plan M3, first half): spec -> candidate windows -> verify, per scene, in parallel.

No model is called here. The agent (agent/orchestrator.py) decides which scenes to search and when to stop;
this module does the searching, so a run is the same whether a model or a script drives it.
"""

import hashlib
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from longtail.contracts import CandidateWindow, Finding, ScenarioSpec
from longtail.data.base import Scene
from longtail.verify.engine import TrackSource
from longtail.verify.run import RUN_FILE, VerifyRun, load_run, verify_and_record

# ASSUMPTION A-213: Glimmer triage is cut (plan cut list item 7): uniform 4 s windows with a 2 s stride
WINDOW_S = 4.0
STRIDE_S = 2.0
# ASSUMPTION A-214: scenes are searched by up to 4 worker threads, one scene per worker
WORKERS = 4
# ASSUMPTION A-230: before verifying, the SAM calls and mask decoding of every (scene, camera, concept) run in
# up to this many worker processes (decoding is pure Python, so threads would not help); 0 turns it off
SAM_WORKERS = int(os.environ.get("LONGTAIL_SAM_WORKERS", "6"))

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


def scene_run_dir(spec: ScenarioSpec, scene_id: str, run_root: Path) -> Path:
    return Path(run_root) / f"{spec_run_id(spec)}-{scene_id}"


def search_scene(spec: ScenarioSpec, scene: Scene, track_source: TrackSource, run_root: Path) -> VerifyRun:
    run_dir = scene_run_dir(spec, scene.scene_id, run_root)
    windows = uniform_windows(scene)
    # ASSUMPTION A-243: with LONGTAIL_RESUME=1 (`longtail demo` sets it), a scene verified for this spec
    # and these windows loads from its verify_run.json instead of being verified again
    if os.environ.get("LONGTAIL_RESUME") == "1" and (run_dir / RUN_FILE).is_file():
        saved = load_run(run_dir)
        if saved.spec == spec and saved.windows == windows and saved.scene_id == scene.scene_id:
            return saved
    return verify_and_record(spec, windows, scene, track_source, run_dir)


def concepts(spec: ScenarioSpec) -> list[str]:
    """The SAM phrases verify() will ask for: every actor's, and every occluder class's (A-87)."""
    phrases = {a.sam_prompt for a in spec.actors}
    for p in spec.predicates:
        cls = p.args.get("cls")
        if isinstance(cls, str):
            phrases.add(next((a.sam_prompt for a in spec.actors if a.cls == cls), cls))
    return sorted(phrases)


def prefetch(
    spec: ScenarioSpec | list[ScenarioSpec],
    scene_ids: list[str],
    open_scene: SceneOpener,
    track_source: TrackSource,
    workers: int,
) -> list[str]:
    """Fill the track source's caches in parallel; returns the failed jobs (verify() retries them itself).

    Only for a source with a `warm` hook (the SAM one, A-229); anything else is left to verify() as before.
    """
    warm = getattr(track_source, "warm", None)
    if warm is None or workers <= 0:
        return []
    allowed = getattr(track_source, "cameras", None)
    jobs = [
        (sid, cam, c)
        for sid in scene_ids
        for cam in open_scene(sid).cameras
        if allowed is None or cam in allowed
        for c in sorted({c for one in (spec if isinstance(spec, list) else [spec]) for c in concepts(one)})
    ]
    failed = []
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as pool:
        futures = {pool.submit(warm, *job): job for job in jobs}
        for f, job in futures.items():
            try:
                f.result()
            except Exception as e:
                failed.append(f"{'/'.join(job)}: {type(e).__name__}: {e}")
    return failed


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

    try:
        prefetch(spec, scene_ids, open_scene, track_source, SAM_WORKERS)
    except Exception:  # prefetch only saves time; verify() makes the same calls if it could not
        pass
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        scenes = list(pool.map(one, scene_ids))
    return PipelineResult(spec, scenes)
