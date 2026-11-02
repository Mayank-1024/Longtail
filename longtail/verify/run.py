"""A saved verification run: what verify() was asked, what it answered, and the Track2Ds it received.

C9 exports findings from this record: a Finding names its tracks but carries no masks or points, and the
record replays the same Track2Ds through the same deterministic association and tracking.
"""

# ASSUMPTION A-101: exports are built from a saved run (spec, windows, findings, every Track2D the engine got)

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from longtail.contracts import CandidateWindow, Finding, ScenarioSpec, Track2D
from longtail.data.base import Scene
from longtail.verify.config import VerifyConfig
from longtail.verify.engine import TrackSource, verify

RUN_FILE = "verify_run.json"


class TrackCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera: str
    concept: str
    tracks: list[Track2D]


class VerifyRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    dataset: str
    scene_id: str
    spec: ScenarioSpec
    windows: list[CandidateWindow]
    findings: list[Finding]
    finding_windows: dict[str, str]  # finding id to window id
    track_calls: list[TrackCall]

    def window_of(self, finding_id: str) -> CandidateWindow:
        wid = self.finding_windows[finding_id]
        return next(w for w in self.windows if w.window_id == wid)

    def replay_source(self) -> TrackSource:
        calls = {(c.camera, c.concept): c.tracks for c in self.track_calls}

        def source(_scene: Scene, camera: str, concept: str) -> list[Track2D]:
            return calls.get((camera, concept), [])

        return source


def verify_and_record(
    spec: ScenarioSpec,
    windows: list[CandidateWindow],
    scene: Scene,
    track_source: TrackSource,
    run_dir: Path,
    cfg: VerifyConfig | None = None,
) -> VerifyRun:
    """verify(), one window at a time so each finding keeps its window, saved to run_dir/verify_run.json."""
    calls: dict[tuple[str, str], list[Track2D]] = {}

    def recording(scn: Scene, camera: str, concept: str) -> list[Track2D]:
        if (camera, concept) not in calls:
            calls[(camera, concept)] = track_source(scn, camera, concept)
        return calls[(camera, concept)]

    findings: list[Finding] = []
    finding_windows: dict[str, str] = {}
    for w in windows:
        for f in verify(spec, [w], scene, track_source=recording, cfg=cfg):
            findings.append(f)
            finding_windows[f.finding_id] = w.window_id
    findings.sort(key=lambda f: (not f.verified, -f.score))
    key = spec.model_dump_json() + "".join(w.model_dump_json() for w in windows) + scene.scene_id
    run = VerifyRun(
        run_id=Path(run_dir).name or hashlib.sha256(key.encode()).hexdigest()[:12],
        dataset=scene.meta.dataset,
        scene_id=scene.scene_id,
        spec=spec,
        windows=windows,
        findings=findings,
        finding_windows=finding_windows,
        track_calls=[TrackCall(camera=c, concept=k, tracks=t) for (c, k), t in sorted(calls.items())],
    )
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_FILE).write_text(run.model_dump_json())
    return run


def load_run(run_dir: Path) -> VerifyRun:
    return VerifyRun.model_validate_json((Path(run_dir) / RUN_FILE).read_text())
