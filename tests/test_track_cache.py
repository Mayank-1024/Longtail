"""The parsed-track cache (A-229) and the pipeline's parallel prefetch (A-230)."""

import functools
from pathlib import Path

import pytest

from longtail.contracts import ScenarioSpec
from longtail.pipeline import concepts, prefetch
from longtail.vision import track_source
from longtail.vision.clips import Clip, ClipIndex

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _clip() -> tuple[str, Clip]:
    text = (FIXTURES / "sam31_SYNTHETIC.txt").read_text()
    index = ClipIndex.model_validate_json((FIXTURES / "sam31_SYNTHETIC.clip_index.json").read_text())
    return text, Clip(path=Path("unused.mp4"), index=index, cached=True)


def test_cached_parse_returns_the_same_tracks_and_parses_once(tmp_path, monkeypatch):
    monkeypatch.setenv("LONGTAIL_CACHE_DIR", str(tmp_path))
    text, clip = _clip()
    calls = []
    real = track_source.parse
    monkeypatch.setattr(track_source, "parse", lambda *a: calls.append(1) or real(*a))
    first = track_source.cached_parse(text, clip, "person")
    second = track_source.cached_parse(text, clip, "person")
    assert len(calls) == 1 and first == second and first
    assert len(list((tmp_path / "sam_tracks").glob("*.json.gz"))) == 1
    # another concept or another response is another entry, never a stale hit
    track_source.cached_parse(text, clip, "car")
    track_source.cached_parse(text + "\n", clip, "person")
    assert len(calls) == 3


def test_concepts_include_occluder_classes():
    spec = ScenarioSpec.model_validate(
        {
            "query": "q",
            "assumptions": [],
            "clarification": None,
            "actors": [{"id": "a1", "cls": "pedestrian", "sam_prompt": "person"}],
            "predicates": [
                {"name": "exists", "actors": ["a1"], "args": {}},
                {"name": "occluded_by", "actors": ["a1"], "args": {"cls": "truck"}},
            ],
        }
    )
    assert concepts(spec) == ["person", "truck"]


def _touch(out: str, scene_id: str, camera: str, concept: str) -> int:
    if camera == "bad_camera":
        raise RuntimeError("boom")
    Path(out, f"{scene_id}-{camera}-{concept}").touch()
    return 0


class _Scene:
    cameras = ("front_camera", "back_camera", "bad_camera")


def _spec() -> ScenarioSpec:
    return ScenarioSpec.model_validate(
        {
            "query": "q",
            "assumptions": [],
            "clarification": None,
            "actors": [
                {"id": "a1", "cls": "pedestrian", "sam_prompt": "person"},
                {"id": "b1", "cls": "car", "sam_prompt": "car"},
            ],
            "predicates": [{"name": "exists", "actors": ["a1"], "args": {}}],
        }
    )


def test_prefetch_warms_every_camera_and_concept_in_processes_and_reports_failures(tmp_path):
    def source(scene, camera, concept):  # pragma: no cover - never called by prefetch
        raise AssertionError

    source.warm = functools.partial(_touch, str(tmp_path))
    source.cameras = None
    failed = prefetch(_spec(), ["001", "002"], lambda sid: _Scene(), source, workers=2)
    made = sorted(p.name for p in tmp_path.iterdir())
    assert made == sorted(
        f"{s}-{c}-{k}"
        for s in ("001", "002")
        for c in ("front_camera", "back_camera")
        for k in ("car", "person")
    )
    assert len(failed) == 4 and all("bad_camera" in f and "boom" in f for f in failed)


@pytest.mark.parametrize("workers", [0, 3])
def test_prefetch_is_a_no_op_without_a_warm_hook_or_with_zero_workers(tmp_path, workers):
    def source(scene, camera, concept):  # pragma: no cover
        raise AssertionError

    if workers:
        assert prefetch(_spec(), ["001"], lambda sid: _Scene(), source, workers=workers) == []
    else:
        source.warm = functools.partial(_touch, str(tmp_path))
        assert prefetch(_spec(), ["001"], lambda sid: _Scene(), source, workers=0) == []
        assert not list(tmp_path.iterdir())
