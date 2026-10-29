"""The pipeline never reads labels (plan 2.1); only eval_ground_truth() opens them."""

import ast
from pathlib import Path

import numpy as np
import pytest

from longtail.data.base import Scene
from longtail.data.guard import LabelAccessError, eval_ground_truth, labels_allowed

ROOT = Path(__file__).resolve().parents[1]


class _StubScene(Scene):
    scene_id = "stub"
    cameras = ()
    meta = None

    def frames(self, camera):
        return []

    def lidar_timestamps(self):
        return np.array([])

    def lidar(self, ts):
        raise KeyError(ts)

    def ego_pose(self, ts):
        return np.eye(4)

    def intrinsics(self, camera):
        return np.eye(3)

    def extrinsics(self, camera, ts):
        return np.eye(4)

    def _labels(self, frame_idx):
        return f"cuboids {frame_idx}"

    def _semseg(self, frame_idx):
        return f"semseg {frame_idx}"


# ASSUMPTION A-15: the guard covers cuboids and semantic segmentation
@pytest.mark.parametrize("method", ["labels", "semseg"])
def test_labels_raise_outside_guard(method):
    with pytest.raises(LabelAccessError):
        getattr(_StubScene(), method)(0)


@pytest.mark.parametrize("method", ["labels", "semseg"])
def test_labels_readable_inside_guard(method):
    with eval_ground_truth():
        assert getattr(_StubScene(), method)(3).endswith("3")


def test_guard_nests_and_closes():
    assert not labels_allowed()
    with eval_ground_truth():
        with eval_ground_truth():
            assert labels_allowed()
        assert labels_allowed()
    assert not labels_allowed()
    with pytest.raises(LabelAccessError):
        _StubScene().labels(0)


def test_guard_closes_after_an_exception():
    with pytest.raises(RuntimeError), eval_ground_truth():
        raise RuntimeError("boom")
    assert not labels_allowed()


def _py_files(*dirs: str) -> list[Path]:
    return [p for d in dirs for p in (ROOT / d).rglob("*.py")]


def _calls(path: Path, name: str, no_args: bool = False) -> list[int]:
    """Line numbers of calls to name(...) or x.name(...) in a file (comments and docstrings ignored)."""
    lines = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        called = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if called == name and not (no_args and (node.args or node.keywords)):
            lines.append(node.lineno)
    return lines


def test_no_sequence_load_anywhere():
    # ASSUMPTION A-15: only per-modality devkit loaders, never sequence.load()
    offenders = [
        f"{p.relative_to(ROOT)}:{n}"
        for p in _py_files("longtail", "scripts")
        for n in _calls(p, "load", no_args=True)
    ]
    assert not offenders, f"argument-free .load() calls found: {offenders}"


def test_eval_ground_truth_used_only_in_eval_code():
    # ASSUMPTION A-36: eval_ground_truth() is entered only in eval/, the index builder and tests
    allowed = ("longtail/eval/", "scripts/build_index.py")
    offenders = [
        rel
        for p in _py_files("longtail", "scripts", "api")
        if _calls(p, "eval_ground_truth") and not (rel := str(p.relative_to(ROOT))).startswith(allowed)
    ]
    assert not offenders, f"eval_ground_truth() used outside evaluation code: {offenders}"
