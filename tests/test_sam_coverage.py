"""scripts/sam_coverage.py on a fake cache: clip indexes and response files only, no images or network."""

import importlib.util
from pathlib import Path

from longtail.data.pandaset import CAMERA_ORDER
from longtail.vision.clips import Clip, ClipCrop, ClipFrame, ClipIndex
from longtail.vision.sam_client import SamClient

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sam_coverage", ROOT / "scripts" / "sam_coverage.py")
cov_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cov_mod)

FRAMES = 5
CONCEPTS = ["car", "person"]


def _index(scene: str, camera: str, crop: str = "full") -> ClipIndex:
    return ClipIndex(
        clip_id=f"{scene}-{camera}-{crop}",
        scene_id=scene,
        camera=camera,
        fps=10.0,
        codec="h264",
        pix_fmt="yuv420p",
        crop=ClipCrop(name=crop, x0=0, y0=0, width=8, height=8, full_width=8, full_height=8),
        frames=[ClipFrame(clip_frame=k, frame_idx=k, ts=k / 10) for k in range(FRAMES)],
    )


def _cache(tmp_path: Path, scenes: list[str]) -> Path:
    """Full coverage: each camera of each scene has a full-frame index and a complete response per concept."""
    (tmp_path / "clips").mkdir()
    (tmp_path / "sam" / "responses").mkdir(parents=True)
    for s in scenes:
        for cam in CAMERA_ORDER:
            ix = _index(s, cam)
            (tmp_path / "clips" / f"{ix.clip_id}.clip_index.json").write_text(ix.model_dump_json())
            for c in CONCEPTS:
                _response(tmp_path, ix, c).write_text("".join(f"<{k}f>\n" for k in range(FRAMES)))
    return tmp_path


def _response(cache: Path, ix: ClipIndex, concept: str) -> Path:
    return cache / "sam" / "responses" / f"{cov_mod.response_key(ix.clip_id, concept)}.txt"


def test_key_matches_the_sam_client(tmp_path):
    ix = _index("001", "front_camera")
    client = SamClient(None, cache_dir=tmp_path, replay=True)
    assert cov_mod.response_key(ix.clip_id, "car") == client._key(Clip(tmp_path / "x.mp4", ix, True), "car")


def test_full_cache_has_no_gaps(tmp_path):
    cov = cov_mod.check(_cache(tmp_path, ["001", "005"]), ["001", "005"], CONCEPTS)
    assert cov.needed == 2 * len(CAMERA_ORDER) * len(CONCEPTS)
    assert (cov.gaps, cov.empty, cov.short, cov.rerun_scenes()) == (0, [], [], [])


def test_finds_every_planted_gap(tmp_path):
    cache = _cache(tmp_path, ["001", "005"])
    _response(cache, _index("001", "left_camera"), "person").unlink()  # one response
    (cache / "clips" / f"{_index('005', 'back_camera').clip_id}.clip_index.json").unlink()  # one camera
    _response(cache, _index("005", "front_camera"), "car").write_text("<0f>\n<1f>\n")  # cut off at frame 1
    _response(cache, _index("005", "right_camera"), "car").write_text("")
    crop = _index("001", "back_camera", crop="left")  # a crop clip never stands in for the full frame
    (cache / "clips" / f"{crop.clip_id}.clip_index.json").write_text(crop.model_dump_json())

    cov = cov_mod.check(cache, ["001", "005", "018"], CONCEPTS)  # 018 is not in the cache at all

    assert cov.missing == [("001", "left_camera", "person")]
    assert ("005", "back_camera") in cov.missing_clip
    assert sorted(c for s, c in cov.missing_clip if s == "018") == sorted(CAMERA_ORDER)
    assert cov.gaps == 1 + (1 + len(CAMERA_ORDER)) * len(CONCEPTS)
    assert cov.short == [("005", "front_camera", "car", 1, FRAMES)]
    assert cov.empty == [("005", "right_camera", "car")]
    assert cov.rerun_scenes() == ["001", "005", "018"]
    text = cov_mod.report(cov)
    assert "scene 018" in text and "rerun scenes: 001,005,018" in text


def test_main_exit_status(tmp_path, capsys):
    cache = _cache(tmp_path, ["001"])
    args = ["--cache", str(cache), "--concepts", *CONCEPTS, "--scenes"]
    assert cov_mod.main([*args, "001"]) == 0
    assert cov_mod.main([*args, "001", "002"]) == 1
    assert "rerun scenes: 002" in capsys.readouterr().out
