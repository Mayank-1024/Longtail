"""scripts/overlay.py without the dataset: helpers, and a full run on the fake sequence into tmp_path."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from fake_pandaset import CAMERAS, write_sequence
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("pandaset")


def _load():
    spec = importlib.util.spec_from_file_location("overlay", ROOT / "scripts" / "overlay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


overlay = _load()


def test_pick_frames_takes_busiest_frame_per_third():
    counts = np.array([0, 2, 1, 0, 0, 5, 0, 3, 3])
    assert overlay.pick_frames(counts) == [1, 5, 7]


def test_depth_colours_run_red_to_blue():
    c = overlay.depth_colours(np.array([0.0, overlay.FAR_M, 2 * overlay.FAR_M]))
    assert c[0, 0] > c[0, 2] and c[1, 2] > c[1, 0]
    np.testing.assert_array_equal(c[1], c[2])  # clipped beyond FAR_M


def test_draw_paints_near_points_on_top():
    img = Image.new("RGB", (10, 10))
    out = np.asarray(overlay.draw(img, np.array([[5, 5], [5, 5]]), np.array([40.0, 1.0]), radius=0))
    np.testing.assert_array_equal(out[5, 5], overlay.depth_colours(np.array([1.0]))[0])
    assert out[0, 0].tolist() == [0, 0, 0]


def test_full_run_on_fake_sequence(tmp_path, capsys):
    write_sequence(tmp_path / "pandaset", "001")
    out = tmp_path / "overlays"
    rc = overlay.main(["--root", str(tmp_path / "pandaset"), "--out", str(out)])
    assert rc == 0
    files = sorted(p.name for p in (out / "001").iterdir())
    # 3 frames in the fake sequence; each third holds one frame.
    assert files == sorted(
        [f"{i:02d}_{c}.jpg" for i in range(3) for c in CAMERAS] + [f"{i:02d}_mosaic.jpg" for i in range(3)]
    )
    text = capsys.readouterr().out
    assert "moving objects per picked frame" in text and "camera-to-LiDAR gaps" in text
