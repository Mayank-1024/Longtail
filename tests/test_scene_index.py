"""`longtail scenes ls`, index selection and the download script, all without the dataset."""

import importlib.util
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest
from fake_pandaset import write_sequence

from longtail.cli import main

ROOT = Path(__file__).resolve().parents[1]


def _load_build_index():
    spec = importlib.util.spec_from_file_location("build_index", ROOT / "scripts" / "build_index.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_index = _load_build_index()


def _row(i, night=False, moving=0.8, walking=0, cyclists=0, vehicles=3, pedestrians=None):
    return {
        "scene_id": f"{i:03d}",
        "time_of_day": "night" if night else "day",
        "ego_moving_frac": moving,
        "pedestrians": walking if pedestrians is None else pedestrians,
        "walking": walking,
        "cyclists": cyclists,
        "vehicles": vehicles,
    }


# ASSUMPTION A-32: quotas for night, walking pedestrians with the ego moving, and cyclists
def test_select_fills_quotas_and_is_deterministic():
    rows = (
        [_row(i, night=True, walking=1) for i in range(8)]
        + [_row(i, walking=20 - i) for i in range(8, 22)]
        + [_row(i, cyclists=i) for i in range(22, 30)]
        + [_row(i, walking=1, cyclists=1) for i in range(30, 36)]
        + [_row(i, moving=0.0) for i in range(36, 50)]
    )
    index, phrase, reasons = build_index.select(rows)
    assert (len(index), len(phrase)) == (30, 3)
    # ASSUMPTION A-14: phrase-tuning scenes stay outside the index
    assert not set(index) & set(phrase)
    assert all(rows[int(s)]["time_of_day"] == "day" and rows[int(s)]["cyclists"] > 0 for s in phrase)
    counts = {r: list(reasons.values()).count(r) for r in set(reasons.values())}
    assert counts["night"] == 6 and counts["walking-pedestrians-ego-moving"] == 10 and counts["cyclists"] == 6
    walking = [s for s, r in reasons.items() if r == "walking-pedestrians-ego-moving"]
    assert walking[:3] == ["008", "009", "010"]  # most walking pedestrians first
    assert build_index.select(rows) == (index, phrase, reasons)
    assert build_index.select(rows, seed=1)[1] != phrase or build_index.select(rows, seed=1)[0] != index


def test_select_needs_phrase_tuning_candidates():
    with pytest.raises(ValueError, match="only 0 day scenes"):
        build_index.select([_row(i) for i in range(40)])


def test_scenes_ls_without_index(tmp_path, capsys):
    assert main(["scenes", "ls", "--index", str(tmp_path / "index.json")]) == 1
    assert "No scene index" in capsys.readouterr().err


def test_scenes_ls_lists_index(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("LONGTAIL_DATA_DIR", str(tmp_path))
    write_sequence(tmp_path / "pandaset", "001")
    index = tmp_path / "index.json"
    entry = {"n_frames": 80, "time_of_day": "day", "ego_moving_frac": 0.75}
    index.write_text(
        json.dumps(
            {
                "scenes": [{"scene_id": "001", **entry}, {"scene_id": "002", **entry}],
                "phrase_tuning_scenes": [{"scene_id": "003", **entry, "time_of_day": "night"}],
            }
        )
    )
    assert main(["scenes", "ls", "--index", str(index)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[1].split() == ["001", "index", "day", "75%", "yes"]
    assert out[3].split() == ["003", "phrase-tuning", "night", "75%", "no"]
    assert out[-1] == "2 indexed, 1 phrase-tuning, 1 on disk"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_download_script_unpacks_nested_archive(tmp_path):
    write_sequence(tmp_path / "src" / "pandaset" / "nested", "004")
    archive = tmp_path / "pandaset.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(tmp_path / "src", arcname="release")
    env = {"LONGTAIL_DATA_DIR": str(tmp_path / "data"), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
    script = ROOT / "scripts" / "download_data.sh"
    for expected in ("added 1 sequences, skipped 0", "added 0 sequences, skipped 1"):
        out = subprocess.run(["bash", str(script), str(archive)], env=env, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert expected in out.stdout
    assert (tmp_path / "data" / "pandaset" / "004" / "lidar" / "00.pkl.gz").is_file()
    bad = subprocess.run(
        ["bash", str(script), str(archive), "0" * 64], env=env, capture_output=True, text=True
    )
    assert bad.returncode == 1 and "sha256 mismatch" in bad.stderr
