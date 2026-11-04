"""`longtail demo`: the replay cache unpacks only under cache/ and runs/, only with the published digest."""

import hashlib
import io
import json
import tarfile

import pytest

from longtail import demo
from longtail.cli import main


def _tgz(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_unpack_writes_cache_files_and_checks_the_digest(tmp_path):
    data = _tgz({"cache/sam/responses/abc.txt": b"x", "cache/llm/q.json": b"{}"})
    assert demo.unpack(data, hashlib.sha256(data).hexdigest(), tmp_path) == 2
    assert (tmp_path / "cache/sam/responses/abc.txt").read_bytes() == b"x"
    with pytest.raises(ValueError, match="sha256"):
        demo.unpack(data, "0" * 64, tmp_path)


def test_unpack_takes_the_recorded_search_under_runs(tmp_path):
    data = _tgz({"runs/search-abc/search.json": b"{}", "runs/abc-001/verify_run.json": b"{}"})
    assert demo.unpack(data, "", tmp_path) == 2
    assert (tmp_path / "runs/abc-001/verify_run.json").is_file()


@pytest.mark.parametrize("name", ["../evil.py", "/etc/passwd", "longtail/cli.py", "cache/../x", "runs/../x"])
def test_unpack_refuses_anything_outside_cache_and_runs(tmp_path, name):
    data = _tgz({name: b"x"})
    with pytest.raises(ValueError, match="unexpected entry"):
        demo.unpack(data, "", tmp_path)


def test_demo_without_replay_refuses_to_run(capsys):
    assert main(["demo"]) == 2
    assert "--replay" in capsys.readouterr().out


def test_list_prints_the_recorded_searches(tmp_path, monkeypatch, capsys):
    d = tmp_path / "runs" / "search-abc"
    d.mkdir(parents=True)
    finding = {"scene_id": "001", "t0": 1.0, "t1": 2.5, "score": 0.9, "finding_id": "f1"}
    doc = {"query": "q", "findings": [finding], "rejected": [], "scenes_searched": ["001"]}
    (d / "search.json").write_text(json.dumps(doc))
    monkeypatch.chdir(tmp_path)
    assert main(["demo", "--list"]) == 0
    out = capsys.readouterr().out
    assert "VERIFIED 001 1.00-2.50 score 0.900 (f1)" in out and "1 verified findings" in out


def test_list_without_a_recorded_search_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["demo", "--list"]) == 1
