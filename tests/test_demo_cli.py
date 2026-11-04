"""`longtail demo`: the replay cache unpacks only under cache/, and only with the published digest."""

import hashlib
import io
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


@pytest.mark.parametrize("name", ["../evil.py", "/etc/passwd", "longtail/cli.py", "cache/../x"])
def test_unpack_refuses_anything_outside_cache(tmp_path, name):
    data = _tgz({name: b"x"})
    with pytest.raises(ValueError, match="unexpected entry"):
        demo.unpack(data, "", tmp_path)


def test_demo_without_replay_refuses_to_run(capsys):
    assert main(["demo"]) == 2
    assert "--replay" in capsys.readouterr().out
