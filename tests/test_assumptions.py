"""The assumption tags in the code must match docs/assumptions.md."""

import re
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTER = "docs/assumptions.md"
# Files that talk about tags rather than carry them.
NOT_SCANNED = {REGISTER, "tests/test_assumptions.py"}

TAG = re.compile(r"#\s*ASSUMPTION\s+(A-\d{2}):\s+\S")
LOOSE_TAG = re.compile(r"ASSUMPTION\s*A-?\d")
ROW = re.compile(r"^\|\s*(A-\d{2})\s*\|(.*)\|\s*(C|M|both)\s*\|\s*(ASSUMED|CONFIRMED)\s*\|(.*)\|\s*$")


def _repo_files() -> list[str]:
    """Tracked and new files, honouring .gitignore (so the dataset and caches are never scanned)."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted({f for f in out.stdout.splitlines() if (ROOT / f).is_file()} - NOT_SCANNED)


def _register() -> dict[str, set[str]]:
    rows: dict[str, set[str]] = {}
    for line in (ROOT / REGISTER).read_text().splitlines():
        if not line.startswith("| A-"):
            continue
        m = ROW.match(line)
        assert m, f"malformed register row: {line}"
        aid, files = m.group(1), m.group(5).strip()
        assert aid not in rows, f"{aid} appears twice in the register"
        rows[aid] = set() if files == "—" else set(re.findall(r"`([^`]+)`", files))
        assert files == "—" or rows[aid], f"{aid}: files column must be backticked paths or —"
    assert rows, "no rows found in the register"
    return rows


def _tags() -> tuple[dict[str, set[str]], list[str]]:
    used: dict[str, set[str]] = defaultdict(set)
    malformed: list[str] = []
    for path in _repo_files():
        try:
            text = (ROOT / path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            ids = TAG.findall(line)
            for aid in ids:
                used[aid].add(path)
            if len(LOOSE_TAG.findall(line)) != len(ids):
                malformed.append(f"{path}:{n}: {line.strip()}")
    return used, malformed


def test_tags_are_well_formed():
    _, malformed = _tags()
    assert not malformed, "tags must read '# ASSUMPTION A-NN: text':\n" + "\n".join(malformed)


def test_every_tag_is_registered():
    used, _ = _tags()
    unknown = {aid: sorted(files) for aid, files in used.items() if aid not in _register()}
    assert not unknown, f"tags with IDs missing from {REGISTER}: {unknown}"


def test_files_column_matches_tags():
    register = _register()
    used, _ = _tags()
    diffs = []
    for aid, listed in sorted(register.items()):
        actual = used.get(aid, set())
        if listed != actual:
            diffs.append(
                f"{aid}: register lists {sorted(listed) or '—'}, tags appear in {sorted(actual) or '—'}"
            )
    assert not diffs, "update the 'Files where used' column:\n" + "\n".join(diffs)
