from longtail.env import load_dotenv


def test_dotenv_sets_missing_keys_only(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# c\nA=1   # note\nexport B='two'\nC=\nKEEP=file\nnot a line\n")
    monkeypatch.setenv("KEEP", "shell")
    for k in ("A", "B", "C"):
        monkeypatch.delenv(k, raising=False)
    added = load_dotenv(f)
    import os

    assert sorted(added) == ["A", "B", "C"]
    assert os.environ["A"] == "1" and os.environ["B"] == "two" and os.environ["C"] == ""
    assert os.environ["KEEP"] == "shell"


def test_missing_file_is_fine(tmp_path):
    assert load_dotenv(tmp_path / "nope") == []
