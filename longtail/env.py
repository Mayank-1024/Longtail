"""Load .env into the environment (no dependency): KEY=VALUE lines, # comments; the shell's values win."""

import os
from pathlib import Path


def load_dotenv(path: Path | str = ".env") -> list[str]:
    """Set every key in the file that the environment does not already have; returns the keys set."""
    p = Path(path)
    if not p.is_file():
        return []
    added = []
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.split(" #", 1)[0].split("\t#", 1)[0].strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            added.append(key)
    return added
