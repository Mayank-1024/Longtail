"""Cost ledger (plan M1): one JSON line per model call, cached or not, in runs/ledger.jsonl.

The same file and line shape as the interim SAM client (A-116), so every cost adds up in one place.
"""

# ASSUMPTION A-201: one ledger for every model call at runs/ledger.jsonl; a cache hit is logged at $0

import json
import os
import time
from pathlib import Path

DEFAULT_LEDGER = Path("runs") / "ledger.jsonl"


def ledger_path() -> Path:
    # ASSUMPTION A-234: LONGTAIL_LEDGER moves the ledger, so each cluster task has its own file and cap:
    # concurrent appends to one file on a network filesystem can interleave lines
    return Path(os.environ.get("LONGTAIL_LEDGER") or DEFAULT_LEDGER)


class Ledger:
    """Append-only JSON lines: model, purpose, tokens or frames, dollars, cache hit, latency, attempts."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else ledger_path()

    def record(self, **entry: object) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps({"logged_at": time.time(), **entry}) + "\n")

    def entries(self) -> list[dict]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def total_usd(self) -> float:
        """Dollars spent on real calls (cache hits are $0)."""
        return float(sum(e.get("usd", 0.0) for e in self.entries()))
