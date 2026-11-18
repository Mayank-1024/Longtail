"""Cost ledger (plan M1): one JSON line per model call, cached or not, in runs/ledger.jsonl.

The same file and line shape as the interim SAM client (A-116), so every cost adds up in one place.

Spend cap without a race (A-249): a paid call first reserves its estimate under a file lock, so the check and
the reservation are one step, then settles by appending its real cost and dropping the reservation in one
step under the same lock. Threads, prefetch processes and concurrent searches therefore never all pass the
check on the same remaining dollars.
"""

# ASSUMPTION A-201: one ledger for every model call at runs/ledger.jsonl; a cache hit is logged at $0

import json
import os
import socket
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

try:  # POSIX (macOS, Linux, WSL2); without it the lock covers this process only
    import fcntl
except ImportError:  # pragma: no cover - native Windows
    fcntl = None  # type: ignore[assignment]

DEFAULT_LEDGER = Path("runs") / "ledger.jsonl"
# ASSUMPTION A-249: a reservation held by a dead process on this host, or older than an hour, no longer counts
RESERVATION_TTL_S = 3600.0

_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class BudgetExceeded(RuntimeError):
    """The estimated cost does not fit the per-query budget or the spend cap."""


def ledger_path() -> Path:
    # ASSUMPTION A-234: LONGTAIL_LEDGER moves the ledger, so each cluster task has its own file and cap:
    # concurrent appends to one file on a network filesystem can interleave lines
    return Path(os.environ.get("LONGTAIL_LEDGER") or DEFAULT_LEDGER)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by another user
        return True
    return True


class Ledger:
    """Append-only JSON lines: model, purpose, tokens or frames, dollars, cache hit, latency, attempts."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else ledger_path()

    # Locking and reservations (A-249)

    @property
    def _reserved_path(self) -> Path:
        return self.path.with_name(self.path.name + ".reserved.json")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """One writer at a time: a thread lock within this process, a file lock across processes."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        key = str(self.path.resolve())
        with _THREAD_LOCKS_GUARD:
            tlock = _THREAD_LOCKS.setdefault(key, threading.Lock())
        with tlock, open(self.path.with_name(self.path.name + ".lock"), "a") as fh:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def _read_reserved(self) -> dict[str, dict]:
        try:
            held = json.loads(self._reserved_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        host, now = socket.gethostname(), time.time()
        return {
            rid: r
            for rid, r in held.items()
            if now - float(r.get("at", 0)) < RESERVATION_TTL_S
            and (r.get("host") != host or _pid_alive(int(r.get("pid", 0))))
        }

    def _write_reserved(self, held: dict[str, dict]) -> None:
        tmp = self._reserved_path.with_name(f"{self._reserved_path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(held))
        tmp.replace(self._reserved_path)

    def reserved_usd(self) -> float:
        """Dollars held by calls that have reserved but not yet settled."""
        return float(sum(r["usd"] for r in self._read_reserved().values()))

    def reserve(self, usd: float, cap_usd: float) -> str:
        """Hold usd against the cap, in the same locked step as the check; returns the reservation id.

        Raises BudgetExceeded, holding nothing, when spent plus reserved plus usd would pass the cap.
        """
        with self._locked():
            held = self._read_reserved()
            spent = self.total_usd()
            in_flight = float(sum(r["usd"] for r in held.values()))
            if spent + in_flight + usd > cap_usd:
                raise BudgetExceeded(
                    f"estimated ${usd:.4f} would pass the spend cap "
                    f"(${spent:.4f} spent and ${in_flight:.4f} reserved, of ${cap_usd:.2f})"
                )
            rid = uuid.uuid4().hex
            held[rid] = {"usd": usd, "pid": os.getpid(), "host": socket.gethostname(), "at": time.time()}
            self._write_reserved(held)
            return rid

    def release(self, reservation: str | None) -> None:
        """Drop a reservation whose call was never charged (it failed before a response came back)."""
        if reservation is None:
            return
        with self._locked():
            held = self._read_reserved()
            if held.pop(reservation, None) is not None:
                self._write_reserved(held)

    # Lines

    def record(self, *, reservation: str | None = None, **entry: object) -> None:
        """Append one line; with a reservation, the line replaces it in the same locked step (settle)."""
        with self._locked():
            with open(self.path, "a") as f:
                f.write(json.dumps({"logged_at": time.time(), **entry}) + "\n")
            if reservation is not None:
                held = self._read_reserved()
                if held.pop(reservation, None) is not None:
                    self._write_reserved(held)

    def entries(self) -> list[dict]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def total_usd(self) -> float:
        """Dollars spent on real calls (cache hits are $0)."""
        return float(sum(e.get("usd", 0.0) for e in self.entries()))
