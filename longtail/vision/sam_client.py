"""SAM 3.1 client (plan C3): upload a clip once, one streamed request per (clip, concept), retries, ledger.

Mayank's llm/client.py (M1) does not exist yet, so this module defines the minimal interface it needs: a
transport that uploads a file and streams a response's text, a response cache, replay mode and a cost ledger.
When M1 lands, MetaTransport and the cache and ledger here should give way to his client; its caching then
replaces this one (no second layer).
"""

# ASSUMPTION A-114: interim SAM client until llm/client.py exists: transport (upload, stream), cache, ledger

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from longtail.vision.clips import Clip, cache_root

SAM_MODEL = "sam-3.1"
# Plan 3: $0.20 per 1,000 frames, 50-frame minimum per request.
USD_PER_FRAME = 0.20 / 1000
MIN_BILLED_FRAMES = 50
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
# ASSUMPTION A-115: up to 4 retries, backoff 1 s doubling (1, 2, 4, 8 s), on 429 and 5xx only
MAX_RETRIES = 4
BACKOFF_S = 1.0


class TransportError(Exception):
    def __init__(self, status: int, message: str = "") -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


class ReplayMissError(LookupError):
    """Replay mode was asked for a response that is not in the cache."""


class NoKeyError(RuntimeError):
    """No transport: MODEL_API_KEY is not set."""


class SamTransport(Protocol):
    def upload(self, path: Path) -> str:
        """Upload a clip; return its file id."""

    def stream(self, *, model: str, file_id: str, concept: str, include_confidence: bool) -> Iterator[str]:
        """Start one streamed request and yield its output text deltas."""


def billed_usd(frames: int) -> float:
    return max(frames, MIN_BILLED_FRAMES) * USD_PER_FRAME


class Ledger:
    """Append-only JSON lines of every model call: model, frames, dollars, cache hit, latency."""

    # ASSUMPTION A-116: ledger lines go to runs/ledger.jsonl, the file the plan gives M1's client
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else Path("runs") / "ledger.jsonl"

    def record(self, **entry: object) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps({"logged_at": time.time(), **entry}) + "\n")

    def entries(self) -> list[dict]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]


@dataclass(frozen=True)
class SamCall:
    text: str
    clip_id: str
    concept: str
    frames: int
    usd: float
    cache_hit: bool
    attempts: int


class SamClient:
    def __init__(
        self,
        transport: SamTransport | None,
        *,
        cache_dir: Path | None = None,
        ledger: Ledger | None = None,
        replay: bool | None = None,
        include_confidence: bool = True,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.transport = transport
        self.cache_dir = cache_dir or cache_root() / "sam"
        self.ledger = ledger or Ledger()
        # LONGTAIL_REPLAY=1 (.env.example): serve every call from the cache and fail on a miss.
        self.replay = os.environ.get("LONGTAIL_REPLAY") == "1" if replay is None else replay
        self.include_confidence = include_confidence
        self.sleep = sleep

    # Retries

    def _with_retries(self, op: Callable[[], object]) -> tuple[object, int]:
        attempt = 0
        while True:
            attempt += 1
            try:
                return op(), attempt
            except TransportError as e:
                if e.status not in RETRY_STATUS or attempt > MAX_RETRIES:
                    raise
                self.sleep(BACKOFF_S * 2 ** (attempt - 1))

    # File ids

    def _files_path(self) -> Path:
        return self.cache_dir / "file_ids.json"

    def file_id(self, clip: Clip) -> str:
        """Upload a clip once; later calls reuse the cached file id."""
        known = json.loads(self._files_path().read_text()) if self._files_path().is_file() else {}
        if clip.index.clip_id in known:
            return known[clip.index.clip_id]
        transport = self._transport()
        file_id, _ = self._with_retries(lambda: transport.upload(clip.path))
        known[clip.index.clip_id] = file_id
        self._files_path().parent.mkdir(parents=True, exist_ok=True)
        self._files_path().write_text(json.dumps(known, indent=1))
        return str(file_id)

    def _transport(self) -> SamTransport:
        if self.replay:
            raise ReplayMissError("replay mode makes no network calls")
        if self.transport is None:
            raise NoKeyError("MODEL_API_KEY is not set: SAM 3.1 calls need a Meta API key (or use replay)")
        return self.transport

    # Requests

    def _key(self, clip: Clip, concept: str) -> str:
        raw = json.dumps([SAM_MODEL, clip.index.clip_id, concept, self.include_confidence])
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def segment(self, clip: Clip, concept: str) -> SamCall:
        """One streamed request for one concept on one clip, served from the cache when possible."""
        frames = len(clip.index.frames)
        cached = self.cache_dir / "responses" / f"{self._key(clip, concept)}.txt"
        common = {
            "model": SAM_MODEL,
            "kind": "sam",
            "clip_id": clip.index.clip_id,
            "concept": concept,
            "scene_id": clip.index.scene_id,
            "camera": clip.index.camera,
            "frames": frames,
        }
        if cached.is_file():
            self.ledger.record(**common, usd=0.0, cache_hit=True, latency_s=0.0, attempts=0)
            return SamCall(cached.read_text(), clip.index.clip_id, concept, frames, 0.0, True, 0)
        if self.replay:
            raise ReplayMissError(
                f"replay: no cached SAM response for {clip.index.scene_id}/{clip.index.camera} "
                f"concept {concept!r} (clip {clip.index.clip_id}); expected {cached}"
            )
        file_id = self.file_id(clip)
        transport = self._transport()
        start = time.perf_counter()

        def run() -> str:
            return "".join(
                transport.stream(
                    model=SAM_MODEL,
                    file_id=file_id,
                    concept=concept,
                    include_confidence=self.include_confidence,
                )
            )

        text, attempts = self._with_retries(run)
        usd = billed_usd(frames)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(str(text))
        latency = time.perf_counter() - start
        self.ledger.record(**common, usd=usd, cache_hit=False, latency_s=round(latency, 3), attempts=attempts)
        return SamCall(str(text), clip.index.clip_id, concept, frames, usd, False, attempts)


class MetaTransport:
    """The Meta Model API through the OpenAI Python SDK (plan 3.1). Untested until the smoke test.

    VERIFY on the smoke test: the include_confidence placement (metadata) and the event types.
    """

    # ASSUMPTION A-117: the real transport follows plan 3.1 (files.create, responses.stream, metadata)
    def __init__(self, api_key: str, base_url: str = "https://api.meta.ai/v1") -> None:
        try:
            from openai import APIStatusError, OpenAI
        except ImportError as e:  # not a project dependency until M1 adds it
            raise RuntimeError("the openai package is needed: uv run --with openai ...") from e
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._status_error = APIStatusError

    def upload(self, path: Path) -> str:
        try:
            with open(path, "rb") as f:
                return self._client.files.create(file=f, purpose="user_data").id
        except self._status_error as e:
            raise TransportError(e.status_code, str(e)) from e

    def stream(self, *, model: str, file_id: str, concept: str, include_confidence: bool) -> Iterator[str]:
        content = [{"type": "input_text", "text": concept}, {"type": "input_video", "file_id": file_id}]
        metadata = {"include_confidence": "true"} if include_confidence else {}
        try:
            with self._client.responses.stream(
                model=model,
                input=[{"type": "message", "role": "user", "content": content}],
                metadata=metadata,
            ) as stream:
                deltas = 0
                for event in stream:
                    if event.type == "response.output_text.delta":
                        deltas += 1
                        yield event.delta
                    elif event.type == "response.content_part.done" and deltas == 0:
                        # meta-sam-parser notes the live API finalises lanes with content_part.done.
                        part = getattr(event, "part", None)
                        if getattr(part, "type", None) == "output_text":
                            yield part.text
        except self._status_error as e:
            raise TransportError(e.status_code, str(e)) from e


# ASSUMPTION A-232: real SAM calls obey the M1 paid-call gate and spend cap (A-206) over the shared ledger;
# each stream is costed up front at one 80-frame PandaSet clip, the most a clip here is billed
SAM_CALL_ESTIMATE_USD = billed_usd(80)


class GatedTransport:
    """The real transport behind LONGTAIL_ALLOW_PAID=1 and LONGTAIL_SPEND_CAP_USD, like every M1 call."""

    def __init__(self, inner: SamTransport, ledger_path: Path | None = None) -> None:
        self.inner, self.ledger_path = inner, ledger_path

    def _check(self, estimate: float) -> None:
        from longtail.llm.client import DEFAULT_SPEND_CAP_USD, BudgetExceeded, PaidCallsDisabled
        from longtail.llm.ledger import Ledger as SpendLedger

        if os.environ.get("LONGTAIL_ALLOW_PAID") != "1":
            raise PaidCallsDisabled("paid calls are off: set LONGTAIL_ALLOW_PAID=1 (or use replay)")
        cap = float(os.environ.get("LONGTAIL_SPEND_CAP_USD") or DEFAULT_SPEND_CAP_USD)
        spent = SpendLedger(self.ledger_path).total_usd()
        if spent + estimate > cap:
            raise BudgetExceeded(
                f"SAM call of ${estimate:.4f} would pass the spend cap (${spent:.4f} of ${cap:.2f})"
            )

    def upload(self, path: Path) -> str:
        self._check(0.0)
        return self.inner.upload(path)

    def stream(self, **kwargs) -> Iterator[str]:
        self._check(SAM_CALL_ESTIMATE_USD)
        return self.inner.stream(**kwargs)


def transport_from_env() -> SamTransport | None:
    key = os.environ.get("MODEL_API_KEY", "").strip()
    return GatedTransport(MetaTransport(key)) if key else None
