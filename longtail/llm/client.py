"""The one client every model call goes through (plan M1): cache, cost ledger, budgets, retries, replay.

Paid calls are off by default. A request reaches the network only when all of these hold:
  1. LONGTAIL_ALLOW_PAID=1 (or allow_paid=True) is set,
  2. a transport exists (MODEL_API_KEY is set),
  3. the response is not already cached and replay mode is off,
  4. the estimated cost fits the per-query budget and the spend cap (LONGTAIL_SPEND_CAP_USD).
Otherwise the client raises before anything is sent, so nothing can be charged by accident.

Calls:
  respond(request)      non-streamed /v1/responses call (planner, cards); returns the output items and text
  stream_text(request)  streamed call whose text deltas are joined (SAM 3.1); billed by frames when given
  upload(path)          /v1/files upload, cached by the file's content hash
"""

import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from longtail.llm.cache import ResponseCache, file_digest, request_key
from longtail.llm.ledger import Ledger

BASE_URL = "https://api.meta.ai/v1"

# Plan section 3 prices. VERIFY the model ids against the console before the first paid call.
# ASSUMPTION A-203: model ids muse-wonder-1.0, muse-spark-1.3 and sam-3.1, priced per plan section 3
TOKEN_PRICES_PER_M: dict[str, tuple[float, float]] = {
    "muse-wonder-1.0": (4.00, 20.00),
    "muse-spark-1.3": (1.25, 4.25),
}
SAM_USD_PER_FRAME = 0.20 / 1000
SAM_MIN_BILLED_FRAMES = 50

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
# ASSUMPTION A-204: up to 4 retries with backoff 1, 2, 4, 8 s on 429 and 5xx; never on 400 (same as A-115)
MAX_RETRIES = 4
BACKOFF_S = 1.0
# ASSUMPTION A-205: pre-call estimate = request chars / 4 input tokens + max_output_tokens (default 4,096)
DEFAULT_MAX_OUTPUT_TOKENS = 4096
# ASSUMPTION A-206: default spend cap $5 until the Meta credits are confirmed; raise it in .env
DEFAULT_SPEND_CAP_USD = 5.0


class TransportError(Exception):
    def __init__(self, status: int, message: str = "") -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


class PaidCallsDisabled(RuntimeError):
    """A call would reach the paid API, but LONGTAIL_ALLOW_PAID is not 1."""


class NoKeyError(RuntimeError):
    """No transport: MODEL_API_KEY is not set."""


class ReplayMissError(LookupError):
    """Replay mode was asked for a response that is not in the cache."""


class BudgetExceeded(RuntimeError):
    """The estimated cost does not fit the per-query budget or the spend cap."""


class Transport(Protocol):
    def upload(self, path: Path) -> str:
        """Upload a file; return its file id."""

    def create(self, request: dict[str, Any]) -> dict[str, Any]:
        """One non-streamed /v1/responses call; return the response as a dict."""

    def stream(self, request: dict[str, Any]) -> Iterator[str]:
        """One streamed /v1/responses call; yield its output text deltas."""


@dataclass
class Budget:
    """A per-query spending limit (plan: LONGTAIL_QUERY_BUDGET_USD, $0.75)."""

    limit_usd: float
    spent_usd: float = 0.0

    @classmethod
    def from_env(cls) -> "Budget":
        return cls(float(os.environ.get("LONGTAIL_QUERY_BUDGET_USD", "0.75")))

    @property
    def remaining_usd(self) -> float:
        return self.limit_usd - self.spent_usd


@dataclass(frozen=True)
class ModelResponse:
    output: list[dict[str, Any]]
    text: str
    usage: dict[str, int]
    usd: float
    cache_hit: bool
    attempts: int
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    def function_calls(self) -> list[dict[str, Any]]:
        return [item for item in self.output if item.get("type") == "function_call"]


def token_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in TOKEN_PRICES_PER_M:
        raise KeyError(f"no price for model {model!r}; add it to TOKEN_PRICES_PER_M first")
    price_in, price_out = TOKEN_PRICES_PER_M[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1e6


def sam_usd(frames: int) -> float:
    return max(frames, SAM_MIN_BILLED_FRAMES) * SAM_USD_PER_FRAME


def estimate_usd(model: str, request: dict[str, Any], frames: int | None = None) -> float:
    if frames is not None:
        return sam_usd(frames)
    input_tokens = len(repr(request)) // 4
    return token_usd(model, input_tokens, int(request.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)))


def output_text(output: list[dict[str, Any]]) -> str:
    parts = []
    for item in output:
        for c in item.get("content") or []:
            if c.get("type") == "output_text":
                parts.append(c.get("text", ""))
    return "".join(parts)


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip() == "1"


class ModelClient:
    def __init__(
        self,
        transport: Transport | None = None,
        *,
        cache: ResponseCache | None = None,
        ledger: Ledger | None = None,
        replay: bool | None = None,
        allow_paid: bool | None = None,
        spend_cap_usd: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.transport = transport
        self.cache = cache or ResponseCache()
        self.ledger = ledger or Ledger()
        self.replay = _flag("LONGTAIL_REPLAY") if replay is None else replay
        self.allow_paid = _flag("LONGTAIL_ALLOW_PAID") if allow_paid is None else allow_paid
        cap = os.environ.get("LONGTAIL_SPEND_CAP_USD")
        self.spend_cap_usd = (
            spend_cap_usd if spend_cap_usd is not None else (float(cap) if cap else DEFAULT_SPEND_CAP_USD)
        )
        self.sleep = sleep

    @classmethod
    def from_env(cls) -> "ModelClient":
        key = os.environ.get("MODEL_API_KEY", "").strip()
        return cls(MetaTransport(key) if key else None)

    # Gates

    def _network(self) -> Transport:
        if self.replay:
            raise ReplayMissError("replay mode makes no network calls")
        if not self.allow_paid:
            raise PaidCallsDisabled(
                "paid calls are off: set LONGTAIL_ALLOW_PAID=1 once the Meta credits are in (or use replay)"
            )
        if self.transport is None:
            raise NoKeyError("MODEL_API_KEY is not set")
        return self.transport

    def _check_budget(self, estimate: float, budget: Budget | None) -> None:
        spent = self.ledger.total_usd()
        if spent + estimate > self.spend_cap_usd:
            raise BudgetExceeded(
                f"estimated ${estimate:.4f} would pass the spend cap "
                f"(${spent:.4f} of ${self.spend_cap_usd:.2f} used)"
            )
        if budget is not None and estimate > budget.remaining_usd:
            raise BudgetExceeded(
                f"estimated ${estimate:.4f} exceeds the query budget (${budget.remaining_usd:.4f} left)"
            )

    def _with_retries(self, op: Callable[[], Any]) -> tuple[Any, int]:
        attempt = 0
        while True:
            attempt += 1
            try:
                return op(), attempt
            except TransportError as e:
                if e.status not in RETRY_STATUS or attempt > MAX_RETRIES:
                    raise
                self.sleep(BACKOFF_S * 2 ** (attempt - 1))

    # Calls

    def upload(self, path: Path) -> str:
        digest = file_digest(path)
        known = self.cache.file_id(digest)
        if known is not None:
            return known
        transport = self._network()
        file_id, _ = self._with_retries(lambda: transport.upload(path))
        self.cache.put_file_id(digest, str(file_id))
        return str(file_id)

    def respond(
        self, request: dict[str, Any], *, purpose: str, budget: Budget | None = None
    ) -> ModelResponse:
        model = request["model"]
        key = request_key(model, request)
        common = {"model": model, "purpose": purpose, "key": key}
        cached = self.cache.get(key)
        if cached is not None:
            self.ledger.record(**common, usd=0.0, cache_hit=True, latency_s=0.0, attempts=0, **_usage(cached))
            return _response(cached, usd=0.0, cache_hit=True, attempts=0)
        if self.replay:
            raise ReplayMissError(f"replay: no cached response for {purpose} (key {key})")
        transport = self._network()
        self._check_budget(estimate_usd(model, request), budget)
        start = time.perf_counter()
        raw, attempts = self._with_retries(lambda: transport.create(request))
        usage = _usage(raw)
        usd = token_usd(model, usage["input_tokens"], usage["output_tokens"])
        self.cache.put(key, raw)
        latency = round(time.perf_counter() - start, 3)
        self.ledger.record(**common, usd=usd, cache_hit=False, latency_s=latency, attempts=attempts, **usage)
        if budget is not None:
            budget.spent_usd += usd
        return _response(raw, usd=usd, cache_hit=False, attempts=attempts)

    def stream_text(
        self,
        request: dict[str, Any],
        *,
        purpose: str,
        frames: int | None = None,
        cache_key: str | None = None,
        budget: Budget | None = None,
    ) -> ModelResponse:
        """A streamed call; frames set means SAM pricing.

        cache_key overrides the request hash (e.g. a clip id, since a file id changes between uploads).
        """
        model = request["model"]
        key = cache_key or request_key(model, request)
        common = {"model": model, "purpose": purpose, "key": key, "frames": frames}
        cached = self.cache.get(key)
        if cached is not None:
            self.ledger.record(**common, usd=0.0, cache_hit=True, latency_s=0.0, attempts=0)
            return ModelResponse([], cached["text"], {}, 0.0, True, 0, cached)
        if self.replay:
            raise ReplayMissError(f"replay: no cached response for {purpose} (key {key})")
        transport = self._network()
        if frames is None and model not in TOKEN_PRICES_PER_M:
            raise KeyError(f"no price for model {model!r}")
        self._check_budget(estimate_usd(model, request, frames), budget)
        start = time.perf_counter()
        text, attempts = self._with_retries(lambda: "".join(transport.stream(request)))
        # ASSUMPTION A-207: a streamed token-priced call is costed at its estimate (deltas carry no usage)
        usd = sam_usd(frames) if frames is not None else estimate_usd(model, request)
        self.cache.put(key, {"text": text})
        latency = round(time.perf_counter() - start, 3)
        self.ledger.record(**common, usd=usd, cache_hit=False, latency_s=latency, attempts=attempts)
        if budget is not None:
            budget.spent_usd += usd
        return ModelResponse([], str(text), {}, usd, False, attempts, {"text": text})


def _usage(raw: dict[str, Any]) -> dict[str, int]:
    u = raw.get("usage") or {}
    return {"input_tokens": int(u.get("input_tokens", 0)), "output_tokens": int(u.get("output_tokens", 0))}


def _response(raw: dict[str, Any], *, usd: float, cache_hit: bool, attempts: int) -> ModelResponse:
    output = list(raw.get("output") or [])
    return ModelResponse(output, output_text(output), _usage(raw), usd, cache_hit, attempts, raw)


class MetaTransport:
    """The Meta Model API through the OpenAI Python SDK (plan 3). Exercised only by the live smoke test.

    VERIFY on the first paid call: the response shape (output items, usage) and the stream event types.
    """

    # ASSUMPTION A-208: the API is OpenAI-compatible: files.create, responses.create and responses.stream
    def __init__(self, api_key: str, base_url: str = BASE_URL) -> None:
        from openai import APIStatusError, OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self._status_error = APIStatusError

    def upload(self, path: Path) -> str:
        try:
            with open(path, "rb") as f:
                return self._client.files.create(file=f, purpose="user_data").id
        except self._status_error as e:
            raise TransportError(e.status_code, str(e)) from e

    def create(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._client.responses.create(**request).model_dump()
        except self._status_error as e:
            raise TransportError(e.status_code, str(e)) from e

    def stream(self, request: dict[str, Any]) -> Iterator[str]:
        try:
            with self._client.responses.stream(**request) as stream:
                deltas = 0
                for event in stream:
                    if event.type == "response.output_text.delta":
                        deltas += 1
                        yield event.delta
                    elif event.type == "response.content_part.done" and deltas == 0:
                        part = getattr(event, "part", None)
                        if getattr(part, "type", None) == "output_text":
                            yield part.text
        except self._status_error as e:
            raise TransportError(e.status_code, str(e)) from e
