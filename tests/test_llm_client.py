"""M1 client: every gate that keeps a call off the paid API, plus cache, ledger, retries and budgets.

A fake transport records every call, so a test that expects no network call checks it really made none.
"""

from pathlib import Path

import pytest

from longtail.llm.cache import ResponseCache
from longtail.llm.client import (
    Budget,
    BudgetExceeded,
    ModelClient,
    NoKeyError,
    PaidCallsDisabled,
    ReplayMissError,
    TransportError,
    sam_usd,
    token_usd,
)
from longtail.llm.ledger import Ledger

REQ = {"model": "muse-spark-1.3", "input": "hello", "max_output_tokens": 100}


class FakeTransport:
    def __init__(self, fail: list[int] | None = None, text: str = "ok") -> None:
        self.fail = list(fail or [])
        self.text = text
        self.creates: list[dict] = []
        self.streams: list[dict] = []
        self.uploads: list[Path] = []

    def _maybe_fail(self) -> None:
        if self.fail:
            raise TransportError(self.fail.pop(0), "fake")

    def upload(self, path: Path) -> str:
        self._maybe_fail()
        self.uploads.append(path)
        return f"file-{len(self.uploads)}"

    def create(self, request: dict) -> dict:
        self._maybe_fail()
        self.creates.append(request)
        return {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": self.text}]}],
            "usage": {"input_tokens": 1000, "output_tokens": 200},
        }

    def stream(self, request: dict):
        self._maybe_fail()
        self.streams.append(request)
        yield from ["<1f>", " a", " b"]


def make(tmp_path: Path, transport=None, **kw) -> ModelClient:
    kw.setdefault("allow_paid", True)
    kw.setdefault("replay", False)
    return ModelClient(
        transport,
        cache=ResponseCache(tmp_path / "cache"),
        ledger=Ledger(tmp_path / "ledger.jsonl"),
        sleep=lambda s: None,
        **kw,
    )


def test_paid_calls_are_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("LONGTAIL_ALLOW_PAID", raising=False)
    t = FakeTransport()
    client = ModelClient(t, cache=ResponseCache(tmp_path / "c"), ledger=Ledger(tmp_path / "l.jsonl"))
    with pytest.raises(PaidCallsDisabled):
        client.respond(REQ, purpose="test")
    with pytest.raises(PaidCallsDisabled):
        client.stream_text({"model": "sam-3.1"}, purpose="sam", frames=80)
    assert t.creates == [] and t.streams == [] and t.uploads == []
    assert client.ledger.entries() == []


def test_no_key_means_no_call(tmp_path):
    with pytest.raises(NoKeyError):
        make(tmp_path, None).respond(REQ, purpose="test")


def test_second_call_is_a_free_cache_hit(tmp_path):
    t = FakeTransport(text="spec")
    client = make(tmp_path, t)
    first = client.respond(REQ, purpose="planner")
    second = client.respond(REQ, purpose="planner")
    assert len(t.creates) == 1
    assert first.text == second.text == "spec"
    assert not first.cache_hit and second.cache_hit and second.usd == 0.0
    assert first.usd == pytest.approx(token_usd("muse-spark-1.3", 1000, 200))
    lines = client.ledger.entries()
    assert [e["cache_hit"] for e in lines] == [False, True]
    assert client.ledger.total_usd() == pytest.approx(first.usd)


def test_replay_serves_the_cache_and_fails_on_a_miss(tmp_path):
    t = FakeTransport()
    make(tmp_path, t).respond(REQ, purpose="planner")
    replay = make(tmp_path, t, replay=True)
    assert replay.respond(REQ, purpose="planner").cache_hit
    with pytest.raises(ReplayMissError):
        replay.respond({**REQ, "input": "new"}, purpose="planner")
    assert len(t.creates) == 1


def test_retries_429_and_5xx_with_backoff(tmp_path):
    waits: list[float] = []
    t = FakeTransport(fail=[429, 503])
    client = make(tmp_path, t)
    client.sleep = waits.append
    resp = client.respond(REQ, purpose="planner")
    assert resp.attempts == 3 and waits == [1.0, 2.0]


def test_never_retries_400(tmp_path):
    t = FakeTransport(fail=[400])
    with pytest.raises(TransportError) as e:
        make(tmp_path, t).respond(REQ, purpose="planner")
    assert e.value.status == 400 and t.fail == []


def test_gives_up_after_four_retries(tmp_path):
    t = FakeTransport(fail=[500] * 5)
    with pytest.raises(TransportError):
        make(tmp_path, t).respond(REQ, purpose="planner")
    assert t.creates == []


def test_spend_cap_refuses_before_sending(tmp_path):
    t = FakeTransport()
    client = make(tmp_path, t, spend_cap_usd=0.0001)
    with pytest.raises(BudgetExceeded):
        client.respond(REQ, purpose="planner")
    assert t.creates == []


def test_spend_cap_counts_earlier_spend_in_the_ledger(tmp_path):
    Ledger(tmp_path / "ledger.jsonl").record(model="muse-spark-1.3", usd=4.9999, cache_hit=False)
    t = FakeTransport()
    with pytest.raises(BudgetExceeded):
        make(tmp_path, t, spend_cap_usd=5.0).respond(REQ, purpose="planner")
    assert t.creates == []


def test_query_budget_is_charged_and_enforced(tmp_path):
    t = FakeTransport()
    client = make(tmp_path, t)
    budget = Budget(limit_usd=0.01)
    resp = client.respond(REQ, purpose="planner", budget=budget)
    assert budget.spent_usd == pytest.approx(resp.usd)
    with pytest.raises(BudgetExceeded):
        client.respond({**REQ, "input": "x" * 40_000, "max_output_tokens": 4000}, purpose="p", budget=budget)


def test_sam_stream_is_joined_billed_by_frames_and_cached_by_key(tmp_path):
    t = FakeTransport()
    client = make(tmp_path, t)
    req = {"model": "sam-3.1", "input": [{"file_id": "file-1"}]}
    first = client.stream_text(req, purpose="sam", frames=80, cache_key="clip-abc|person")
    again = client.stream_text(
        {**req, "input": [{"file_id": "file-2"}]}, purpose="sam", frames=80, cache_key="clip-abc|person"
    )
    assert first.text == "<1f> a b" and first.usd == pytest.approx(sam_usd(80))
    assert again.cache_hit and len(t.streams) == 1
    assert sam_usd(10) == pytest.approx(50 * 0.20 / 1000)


def test_upload_happens_once_per_file_content(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00" * 64)
    t = FakeTransport()
    client = make(tmp_path, t)
    assert client.upload(clip) == client.upload(clip) == "file-1"
    assert len(t.uploads) == 1


def test_unknown_model_has_no_price(tmp_path):
    with pytest.raises(KeyError):
        make(tmp_path, FakeTransport()).respond({**REQ, "model": "mystery"}, purpose="p")
