"""Spend guards: SAM calls count against the query budget (A-250), and the spend cap holds under concurrency
(A-249). Every transport here is a fake; nothing reaches a paid API."""

import json
import multiprocessing as mp
import threading
import time

import pytest
from sam_wire import FakeSamTransport
from synthetic import ImageScene, SyntheticConfig

from longtail.agent.orchestrator import SearchAgent, open_with
from longtail.contracts import ScenarioSpec
from longtail.llm.cache import ResponseCache
from longtail.llm.client import Budget, BudgetExceeded, ModelClient, estimate_usd
from longtail.llm.ledger import Ledger as SpendLedger
from longtail.pipeline import prefetch
from longtail.vision.clips import build_clip
from longtail.vision.sam_client import GatedTransport, Ledger, SamClient, TransportError, billed_usd
from longtail.vision.track_source import sam_track_source

SPEC = {
    "query": "a pedestrian crossing in front of us while we move",
    "actors": [{"id": "a1", "cls": "pedestrian", "sam_prompt": "person"}],
    "predicates": [
        {"name": "exists", "actors": ["a1"], "args": {}},
        {"name": "crossing_ego_path", "actors": ["a1"], "args": {}},
        {"name": "ego_moving", "actors": [], "args": {}},
    ],
    "assumptions": [],
}


@pytest.fixture
def scene(tmp_path):
    return ImageScene(tmp_path / "images", SyntheticConfig(n_frames=12))


def sam_client(tmp_path, transport) -> SamClient:
    return SamClient(
        transport,
        cache_dir=tmp_path / "cache" / "sam",
        ledger=Ledger(tmp_path / "ledger.jsonl"),
        replay=False,
        sleep=lambda s: None,
    )


# Finding 1: SAM spend counts against the per-query budget


def test_a_sam_call_is_charged_to_the_query_budget_once(scene, tmp_path):
    clip = build_clip(scene, "front_camera", cache_dir=tmp_path / "clips")
    client = sam_client(tmp_path, FakeSamTransport(scene, {"person": ["crosser"]}))
    budget = Budget(1.0)
    first = client.segment(clip, "person", budget)
    again = client.segment(clip, "person", budget)  # a cache hit costs nothing
    assert first.usd == pytest.approx(billed_usd(12)) and again.cache_hit
    assert budget.spent_usd == pytest.approx(billed_usd(12)) and budget.reserved_usd == 0.0


def test_a_sam_call_over_the_query_budget_is_refused_before_anything_is_sent(scene, tmp_path):
    clip = build_clip(scene, "front_camera", cache_dir=tmp_path / "clips")
    transport = FakeSamTransport(scene, {"person": ["crosser"]})
    budget = Budget(billed_usd(12) / 2)
    with pytest.raises(BudgetExceeded, match="query budget"):
        sam_client(tmp_path, transport).segment(clip, "person", budget)
    assert transport.upload_calls == 0 and transport.stream_calls == 0
    assert budget.spent_usd == 0.0 and budget.reserved_usd == 0.0


def test_a_failed_sam_call_releases_its_reservations(scene, tmp_path, monkeypatch):
    monkeypatch.setenv("LONGTAIL_ALLOW_PAID", "1")  # the gate's switch; the inner transport is a fake
    monkeypatch.setenv("LONGTAIL_SPEND_CAP_USD", "1.00")
    clip = build_clip(scene, "front_camera", cache_dir=tmp_path / "clips")
    ledger = tmp_path / "ledger.jsonl"
    gated = GatedTransport(FakeSamTransport(scene, {"person": ["crosser"]}, errors=[400]), ledger)
    budget = Budget(1.0)
    with pytest.raises(TransportError):
        sam_client(tmp_path, gated).segment(clip, "person", budget)
    assert budget.reserved_usd == 0.0 and budget.spent_usd == 0.0
    assert SpendLedger(ledger).reserved_usd() == 0.0 and SpendLedger(ledger).total_usd() == 0.0


def test_a_budgeted_prefetch_makes_the_paid_calls_in_this_process(scene, tmp_path):
    transport = FakeSamTransport(scene, {"person": ["crosser"]})
    source = sam_track_source(sam_client(tmp_path, transport), clip_cache=tmp_path / "clips")
    budget = Budget(5.0)
    bound = source.with_budget(budget)
    assert not hasattr(source, "fetch") and bound.warm.keywords["replay"] is True
    spec = ScenarioSpec.model_validate({**SPEC, "clarification": None})
    prefetch(spec, [scene.scene_id], open_with({scene.scene_id: scene}), bound, workers=2)
    # one call per camera for the one concept, each charged here; the worker processes make none
    # (they open scenes from the dataset, which a synthetic scene is not in, so their decoding fails)
    cameras = len(scene.cameras)
    assert transport.stream_calls == cameras
    assert budget.spent_usd == pytest.approx(cameras * billed_usd(12))


class Scripted:
    def __init__(self, steps) -> None:
        self.steps = list(steps)

    def create(self, request: dict) -> dict:
        step = self.steps.pop(0)
        return {"output": step(request), "usage": {"input_tokens": 2000, "output_tokens": 300}}


def fc(name: str, args: dict, call_id: str) -> dict:
    return {"type": "function_call", "name": name, "arguments": json.dumps(args), "call_id": call_id}


def test_search_usd_includes_the_sam_calls(scene, tmp_path, monkeypatch):
    monkeypatch.setattr("longtail.pipeline.SAM_WORKERS", 0)  # verify() makes the SAM calls itself
    transport = FakeSamTransport(scene, {"person": ["crosser"]})
    sam = sam_client(tmp_path, transport)
    model = ModelClient(
        Scripted(
            [
                lambda r: [fc("emit_scenario_spec", SPEC, "p1")],
                lambda r: [fc("search_scenes", {"scene_ids": [scene.scene_id]}, "s1")],
                lambda r: [fc("finish", {"finding_ids": [], "summary": "done"}, "f1")],
            ]
        ),
        cache=ResponseCache(tmp_path / "cache" / "llm"),
        ledger=SpendLedger(tmp_path / "ledger.jsonl"),
        allow_paid=True,
        replay=False,
        sleep=lambda s: None,
    )
    agent = SearchAgent(
        model,
        [scene.scene_id],
        open_with({scene.scene_id: scene}),
        sam_track_source(sam, clip_cache=tmp_path / "clips"),
        tmp_path / "runs",
    )
    result = agent.search(SPEC["query"], Budget(5.0))
    sam_usd = transport.stream_calls * billed_usd(12)
    model_usd = sum(e.usd for e in result.trace if e.kind in ("plan", "model"))
    assert transport.stream_calls > 0 and sam_usd > 0
    assert result.usd == pytest.approx(model_usd + sam_usd)
    # the same dollars as the shared ledger, and the trace adds up to them
    assert result.usd == pytest.approx(SpendLedger(tmp_path / "ledger.jsonl").total_usd())
    assert result.usd == pytest.approx(sum(e.usd for e in result.trace))
    assert next(e for e in result.trace if e.name == "search_scenes").usd == pytest.approx(sam_usd)


# Finding 5: the spend cap under concurrency


def _reserve_in_a_process(path: str, barrier, results) -> None:
    ledger = SpendLedger(__import__("pathlib").Path(path))
    barrier.wait()
    try:
        rid = ledger.reserve(1.0, cap_usd=3.0)
    except BudgetExceeded:
        results.put("refused")
        return
    time.sleep(0.3)  # the call is in flight: its reservation must keep the others out
    ledger.record(model="sam-3.1", usd=1.0, cache_hit=False, reservation=rid)
    results.put("charged")


def test_the_spend_cap_holds_across_processes_racing_for_it(tmp_path):
    ctx = mp.get_context("spawn")
    n = 6  # the pipeline's default prefetch workers
    barrier, results = ctx.Barrier(n), ctx.Queue()
    path = str(tmp_path / "ledger.jsonl")
    procs = [ctx.Process(target=_reserve_in_a_process, args=(path, barrier, results)) for _ in range(n)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    outcomes = sorted(results.get(timeout=5) for _ in range(n))
    assert outcomes.count("charged") == 3 and outcomes.count("refused") == 3
    ledger = SpendLedger(tmp_path / "ledger.jsonl")
    assert ledger.total_usd() == pytest.approx(3.0) and ledger.reserved_usd() == 0.0


class SlowTransport:
    """Answers after a pause, so concurrent calls overlap while in flight."""

    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    def create(self, request: dict) -> dict:
        with self._lock:
            self.calls += 1
        time.sleep(0.2)
        return {"output": [], "usage": {"input_tokens": 10, "output_tokens": 10}}


def test_concurrent_model_calls_cannot_all_pass_the_spend_cap(tmp_path):
    transport = SlowTransport()
    requests = [{"model": "muse-spark-1.3", "input": f"q{i}", "max_output_tokens": 1000} for i in range(8)]
    estimate = estimate_usd("muse-spark-1.3", requests[0])
    client = ModelClient(
        transport,
        cache=ResponseCache(tmp_path / "cache"),
        ledger=SpendLedger(tmp_path / "ledger.jsonl"),
        allow_paid=True,
        replay=False,
        spend_cap_usd=2.5 * estimate,  # room for two estimates, not three
        sleep=lambda s: None,
    )
    barrier, outcomes = threading.Barrier(8), []

    def one(req):
        barrier.wait()
        try:
            client.respond(req, purpose="race")
            outcomes.append("charged")
        except BudgetExceeded:
            outcomes.append("refused")

    threads = [threading.Thread(target=one, args=(r,)) for r in requests]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert transport.calls == 2 and outcomes.count("charged") == 2 and outcomes.count("refused") == 6
    assert SpendLedger(tmp_path / "ledger.jsonl").reserved_usd() == 0.0


def test_concurrent_sam_calls_cannot_all_pass_the_spend_cap(scene, tmp_path, monkeypatch):
    monkeypatch.setenv("LONGTAIL_ALLOW_PAID", "1")  # the gate's switch; the inner transport is a fake
    monkeypatch.setenv("LONGTAIL_SPEND_CAP_USD", str(2.5 * billed_usd(12)))
    inner = FakeSamTransport(scene, {"person": ["crosser"]})
    real_stream = inner.stream

    def slow_stream(**kw):
        time.sleep(0.2)
        return real_stream(**kw)

    inner.stream = slow_stream
    client = sam_client(tmp_path, GatedTransport(inner, tmp_path / "ledger.jsonl"))
    clip = build_clip(scene, "front_camera", cache_dir=tmp_path / "clips")
    concepts = [f"thing {i}" for i in range(6)]
    barrier, outcomes = threading.Barrier(len(concepts)), []

    def one(concept):
        barrier.wait()
        try:
            client.segment(clip, concept)
            outcomes.append("charged")
        except BudgetExceeded:
            outcomes.append("refused")

    threads = [threading.Thread(target=one, args=(c,)) for c in concepts]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count("charged") == 2 and outcomes.count("refused") == 4
    ledger = SpendLedger(tmp_path / "ledger.jsonl")
    assert ledger.total_usd() == pytest.approx(2 * billed_usd(12)) and ledger.reserved_usd() == 0.0


def test_a_reservation_of_a_dead_process_stops_counting(tmp_path):
    ledger = SpendLedger(tmp_path / "ledger.jsonl")
    rid = ledger.reserve(4.0, cap_usd=5.0)
    held = json.loads(ledger._reserved_path.read_text())
    held[rid]["pid"] = 2**22 + 12345  # no such process
    ledger._reserved_path.write_text(json.dumps(held))
    assert ledger.reserved_usd() == 0.0
    ledger.reserve(4.0, cap_usd=5.0)  # fits again
