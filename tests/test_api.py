"""M6 API, end to end in offline-demo mode: synthetic scenes, the real verifier and C9 export, a scripted
model. Covers the live trace (SSE), finding detail with overlays, cards, review and export."""

import time

import pytest
from fastapi.testclient import TestClient

from api.demo import DEMO_QUERY, demo_client, demo_scenes
from api.server import create_app
from longtail.agent.orchestrator import SearchAgent, open_with


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    run_root = tmp_path_factory.mktemp("work") / "runs"
    scenes, source = demo_scenes()
    ids, opener = list(scenes), open_with(scenes)
    client = demo_client(run_root, ids)

    def factory(on_event):
        return SearchAgent(client, ids, opener, source, run_root, on_event=on_event)

    return TestClient(create_app(run_root, factory, opener, "offline-demo")), run_root


def wait(api_client, job_id, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        j = api_client.get(f"/api/jobs/{job_id}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.2)
    raise TimeoutError(job_id)


@pytest.fixture(scope="module")
def search(api):
    c, _ = api
    job = c.post("/api/searches", json={"query": DEMO_QUERY}).json()
    assert job["mode"] == "offline-demo"
    done = wait(c, job["job_id"])
    assert done["status"] == "done", done["error"]
    return done


def test_health_reports_the_mode(api):
    assert api[0].get("/api/health").json() == {"ok": True, "mode": "offline-demo"}


def test_a_search_streams_its_trace_and_is_listed(api, search):
    c, _ = api
    kinds = [(e["kind"], e["name"]) for e in search["events"]]
    assert (
        kinds[0] == ("plan", "planner") and ("tool", "search_scenes") in kinds and ("tool", "finish") in kinds
    )
    with c.stream("GET", f"/api/jobs/{search['job_id']}/events") as r:
        body = "".join(r.iter_text())
    assert body.count("event: trace") == len(search["events"]) and "event: done" in body
    listed = c.get("/api/searches").json()
    assert listed[0]["search_id"] == search["search_id"] and listed[0]["verified"] >= 1


def test_finding_detail_exports_overlays_bev_and_timeline(api, search):
    c, run_root = api
    s = c.get(f"/api/searches/{search['search_id']}").json()
    fid = s["findings"][0]["finding_id"]
    d = c.get(f"/api/searches/{search['search_id']}/findings/{fid}").json()
    viz = d["viz"]
    assert viz["verified"] and viz["overlays"] and viz["timeline"]["predicates"]
    assert c.get(viz["overlays"][0]["url"]).headers["content-type"] == "image/jpeg"
    bev = c.get(viz["bev_url"]).json()
    assert bev["frame"] == "ego" and bev["frames"]
    assert d["card"]["source"] == "template" and d["card"]["sentences"]


def test_rejected_findings_have_details_too(api, search):
    c, _ = api
    s = c.get(f"/api/searches/{search['search_id']}").json()
    assert s["rejected"]
    fid = s["rejected"][0]["finding_id"]
    d = c.get(f"/api/searches/{search['search_id']}/findings/{fid}").json()
    assert d["viz"] is not None and not d["viz"]["verified"]


def test_card_review_and_export(api, search):
    c, _ = api
    sid = search["search_id"]
    s = c.get(f"/api/searches/{sid}").json()
    good, bad = s["findings"][0]["finding_id"], s["rejected"][0]["finding_id"]
    card = c.post(f"/api/searches/{sid}/findings/{good}/card").json()
    assert card["source"] == "template"  # offline demo never calls a model for cards
    assert (
        c.post(f"/api/searches/{sid}/reviews", json={"finding_id": bad, "decision": "approve"}).status_code
        == 400
    )
    assert (
        c.post(
            f"/api/searches/{sid}/reviews",
            json={"finding_id": good, "decision": "approve", "note": "clear crosser"},
        ).status_code
        == 200
    )
    exported = c.get(f"/api/searches/{sid}/export").json()
    assert [r["finding_id"] for r in exported["approved"]] == [good]
    csv_text = c.get(f"/api/searches/{sid}/export?format=csv").text
    assert (
        csv_text.splitlines()[0].startswith("finding_id,")
        and good in csv_text
        and "clear crosser" in csv_text
    )


def test_unknown_query_gets_the_scripted_clarification(api):
    c, _ = api
    done = wait(c, c.post("/api/searches", json={"query": "a car running a red light"}).json()["job_id"])
    assert done["status"] == "done"
    s = c.get(f"/api/searches/{done['search_id']}").json()
    assert s["stop_reason"] == "clarification" and "offline demo" in s["spec"]["clarification"]


def test_paths_outside_searches_are_refused(api):
    c, _ = api
    assert c.get("/api/searches/..%2F..%2Fetc").status_code == 404
    assert c.get("/api/searches/search-nope").status_code == 404


# Limits (A-253): a bounded number of searches, finished jobs expire, an optional token on every POST.


class _Blocked:
    """An agent whose search waits until released, so a test controls how many are running."""

    def __init__(self, gate) -> None:
        self.gate = gate

    def search(self, query):
        self.gate.wait(10)

        class R:
            search_id = "search-none"

        return R()


def _limited_app(tmp_path, **kw):
    import threading

    gate = threading.Event()
    now = [1000.0]
    app = create_app(
        tmp_path / "runs",
        lambda on_event: _Blocked(gate),
        lambda sid: None,
        "offline-demo",
        clock=lambda: now[0],
        **kw,
    )
    return TestClient(app), gate, now


def test_a_search_over_the_limit_gets_busy_and_a_slot_frees_when_one_ends(tmp_path):
    c, gate, _ = _limited_app(tmp_path, max_searches=1)
    first = c.post("/api/searches", json={"query": "a bus"})
    assert first.status_code == 200
    busy = c.post("/api/searches", json={"query": "a bus"})
    assert busy.status_code == 429 and "busy" in busy.json()["detail"] and busy.headers["retry-after"]
    gate.set()
    wait(c, first.json()["job_id"], timeout=10)
    assert c.post("/api/searches", json={"query": "a bus"}).status_code == 200


def test_finished_jobs_expire(tmp_path):
    c, gate, now = _limited_app(tmp_path, job_ttl_s=60)
    gate.set()
    job = c.post("/api/searches", json={"query": "a bus"}).json()["job_id"]
    assert wait(c, job, timeout=10)["status"] == "done"
    now[0] += 30
    assert c.get(f"/api/jobs/{job}").status_code == 200  # still within its time to live
    now[0] += 31
    assert c.get(f"/api/jobs/{job}").status_code == 404


def test_the_token_is_off_by_default_and_guards_every_post_when_set(tmp_path):
    c, gate, _ = _limited_app(tmp_path, token="s3cret")
    gate.set()
    assert c.get("/api/health").status_code == 200 and c.get("/api/searches").status_code == 200
    assert c.post("/api/searches", json={"query": "a bus"}).status_code == 401
    bad = {"Authorization": "Bearer nope"}
    assert c.post("/api/searches", json={"query": "a bus"}, headers=bad).status_code == 401
    assert (
        c.post("/api/searches/search-x/reviews", json={"finding_id": "f", "decision": "reject"}).status_code
        == 401
    )
    assert c.post("/api/searches/search-x/findings/f/card").status_code == 401
    ok = c.post("/api/searches", json={"query": "a bus"}, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
    alt = c.post("/api/searches", json={"query": "a bus"}, headers={"X-LongTail-Token": "s3cret"})
    assert alt.status_code in (200, 429)  # authorised; it may find the earlier search still running
    # with no token configured, a POST needs nothing (the default local demo)
    open_c, open_gate, _ = _limited_app(tmp_path / "open")
    open_gate.set()
    assert open_c.post("/api/searches", json={"query": "a bus"}).status_code == 200
