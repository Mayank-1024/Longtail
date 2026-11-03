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
