"""LongTail API for the web view (plan M6): searches with a live trace, findings, cards, review, export.

Run:  uv run uvicorn api.server:create_app_from_env --factory --port 8000
Modes (LONGTAIL_MODE):
  live          the real agent: Meta client, scene index, SAM track source (C3); paid calls still need
                LONGTAIL_ALLOW_PAID=1
  replay        the same, served only from the cache (LONGTAIL_REPLAY=1): the deterministic demo
  offline-demo  synthetic scenes and a SCRIPTED model (api/demo.py), for building the UI; never for footage

Endpoints:
  GET  /api/health
  GET  /api/searches                              saved searches, newest first
  POST /api/searches {query}                      start a search; returns {job_id}
  GET  /api/jobs/{job_id}                         status, events so far, search_id when done
  GET  /api/jobs/{job_id}/events                  server-sent events: one "trace" per step, then "done"
  GET  /api/searches/{sid}                        the search with its reviews
  GET  /api/searches/{sid}/findings/{fid}         finding, spec, card, viz assets (exported on first view)
  POST /api/searches/{sid}/findings/{fid}/card    write the card (Spark Open if allowed, else the template)
  POST /api/searches/{sid}/reviews {finding_id, decision, note}
  GET  /api/searches/{sid}/export?format=json|csv approved findings
  /files/...                                      everything under the runs directory (overlays, bev.json)

Limits (A-253), all from the environment:
  LONGTAIL_API_MAX_SEARCHES  searches running at once (default 2); one more gets 429 "busy"
  LONGTAIL_API_JOB_TTL_S     a finished job's status and trace are kept this long (default 3600 s), then
                             dropped; at most MAX_KEPT_JOBS finished jobs are kept in any case
  LONGTAIL_API_TOKEN         unset (the default, the local demo): no check. Set: every POST (the requests
                             that spend money or write state) needs the header "Authorization: Bearer <token>"
                             or "X-LongTail-Token: <token>"; the web view sends it when opened as /?token=...
"""

import asyncio
import csv
import hmac
import io
import json
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from longtail.agent.orchestrator import SearchAgent, TraceEvent, open_with
from longtail.contracts import Finding, ScenarioSpec
from longtail.data.base import Scene
from longtail.env import load_dotenv
from longtail.explain.cards import make_card
from longtail.llm.client import BudgetExceeded, NoKeyError, PaidCallsDisabled, ReplayMissError
from longtail.pipeline import SceneOpener
from longtail.verify.run import load_run
from longtail.viz.export import export_run

Mode = Literal["live", "replay", "offline-demo"]
AgentFactory = Callable[[Callable[[TraceEvent], None]], SearchAgent]
GATED = (PaidCallsDisabled, NoKeyError, ReplayMissError, BudgetExceeded)
WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"
# ASSUMPTION A-253: at most 2 searches at once, finished jobs kept 1 h (and at most 200), no token by default
DEFAULT_MAX_SEARCHES = 2
DEFAULT_JOB_TTL_S = 3600.0
MAX_KEPT_JOBS = 200


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)


class ReviewRequest(BaseModel):
    finding_id: str
    decision: Literal["approve", "reject"]
    note: str = Field(default="", max_length=500)


@dataclass
class Job:
    job_id: str
    query: str
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "running"  # running | done | error
    search_id: str | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    finished_at: float | None = None


def _event_json(e: TraceEvent) -> dict[str, Any]:
    return json.loads(json.dumps(e.__dict__, default=str))


def create_app(
    run_root: Path,
    agent_factory: AgentFactory,
    open_scene: SceneOpener,
    mode: Mode,
    card_client: Callable[[], Any] | None = None,
    *,
    max_searches: int = DEFAULT_MAX_SEARCHES,
    job_ttl_s: float = DEFAULT_JOB_TTL_S,
    token: str | None = None,
    clock: Callable[[], float] = time.time,
) -> FastAPI:
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="LongTail", version="0.1")
    jobs: dict[str, Job] = {}
    jobs_lock = threading.Lock()
    slots = threading.BoundedSemaphore(max(1, max_searches))

    def require_token(
        authorization: str | None = Header(default=None), x_longtail_token: str | None = Header(default=None)
    ) -> None:
        """Off unless a token is configured; then every POST must carry it."""
        if not token:
            return
        given = x_longtail_token or ""
        if authorization and authorization.lower().startswith("bearer "):
            given = authorization[7:].strip()
        if not hmac.compare_digest(given.encode(), token.encode()):
            raise HTTPException(401, "a valid API token is required")

    guarded = [Depends(require_token)]

    def prune_jobs() -> None:
        """Drop finished jobs past their time to live, and the oldest beyond MAX_KEPT_JOBS (caller locks)."""
        now = clock()
        finished = sorted(
            (j for j in jobs.values() if j.finished_at is not None), key=lambda j: j.finished_at or 0.0
        )
        for j in finished:
            if now - (j.finished_at or now) > job_ttl_s:
                jobs.pop(j.job_id, None)
        finished = [j for j in finished if j.job_id in jobs]
        for j in finished[: max(0, len(finished) - MAX_KEPT_JOBS)]:
            jobs.pop(j.job_id, None)

    def search_dir(sid: str) -> Path:
        d = (run_root / sid).resolve()
        if (
            d.parent != run_root.resolve()
            or not sid.startswith("search-")
            or not (d / "search.json").is_file()
        ):
            raise HTTPException(404, f"no search {sid}")
        return d

    def load_search(sid: str) -> dict[str, Any]:
        return json.loads((search_dir(sid) / "search.json").read_text())

    def reviews(sid: str) -> dict[str, dict[str, Any]]:
        f = search_dir(sid) / "reviews.jsonl"
        latest: dict[str, dict[str, Any]] = {}
        if f.is_file():
            for line in f.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    latest[r["finding_id"]] = r
        return latest

    def find(search: dict[str, Any], fid: str) -> Finding:
        for f in search["findings"] + search["rejected"]:
            if f["finding_id"] == fid:
                return Finding.model_validate(f)
        raise HTTPException(404, f"no finding {fid} in this search")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "mode": mode}

    @app.get("/api/searches")
    def list_searches() -> list[dict[str, Any]]:
        out = []
        for f in sorted(run_root.glob("search-*/search.json"), key=lambda p: -p.stat().st_mtime):
            s = json.loads(f.read_text())
            out.append(
                {
                    "search_id": f.parent.name,
                    "query": s["query"],
                    "stop_reason": s["stop_reason"],
                    "verified": len(s["findings"]),
                    "rejected": len(s["rejected"]),
                    "usd": s["usd"],
                    "scenes_searched": len(s["scenes_searched"]),
                    "modified": f.stat().st_mtime,
                }
            )
        return out

    @app.post("/api/searches", dependencies=guarded)
    def start_search(req: SearchRequest) -> dict[str, Any]:
        if not slots.acquire(blocking=False):
            raise HTTPException(
                429,
                f"busy: {max(1, max_searches)} searches are already running; try again shortly",
                headers={"Retry-After": "30"},
            )
        job = Job(uuid.uuid4().hex[:12], req.query)
        with jobs_lock:
            prune_jobs()
            jobs[job.job_id] = job

        def on_event(e: TraceEvent) -> None:
            with job.lock:
                job.events.append(_event_json(e))

        def work() -> None:
            try:
                result = agent_factory(on_event).search(req.query)
                job.search_id = result.search_id
                job.status = "done"
            except Exception as e:  # reported to the client; the server keeps running
                job.error, job.status = f"{type(e).__name__}: {e}", "error"
            finally:
                job.finished_at = clock()
                slots.release()

        try:
            threading.Thread(target=work, daemon=True).start()
        except BaseException:
            slots.release()
            with jobs_lock:
                jobs.pop(job.job_id, None)
            raise
        return {"job_id": job.job_id, "mode": mode}

    def get_job(job_id: str) -> Job:
        with jobs_lock:
            prune_jobs()
            job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"no job {job_id} (unknown, or finished and expired)")
        return job

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict[str, Any]:
        job = get_job(job_id)
        with job.lock:
            events = list(job.events)
        return {
            "job_id": job_id,
            "status": job.status,
            "search_id": job.search_id,
            "error": job.error,
            "events": events,
        }

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str) -> StreamingResponse:
        job = get_job(job_id)

        async def stream():
            sent = 0
            while True:
                with job.lock:
                    new, status = job.events[sent:], job.status
                for e in new:
                    yield f"event: trace\ndata: {json.dumps(e)}\n\n"
                sent += len(new)
                if status != "running" and not new:
                    done = {"status": status, "search_id": job.search_id, "error": job.error}
                    yield f"event: done\ndata: {json.dumps(done)}\n\n"
                    return
                await asyncio.sleep(0.2)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    @app.get("/api/searches/{sid}")
    def get_search(sid: str) -> dict[str, Any]:
        return {**load_search(sid), "search_id": sid, "reviews": reviews(sid), "mode": mode}

    def viz_assets(search: dict[str, Any], f: Finding) -> dict[str, Any] | None:
        rel = search.get("run_dirs", {}).get(f.scene_id)
        if rel is None:
            return None
        run_dir = run_root / rel
        manifest_path = run_dir / "viz" / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"findings": []}
        entry = next((x for x in manifest["findings"] if x["finding_id"] == f.finding_id), None)
        if entry is None:
            run = load_run(run_dir)
            if f.finding_id not in {x.finding_id for x in run.findings}:
                return None
            # ASSUMPTION A-225: export on first view, every finding of the run at once, then reused
            export_run(run, open_scene(f.scene_id), run_dir / "viz")
            manifest = json.loads(manifest_path.read_text())
            entry = next(x for x in manifest["findings"] if x["finding_id"] == f.finding_id)
        base = f"/files/{(run_dir / 'viz').relative_to(run_root).as_posix()}"
        timeline = json.loads((run_dir / "viz" / entry["timeline_path"]).read_text())
        return {
            **entry,
            "overlays": [{**o, "url": f"{base}/{o['path']}"} for o in entry["overlays"]],
            "bev_url": f"{base}/{entry['bev_path']}",
            "timeline": timeline,
        }

    def card_path(sid: str, fid: str) -> Path:
        return search_dir(sid) / "cards" / f"{fid}.json"

    @app.get("/api/searches/{sid}/findings/{fid}")
    def get_finding(sid: str, fid: str) -> dict[str, Any]:
        search = load_search(sid)
        f = find(search, fid)
        spec = ScenarioSpec.model_validate(search["spec"])
        cp = card_path(sid, fid)
        card = json.loads(cp.read_text()) if cp.is_file() else make_card(f, spec, None).to_json()
        return {
            "finding": f.model_dump(mode="json"),
            "spec": search["spec"],
            "card": card,
            "review": reviews(sid).get(fid),
            "viz": viz_assets(search, f),
            "mode": mode,
        }

    @app.post("/api/searches/{sid}/findings/{fid}/card", dependencies=guarded)
    def write_card(sid: str, fid: str) -> dict[str, Any]:
        search = load_search(sid)
        f = find(search, fid)
        spec = ScenarioSpec.model_validate(search["spec"])
        note = None
        client = card_client() if card_client is not None and mode != "offline-demo" else None
        try:
            card = make_card(f, spec, client)
        except GATED as e:
            card, note = make_card(f, spec, None), f"template card: {e}"
        out = {**card.to_json(), "errors": card.errors, "note": note}
        cp = card_path(sid, fid)
        cp.parent.mkdir(exist_ok=True)
        cp.write_text(json.dumps(out, indent=1))
        return out

    @app.post("/api/searches/{sid}/reviews", dependencies=guarded)
    def review(sid: str, req: ReviewRequest) -> dict[str, Any]:
        search = load_search(sid)
        f = find(search, req.finding_id)
        if req.decision == "approve" and not f.verified:
            raise HTTPException(400, "an unverified finding cannot be approved")
        entry = {**req.model_dump(), "at": time.time()}
        with open(search_dir(sid) / "reviews.jsonl", "a") as fh:
            fh.write(json.dumps(entry) + "\n")
        return entry

    @app.get("/api/searches/{sid}/export")
    def export(sid: str, format: Literal["json", "csv"] = "json"):
        search = load_search(sid)
        decided = reviews(sid)
        spec = ScenarioSpec.model_validate(search["spec"])
        rows = []
        for raw in search["findings"]:
            if decided.get(raw["finding_id"], {}).get("decision") != "approve":
                continue
            f = Finding.model_validate(raw)
            cp = card_path(sid, f.finding_id)
            card = json.loads(cp.read_text()) if cp.is_file() else make_card(f, spec, None).to_json()
            rows.append(
                {
                    "finding_id": f.finding_id,
                    "scene_id": f.scene_id,
                    "t0": f.t0,
                    "t1": f.t1,
                    "score": round(f.score, 4),
                    "card_title": card["title"],
                    "card": " ".join(s["text"] for s in card["sentences"]),
                    "predicates": "; ".join(f"{r.name}={r.status}" for r in f.results),
                    "note": decided[f.finding_id].get("note", ""),
                }
            )
        if format == "csv":
            buf = io.StringIO()
            fields = [
                "finding_id",
                "scene_id",
                "t0",
                "t1",
                "score",
                "card_title",
                "card",
                "predicates",
                "note",
            ]
            w = csv.DictWriter(buf, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
            return PlainTextResponse(
                buf.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": f'attachment; filename="{sid}.csv"'},
            )
        return JSONResponse(
            {"search_id": sid, "query": search["query"], "approved": rows},
            headers={"Content-Disposition": f'attachment; filename="{sid}.json"'},
        )

    app.mount("/files", StaticFiles(directory=run_root), name="files")
    if WEB_DIST.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
    return app


def _limits_from_env() -> dict[str, Any]:
    return {
        "max_searches": int(os.environ.get("LONGTAIL_API_MAX_SEARCHES") or DEFAULT_MAX_SEARCHES),
        "job_ttl_s": float(os.environ.get("LONGTAIL_API_JOB_TTL_S") or DEFAULT_JOB_TTL_S),
        "token": os.environ.get("LONGTAIL_API_TOKEN") or None,
    }


def create_app_from_env() -> FastAPI:
    load_dotenv()
    mode: Mode = os.environ.get("LONGTAIL_MODE", "replay")  # type: ignore[assignment]
    run_root = Path(os.environ.get("LONGTAIL_RUNS", "runs"))
    limits = _limits_from_env()
    if mode == "offline-demo":
        from api.demo import demo_client, demo_scenes

        scenes, source = demo_scenes()
        ids = list(scenes)
        opener: SceneOpener = open_with(scenes)
        client = demo_client(run_root, ids)

        def factory(on_event):
            return SearchAgent(client, ids, opener, source, run_root, on_event=on_event)

        return create_app(run_root, factory, opener, mode, **limits)
    if mode == "replay":
        os.environ["LONGTAIL_REPLAY"] = "1"
    from longtail.data.index import load_index, open_scene
    from longtail.llm.baseten import spark_open_client
    from longtail.llm.client import ModelClient

    index = load_index()
    ids = [s["scene_id"] for s in index["scenes"]]
    meta = {s["scene_id"]: {k: v for k, v in s.items() if k != "scene_id"} for s in index["scenes"]}

    def live_factory(on_event):
        try:
            from longtail.vision.sam_client import SamClient, transport_from_env
            from longtail.vision.track_source import sam_track_source
        except ImportError as e:  # C3 not merged yet
            raise RuntimeError("the SAM track source (C3, PR #9) is not on this branch") from e
        source = sam_track_source(SamClient(transport_from_env()))
        return SearchAgent(
            ModelClient.from_env(), ids, open_scene, source, run_root, scene_meta=meta, on_event=on_event
        )

    def open_by_id(scene_id: str) -> Scene:
        return open_scene(scene_id)

    return create_app(run_root, live_factory, open_by_id, mode, card_client=spark_open_client, **limits)
