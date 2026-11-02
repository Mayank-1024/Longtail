# Mayank's status

Agent, models, product and story (M0 to M10). Updated at the end of every task. Last update: 2026-11-02.

No paid call has been made: the Meta account is being verified and the credits are not in yet. Paid calls are off until `LONGTAIL_ALLOW_PAID=1` (A-206).

## a. Status

| Task | Status | Branch | Tested | Pending |
|---|---|---|---|---|
| M0 Setup | Repo, CI (PR #13, merged), gitleaks | `main` | CI green | Meta key and credits (verification in progress); PandaSet licence terms |
| M1 Meta client | Done, offline | `m/m1-client` | Fake transport: paid-call gate, cache, replay, retries (429/5xx, not 400), spend cap, query budget, upload once | First paid call: VERIFY model ids, response shape, stream events (A-203, A-208) |
| M2 Planner | Done, offline | `m/m1-client` | All 25 gold specs parse; retry with the error sent back; missing `exists` caught; clarification path | Live golden test on the 10 dev queries (`tests/test_planner.py`, skipped without a key) |
| M7 Queries and gold specs | Done | `m/m7-queries` (PR #14) | All 25 validate; C7's `load_queries` reads them | Zero-positive check with `longtail gt build` once PandaSet is down; freeze by Nov 3, 6 PM |
| M3 Pipeline and agent | Done, offline | `m/m3-agent` | Linear `pipeline.run` on uniform windows (Glimmer cut, A-213), parallel scenes, a failing scene isolated; `SearchAgent`: plan, then list_scenes / search_scenes / finish on Wonder, budget and 6-step stop rule, auto-finish, finish drops unverified findings, trace saved to `runs/search-<id>/search.json`; end to end on the synthetic scene with the real verifier | Real track source (C3 merge); live run once the key works |
| M4 to M10 | Not started (M4 Glimmer cut) | | | |

## b. For Chethan

- **Client for C3:** `ModelClient` in `longtail/llm/client.py` covers your list: `upload(path)` (cached by content hash), `stream_text(request, frames=..., cache_key=...)` (joined deltas, billed by frames, cached by your clip key), replay (`LONGTAIL_REPLAY=1`), retries, and ledger lines in `runs/ledger.jsonl`. It also refuses any paid call unless `LONGTAIL_ALLOW_PAID=1` and the estimate fits the spend cap (`LONGTAIL_SPEND_CAP_USD`, default $5).
- **Agent (M3):** `SearchAgent(client, scene_ids, open_scene, track_source).search(query)`; the linear core is `longtail.pipeline.run(spec, scene_ids, open_scene, track_source, run_root)`. You don't need to build `pipeline.run`; please use it and tell me what's missing.
- **Planner (M2):** `from longtail.agent.planner import plan; spec = plan(query, ModelClient.from_env()).spec`. CLI: `longtail plan "a cyclist within 15 m"`.
- **Held-out queries** are for the final run only; the planner's live test uses the dev split.
