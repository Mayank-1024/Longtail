# Chethan's status

Perception and verification (C1 to C10). This copy on `main` is the source of truth (see `CLAUDE.md`). Last update: 2026-10-30.

Everything is tested on a synthetic scene only (`tests/synthetic.py`); PandaSet is not downloaded and there is no Meta key yet, so no paid call has been made. The assumption register (`docs/assumptions.md`) and the evaluation report (`docs/eval-report.md`) are complete on the top of the stack (#11) and land with the PRs below.

## a. Status

| Task | Status | Branch | PR | Tested on synthetic data | Real-data checks pending |
|---|---|---|---|---|---|
| C1 Dataset adapter | Merged | `c/c1-adapter` | [#2](https://github.com/Mayank-1024/longtail/pull/2) | Fake on-disk PandaSet sequence through the real devkit; label guard; download script; scene-index selection | Build `index.json` (30 scenes + 3 phrase-tuning); one scene loads in under 5 s; ego axes and camera poses against the devkit calibration |
| C2 Projection and sync | Written, in review | `c/c2-projection` | [#3](https://github.com/Mayank-1024/longtail/pull/3) | Projection, transforms, sync, gap report | Projection within 1 px of the devkit; overlays for six cameras by eye; real gap distribution |
| C3 SAM 3.1 client | Offline parts written, in review | `c/c3-sam` | [#9](https://github.com/Mayank-1024/longtail/pull/9) | Clips (H.264, frame-checked index, crops, cache); parser on a SYNTHETIC fixture via Meta's `meta-sam-parser`; client with a fake transport (retries, file-id reuse, ledger, replay); SAM tracks into `verify()` | The smoke test with a Meta key (section f); concept-phrase test on 3 scenes; move onto `llm/client.py` |
| C4 Mask to LiDAR | Written, in review | `c/c4-association` | [#4](https://github.com/Mayank-1024/longtail/pull/4) | Clusters, local ground planes (incl. slopes), visibility filter, cross-camera merge with a 15% shared-point floor, frustum and box-only modes | 17 of 20 real SAM masks correct by eye |
| C5 Tracks and ego path | Written, in review | `c/c5-tracks` | [#5](https://github.com/Mayank-1024/longtail/pull/5) | Speeds, gaps, ego motion cancelling, face-anchored box centre, corridor (straight, arc, stopped) | Parked car under 0.3 m/s on a real scene; corridor on the bird's-eye export |
| C6 Verifier | Written, in review | `c/c6-verifier` | [#6](https://github.com/Mayank-1024/longtail/pull/6) | PASS, FAIL and UNVERIFIABLE for all 10 predicates; binding, pruning, cap; each motion verdict names its 1 s window (A-94) | Thresholds on PandaSet; tuning on development queries |
| C7 Ground truth | Written, in review | `c/c7-gt` | [#10](https://github.com/Mayank-1024/longtail/pull/10) | Four synthetic gold specs; sibling dedup; 3 s horizon; events; one-to-one matching at IoU 0.3; A-10 claims (50% inside); label guard; the verifier's crossing finding matches its positive | `longtail gt build` on 30 scenes with your 25 gold specs; 5 positives by eye; geometry vs attribute motion agreement; sibling uuids (A-16) |
| C8 Metrics, ablation, breaks | Written, in review | `c/c8-metrics` | [#11](https://github.com/Mayank-1024/longtail/pull/11) | Precision, recall, P@10 (over 10, with the returned count), catch and false-rejection rates, bootstrap; (b) and (c) run; (a) and (d) adapters on sample files; all three breaks pass | The held-out run (Nov 3); (a) and (d) need your files; whether the LiDAR offset still fails safe on real scenes |
| C9 Visualization export | Written, in review | `c/c9-viz` | [#7](https://github.com/Mayank-1024/longtail/pull/7) | Overlays, bird's-eye JSON under 5 MB, timeline, manifest, schema, CLI | Overlays on real photos; real export size and time |
| C10 Fresh-clone test | Pending (Nov 4) | | | | Clone, `uv sync`, download, `longtail demo --replay`, rules checklist |

Synthetic results so far (one 3 s scene, 4 queries, two injected hallucinations; the numbers test the machinery, not PandaSet):

| | Precision | Recall | P@10 (returned / 10) | Catch rate | False rejection |
|---|---|---|---|---|---|
| (b) SAM tracks, no LiDAR | 0.25 | 0.33 | 0.025 (1.0) | n/a | n/a |
| (c) with the verifier | 1.00 | 0.67 | 0.050 (0.5) | 1.00 | 0.29 |

Breaks on (c): verifier off moves precision and the catch rate; the calibration shift moves precision and recall; the 200 ms LiDAR offset fails safe: shows up as lost recall.

Open failures, kept as strict expected failures and explained in `docs/eval-report.md`: C4's centroid sits about 0.3 m from a box's centre (LiDAR sees surfaces only), and C5 reads a parked car partly hidden by another car as moving for 1.2 s (C6 never lets it PASS `moving`).

## b. Decisions needed from Mayank

Every assumption owned by you or by both of us (none is yours alone). The ones that most affect your work come first; the rest follow in ID order. Full text in `docs/assumptions.md` (on #11 until the stack lands).

| Topic | ID | What is assumed | Why it matters |
|---|---|---|---|
| Contracts | A-02 | `contracts.py` follows plan section 4, except where another assumption changes it | The only coupling between our halves |
| Contracts | A-06 | `PredicateCall.name` gains `ego_moving` and `ego_stopped`, with no actors | Needed for "while we move"; your planner must emit them |
| Contracts | A-08, A-24 | Every result with a value has a numeric `evidence["margin"]`; none when the value is null | Your ranking (M3) reads it |
| Contracts | A-17 | The engine returns unverified findings; only your `finish` tool drops them | The "rejected" view and the catch rate need them |
| Contracts | A-19 | Taxonomy: pedestrian, cyclist, car, truck, bus | Your planner's classes |
| Contracts | A-20 | `Frame2D` boxes are inclusive pixels; masks are COCO compressed RLE; mask and conf may be null | Fixes the SAM and VLM-claim formats |
| Contracts | A-21 | Every contract model rejects unknown fields | Planner typos fail validation and go back for retry |
| Contracts | A-22 | `PredicateResult.required` is copied from the call, so `verified` is computed from the finding alone | Without it `verified` cannot be computed |
| Contracts | A-23 | Predicate signatures (arity, argument names) per the plan's C6 table; unknown arguments rejected | Your planner's output must match |
| Ground truth | A-126, revises A-95 | A positive is an **event**: a maximal interval where the gold spec holds for at least `min_duration_s`; overlapping objects' intervals merge. Not windows | Recall is over events; how the README describes it |
| Ground truth | A-123 | PandaSet labels to classes: Pickup Truck, Medium-sized Truck and Semi-truck are `truck`; a Bicycle is a `cyclist` only with a rider | Decides which queries have positives |
| Queries | A-128 | Queries file: `format: longtail.queries/v1`, then per query `query_id`, `text`, `split` (dev or heldout), `tier`, `gold` (a ScenarioSpec). Example: `tests/fixtures/gold_synthetic.yaml` | `longtail gt build <file>` reads it; write M7 in it |
| Ablation inputs | A-135 | Baseline (a): `runs/baseline_a/claims.jsonl` and `rankings.json`; triage (d): `runs/triage_d/windows.json`. Formats with examples in `docs/eval-formats.md` | Without them (a) and (d) stop with "waiting on M5" / "waiting on M4" |
| Metrics | A-134 | Precision (correct over returned, pooled over queries) is the headline next to recall (matched positives over all positives, pooled). P@10 divides by 10, with the number returned beside it | How the README reports results |
| Matching | A-127 | Findings with only image boxes (configs a and b) match a positive by box IoU >= 0.3. **Proposed:** switch to the 50%-inside rule that A-10 now uses for claims, for consistency | With IoU, an occluded object's visible-part box never matches its positive, so (a) and (b) are scored more harshly than (c) |
| Client (M1) | A-114 | C3 runs on an interim client in `longtail/vision/sam_client.py`. Your `llm/client.py` should replace it, covering: `upload(path) -> file id` (cached per clip hash); a streamed call `(model, file_id, concept, include_confidence)` yielding text deltas; caching of the final text keyed by (model, clip hash, concept, confidence flag); replay mode (`LONGTAIL_REPLAY=1`, fail on a miss); retries on 429 and 5xx only (A-115); a ledger line per call to `runs/ledger.jsonl` with frames and dollars (A-116) | I then switch C3 to yours and drop my cache layer |
| SAM licence | A-109 | We depend on Meta's `meta-sam-parser` 0.1.0 under the "SAM License" (not open source) | The README and the video must acknowledge SAM |
| Web view (M6) | | Overlays are about 130 MB of JPEGs per finding at full resolution (six cameras, 8 s) | Choose: downscale (e.g. 960 px wide, about 4x smaller) or export only the frames you show |
| Pipeline (M3) | A-101 | The web view exports from a saved run: call `verify_and_record()` (writes `runs/<run>/verify_run.json`) | Without a saved run the web view has nothing to show |
| CI | | The workflow is still in `.github/ci-pending/` | Needs a token with the `workflow` scope and a move to `.github/workflows/`; no PR has CI yet |
| Dataset | A-01 | PandaSet; download by signing up at https://pandaset.org/. It ships its own `LICENSE.txt`, which I have not seen | Send me the link; confirm the licence allows our use and showing frames |
| Units | A-03 | Metres, seconds; ego x forward, y left, z up; 0-based `frame_idx`; `ts` in dataset seconds | Every number crossing the contract |
| Shared CLI | A-34 | `longtail/cli.py` is a thin argparse dispatcher; areas register subcommands (`scenes`, `viz`, `sam`, `gt`) | The plan gives `cli.py` to you; keep or replace it |
| Verifier | A-79 | Reason codes (section c) | The cards and the web view show them |
| Verifier | A-85 | Score = mean of tanh(margin) for PASS, -0.5 UNVERIFIABLE, -1 FAIL; one binding kept per window | Your ranking |
| Verifier | A-86 | `verify()` takes a `track_source(scene, camera, concept)` | How your pipeline feeds SAM tracks |
| Verifier | A-88 | Window `t0`, `t1` are dataset timestamps in seconds | Your windows must use them |
| Verifier | A-90 | Evidence records the binding search (cap hit, tried, upper bound) | Shown in the trace |
| Verifier | A-91 | A finding's interval is the longest joint PASS run, at least `min_duration_s` | What the card's time range means |
| Verifier | A-92 | The verifier leaves `Finding.cost_usd` at 0 | Your pipeline fills in model costs |
| Viz | A-99 | `bev.json` under 5 MB per finding | Web view load time |
| Viz | A-100 | The ego box is a 4.8 x 1.9 m placeholder centred on the LiDAR | Replace if you know the body box |
| Viz | A-102 | Every result's evidence has `timeline` | Drives the predicate timeline |
| Viz | A-103 | `longtail viz runs/<run>[/<finding>]`; `--write-schema` | How exports are made |
| Viz | A-106 | The timeline covers the whole window; overlays and BEV cover [t0, t1] | What the scrubber shows |
| Viz | A-107 | BEV shows bound tracks only, positions from C4, boxes from C5 | What the canvas draws |
| Caches | A-112 | Caches under `$LONGTAIL_CACHE_DIR` (default `./cache`) | Your client should use the same root |
| Costs | A-116 | Ledger at `runs/ledger.jsonl`; $0.20 per 1,000 frames, 50-frame minimum | Costs add up in one place |
| SAM transport | A-117 | `MetaTransport` follows plan 3.1 (OpenAI SDK, `files.create`, `responses.stream`, `include_confidence` in metadata); untested until the smoke test | Your client inherits whatever the smoke test finds |
| CLI | A-119 | `longtail sam` refuses without a key; `--replay` and `--fixture` need none | The demo runs in replay mode |
| Ground truth | A-129 | `gt/<query>.json` and `gt/summary.json`, only under `gt/` (git-ignored) | Where to look for zero-positive queries |
| CLI | A-131 | `longtail gt build`, `spotcheck`, `motion-report` | Section c |
| Ablation | A-136 | Uniform windows: 2 s every 1 s, padded by 1 s (the triage grid unfiltered) | (c) and (d) then differ only in the windows kept |
| Ablation | A-137 | Findings of the same binding over overlapping windows merge before ranking | Your pipeline should do the same, or duplicates fill the top 10 |
| Ablation | A-140 | Results go to `longtail/eval/results/` with the commit and a thresholds hash | Reproducibility of the README numbers |

## c. Interfaces your code calls

All on the stacked branches until merged.

**Scene API** (`longtail/data/base.py`; PandaSet in `longtail/data/pandaset.py`): times in seconds, poses as 4x4 `world_from_X` matrices.

```python
from longtail.data.index import open_scene

scene = open_scene("002")  # $LONGTAIL_DATA_DIR/pandaset/002
scene.cameras  # ('front_camera', ...)
scene.frames("front_camera")  # [CameraFrame(frame_idx, ts, image_path), ...]
scene.lidar(scene.lidar_timestamps()[0]).points  # (N, 5): x, y, z, intensity, t; world frame
scene.ego_pose(ts), scene.extrinsics(camera, ts), scene.intrinsics(camera), scene.image_size(camera)
scene.labels(i)  # raises LabelAccessError outside eval_ground_truth()
```

**verify()** (`longtail/verify/engine.py`), with a track source (A-86):

```python
from longtail.verify.engine import verify
from longtail.vision.sam_client import SamClient, transport_from_env
from longtail.vision.track_source import sam_track_source

source = sam_track_source(SamClient(transport_from_env()))  # replay=True or LONGTAIL_REPLAY=1: cache only
findings = verify(spec, windows, scene, track_source=source)  # list[Finding], verified first
```

One finding per window with frames: the best binding, verified or not. Raises if `spec.clarification` is set, a window is for another scene, or no `track_source` is given. Optional `pool=TrackPool(scene, source, cfg)` keeps the tracks for later reading.

**Reason codes** (A-79, `Reason` in `longtail/verify/predicates/base.py`): PASS is `ok`. UNVERIFIABLE: `no_coverage`, `out_of_range`, `unsynced`, `truncated`, `not_observed`, `track_too_short`, `unreliable_speed`, `implausible_jump`, `requires_exists`. FAIL: `wrong_size`, `too_few_points`, `no_object_points`, `too_far`, `not_moving`, `not_stopped`, `outside_corridor`, `not_toward_corridor`, `wrong_side`, `not_near`, `not_occluded`, `shorter_than_min_duration`.

**Finding evidence** (each `PredicateResult.evidence`):

| Key | Meaning |
|---|---|
| `reason` | The result's reason code |
| `margin` | How far the value clears the threshold (present exactly when `value` is set) |
| `frames` | Sweep indices of the finding's interval |
| `frame_status`, `frame_reasons` | Counts of PASS, FAIL, UNVERIFIABLE and of each reason over the interval |
| `example_frame`, `example` | One representative frame and its numbers (e.g. `range_m`, `net_speed_mps`, `judged_window`) |
| `longest_pass_s` | This predicate's own longest PASS run in the window |
| `timeline` | Every window frame as `[sweep index, status, reason]` |
| `binding_search` | `binding_cap_hit`, `bindings_tried`, `bindings_total_upper_bound`, `pruning_relaxed_for` |

**verify_and_record()** (`longtail/verify/run.py`, A-101): use it instead of `verify()` for any run the web view shows.

```python
from longtail.verify.run import verify_and_record

run = verify_and_record(spec, windows, scene, source, Path("runs") / run_id)  # runs/<run_id>/verify_run.json
```

**Viz export and schema** (`longtail/viz/`, `docs/viz-schema.json`, `docs/viz-schema.md`):

```
longtail viz runs/<run_id>              # every finding -> runs/<run_id>/viz/manifest.json
longtail viz runs/<run_id>/<finding>    # one finding
longtail viz --write-schema docs/viz-schema.json
```

Generate TypeScript from `$defs` (`VizManifest`, `BevExport`, `Timeline`); drive the scrubber by `frame_idx`.

**SAM** (`longtail/vision/`):

```
longtail sam 002 front_camera person            # needs MODEL_API_KEY; prints tracks and cost
longtail sam 002 front_camera person --replay   # cache only, fails on a miss
longtail sam 002 front_camera person --fixture tests/fixtures/sam31_SYNTHETIC.txt
```

**Ground truth and evaluation** (`longtail/eval/`; formats in `docs/eval-formats.md`):

```
longtail gt build longtail/eval/queries/queries.yaml   # gt/<query>.json + gt/summary.json; zero-positive queries flagged
longtail gt spotcheck q07 --n 5                        # five random positives: scene, time, label ids
longtail gt motion-report                              # geometric motion vs the labelled attributes
uv run python -m longtail.eval.ablation --config c --queries <file> --split heldout --replay
uv run python -m longtail.eval.breaks --queries <file> --split heldout --replay
```

## d. How to review and merge

1. **Order:** #3 (C2), #4 (C4), #5 (C5), #6 (C6), #7 (C9), #9 (C3), #10 (C7), #11 (C8).
2. Each stacked PR targets the branch below it, so its diff shows only its own task. #9 also shows `main`'s handoff docs (it merged `main` in); that is cosmetic.
3. Merge with **"Create a merge commit"**, never squash. Squashing rewrites the commits the PRs above are built on.
4. After each merge, check that the next PR's base is now `main`; if not, change it by hand (Edit, next to the title).
5. If `docs/status-chethan.md` conflicts during a merge, keep `main`'s version.
6. If your review changes the contracts, tell me and I rebase the stack; please don't rebase it yourself.

## e. What I need, in priority order

1. The PandaSet download link (and the licence check).
2. Reviews of #3 to #11, and CI turned on.
3. The 25 queries and gold specs (M7) in the A-128 format, so `longtail gt build` can flag zero-positive queries before the freeze.
4. Your `llm/client.py` (A-114), so C3 moves onto it.
5. The baseline (a) and triage (d) outputs in the A-135 formats (`docs/eval-formats.md`).

## f. The SAM smoke test, once a Meta key exists

From `scripts/sam_smoke.py`. One paid call, about $0.016.

1. Put `MODEL_API_KEY=...` in `.env` (git-ignored) and load it: `set -a; source .env; set +a`.
2. Make sure the scene is on disk: `uv run longtail scenes ls` (or check `data/pandaset/<scene>/`).
3. Run `uv run --with openai python scripts/sam_smoke.py --scene 002 --camera front_camera --concept person`.
4. Read the output:
   - "file id" and "streamed N characters": the upload and streaming work.
   - "parsed K tracks": the real format parses. If not, the raw response is saved for fixing `sam_parse.py`.
   - "cost per frame": should be $0.000200.
   - "confidence present": whether `include_confidence` in the metadata works (plan 3.1 marks it unverified).
5. Before committing the saved fixture under `tests/fixtures/`, check the PandaSet licence (its masks come from dataset images). Then replace the SYNTHETIC fixture in the parser test and log the cost in `docs/costs.md` and the build log.
6. Re-run the same command: it must report a $0 cache hit.
