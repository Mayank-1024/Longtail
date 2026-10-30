# Chethan's status

Perception and verification (C1 to C10). Updated at the end of every task, on that task's branch. Last update: 2026-10-30, after the A-10, P@10 and break-rule changes (C7 and C8).

Everything below is tested on a synthetic scene only (`tests/synthetic.py`); PandaSet is not downloaded yet. The assumptions register (`docs/assumptions.md`, on the stacked branches) holds every choice the plan does not fix, with an owner; numbers and deviations from the plan are in `docs/eval-report.md`.

## a. Status

| Task | Status | Branch | PR | Tested on synthetic data | Real-data checks pending |
|---|---|---|---|---|---|
| Contracts (joint, M1) | Merged (merge commit) | `c/contracts-and-assumptions` | [#1](https://github.com/Mayank-1024/longtail/pull/1) | Validation, computed `verified`, assumption-tag test | none |
| C1 Dataset adapter | Merged (merge commit) | `c/c1-adapter` | [#2](https://github.com/Mayank-1024/longtail/pull/2) | Fake on-disk PandaSet sequence read through the real devkit; label guard; download script; scene-index selection | Build `index.json` (30 scenes + 3 phrase-tuning); one scene loads in under 5 s; ego axes, world up and camera poses against the devkit calibration |
| C2 Projection and sync | Done | `c/c2-projection` | [#3](https://github.com/Mayank-1024/longtail/pull/3) | Projection, transforms, sync, gap report | Projection within 1 px of the devkit; overlays for six cameras by eye (Oct 30 gate); real gap distribution |
| C3 SAM 3.1 client | Offline parts done (draft) | `c/c3-sam` | [#9](https://github.com/Mayank-1024/longtail/pull/9) | Clips (H.264, round trip, crops, cache); parser on a SYNTHETIC fixture via meta-sam-parser; client with a fake transport (retries, file-id reuse, ledger, replay); SAM tracks into verify() end to end | Smoke test with a Meta key (`scripts/sam_smoke.py`), replacing the SYNTHETIC fixture with a recorded one; concept-phrase test on 3 scenes; move onto `llm/client.py` when it exists |
| C4 Mask to LiDAR | Done | `c/c4-association` | [#4](https://github.com/Mayank-1024/longtail/pull/4) | Clusters, ground (incl. slopes), visibility filter, merging, frustum and box-only | 17 of 20 real SAM masks correct by eye (needs C3) |
| C5 Tracks and ego path | Done | `c/c5-tracks` | [#5](https://github.com/Mayank-1024/longtail/pull/5) | Speeds, gaps, ego motion cancelling, corridor (straight, arc, stopped) | Parked car on a real scene under 0.3 m/s; corridor on the bird's-eye export |
| C6 Verifier | Done | `c/c6-verifier` | [#6](https://github.com/Mayank-1024/longtail/pull/6) | PASS, FAIL and UNVERIFIABLE for all 10 predicates; binding, pruning, cap | Thresholds on PandaSet and tuning on development queries |
| C7 Ground truth | Done on synthetic (draft) | `c/c7-gt` | [#10](https://github.com/Mayank-1024/longtail/pull/10) | Four synthetic gold specs (crossing, parked car, pedestrian hidden by a car, zero-positive cyclist); sibling dedup; 3 s horizon; event merging; one-to-one matching at IoU 0.3; label guard; the verifier's crossing finding matches its positive; `longtail gt build`, `spotcheck`, `motion-report` | Your 25 queries and gold specs (M7); `longtail gt build` on 30 scenes; 5 positives checked by eye; geometry vs attribute motion agreement; sibling uuids (A-16) |
| C8 Metrics, ablation, breaks | Done on synthetic (draft) | `c/c8-metrics` | [#11](https://github.com/Mayank-1024/longtail/pull/11) | Metrics on hand-computed cases; deterministic bootstrap; catch and false-rejection rates; (b) and (c) on the synthetic scene ((c) precision 1.00 vs (b) 0.25; P@10 0.050 vs 0.025); all three breaks pass (lower precision, catch rate or recall); (a) and (d) adapters on sample files and clear "waiting on M4/M5" errors; results JSON and chart | Held-out run on Nov 3 from `eval-final`; (a) needs your M5 files, (d) your M4 file (`docs/eval-formats.md`); box-only matching (A-127) still IoU, so occluded objects in (a) and (b) will not match |
| C9 Visualization export | Done | `c/c9-viz` | [#7](https://github.com/Mayank-1024/longtail/pull/7) | Overlays, bird's-eye JSON, timeline, manifest, schema, CLI | Overlays on real photos; real export size and time |
| C10 Fresh-clone test | Not started (Nov 4) | | | | |

Open failures, kept as strict expected failures and explained in `docs/eval-report.md`: C4's centroid sits about 0.3 m from a box's centre (surface-only LiDAR), and C5 still reads a parked car partly hidden by another car as moving for 1.2 s (C6 does not let it PASS `moving`).

## b. Decisions needed from Mayank

| Topic | ID | What is assumed | Why it matters |
|---|---|---|---|
| Contracts | A-02 | `contracts.py` follows plan section 4 except where another assumption changes it | The draft in #1 is the only coupling between our halves |
| Contracts | A-06 | `PredicateCall.name` gains `ego_moving` and `ego_stopped`, with no actors | Needed for "while we move"; your planner must emit them |
| Contracts | A-08, A-24 | Every result with a value has a numeric `evidence["margin"]`; none when the value is null | Your ranking (M3) reads it |
| Contracts | A-17 | The engine returns unverified findings; only your `finish` tool drops them | The "rejected by verifier" tab and the catch rate need them |
| Contracts | A-19 | Taxonomy: pedestrian, cyclist, car, truck, bus | Your planner's classes; dataset labels map onto these |
| Contracts | A-20 | `Frame2D` boxes are inclusive pixels; masks are COCO compressed RLE; mask and conf may be null | Fixes the SAM and VLM-claim formats |
| Contracts | A-21 | Every contract model rejects unknown fields | Planner typos fail validation and go back for retry |
| Contracts | A-22 | `PredicateResult.required` is copied from the call, so `verified` is computed from the finding alone | Without it `verified` cannot be computed |
| Contracts | A-23 | Predicate signatures: arity and argument names per the plan's C6 table; unknown arguments rejected | Your planner's output must match |
| Dataset | A-01 | PandaSet. Download: sign up at https://pandaset.org/ (the devkit README's instructions), which leads to the download links. Licence: the dataset ships its own `LICENSE.txt`; I have not seen it (the devkit code is Apache-2.0) | Confirm the choice and that the licence allows our use and showing frames (your M0 licence check); send me the link |
| Shared config | A-34 | `longtail/cli.py` is a thin argparse dispatcher; areas register subcommands (`scenes`, `viz`) | The plan gives `cli.py` to you; keep or replace it |
| Shared config | | The pinned PandaSet devkit lists Jupyter in its requirements, pulling in about 118 packages | Slow CI; option: install the devkit with `--no-deps` (it needs only numpy, pandas, Pillow, transforms3d) or override the extras in uv |
| Shared config | | `pyproject.toml` sets pytest `pythonpath = ["."]` | Tests import from the checkout even when an editable install is skipped (it is on my Mac); keep or drop |
| CI | | The workflow is still in `.github/ci-pending/` | Needs a token with the `workflow` scope and a move to `.github/workflows/`; no PR has CI yet |
| Evaluation (M7) | A-95, A-126 | **Changed:** positives are now events (maximal intervals where the gold spec holds for at least `min_duration_s`), not windows. A-95 is owned by both: please agree or object | Recall is over events; overlapping findings of one event count once |
| Evaluation (M7) | A-128 | Queries file format: `format: longtail.queries/v1`, then per query `query_id`, `text`, `split` (dev or heldout), `tier`, `gold` (a ScenarioSpec); example in `tests/fixtures/gold_synthetic.yaml` | `longtail gt build <file>` reads it; write M7's queries in it, or tell me your format |
| Evaluation (M7) | A-123 | PandaSet labels to classes: Pickup Truck, Medium-sized Truck and Semi-truck are `truck`; a Bicycle is a `cyclist` only with a rider | Decides which queries have positives (e.g. "a truck" includes pickups) |
| Evaluation (M4, M5) | A-135 | Baseline (a) writes `runs/baseline_a/claims.jsonl` and `rankings.json`; triage (d) writes `runs/triage_d/windows.json`. Formats with examples in `docs/eval-formats.md` | Without them (a) and (d) stop with "waiting on M5" / "waiting on M4" |
| Evaluation (M3) | A-136, A-137 | Uniform windows are the triage grid unfiltered (2 s every 1 s, padded 1 s); findings of the same binding over overlapping windows merge before ranking | Your pipeline's windows and ranking should do the same, or duplicates eat the top 10 |
| Evaluation | A-134 | Precision (correct over returned, pooled) is the headline next to recall; P@10 divides by 10, with the returned count beside it | How the README reads precision; with few findings per query P@10 counts hits only |
| Evaluation | A-10 | A model claim is true when half of its box lies inside a projected label box of its class (was IoU >= 0.3) | Your baseline (a) boxes are judged this way for the catch rate |
| Evaluation (M7) | A-127 | Matching: same scene, temporal IoU >= 0.3, every bound actor within 2 m of a labelled object (boxes-only findings: box IoU >= 0.3) | How your baseline (a) and pipeline findings are scored |
| Evaluation (M7) | A-96 | Ground truth uses crossing_ego_path's 3 s look-ahead, so it cannot count a FAIL in a scene's last 3 s | Avoid gold specs whose only evidence is in a scene's last 3 s |
| Pipeline (M3) | A-101 | Exports come from a saved run (`verify_and_record()` writes `runs/<run>/verify_run.json`) | Without a saved run the web view has nothing to show |
| Web view (M6) | | Overlays are about 130 MB of JPEGs per finding at full resolution (six cameras, 8 s, measured without real photos) | Choose: downscale (e.g. 960 px wide, about 4x smaller) or export only the frames you show |
| Web view (M6) | A-100 | The ego box is a 4.8 x 1.9 m placeholder centred on the LiDAR | Fine for drawing; replace if you know the vehicle's body box |
| Client (M1) | A-114, A-116, A-117 | C3 now runs on an interim client in `longtail/vision/sam_client.py` (fake transport in tests; `MetaTransport` for the smoke test) | When you build `llm/client.py`, please cover what it does: `upload(path) -> file id` (cached per clip), a streamed call yielding text deltas, caching of the final text, ledger lines in frames and dollars to `runs/ledger.jsonl`, retry on 429 and 5xx but not 400, and replay (`LONGTAIL_REPLAY=1`). I then switch C3 to yours and drop my cache layer |
| SAM parser licence | A-109 | We depend on Meta's `meta-sam-parser` 0.1.0, under the "SAM License" (not open source; publications must acknowledge SAM) | It is a dependency, not vendored; flag it if that conflicts with how we release the repo |
| Shared cache and ledger paths | A-112, A-116 | Caches under `$LONGTAIL_CACHE_DIR` (default `./cache`); ledger at `runs/ledger.jsonl` | Your client should use the same paths so costs add up in one place |

Other assumptions owned by "both" that touch your side, listed in each PR's "For Mayank" section: A-03 (units and frames), A-79 (reason codes), A-85 (score and binding choice), A-86 (track source), A-88 (window times), A-90 (binding search evidence), A-91 (joint timing), A-92 (`cost_usd` left at 0), A-99, A-102, A-103, A-106, A-107 (viz).

## c. Interfaces your code calls

All on the stacked branches until merged.

**Scene API** (`longtail/data/base.py`, PandaSet in `longtail/data/pandaset.py`): times in seconds, poses as 4x4 `world_from_X` matrices, ego frame x forward, y left, z up.

```python
from longtail.data.index import open_scene

scene = open_scene("002")  # from $LONGTAIL_DATA_DIR/pandaset/002
scene.cameras  # ('front_camera', ...)
scene.frames("front_camera")  # [CameraFrame(frame_idx, ts, image_path), ...]
scene.lidar(scene.lidar_timestamps()[0]).points  # (N, 5): x, y, z, intensity, t, world frame
scene.ego_pose(ts), scene.extrinsics(camera, ts), scene.intrinsics(camera), scene.image_size(camera)
scene.labels(i)  # raises LabelAccessError outside eval_ground_truth()
```

**verify()** (`longtail/verify/engine.py`):

```python
from longtail.verify.engine import verify

findings = verify(spec, windows, scene, track_source=my_source)  # list[Finding], verified first
```

One finding per window that has frames: the best binding, verified or not. Raises if `spec.clarification` is set, if a window is for another scene, or if no `track_source` is given.

**track_source** (A-86): any callable `(scene, camera, concept) -> list[Track2D]`; C3's SAM client will provide the real one.

```python
def my_source(scene, camera, concept):
    return sam_tracks_for(scene.scene_id, camera, concept)  # Track2D list from SAM 3.1
```

The real one (C3, `longtail/vision/track_source.py`), once there is a key:

```python
from longtail.vision.sam_client import SamClient, transport_from_env
from longtail.vision.track_source import sam_track_source

source = sam_track_source(SamClient(transport_from_env()))  # replay=True or LONGTAIL_REPLAY=1: cache only
findings = verify(spec, windows, scene, track_source=source)
```

**Reason codes** (A-79, `longtail/verify/predicates/base.py`, `Reason`): PASS is `ok`. UNVERIFIABLE: `no_coverage`, `out_of_range`, `unsynced`, `truncated`, `not_observed`, `track_too_short`, `unreliable_speed`, `implausible_jump`, `requires_exists`. FAIL: `wrong_size`, `too_few_points`, `no_object_points`, `too_far`, `not_moving`, `not_stopped`, `outside_corridor`, `not_toward_corridor`, `wrong_side`, `not_near`, `not_occluded`, `shorter_than_min_duration`.

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

**verify_and_record()** (`longtail/verify/run.py`, A-101): call this instead of `verify()` for runs the web view will show.

```python
from longtail.verify.run import verify_and_record

run = verify_and_record(spec, windows, scene, my_source, Path("runs") / run_id)
run.findings  # same findings as verify(); saved to runs/<run_id>/verify_run.json
```

**Ground truth** (C7, `longtail/eval/gt.py`; reads labels, so evaluation only):

```
longtail gt build longtail/eval/queries/queries.yaml   # gt/<query_id>.json + gt/summary.json; zero-positive queries flagged
longtail gt spotcheck <query_id> --n 5                 # five random positives: scene, time, label ids
longtail gt motion-report                              # geometric motion vs the labelled attributes, per class
```

Pass `verify(..., pool=TrackPool(scene, source, cfg))` to keep the engine's tracks; `gt.scored_finding(finding, pool)` turns a finding into what the matcher scores.

**Ablation and breaks** (C8; formats in `docs/eval-formats.md`):

```
uv run python -m longtail.eval.ablation --config c --queries <queries file> --split heldout --replay
uv run python -m longtail.eval.breaks --queries <queries file> --split heldout --replay
```

Results go to `longtail/eval/results/` (`ablation_<config>.json`, `ablation.png`, `breaks.json`). `--replay` serves SAM from the cache only; without it a key makes paid calls.

**Viz export and schema** (`longtail/viz/`, `docs/viz-schema.json`, `docs/viz-schema.md`):

```
longtail viz runs/<run_id>              # every finding -> runs/<run_id>/viz/manifest.json
longtail viz runs/<run_id>/<finding>    # one finding
longtail viz --write-schema docs/viz-schema.json
```

Generate TypeScript from `$defs` (`VizManifest`, `BevExport`, `Timeline`). Drive the scrubber by `frame_idx`; the bird's-eye view is ego-centred per frame.

## d. How to review and merge

1. **Order:** #3 (C2), #4 (C4), #5 (C5), #6 (C6), #7 (C9), #9 (C3), #10 (C7), #11 (C8) (#1, #2 and #8 are merged). Each stacked PR targets the branch below it, so its diff shows only its own task.
2. Read each PR's "For Mayank" section first; that is what touches your side.
3. Merge bottom-up with **"Create a merge commit"**, not squash. Squashing rewrites the commits the PRs above it are built on, and their diffs fill with already-merged changes.
4. After each merge, check that the next PR's base is now `main`. GitHub usually retargets it when the merged branch is deleted; if not, change the base by hand (Edit, next to the title).
5. If your review of #1 changes the contracts, tell me and I rebase the stack; please don't rebase it yourself.

## e. What I need, in priority order

1. The PandaSet download link (and the licence check), so the real-data checks and the Oct 30 projection gate can run. A Meta API key with a spend limit for me too, for the SAM smoke test.
2. A review of #1 (contracts) and CI turned on.
3. The evaluation queries and gold specs (M7) in the A-128 format, so `longtail gt build` can flag zero-positive queries before the Oct 31 freeze.
4. The `llm/client.py` interface, for the C3 SAM client.

A matching `docs/status-mayank.md` on your side would let both of us (and any Claude Code session, see `CLAUDE.md`) pick up the other's state quickly.
