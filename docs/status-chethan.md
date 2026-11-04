# Chethan's status

Perception and verification (C1 to C10). This copy on `main` is the source of truth (see `CLAUDE.md`). Last update: 2026-11-03.

PandaSet is now on disk here (scenes 001, 005, 018 and 058, from `scripts/fetch_pandaset.py`), and the real-data checks that need no SAM output are done. Everything here runs in replay mode: I make no paid calls. Mayank's SAM cache for those scenes arrived on Nov 3, and the two checks that needed it are done; they are below the plan's bars, and `docs/eval-report.md` gives the numbers and the known limitations.

## a. Status

| Task | Status | PR | Real-data checks |
|---|---|---|---|
| C1 Dataset adapter | Merged | [#2](https://github.com/Mayank-1024/longtail/pull/2); ego-frame fix [#20](https://github.com/Mayank-1024/longtail/pull/20) | Done. Real-data tests pass on the 4 scenes. The ego frame was 180 degrees off (A-25); Mayank found it on real data and fixed it in #20, which I checked. Scene 001 loads in 3.7 s and 3.6 s here against the plan's 5 s; Mayank measured about 10 s, and the cluster sets its own budget (`LONGTAIL_LOAD_BUDGET_S`). The test keeps 5 s, unmarked. Scene 054 is left out (A-240, broken camera timestamps). |
| C2 Projection and sync | Merged | [#3](https://github.com/Mayank-1024/longtail/pull/3) | Done. Projection matches the devkit (`test_projection_real`). Six-camera overlays checked by eye on scenes 001 and 058 (day and night): points sit on building edges, poles, curbs and car bodies. One thing to watch: a thin band of near points 10 to 45 px above some close car roofs, probably the roof seen from the higher LiDAR (the overlay has no occlusion test), possibly a small calibration residual; static structure does not show it. 3 frames per scene are unsynced and marked so. |
| C3 SAM 3.1 client | Merged (#21) | [#9](https://github.com/Mayank-1024/longtail/pull/9) via [#21](https://github.com/Mayank-1024/longtail/pull/21) | Running live on the cluster (Mayank). Concept-phrase test skipped: changing a phrase now means re-running SAM on every scene; recorded as a limitation. New: `scripts/sam_coverage.py` lists missing (scene, camera, concept) responses in a cache ([#23](https://github.com/Mayank-1024/longtail/pull/23)). |
| C4 Mask to LiDAR | Merged | [#4](https://github.com/Mayank-1024/longtail/pull/4) | Done, **below the plan's bar:** 12 of 20 real SAM masks correct by eye (plan: 17). 2 wrong clusters, 3 thin masks read as FAIL (A-50, A-46), 2 correctly UNVERIFIABLE, 1 unclear. Clip ids match Mayank's (24 of 24). |
| C5 Tracks and ego path | Merged | [#5](https://github.com/Mayank-1024/longtail/pull/5) | Done, **below the plan's bar:** 34 parked cars; positions within 0.1 to 0.4 m of the labels, but C5's speed jitters (median under 0.3 m/s on 5 of 34). `stopped()` is never wrong on them but judges only 16% of steps (A-77). |
| C6 Verifier | Merged | [#6](https://github.com/Mayank-1024/longtail/pull/6) | No separate real-data check; the evaluation exercises it. |
| C7 Ground truth | Merged (#21) | [#10](https://github.com/Mayank-1024/longtail/pull/10) via #21 | Done on the 4 local scenes: 5 random positives checked by eye, 5 of 5 correct. Two notes for the report: the ground truth has no occlusion test (A-122), so some positives are hard to see (a night scene) and recall is a lower bound; a parked pickup counts as a truck (A-123). Mayank built the ground truth on all scenes. |
| C8 Metrics, ablation, breaks | Merged (#21) | [#11](https://github.com/Mayank-1024/longtail/pull/11) via #21 | The evaluation run is Mayank's, on the cluster. Configurations (a) to (c) only: (d) is not run, since Glimmer was cut (A-213). Held-out results on 102 scenes are on `main` (#26, Mayank), with the tables written by `scripts/report_results.py`. |
| C9 Visualization export | Merged | [#7](https://github.com/Mayank-1024/longtail/pull/7) | **Pending:** an export on a real scene, for its size and time. |
| C10 Fresh-clone test | Run on Nov 3 (we submit Nov 3); rerun once the demo cache is published | [#25](https://github.com/Mayank-1024/longtail/pull/25) | `scripts/fresh_clone.sh` on a fresh clone of `main` (8e8830c), scene 001 downloaded: 13 pass (clone, `uv sync`, ruff, 315 synthetic tests, the download, 29 real-data tests, no `.env`, gitleaks, LICENSE, PandaSet credit, SAM acknowledgement, tags `eval-freeze` and `eval-final`). Open: the repo is private; `longtail demo --fetch-cache` gets a 404 (no GitHub release yet), so `--replay --query` misses at the planner (an error, no charge); the hero search needs the camera images of all 29 index scenes; the web view needs `npm run build`. The "media tracked" failure is the 5 result-chart PNGs, not data. The script's demo step did not run (a shell function under `env -i`); fixed in a separate PR. |

Open failures, kept as strict expected failures and explained in `docs/eval-report.md`: C4's centroid sits about 0.3 m from a box's centre (LiDAR sees surfaces only), and C5 reads a parked car partly hidden by another car as moving for 1.2 s (C6 never lets it PASS `moving`).

## b. Decisions

Agreed on 2026-11-03 (from my side):

- **A-213:** Glimmer cut, so no configuration (d). The README and the video present (a) to (c) only.
- **A-142:** 102 scenes are the headline; confirmed by both (#26).
- **A-228:** 4 queries replaced on Nov 3 by the rule written down before counting.

Waiting on Mayank:

- **A-142:** confirmed by both in #26 (above). Earlier proposal text, kept for the record: the evaluation runs on every usable PandaSet scene: 103 minus 054 (A-240), so 102. This supersedes the plan's 30 and the 30-scene index (29 after A-240). Headline numbers are over the 102; index-only results are secondary. Registered in `docs/assumptions.md` (still needs Mayank's agreement), with the limits in `docs/eval-report.md`: the held-out split is by query, not by scene; development tuning used the same scenes; phrases untuned; recall a lower bound (A-122).
- **#21 review comments:** none blocking (sweep-cache memory against `--mem=24G`, the track-cache temp file name, the spend cap with parallel workers, dropped streams that may be billed).

The full register is `docs/assumptions.md`. The Oct 30 version of this file lists every assumption that touches Mayank's half.

## c. Interfaces your code calls

C1, C2, C4, C5, C6 and C9 are on `main`; C3, C7 and C8 come with #21.

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

## d. How the open work merges

1. Merged on 2026-11-03: #21 (`m/integration`, with Mayank's OK: C3, C7, C8 and his M1 to M8), #23 (`scripts/sam_coverage.py`) and #24 (A-142 and the eval-report notes). #9 and #14 were marked merged; #10 and #11 closed as superseded; Mayank's #15 to #19 are his to close. Then #25 (C10 prep), #26 (Mayank: held-out results, `longtail demo`, README) and the docs PR with the Nov 3 real-data checks.
2. #25 (C10): `scripts/fresh_clone.sh`, `scripts/report_results.py`, `docs/submission-checklist.md` and this status.
3. If `docs/status-chethan.md` conflicts during a merge, keep `main`'s version.

## e. What I need, in priority order

1. Done (Nov 3). **The SAM cache for scenes 001, 005, 018 and 058**, for the C4 mask check and the C5 parked-car check: the 24 full-frame clip indexes and their 120 responses (6 cameras x 5 concepts), nothing else. Save this as `bundle.py` in the repo root on the machine that holds the cache and run `uv run python bundle.py`; it writes `sam-cache-4-scenes.tgz` (I unpack it into `~/longtail-data/`, then check that the clip ids I build locally match yours):

   ```python
   import importlib.util, json, tarfile
   from pathlib import Path
   spec = importlib.util.spec_from_file_location("cov", "scripts/sam_coverage.py")
   cov = importlib.util.module_from_spec(spec); spec.loader.exec_module(cov)
   cache, scenes, concepts = Path("cache"), {"001", "005", "018", "058"}, cov.gold_concepts()
   n = 0
   with tarfile.open("sam-cache-4-scenes.tgz", "w:gz") as tar:
       for f in sorted((cache / "clips").glob("*.clip_index.json")):
           ix = json.loads(f.read_text())
           if ix["scene_id"] in scenes and ix["crop"]["name"] == "full":
               tar.add(f); n += 1
               for c in concepts:
                   r = cache / "sam" / "responses" / f"{cov.response_key(ix['clip_id'], c)}.txt"
                   if r.is_file():
                       tar.add(r); n += 1
   print(n, "files in sam-cache-4-scenes.tgz")
   ```

2. A coverage run over all 102 scenes before the evaluation, especially after the A-239 outage: `uv run python scripts/sam_coverage.py --cache cache --scenes-from data/pandaset` on the machine holding the cache (no images or key needed). It prints every missing (scene, camera, concept) and the scenes to re-run; exit status 1 on any gap. Full coverage is 102 x 6 x 5 = 3,060 responses.
3. Done (#26). Agreement on A-142.
4. Done (#26). The results directory of the held-out run (`longtail/eval/runs/all-heldout/results/`: `ablation_a.json`, `ablation_b.json`, `ablation_c.json`, `breaks.json`). Then `uv run python scripts/report_results.py <dir> --write docs/eval-report.md` fills in the tables.
5. Partly done (#26: `longtail demo` and the SAM acknowledgement; the Nov 4 run checks them). For C10 (Mayank's areas): a `longtail demo --replay` command (or the documented replay command it stands for) and a way for a fresh clone to get the replay cache; the repo made public; the SAM acknowledgement in the README.
