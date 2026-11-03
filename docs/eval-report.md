# Evaluation report

Results and every deviation from the build plan, with the evidence for it. Numbers here come from tests or scripts in the repository; real-data results are added when PandaSet and SAM 3.1 output exist. Threshold changes after the freeze (plan C6) are recorded here with their justification on development queries only.

## C1: dataset adapter

### Load time: the plan's 5 s depends on the machine

The plan's done check is that one scene loads all six cameras and LiDAR for every frame in under 5 s (`test_real_scene_loads_in_under_5s`). On scene 001, 2026-11-03:

| Machine | Time |
|---|---|
| Mayank's M-series Mac, quiet | about 10 s |
| Chethan's Mac, files just downloaded (probably in the OS file cache), two runs | 3.7 s, 3.6 s |

Most of the time is the devkit reading the 80 LiDAR sweeps, so a cold disk costs the most. Nothing downstream depends on the 5 s: the evaluation loads each scene once per job. The test keeps the plan's 5 s and is not marked as an expected failure; `LONGTAIL_LOAD_BUDGET_S` raises the budget for the cluster's shared disk, which reads a cold scene far slower than a laptop.

### Ego frame (A-25)

The first real-data run found the ego frame 180 degrees off: forward was read as -y of the Pandar64 frame, from the devkit's static calibration file. On scene 001 the ego travels along +y, and the front, back, left and right cameras look along +y, -y, -x and +x. Fixed in #20, before any evaluation run. The verifier and the ground truth were wrong the same way, so their agreement would not have shown it; the real-data axis tests did.

## C3: SAM 3.1 client

### Concept phrases were not tuned

The plan's concept-phrase test (2 or 3 phrases per class on 3 development scenes, with the winners going into `configs/taxonomy.yaml`) was not run. Changing a phrase now would mean re-running SAM on every scene, so the phrases are the ones first written in the taxonomy. SAM's recall per class is not measured, and a class whose phrase SAM finds poorly loses recall in configurations (b) and (c). The 3 phrase-tuning scenes in the index were never used for tuning.

## C4: mask to LiDAR association

Everything below was measured on the synthetic scene (`tests/synthetic.py`), with the camera-side and LiDAR-side occlusion model on (A-43). The real-mask check (17 of 20 correct by eye) is still to do; see `tests/test_associate_real.py`.

### Centroid check: the plan's 0.15 m does not hold against the box centre

The plan's done check is a centroid error under 0.15 m. LiDAR returns lie only on the faces turned toward the sensor, so the median of a cluster sits on the near face, about half the object's depth in front of its centre.

| Crosser (0.6 x 0.6 x 1.7 m), 30 frames | Horizontal error to the box centre |
|---|---|
| Faces the LiDAR can see (realistic) | 0.29 to 0.36 m, median 0.30 m |
| All faces sampled (closer to the plan's "box of points") | 0.00 to 0.32 m, median 0.09 m; 21 of 30 frames under 0.15 m |

With all faces sampled, frames 8 to 16 still miss: DBSCAN's eps of 0.5 m splits the hollow 0.6 m box into front and back faces and keeps one.

**Revised check:** every frame's cluster consists only of the crosser's points (it does, in all 30 frames), and the error stays under 0.15 m plus half the box's larger horizontal side. The plan's literal check is kept as a strict expected failure (`test_crosser_centroid_error_plan_criterion`), so it flags itself if a later change (for example an amodal centre) makes it pass. Downstream, C5 and C6 should expect up to about 0.3 m of near-face bias for a pedestrian; C6's 0.5 m distance tolerance absorbs it.

### Visibility filter (A-63), not in the plan

The LiDAR sits above the cameras (0.3 m in the synthetic rig; more on a real roof mount), so it sees past an object onto what is behind it, and those points project inside the object's mask. In synthetic frames 22 to 24, about 350 wall points landed in the crosser's mask against about 75 crosser points, and the plan's "nearest depth segment holding 30%" rule returned a 13 to 18 m slab of wall.

Fix: after ground removal, per 8 px image cell, drop points more than 1.0 m behind the nearest point in the cell. With it, every frame's cluster is pure crosser; without it (`visibility_cell_px: 0`), those frames are mostly wall (`test_wall_points_seen_past_the_object_are_dropped`).

### Local ground planes (A-49) instead of one plane per sweep

The plan fits one RANSAC plane per sweep. PandaSet is largely San Francisco, and one plane misses the ground on a hill. Ground is instead fitted per object, on sweep points within 3 m of its in-mask points, falling back to the sweep's plane.

On a synthetic slope (flat, then a 15% grade from 8 m ahead), the in-mask ground points of a dilated crosser mask over three frames were:

| Plane | Ground points kept | Crosser points kept |
|---|---|---|
| Local (A-49) | 0 | over 80% (the bottom 0.2 m is cut by design) |
| Global, one per sweep | all of them | all |

Two details make the RANSAC fit reliable: seeds come from the lowest 30% of points (plain sampling rarely found the ground when it was a minority of nearby points), and a plane is rejected when more than 5% of the region lies over 0.3 m below it, so a car roof is never taken for ground.

Height (A-53) is now the cluster's top minus the local plane's height under the cluster's median footprint, not the z range: the z range lost the 0.2 m ground cut and gave 1.4 m for the 1.7 m crosser. Measured from the plane it is 1.6 to 1.7 m on flat ground and 1.61 to 1.63 m on the slope.

### Fragment merging by extrapolation (A-56)

The plan merges tracks whose centroids stay within 1.0 m over at least half of their shared timestamps. Two fragments of one camera's track never share timestamps, so that rule cannot apply to them. The earlier fragment's centroid is extrapolated at constant velocity (fitted over its last 1 s) to the later fragment's first sweep; they merge when within 1.0 m and the gap is at most 1.0 s. A crosser track split into frames 0 to 11 and 16 to 29 merges back into one track; fragments of different objects stay apart.

### Cross-camera merging on shared LiDAR points (A-74)

Found while testing C5. A 4.5 m car passed 2.0 m away is seen in parts by the front, front-right and right cameras; the parts' centroids are more than 1 m apart, so the plan's rule left them as separate tracks, none covering the whole car. Clusters from adjacent cameras now also merge when they share LiDAR points at a timestamp: at least 5, and at least 15% of the smaller cluster.

The floor matters. Two cars parked nose to tail 0.3 m apart, where one car's mask spills slightly onto the other, share 2.3% of the smaller cluster; on shared points alone (5 or more) the two cars merged into one track, with the floor they stay apart (`test_cars_parked_nose_to_tail_stay_apart`).

The floor was first 30%. That split the close-pass car: its front and front-right views share only 21% to 28% of the smaller cluster (the synthetic cameras overlap by about 5 degrees). At 15% those views merge again, and the nose-to-tail cars, at 2.3%, still stay apart.

## C5: 3D tracks and the ego path

Synthetic scene, Savitzky-Golay window 5 (plan), quadratic fit (A-66). "Run" is the longest continuous time above 0.5 m/s, which is what C6's `moving` predicate needs for 1 s; a parked object must never get a run of 1 s or more.

### The median drifts with the visible part; the box centre mostly does not (A-68)

| Object | Median position: median / max speed, longest run | Box centre (A-68): median / max speed, longest run |
|---|---|---|
| Crosser, 1.5 m/s (front camera) | 1.49 / range 0.96 to 1.54 m/s | 1.50 / range 1.11 to 1.94 m/s |
| Parked car, ego at 2 m/s, crosser passes in front of it | 0.29 / 3.61 m/s, 0.7 s | 0.00 / 0.45 m/s, 0 s |
| Close-pass car, ego at 5 m/s: front, front-right and right merged (A-74) | 1.29 / 4.82 m/s, 1.6 s | 0.04 / 5.31 m/s, 0.3 s |
| Close-pass car: back camera | 1.30 / 2.60 m/s, 1.2 s | 0.02 / 5.08 m/s, 0.6 s |
| Parked car partly hidden by the close-pass car | 1.37 / 5.12 m/s, 2.1 s | 0.91 / 20.78 m/s, 1.2 s |

With the median, C6 would call the close-pass car moving (runs of 1.6 s and 1.2 s). With the box centre no run reaches 1 s (at most 0.6 s), so C6 reads it parked.

Why the median fails: LiDAR sees only the faces toward the sensor, and each camera sees only the part of an object inside its field of view, so the cluster's median follows whichever part is visible. As the ego passes a long car, that part slides along the car at the ego's speed.

Why a plain box fit is not enough: the first L-shape box centre (nearest edge plus half the track-wide size) still read 2 to 7 m/s on the close pass. It needed three refinements, all in A-68 and A-73:

1. Erosion shaves a few centimetres off a face seen head-on but up to 0.3 m off a side face seen at a grazing angle, so the face toward the sensor (whose plane does not move) anchors the centre, and only when the sensor is outside the cluster's extent by more than the erosion margin.
2. An end cut by the image border or an occluder is not a real boundary; such ends are detected per camera (with distances scaled to the cluster's point spacing in pixels: the synthetic car's points are about 25 px apart at 2 m) and never used as anchors.
3. Long objects need the cross-camera merge on shared points (A-74), or no track ever sees both ends.

Tried and dropped: registering each frame's points onto the previous frame's (translation-only ICP). It read the occluded parked car as exactly 0 m/s, but a flat face sliding along itself fits equally well at any in-plane shift, so it misread the crosser (0.6 m/s) and the close pass (up to 5.3 m/s).

### Still failing

- The crosser's speed is right as a track (median 1.50 m/s) but swings from 1.11 to 1.94 m/s step to step, as its visible faces change. The plan's "1.5 plus or minus 0.1" holds for the track's median, not for every step; C6's 1 s duration rule absorbs the swings.
- A parked car partly hidden by the close-pass car still reads as moving for 1.2 s (`test_occluded_parked_car_during_close_pass`, a strict expected failure). Occlusion by a nearer object that itself slides across the view is not caught by the end tests.
- A 1 s smoothing window (11) trims the spikes (close pass 1.87 m/s max, the occluded car 0.8 s) but was not adopted: the plan fixes 5, and the change belongs with C6 tuning on development queries.

### Ego path

The corridor follows the ego's actual future path for 3 s, half-width 1.5 m, checked against the analytic arc for a turning ego. A stopped or slow ego gets at least 10 m straight ahead (A-69), so a pedestrian crossing in front of a car waiting at a light is still in its path.

## C6: verifier predicates and engine

All on the synthetic scene. Every predicate has a PASS, a FAIL and an UNVERIFIABLE test with readable evidence (`tests/test_verify.py`): a reason code (A-79), the frames, per-status and per-reason counts, an example frame with its numbers, the threshold, the predicate's own longest PASS run, the binding search, and a margin whenever there is a value.

### Where the plan's predicates did not work as written

- **exists.** The plan gives a size prior only for pedestrians; priors for every class were added (A-76), with loose lower bounds because a cluster is only the visible surfaces (a car seen from behind measures 1.9 x 0.9 m). Height uses C4's ground-based measure (A-53). A frame where coverage was fine but nothing survived ground removal and clustering FAILs (`no_object_points`) rather than UNVERIFIABLE. The wall-only cluster FAILs on size (`wrong_size`: 2.75 m long against the pedestrian's 1.2 m); empty space is UNVERIFIABLE (`no_coverage`).
- **The 50 ms sync rule rarely fires at 10 Hz.** With sweeps every 100 ms, every camera frame is within 50 ms of some sweep, so `unsynced` can only happen at a log's ends or beside a dropped sweep. The reason code is tested with a 5 Hz LiDAR and a 70 ms camera offset.
- **within_distance.** Uses the range of C4's median, which sits on the near face, about 0.3 m nearer than the object's centre for a pedestrian (C4 above). The plan's 0.5 m tolerance absorbs it.
- **moving / stopped.** Judged by net displacement over 1 s (as asked) with the plan's hysteresis. The plan says nothing about where the hysteresis starts (A-80) or about position jumps; jumps faster than 10 m/s^2 are now UNVERIFIABLE (`implausible_jump`, A-77). That limit catches the C5 spikes, but it also shows how often they occur: see the occluded parked car below.
- **crossing_ego_path.** A FAIL ("stays outside") needs the full 3 s look-ahead, so in the last 3 s of any track the predicate can only PASS or be UNVERIFIABLE (`track_too_short`); on PandaSet's 8 s scenes that is the last 3 s of every scene. An actor already inside the corridor and moving across it now PASSes (A-81). With the ego stopped, the 10 m minimum corridor (A-69) lets a pedestrian crossing 8 m ahead PASS.
- **relative_position.** "At least half of the window" is judged in a 1 s window around each frame (A-78), so it can combine with the other predicates frame by frame; sides are half-planes (A-82).
- **near.** No tolerance in the plan; none added. It measures between C4 medians, so each actor's near-face bias applies (A-83).
- **occluded_by.** The amodal box as specified (the visible box extended down to the ground and up to the class height, A-07) never sees an object hidden at its side: the visible box simply ends where the occluder begins, so in the synthetic scene the parked car partly behind the crosser read 0% covered in every frame. The box now also includes the C5 box fit, which knows the car's track-wide size (4.2 x 1.7 m measured, 4.5 x 1.8 m true), projected from the ground to the typical height (A-93). With it the parked car PASSes (30%, 22%, 22% covered) while the crosser is in front of it, and FAILs once the crosser has moved on. Coverage uses bounding boxes (A-84), which overstates it for occluders that are not box-shaped.

### Engine

- Findings: one per window, the best binding (plan). The plan's "best-scoring" picked, among unverified bindings, the one with the least evidence: UNVERIFIABLE (-0.5) outscores FAIL (-1), so a candidate the sensors barely saw beat the track that clearly showed the claim false. Unverified bindings now rank first by the number of frames judged PASS or FAIL (A-85).
- Joint timing: verified needs one run of frames where every required predicate PASSes together for min_duration_s (A-91). Without such a run each predicate is summarised over the whole window, and its own longest PASS run is reported, so "it did cross, but not while the ego was moving" stays readable.
- One finding per window means a window with two true matches (two pedestrians crossing) reports only the better one; the plan's design, noted here as a limit.

### The occluded parked car and the close pass (from C5)

| Claim | Result | Frames: PASS / FAIL / UNVERIFIABLE | Reasons |
|---|---|---|---|
| Occluded parked car: moving | FAIL (`not_moving`) | 0 / 6 / 34 | implausible_jump 33, not_moving 6, not_observed 1 |
| Occluded parked car: stopped | PASS for 0.6 s | | |
| Close-pass car: stopped | PASS for 0.6 s (back camera's track) | | |
| Close-pass car: moving | FAIL (`not_moving`) | 0 / 6 / 34 | implausible_jump 19, not_observed 15, not_moving 6 |

Why "stopped" PASSes on a 0.6 s run (A-94): every frame is judged on a full 1 s window of clean steps, centred on the frame but kept inside the speed series. For the back camera's track (2.2 to 3.9 s), any window touching the position jump at 2.6 to 2.8 s is UNVERIFIABLE, so only the frames from 3.4 s on have a clean window, and near the series' end they all share the last second, 2.9 to 3.9 s (net speed 0.006 m/s, every step below 0.3 m/s). The 0.6 s is the run of PASS frames, which needs only min_duration_s (0.5 s); the evidence behind it is the full 1 s, and each verdict's window is in its evidence (`judged_window`).

Neither car ever PASSes moving. The implausible-jump rule is what stops the C5 spikes from becoming "moving", but most of their frames end up UNVERIFIABLE rather than stopped: the "stopped" verdict rests on 0.6 s of evidence. That is honest, and it is thin; better positions for speed (C5) would turn those frames into evidence.

## C7: ground truth from labels

Synthetic only (the scene's true boxes as PandaSet-style cuboids). The four synthetic gold specs give the expected positives: the crosser from 0.0 to 2.5 s, the parked car and the pedestrian behind it over the whole 3 s, and none for the cyclist (flagged).

### Where the plan's C7 spec did not work as written

- **"The same predicate definitions with label geometry"** works for eight of the ten predicates as code, not just as definitions: they run on cuboid tracks unchanged. Two read LiDAR clusters and need label versions (A-125): `exists` becomes "the cuboid is visible" (a point count or size prior would make the labels disagree with themselves), and `occluded_by` compares projected cuboid boxes, which are amodal by construction. `within_distance` called `exists` directly; it now goes through `Context.evaluate`, so the label version is used (A-120).
- **"Positive windows (scene, t0, t1, label ids)"**: windows depend on how a pipeline cuts the scene, so positives are events instead (A-126): maximal intervals where the spec holds for some binding, with overlapping bindings merged. This changes A-95 (owned by both) and waits on Mayank's agreement.
- **"Each bound actor lies within 2 m of the labelled object"**: the configurations without LiDAR (a and b) have no 3D position. They match on box IoU >= 0.3 against the label's projected box instead, A-10's rule (A-127).
- **Motion from geometry** is circular in one way: the verifier's motion and the ground truth's are the same code. PandaSet labels moving or stopped per frame (`attributes.object_motion` for vehicles and bicycles, `attributes.pedestrian_behavior` for pedestrians), so `motion_source: attribute` is available and `longtail gt motion-report` measures how often the two agree (A-124). On the synthetic scene they agree on all 90 judged frames; the real number needs PandaSet.

## C8: metrics, ablation and breaks

Synthetic only: one 3 s scene and the four synthetic gold specs (dev and held-out together), with a stand-in SAM that returns the true objects (confidence 0.9) plus two confident hallucinations (0.97): a "person" on the wall and a "cyclist" on the empty sky. These numbers test the machinery. With 4 queries the intervals are wide and say nothing about PandaSet. Re-run on 2026-10-30 after the A-10 and P@10 changes.

| Configuration | Precision | Recall | P@10 (returned / 10) | Catch rate | False rejection | Hallucination rate | Cost / query | Seconds / query |
|---|---|---|---|---|---|---|---|---|
| (b) + SAM | 0.25 [0.00, 0.75] | 0.33 [0.00, 1.00] | 0.025 [0.000, 0.075] (1.0) | n/a (no verifier) | n/a | 0.35 | $0.060 | 0.11 |
| (c) + LiDAR verification | 1.00 [1.00, 1.00] | 0.67 [0.00, 1.00] | 0.050 [0.000, 0.100] (0.5) | 1.00 [1.00, 1.00] | 0.29 [0.00, 0.48] | 0.30 | $0.075 | 1.0 |
| (c), verifier off | 0.50 | 0.67 | 0.050 (1.0) | 0.00 | 0.00 | 0.30 | | |
| (c), calibration 2 degrees + 0.3 m | 0.00 | 0.00 | 0.000 (0.5) | 1.00 | 0.41 | 0.30 | | |
| (c), LiDAR 200 ms late | 1.00 | 0.33 | 0.025 (0.25) | 1.00 | 0.39 | 0.30 | | |

Precision (correct findings over findings returned, pooled over queries) is the headline next to recall (A-134). P@10 divides by 10 (A-134). Each query here has at most one positive, so 0.1 is the most any query can score. (c) returns nothing for two queries:
- **The cyclist:** none exist in the scene.
- **The pedestrian behind the car:** only its head shows above the roof, too few points for exists.

The hidden pedestrian also gives the false-rejection rate. Its 60 head boxes are true claims under A-10, and the verifier rejects them all (60 of 210 true claims). Costs are at list price with a cold cache (A-13): six $0.01 SAM requests for each class a query asks for.

### A-10 changed: intersection over the claimed box

A claim is now true when at least half of its box lies inside a projected label box of its class (`claims.min_inside_frac: 0.5`). Under the old rule, IoU >= 0.3 against the projected cuboid, the hidden pedestrian's 9 x 5 px head box scored IoU 0.06 against its own label: its 60 boxes counted as hallucinations, and the verifier's rejections counted as catches. The two rules on configuration (c):

| A-10 rule | False claims | Caught | Catch rate | True claims | Rejected | False rejection |
|---|---|---|---|---|---|---|
| IoU >= 0.3 (before) | 150 | 150 | 1.00 | 150 | 0 | 0.00 |
| >= 50% of the claimed box inside (now) | 90 | 90 | 1.00 | 210 | 60 | 0.29 |

The catch rate is unchanged. The verifier's real cost shows instead: it rejects occluded objects it cannot see in LiDAR. Box-only matching of findings (A-127) still uses IoU >= 0.3, so in configurations (a) and (b) a heavily occluded object's box will not match its positive.

### Breaks

A break passes when it lowers precision, the catch rate or recall (A-141). All three pass; `breaks.json` records which metrics each one moved.

- **Verifier off, moved precision and the catch rate.** Precision falls from 1.00 to 0.50: the unverified findings, including the ghosts, come back. The catch rate falls from 1.00 to 0.00. P@10 does not move, because the unverified findings rank below the verified hits.
- **Calibration shift, moved precision and recall** (both to 0.00). The crossing finding is lost, and a finding the labels do not support gets verified.
- **LiDAR 200 ms late, moved recall. Fails safe: shows up as lost recall** (0.67 to 0.33). Precision stays at 1.00 and the catch rate at 1.00: nothing wrong is verified and nothing false gets through. Static objects do not move in the world frame, so their points still land in their masks. The moving crosser's points slip about 0.3 m, and the verifier rejects it rather than verifying something wrong. P@10 drops from 0.050 to 0.025 only because the lost finding leaves an empty slot. Real scenes have more moving objects; re-check on PandaSet.

### Where the plan's C8 spec did not work as written

- **"Verifier off" is not a Scene perturbation.** It is a flag on configuration (c): every finding counts as verified and every claim as accepted (A-141). The other two breaks are Scene wrappers.
- **The catch rate needs a verifier.** Configuration (b) runs none, so it reports the hallucination rate instead (A-138). For (a) the verifier checks the baseline's cited boxes, as plan 3.4 says.
- **Uniform windows overlap** (2 s every 1 s, padded by 1 s), so one event yields a finding per window. Without merging the duplicates (A-137), they would fill the top 10, all but one unmatched.
- **P@10 over 10 is insensitive to false positives** when a query returns 10 or fewer findings and the hits rank first: then it counts hits, like recall capped at 10. That is why precision over all returned findings is the headline, with P@10 beside it.

## C9: visualization export

Per finding (`longtail viz`), measured on the synthetic scene; the second row is sized like PandaSet (an 8 s window, 1920x1080 images, about 76k LiDAR points per sweep), with a plain background in place of photos.

| Case | Overlays | JPEG total (mean each) | bev.json | timeline.json | Export time |
|---|---|---|---|---|---|
| Synthetic default: 25 frames, 640x480, 9k points per sweep | 150 | 5.5 MB (37 kB) | 3.4 MB | 7 kB | 1.0 s |
| PandaSet-sized: 80 frames, 1920x1080, 76k points per sweep | 480 | 131.5 MB (274 kB) | 3.7 MB | 21 kB | 13.9 s |

The bird's-eye file stays under its 5 MB budget (A-99) by thinning to about 2,700 points per frame at PandaSet size. The overlays dominate: six full-resolution cameras for every frame of an 8 s window is about 130 MB per finding, more with real photos. Downscaling overlays or exporting only the frames the web view shows would cut that; not done, since the plan fixes neither.

## Limits of the evaluation

- **Scenes:** all 103 PandaSet scenes except 054, whose camera timestamps are broken (A-240): 102 scenes (A-142), not the plan's 30. Results over the index scenes alone (29 once 054 is dropped), where reported, are secondary.
- **The held-out split is by query, not by scene.** The 15 held-out queries are scored on the same 102 scenes as the 10 development queries. Held-out numbers measure how the system does on new queries, not on new scenes or places.
- **Development tuning used the same 102 scenes.** Anything adjusted on development queries (thresholds, prompts) saw every scene the held-out queries are scored on.
- **Concept phrases were not tuned** (C3 above); per-class SAM recall is not measured.
- **Four queries were replaced on Nov 3, after the plan's Oct 31 freeze,** because they had no positives in any of the 103 scenes. The replacement rule was written down before counting (A-228).
- **No triage:** Glimmer was cut, so configuration (d) is not run (A-213).
- **Recall is a lower bound.** The ground truth counts a labelled object wherever its cuboid projects into a synced camera image within 60 m, with no occlusion test (A-122), so it includes objects that are hidden behind others or too dark to see. The system cannot find those, so measured recall understates recall on visible objects. Checked by eye on 5 random positives: in a night scene (058) most of the labelled pedestrians are hard to see or behind parked cars.
- **Pickups count as trucks** (A-123), as documented. A query for a truck can be satisfied by a parked pickup; one of the 5 positives checked by eye was exactly that (scene 058).
- **One dataset,** and the ground truth inherits the labels' errors (plan section 10).
