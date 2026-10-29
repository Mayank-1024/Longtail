# Evaluation report

Results and every deviation from the build plan, with the evidence for it. Numbers here come from tests or scripts in the repository; real-data results are added when PandaSet and SAM 3.1 output exist. Threshold changes after the freeze (plan C6) are recorded here with their justification on development queries only.

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
