# C9 visualization export: schema

What `longtail viz` writes for the web view (M6), field by field. The machine-readable schema is [`viz-schema.json`](viz-schema.json) (JSON Schema 2020-12), generated from `longtail/viz/schema.py`; generate TypeScript types from its `$defs` (`VizManifest`, `BevExport`, `Timeline` and the types they use). A test fails if the file and the models drift apart. Schema version: `1.0`.

## Conventions

| Thing | Convention |
|---|---|
| Distances | Metres |
| Times (`ts`, `t0`, `t1`) | Dataset timestamps, seconds (A-03, A-88). Subtract the first frame's `ts` for a scene-relative clock. |
| `frame_idx` | LiDAR sweep index in the scene (0-based). The one index shared by overlays, the bird's-eye view and the timeline: the frame scrubber should step through it. |
| `camera_frame_idx` | The camera frame drawn for that sweep (C2 sync, A-105). Usually equal to `frame_idx` on PandaSet; not guaranteed. |
| Bird's-eye coordinates | Ego frame of that frame's own sweep (A-03): x forward, y left, z up, origin at the Pandar64 (A-25). Each frame has its own ego frame; nothing is in world coordinates. |
| Angles | Radians, counter-clockwise from +x (a left turn is positive). |
| Vectors | JSON arrays of numbers: `[x, y]` or `[x, y, z]` (fixed length; typed as `number[]`). |
| Paths | Relative to the directory holding `manifest.json`. |
| Reason codes | Strings from A-79 (`docs/assumptions.md`): `ok`, `no_coverage`, `out_of_range`, `unsynced`, `truncated`, `not_observed`, `track_too_short`, `unreliable_speed`, `implausible_jump`, `requires_exists`, `wrong_size`, `too_few_points`, `no_object_points`, `too_far`, `not_moving`, `not_stopped`, `outside_corridor`, `not_toward_corridor`, `wrong_side`, `not_near`, `not_occluded`, `shorter_than_min_duration`. |

## Layout

```
runs/<run>/
  verify_run.json          the saved verification run (spec, windows, findings, Track2Ds); C9's input
  viz/
    manifest.json          VizManifest
    <finding_id>/
      overlays/<frame_idx:03d>_<camera>.jpg
      bev.json             BevExport
      timeline.json        Timeline
```

## manifest.json (`VizManifest`)

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | `"1.0"` | |
| `run_id` | string | The run directory's name |
| `scene_id`, `dataset` | string | |
| `findings` | `FindingAssets[]` | Every exported finding, verified first, then by score |

`FindingAssets`:

| Field | Type | Meaning |
|---|---|---|
| `finding_id`, `window_id` | string | The Finding and the CandidateWindow it came from |
| `verified` | bool | `Finding.verified`; false findings belong in the "rejected by verifier" tab |
| `score` | number | `Finding.score` (A-85) |
| `t0`, `t1` | seconds | The finding's interval: the joint PASS run if verified, else the whole window (A-91) |
| `bindings` | `{actor id: track id}` | `"unbound"` when no track could be bound |
| `overlays` | `OverlayAsset[]` | One per camera per `frame_idx` in [`t0`, `t1`] |
| `missing_overlays` | `MissingOverlay[]` | Camera and frame pairs with no image, and why (no camera frame synced to that sweep) |
| `bev_path`, `bev_bytes` | string, int | The bird's-eye file and its size (under 5 MB, A-99) |
| `timeline_path` | string | |
| `export_seconds` | number | Time to export this finding |

`OverlayAsset`: `camera`, `frame_idx` (sweep), `camera_frame_idx`, `ts` (camera frame time, seconds), `path`.

## Overlays (JPEG)

One per camera and `frame_idx` in [`t0`, `t1`], at the camera's resolution (PandaSet 1920x1080; the synthetic scene 640x480 on a plain background). Drawn (A-108):

- LiDAR points coloured by camera depth: red near, through yellow and green, to blue at 50 m and beyond.
- Each bound actor in its own colour (cyan, magenta, green, orange, violet, in actor order): its mask at 35% opacity, its box, its C4 cluster points as larger dots, and a label `actor = track id`.
- A box with no mask (a box-only claim, e.g. the VLM baseline's) is drawn dashed in yellow and labelled `VLM claim: ...`, beside the verifier's cluster.
- A header: `VERIFIED` (green) or `REJECTED` (red), the finding id, camera, sweep, and up to two failing required predicates with their reason codes.

## bev.json (`BevExport`)

| Field | Type | Meaning |
|---|---|---|
| `frame` | `"ego"` | Every coordinate below is in the ego frame of its own `frame_idx` |
| `decimation` | `BevDecimation` | `max_range_m` (50), `voxel_m` (0.2), `max_points_per_frame` (from the budget), `budget_bytes` (5,000,000) |
| `frames` | `BevFrame[]` | One per `frame_idx` in [`t0`, `t1`] |

`BevFrame`:

| Field | Type | Meaning |
|---|---|---|
| `frame_idx`, `ts` | int, seconds | Sweep index and time |
| `points_xyz` | `[x, y, z][]` | LiDAR points within 50 m, one per 0.2 m voxel, evenly thinned to the cap; centimetre precision. z is useful for colouring (height above the LiDAR, which sits about 1.8 m above the road) |
| `ego_box` | `[x, y][4]` | Ego footprint, 4.8 x 1.9 m centred on the origin (A-100; a placeholder, PandaSet gives no body box) |
| `corridor` | `[x, y][]` | Ego corridor polygon (C5): left edge forward, then right edge back; close it when drawing |
| `corridor_centerline` | `[x, y][]` | The ego's actual next 3 s of path, from the origin; extended straight ahead to 10 m when shorter (A-69) |
| `corridor_extended` | bool | Whether that extension happened (the ego was slow or stopped) |
| `tracks` | `BevTrack[]` | The finding's bound tracks (A-107) |

`BevTrack`:

| Field | Type | Meaning |
|---|---|---|
| `track_id`, `cls` | string | |
| `actors` | string[] | Actor ids bound to this track |
| `position_xyz` | `[x, y, z]` or null | C4's median of the cluster; sits on the near face, about 0.3 m short of a pedestrian's centre. Null where the track has no sample |
| `box` | `BevBox` or null | C5 box fit (A-68): `center_xy`, `yaw_rad` (length axis), `length_m` (along `yaw_rad`), `width_m`. The size is track-wide; `length_m` is not always the longer side. Null where the centre could not be placed |
| `speed_mps` | number or null | Horizontal speed over ground (C5) |
| `velocity_xy_mps` | `[vx, vy]` or null | Velocity over ground in this frame's ego axes (A-70): a parked car reads [0, 0] even while the ego drives |
| `speed_reliable` | bool | False when the series was too short to smooth (A-71) |

## timeline.json (`Timeline`)

| Field | Type | Meaning |
|---|---|---|
| `finding_id`, `verified`, `t0`, `t1` | | As in the manifest |
| `frames` | `TimelineFrame[]` | Every frame of the candidate window, with `in_interval` true for [`t0`, `t1`] (the frames with overlays and bird's-eye data, A-106) |
| `predicates` | `TimelinePredicate[]` | One per predicate call in the spec, in spec order |

`TimelinePredicate`: `result_id` (matches the Finding's `PredicateResult`), `name`, `actors` (actor id to track id), `required`, `status` and `reason` (the summary in the Finding), and `entries`: one `TimelineEntry` per timeline frame with `frame_idx`, `ts`, `status` (`PASS`, `FAIL`, `UNVERIFIABLE` or `NOT_JUDGED`) and `reason` (null when `NOT_JUDGED`). `NOT_JUDGED` is in the schema for frames the engine did not evaluate; the current engine evaluates every window frame, so it does not occur yet.

## For the web view (M6)

- Drive the scrubber by `frame_idx`. Overlays and `bev.json` exist only for frames with `in_interval: true`; the timeline covers the whole window, so the predicate table can show why a rejected finding failed outside its interval too.
- The bird's-eye view is ego-centred per frame: draw each frame in its own coordinates (the ego stays at the origin, facing +x). Do not accumulate points across frames without transforming them.
- The predicate table's numbers (value, margin, threshold, example frame) are in the Finding itself (`PredicateResult.evidence`); the timeline adds the per-frame status strip.
- Box-only claims (the VLM baseline) appear as dashed yellow boxes in the overlays; the verifier's verdict is the header and the predicate table.
