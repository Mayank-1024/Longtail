# Assumptions register

Choices the build plan does not fix, made before the contracts session so work can proceed. Each one can change later; the tags make every dependent line findable.

- **Tag format:** every line of code or config that depends on an assumption carries a comment of the form `# ASSUMPTION A-04: Pandar64 only`.
- **Status:** `ASSUMED` until both owners (or the named owner) agree, or a check on real data confirms it; then `CONFIRMED`.
- **Owner:** `C` is Chethan, `M` is Mayank, `both` needs the two of us.
- **Files where used:** must list exactly the files that carry the tag, or `—` if none yet. `tests/test_assumptions.py` fails when the tags and this column disagree, or when a tag uses an ID that is not here.

| ID | Assumption | Owner | Status | Files where used |
|---|---|---|---|---|
| A-01 | Dataset is PandaSet. | both | ASSUMED | — |
| A-02 | `contracts.py` follows plan section 4, except where another assumption changes it. | both | ASSUMED | `longtail/contracts.py` |
| A-03 | Units are metres and seconds. Ego frame: x forward, y left, z up. `frame_idx` is the 0-based camera frame index; `ts` is that frame's dataset timestamp in seconds. | both | ASSUMED | `longtail/contracts.py` |
| A-04 | LiDAR: Pandar64 only (devkit sensor 0). The sensor id is kept with every point set. | C | ASSUMED | — |
| A-05 | Projection keeps points with camera depth > 0.5 m. | C | ASSUMED | — |
| A-06 | `PredicateCall.name` gains `ego_moving` and `ego_stopped` (no actors). Needs Mayank's approval. | both | ASSUMED | `longtail/contracts.py` |
| A-07 | `occluded_by` uses an amodal target box: the visible box extended down to the ground plane and up to a class-prior height. | C | ASSUMED | — |
| A-08 | Every `PredicateResult` puts a numeric `margin` in evidence (how far the value clears its threshold), for ranking. | both | ASSUMED | `longtail/contracts.py` |
| A-09 | Ground truth uses its own frozen thresholds in `configs/gt.yaml`, not `verify.yaml`. | C | ASSUMED | — |
| A-10 | A model claim is false when no labelled cuboid of the claimed class projects onto the claimed box with IoU >= 0.3. Behaviour isn't scored. | C | ASSUMED | — |
| A-11 | SAM 3.1 output follows plan section 3.1 until the smoke test confirms it. | C | ASSUMED | — |
| A-12 | Track fragments also merge within a single camera, using the same 1 m centroid rule. | C | ASSUMED | — |
| A-13 | Evaluation costs are reported with a cold cache. | C | ASSUMED | — |
| A-14 | Concept phrases are tuned on 3 PandaSet scenes outside the 30-scene index. `taxonomy.yaml` holds a ranked list of phrases per class. | C | ASSUMED | — |
| A-15 | The label guard covers cuboids and semantic segmentation. Only per-modality devkit loaders are used, never `sequence.load()`. | C | ASSUMED | — |
| A-16 | Ground truth deduplicates cuboids seen by both LiDARs. | C | ASSUMED | — |
| A-17 | The verifier engine returns unverified findings too. Only the agent's `finish` tool filters them out. | both | ASSUMED | `longtail/contracts.py` |
| A-18 | LiDAR points keep their per-point timestamp. `lidar(ts)` returns Nx5 (x, y, z, intensity, t) instead of the plan's Nx4. | C | ASSUMED | — |
| A-19 | The taxonomy is a fixed tuple in `contracts.py`: pedestrian, cyclist, car, truck, bus. Dataset labels map onto it outside the contract. | both | ASSUMED | `longtail/contracts.py` |
| A-20 | `Frame2D.box_xyxy` is an inclusive pixel box in the source image; `mask_rle` is COCO-style compressed RLE `{size: (h, w), counts}`; `mask_rle` and `conf` may be `None`. | both | ASSUMED | `longtail/contracts.py` |
| A-21 | Every contract model rejects unknown fields (`extra="forbid"`), so planner typos fail validation and get sent back for retry. | both | ASSUMED | `longtail/contracts.py` |
| A-22 | `PredicateResult` carries `required` (copied from its `PredicateCall`) so `Finding.verified` can be computed from the finding alone. A finding with no required results is not verified. | both | ASSUMED | `longtail/contracts.py` |
| A-23 | Predicate signatures follow the plan C6 table: `within_distance` and `near` take `m` (> 0), `relative_position` takes `side` (left, right, ahead, behind), `occluded_by` takes `cls` (in the taxonomy), `near` takes 2 actors, ego predicates 0, the rest 1. Unknown arguments are rejected. | both | ASSUMED | `longtail/contracts.py` |
| A-24 | `evidence.margin` (A-08) is required exactly when `value` is set; an UNVERIFIABLE result with no value has no margin. | both | ASSUMED | `longtail/contracts.py` |
