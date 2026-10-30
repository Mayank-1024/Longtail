# Evaluation file formats

What C7 and C8 read from Mayank's side, and what they write. Pydantic models: `longtail/eval/gt.py` (queries file) and `longtail/eval/formats.py` (baseline and triage). Times are dataset timestamps in seconds (A-03, A-88); boxes are inclusive full-image pixels `[x1, y1, x2, y2]` (A-20).

## Queries file (M7 → C7, C8), A-128

YAML or JSON. Example with all four tiers' shapes: `tests/fixtures/gold_synthetic.yaml`.

```yaml
format: longtail.queries/v1
queries:
  - query_id: q07                 # unique; becomes gt/q07.json
    text: a pedestrian crossing in front of us while we move
    split: heldout                # dev | heldout
    tier: behaviour               # single | behaviour | relational (optional)
    gold:                         # a ScenarioSpec (contracts.py), clarification null
      query: a pedestrian crossing in front of us while we move
      actors: [{id: a1, cls: pedestrian, sam_prompt: person}]
      predicates:
        - {name: exists, actors: [a1], args: {}}
        - {name: crossing_ego_path, actors: [a1], args: {}}
        - {name: ego_moving, actors: [], args: {}}
      min_duration_s: 0.5
      assumptions: []
      clarification: null
```

`longtail gt build <file>` writes `gt/<query_id>.json` and `gt/summary.json`, and prints any query with zero positives (replace it before the freeze).

## Baseline (a): Spark Open, describe then match (M5 → C8), A-135

Two files in `runs/baseline_a/` (git-ignored). Without them, `--config a` stops with "waiting on M5".

`claims.jsonl`: one VLM claim per line, cached per window and shared across queries.

```json
{"claim_id": "002/w014/c3", "scene_id": "002", "camera": "front_camera", "frame_idx": 41, "ts": 1557540413.85, "cls": "pedestrian", "box_xyxy": [812.0, 402.0, 860.0, 530.0], "behaviour": "walking across"}
```

`rankings.json`: per query, the windows the baseline returns (any order; ranked by `score`), each citing the claims it rests on.

```json
{
  "format": "longtail.baseline_a/v1",
  "model": "muse-spark-open",
  "queries": {
    "q07": {
      "matches": [
        {"window_id": "002/w014", "scene_id": "002", "t0": 1557540412.6, "t1": 1557540416.6, "score": 0.83, "claim_ids": ["002/w014/c3"]}
      ],
      "cost_usd": 0.042,
      "seconds": 11.5
    }
  }
}
```

- `cls` is a taxonomy class (A-19). Cited claims are assigned to the gold spec's actors by class; an actor with no cited claim of its class is unbound, so that match cannot score.
- A match scores when its window overlaps a ground-truth positive with temporal IoU >= 0.3 and a cited box has IoU >= 0.3 with that positive's labelled box in the same camera frame (A-127).
- `cost_usd` is at list price including cache hits (A-13); `seconds` is wall time for the query.
- C8 runs the verifier on every cited box. That gives the hallucination-catch rate for the baseline's claims.

## Configuration (d): Glimmer triage (M4 → C8), A-135

`runs/triage_d/windows.json`. Without it, `--config d` stops with "waiting on M4".

```json
{
  "format": "longtail.triage_d/v1",
  "model": "muse-glimmer",
  "queries": {
    "q07": {
      "windows": [
        {"window_id": "002/g031", "scene_id": "002", "t0": 1557540411.6, "t1": 1557540415.6, "cameras": [], "score": 0.7, "source": "glimmer"}
      ],
      "windows_total": 520,
      "cost_usd": 0.0,
      "seconds": 4.2
    }
  }
}
```

- `windows` are `CandidateWindow`s (contracts.py) after the per-query filter; `cameras: []` means all six.
- `windows_total` is how many windows the filter saw, so the result can report the share kept.
- The tagging run is query-independent. Spread its cost over queries, or put it in each query's `cost_usd` at list price; say which in the README.

Configuration (c) uses the unfiltered grid: 2 s windows every 1 s, padded by 1 s and clipped (A-136). So (d) and (c) differ only in which windows are kept.

## Outputs (C8)

- `longtail/eval/results/ablation_<config>.json` (`longtail.ablation/v1`): commit, thresholds hash, split, scenes, every metric with its 95% bootstrap interval (1,000 resamples over queries, fixed seed), and per-query rows.
- `longtail/eval/results/ablation.png`: P@10, recall and catch rate per configuration, with intervals.
- `longtail/eval/results/breaks.json` (`longtail.breaks/v1`): configuration (c) clean and under each break, which metrics dropped, and any hole.
