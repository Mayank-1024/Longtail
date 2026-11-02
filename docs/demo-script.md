# Demo video: shot list and capture rules (M9)

Three minutes, captions burned in, plus a 30 to 45 s cut for X. Narration in Mayank's own voice (local voice clone; the audio never goes into the repo). Footage is captured **after the Nov 3, 6 PM freeze**, from `LONGTAIL_MODE=replay` so every shot replays the recorded run exactly.

## Rules

- **Never stage a hallucination.** The opening shot is a real claim the verifier rejected on a real scene. If no such catch exists, the opening shows the measured catch rate instead.
- **Never use offline-demo mode** (synthetic scenes, scripted model) in any shot; its banner must never appear on screen.
- Every number on screen comes from `docs/eval-report.md`, `runs/ledger.jsonl` (`longtail costs`) or the held-out run, and is the same in the README.
- Show the mode badge (`replay`) once, and say in the narration that the demo replays a recorded run.

## Shots

| Time | Shot | Source | Capture |
|---|---|---|---|
| 0:00-0:15 | A real caught hallucination: the model's claimed box (dashed yellow, "VLM claim"), the empty LiDAR in that box, the REJECTED header | Baseline (a) claim that C8 found false and the verifier rejected | Overlay JPEG from `runs/<run>/viz/`, then the detail page |
| 0:15-0:35 | The problem: rare scenarios hide in hours of logs; a vision model alone says yes too easily | Narration over a scrub through six cameras | Detail page, play |
| 0:35-1:35 | A live query end to end: type it, (optionally) the clarifying question, the trace streaming, the verified results, a card with its citations, the detail page (cameras, bird's-eye view, predicate strip), approve, export | Web view in replay mode | Screen recording at 1440x900 |
| 1:35-2:15 | Ablation (a) to (c) with intervals, the catch rate and false-rejection rate, the three breaks | `docs/eval-report.md` chart | Chart image |
| 2:15-2:40 | The Meta models and what each does, cost per query | README model table, `longtail costs --markdown` | Slide |
| 2:40-3:00 | Built during the hackathon: commit timeline, repo link | `git log`, GitHub | Slide |

## Before recording

1. `uv run longtail costs --markdown` into `docs/costs.md`.
2. Pick the opening catch from C8's output (a baseline claim with FAIL or UNVERIFIABLE that the labels confirm is false).
3. Run the demo query once live so it is in the cache; then restart the server with `LONGTAIL_MODE=replay` and check it replays with no network.
4. Check the frames for faces and number plates before choosing close-ups (PandaSet's blurring is not confirmed).
