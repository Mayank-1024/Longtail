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

## Narration (about 400 words, 3:00)

Numbers: held-out split, 15 queries, 102 PandaSet scenes (A-142), from `docs/eval-report.md`. Read at an even pace; each block matches a shot above.

**0:00-0:15 (the catch).** "This vision model says there's a pedestrian here. The LiDAR finds no object in this box. LongTail rejects the claim, and that's the point: on our held-out queries, 96 percent of what a vision model claimed on its own was false."

**0:15-0:35 (the problem).** "Self-driving teams need the rare moments: a pedestrian half hidden by a car, a cyclist cutting in. They're buried in hours of logs, and a vision model alone says yes far too easily. LongTail finds them from one sentence, and proves each one before showing it."

**0:35-1:35 (the live query).** "I type: a pedestrian partly hidden by a car. Muse Wonder turns it into a scenario spec: which actors, which relations. The agent picks the scenes to search. SAM 3.1 tracks every person and car through all six cameras. Then the verifier checks each track against the LiDAR: is there a real object in that box, at that distance, at that moment? Only findings that pass come back. Each one gets an evidence card from Muse Spark Open, and every sentence on the card cites a measurement; a sentence that cites nothing is rejected. Here's the finding: six cameras, the bird's-eye view, and the predicate strip showing when the pedestrian was hidden. I approve it, and export it. This replays a recorded run: every model call, SAM track and LiDAR check comes from the recording, so it costs nothing."

**1:35-2:15 (the evidence).** "We measured it against PandaSet's human labels, on held-out queries. Spark Open alone found no correct events. Adding SAM tracks found 28 percent of them, at 9 percent precision. Adding the LiDAR check raises precision to 20 percent and catches 59 percent of the false claims that are left. It costs us: recall drops to 20 percent, the verifier wrongly rejects 46 percent of true claims, and the precision gain isn't clearly significant. Fixing what our real-data checks found raised recall to 24 percent, at a small cost in precision. We also broke the system on purpose: verifier off, calibration shifted, LiDAR out of sync. Every break made the numbers worse, so the evaluation can tell."

**2:15-2:40 (the models).** "Three Meta models do the work. Muse Wonder plans and runs the search. SAM 3.1 tracks. And Muse Spark Open writes the cards. Every paid call we made, the whole 102-scene evaluation included, cost about 68 dollars."

**2:40-3:00 (close).** "Everything here was built during the Meta Global AI Developer Hackathon: the commits are public, every assumption is written down, and every number traces back to a script in the repo. LongTail: find the long tail, and prove it."
