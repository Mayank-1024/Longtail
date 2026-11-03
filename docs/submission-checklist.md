# Submission checklist (C10)

What the build plan asks of the submission (C10, M8 to M10, sections 10 and 12), with how each item is checked. `scripts/fresh_clone.sh` checks the items marked *script*; the rest are checked by hand. The deadline is Wed Nov 4, 9 PM ET; our target is 6 PM ET.

State as of 2026-11-03, from a run of `scripts/fresh_clone.sh` on `main` (0351678) and a gitleaks scan of the full history.

## Fresh clone

| Item | Check | State |
|---|---|---|
| Clone, `uv sync` on a clean machine | script | Pass |
| `ruff check`, `pytest` on synthetic data (no dataset, keys or network) | script | Pass: 302 passed, 18 skipped, 2 xfailed |
| Download script, then the real-data tests on one scene | script | Pass on the 4 scenes already on disk (29 passed, `--data-dir`); the download step itself is to run on Nov 4 |
| `longtail demo --replay` with no key | script | **Missing:** no `demo` subcommand. A fresh clone also has no replay cache (`cache/` is git-ignored), so the demo needs its cache published somewhere a clone can fetch it (owner: Mayank, M6 and M8) |

## Rules

| Item | Check | State |
|---|---|---|
| Public repo | script (gh) | **Fail:** the repo is private |
| Licence: Apache-2.0 `LICENSE` | script | Pass |
| Dataset attribution: PandaSet, CC BY 4.0, in the README; frames shown only with attribution | script, by hand | Pass in the README; check the video's credits by hand |
| SAM acknowledgement: Meta's SAM License requires it (A-109) | script | **Fail:** the README does not mention the SAM License or `meta-sam-parser` |
| No secrets: no `.env` tracked; gitleaks over all history; no keys in screenshots or the video | script, by hand | Pass, 2026-11-03: no `.env` tracked; `gitleaks detect --source . --log-opts="--all" --redact` (8.30.1) on a fresh clone with all 27 branches found no leaks in 57 commits, and `gitleaks dir` on the files found none. CI's `secrets` job runs on every push. Screenshots and video by hand |
| No raw data, frames, cache or runs in the repo | script | Pass |
| No personal data anywhere | by hand | |
| The API binds to localhost; any hosted demo is replay-only with no keys | by hand | The README starts uvicorn without `--host`, so it binds to 127.0.0.1 |
| Built during the hackathon: build log, commit history | by hand | `docs/build-log.md`; its Oct 29 to 30 rows are out of order (cosmetic) |

## Evaluation (plan section 10)

| Item | Check | State |
|---|---|---|
| Queries frozen by the tag `eval-freeze` (Oct 31), before tuning | script | **Fail:** no tag. The 4 queries replaced on Nov 3 (A-228) need noting wherever the tag is described |
| Held-out run from the tag `eval-final`, outputs committed under the results directory (A-140) | script (tag), by hand | **Fail:** no tag; no committed results yet |
| Numbers with 95% bootstrap intervals; catch rate with the false-rejection rate; cost and time per query | script | `scripts/report_results.py <results dir> --write docs/eval-report.md` writes them, with the breaks and what each moved; tested on the synthetic results. README rows are *not measured yet* |
| Limits stated: scene set, one dataset, label errors | by hand | In `docs/eval-report.md`; the README should say 102 scenes (A-142) |
| Configurations (a) to (c) only; no (d) (A-213) | by hand | |

## Video (M9) and form (M10)

| Item | Check | State |
|---|---|---|
| Video 3 minutes, burned-in captions | by hand | |
| X cut, 30 to 45 s | by hand | |
| A real caught hallucination, never staged; if none, the measured rate | by hand | |
| Submission form: both of us listed, video and repo links, model list | by hand | |
| Submitted by 6 PM ET, Nov 4 | by hand | |
