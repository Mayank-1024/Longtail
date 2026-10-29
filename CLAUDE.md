# LongTail: notes for Claude Code sessions

Joint proposal by Chethan and Mayank (proposed in the handoff PR; edit only by agreement).

## Before starting work

Read `docs/status-chethan.md` and, if it exists, `docs/status-mayank.md`: current status, open decisions and how the stacked PRs merge. The build plan is the source of the task IDs (C1 to C10, M0 to M10).

## Ownership

- Chethan: `longtail/data/`, `longtail/geometry/`, `longtail/vision/`, `longtail/verify/`, `longtail/eval/gt.py`, `longtail/eval/metrics.py`, `longtail/viz/`.
- Mayank: `longtail/llm/`, `longtail/agent/`, `longtail/triage/`, `longtail/explain/`, `api/`, `web/`.
- Do not edit the other person's folders. If something there needs to change, raise it in a PR comment.
- `longtail/contracts.py` is the only coupling between the two halves. Change it only through a PR the other person approves.

## Assumptions

- Every choice the build plan does not fix gets an ID in `docs/assumptions.md` (ID, assumption, owner C, M or both, status, files where used).
- Tag each dependent line `# ASSUMPTION A-xx: short text`, and keep `tests/test_assumptions.py` passing (it fails on unregistered or malformed tags and on a stale "files where used" column).
- Never change an assumption owned by the other person, or by both, without their agreement.

## Labels

Dataset labels are evaluation ground truth. Never read them outside `with eval_ground_truth():` (`longtail/data/guard.py`); `tests/test_label_guard.py` enforces where that context may be used.

## Status files

Each of us keeps a `docs/status-<name>.md`, updated at the end of every task on that task's branch. Mayank: a matching `docs/status-mayank.md` is suggested.
