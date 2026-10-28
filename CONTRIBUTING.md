# Working on LongTail

- `main` is protected. Branch as `c/<task>` (Chethan) or `m/<task>` (Mayank), for example `c/c1-adapter`.
- Every pull request needs green CI and the other person's review. If a review waits more than 12 hours, the author may merge with green CI and a note in the PR.
- `longtail/contracts.py` is the only coupling between the two halves: change it only through a PR the other person approves.
- One issue per task ID from the build plan. Reference it in the PR (`Closes #n`).
- Add a line to `docs/build-log.md` every day you work.
- Never commit keys, dataset files, frames, caches or run outputs. Install the hooks once: `uvx pre-commit install`.

## Setup

```bash
uv sync
cp .env.example .env   # fill in your own keys
uvx pre-commit install
uv run pytest
```
