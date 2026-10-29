# Build log

Everything in this repository was created during the Meta Global AI Developer Hackathon build window (Oct 26 to Nov 4, 2026). One dated line per person per day: what was built, what ran, what broke.

| Date | Who | What |
|---|---|---|
| 2026-10-28 | Mayank | Created the repository: licence, CI (ruff, pytest, gitleaks), package layout by owner, PR and issue templates, build log. |
| 2026-10-29 | Chethan | Draft `contracts.py` (plan section 4 plus tagged assumptions) with validation and a computed `Finding.verified`; assumptions register and a test that keeps tags and register in sync; pinned Python 3.11. Contracts await Mayank. |
| 2026-10-29 | Chethan | C1 adapter on the PandaSet devkit (pinned 59be180): Scene API, label guard over cuboids and semseg, download and index scripts, `longtail scenes ls`; ego axes derived from the devkit static calibration (A-25). Tested on a fake on-disk sequence; dataset not downloaded yet, so real-data checks skip. |
| 2026-10-29 | Chethan | C2: transforms, vectorised projection (170k points x 6 cameras in ~15 ms), camera-to-LiDAR sync with gap report, synthetic scene for C2 to C6 tests, overlay script; no lens distortion (A-38). Real-data checks (devkit 1 px, overlays, gap distribution) written but skipped: dataset not downloaded. |
