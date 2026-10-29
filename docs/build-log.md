# Build log

Everything in this repository was created during the Meta Global AI Developer Hackathon build window (Oct 26 to Nov 4, 2026). One dated line per person per day: what was built, what ran, what broke.

| Date | Who | What |
|---|---|---|
| 2026-10-28 | Mayank | Created the repository: licence, CI (ruff, pytest, gitleaks), package layout by owner, PR and issue templates, build log. |
| 2026-10-29 | Chethan | Draft `contracts.py` (plan section 4 plus tagged assumptions) with validation and a computed `Finding.verified`; assumptions register and a test that keeps tags and register in sync; pinned Python 3.11. Contracts await Mayank. |
| 2026-10-29 | Chethan | C1 adapter on the PandaSet devkit (pinned 59be180): Scene API, label guard over cuboids and semseg, download and index scripts, `longtail scenes ls`; ego axes derived from the devkit static calibration (A-25). Tested on a fake on-disk sequence; dataset not downloaded yet, so real-data checks skip. |
| 2026-10-29 | Chethan | C2: transforms, vectorised projection (170k points x 6 cameras in ~15 ms), camera-to-LiDAR sync with gap report, synthetic scene for C2 to C6 tests, overlay script; no lens distortion (A-38). Real-data checks (devkit 1 px, overlays, gap distribution) written but skipped: dataset not downloaded. |
| 2026-10-29 | Chethan | C4: mask-to-LiDAR association with local ground planes (A-49), a visibility filter the plan lacks (A-63), cross-camera and fragment merging, frustum and box-only modes; COCO RLE codec; verify.yaml. Synthetic: every crosser cluster pure, but centroid sits ~0.3 m from the box centre (surface returns), so the plan's 0.15 m check is recorded as an expected failure. |
| 2026-10-29 | Chethan | C4 follow-up: height measured from the local ground plane (1.6 to 1.7 m for the 1.7 m crosser, was 1.4 m); C4 deviations from the plan written up in docs/eval-report.md. |
| 2026-10-29 | Chethan | C4: cross-camera merge on shared LiDAR points (A-74) with a 30% minimum overlap, so two cars parked nose to tail stay apart. |
| 2026-10-29 | Chethan | C4: A-74 overlap floor lowered from 30% to 15% (nose-to-tail cars share 2.3%, the close-pass views 21-28%). |
