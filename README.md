# LongTail

**Find the rare driving moments that break self-driving cars, and prove they are real.**

An engineer types a rare scenario in plain English ("a pedestrian crossing in front of us while we are moving") and gets back clips from multi-camera driving logs that contain it. Every claim a vision model makes about those clips is checked against a second sensor, LiDAR, before it counts. The model proposes; geometry decides.

> Status: in development for the Meta Global AI Developer Hackathon (Indie Developer track, Push the Frontier). Nothing below is a result yet; this README is updated as parts land.

## How it works

1. **Ask:** a scenario in plain English.
2. **Plan:** Muse Wonder turns it into a validated scenario spec, or asks a clarifying question.
3. **Orchestrate:** a budgeted agent runs one Muse Spark sub-agent per scene.
4. **Triage:** Muse Glimmer (open weights) tags keyframes once; only candidate windows go further.
5. **See:** SAM 3.1 segments and tracks the named objects through video from a text prompt.
6. **Verify:** our code matches masks to LiDAR, builds 3D tracks and evaluates each claim as PASS, FAIL or UNVERIFIABLE.
7. **Explain:** Muse Spark Open (open weights) writes a card that may cite only verified results.
8. **Approve:** a human accepts or rejects; approved finds export as a regression test set.

## Evaluation (planned)

25 queries (10 development, 15 held-out) with ground truth computed from the dataset's 3D labels by a script; precision@10, recall, hallucination-catch rate, an ablation of each Meta model, and deliberate breaks that must lower the scores.

## Setup

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Built during the hackathon

All code here was written during the build window; see [docs/build-log.md](docs/build-log.md) and the commit history.

## Team

Mayank Bhadrasen and Chethan.

## Licence

Apache-2.0 for the code. Dataset files are never included; they are downloaded under their own licence.
