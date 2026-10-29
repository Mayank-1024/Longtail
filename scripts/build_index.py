"""Build longtail/data/index.json: about 30 varied PandaSet scenes plus 3 phrase-tuning scenes.

Run once after scripts/download_data.sh:

    uv run python scripts/build_index.py

Variety (pedestrians, cyclists) can only be judged from the labels, so that part runs inside
eval_ground_truth(). The label-derived numbers go to gt/index_selection.json (git-ignored); the index
the pipeline reads holds only scene ids and metadata that do not come from labels.
"""

# ASSUMPTION A-31: selection reads labels inside eval_ground_truth(); the index gets no label data
# ASSUMPTION A-14: 3 phrase-tuning scenes are chosen outside the 30-scene index

import argparse
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from longtail.data.guard import eval_ground_truth
from longtail.data.index import INDEX_PATH, pandaset_root
from longtail.data.pandaset import DEVKIT_COMMIT, PandaSetScene, list_scenes

SEED = 20261028
N_INDEX = 30
N_PHRASE = 3
MOVING_MPS = 0.5  # plan C6 "moving" threshold

# ASSUMPTION A-31: PandaSet label strings and attributes as documented in the devkit's Cuboids docstring
PEDESTRIAN_LABELS = {"Pedestrian", "Pedestrian with Object"}
CYCLIST_LABEL = "Bicycle"
VEHICLE_LABELS = {"Car", "Pickup Truck", "Medium-sized Truck", "Semi-truck", "Bus"}

# ASSUMPTION A-32: selection quotas; walking pedestrians while the ego moves stand in for "urban crossings"
QUOTAS = (("night", 6), ("walking-pedestrians-ego-moving", 10), ("cyclists", 6))


def ego_moving_frac(scene: PandaSetScene) -> float:
    ts = scene.lidar_timestamps()
    xyz = np.array([scene.ego_pose(t)[:3, 3] for t in ts])
    speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1) / np.diff(ts)
    return float(np.mean(speed > MOVING_MPS))


def label_stats(scene: PandaSetScene, labels_seen: set[str]) -> dict[str, int]:
    """Distinct objects per group across the scene. Call only inside eval_ground_truth()."""
    groups: dict[str, set[str]] = {
        "pedestrians": set(),
        "walking": set(),
        "cyclists": set(),
        "vehicles": set(),
    }
    for i in range(scene.meta.n_frames):
        df = scene.labels(i)
        labels_seen.update(df["label"])
        ped = df[df["label"].isin(PEDESTRIAN_LABELS)]
        groups["pedestrians"].update(ped["uuid"])
        groups["walking"].update(ped.loc[ped["attributes.pedestrian_behavior"] == "Walking", "uuid"])
        bikes = df[(df["label"] == CYCLIST_LABEL) & (df["attributes.rider_status"] == "With Rider")]
        groups["cyclists"].update(bikes["uuid"])
        groups["vehicles"].update(df.loc[df["label"].isin(VEHICLE_LABELS), "uuid"])
    return {k: len(v) for k, v in groups.items()}


def select(rows: list[dict], seed: int = SEED) -> tuple[list[str], list[str], dict[str, str]]:
    """Pick phrase-tuning and index scenes from per-scene rows. Deterministic for a given seed.

    Each row has scene_id, time_of_day, ego_moving_frac, pedestrians, walking, cyclists, vehicles.
    """
    rng = random.Random(seed)
    by_id = {r["scene_id"]: r for r in sorted(rows, key=lambda r: r["scene_id"])}

    eligible = [
        s
        for s, r in by_id.items()
        if r["time_of_day"] == "day" and r["walking"] > 0 and r["cyclists"] > 0 and r["vehicles"] > 0
    ]
    if len(eligible) < N_PHRASE:
        raise ValueError(f"only {len(eligible)} day scenes have walking pedestrians, cyclists and vehicles")
    phrase = sorted(rng.sample(eligible, N_PHRASE))

    pool = [s for s in by_id if s not in phrase]
    chosen: dict[str, str] = {}

    def take(candidates: list[str], key, n: int, reason: str) -> None:
        for s in sorted(candidates, key=lambda s: (key(by_id[s]), s)):
            if sum(r == reason for r in chosen.values()) >= n or len(chosen) >= N_INDEX:
                break
            if s not in chosen:
                chosen[s] = reason

    quota = dict(QUOTAS)
    take(
        [s for s in pool if by_id[s]["time_of_day"] == "night"],
        lambda r: -(r["pedestrians"] + r["cyclists"]),
        quota["night"],
        "night",
    )
    take(
        [s for s in pool if by_id[s]["ego_moving_frac"] >= 0.5 and by_id[s]["walking"] > 0],
        lambda r: -r["walking"],
        quota["walking-pedestrians-ego-moving"],
        "walking-pedestrians-ego-moving",
    )
    take(
        [s for s in pool if by_id[s]["cyclists"] > 0],
        lambda r: -r["cyclists"],
        quota["cyclists"],
        "cyclists",
    )
    rest = [s for s in pool if s not in chosen]
    for s in rng.sample(rest, max(0, min(N_INDEX - len(chosen), len(rest)))):
        chosen[s] = "random-fill"
    return sorted(chosen), phrase, {**chosen, **{s: "phrase-tuning" for s in phrase}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=pandaset_root())
    ap.add_argument("--out", type=Path, default=INDEX_PATH)
    ap.add_argument("--stats-out", type=Path, default=Path("gt/index_selection.json"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    ids = list_scenes(args.root)
    if len(ids) < N_INDEX + N_PHRASE:
        print(f"found {len(ids)} sequences under {args.root}; need {N_INDEX + N_PHRASE}", file=sys.stderr)
        return 1

    rows, meta = [], {}
    labels_seen: set[str] = set()
    with eval_ground_truth():
        for sid in ids:
            scene = PandaSetScene(args.root, sid)
            if not scene.has_cuboids:
                print(f"skip {sid}: no cuboid annotations", file=sys.stderr)
                continue
            m = scene.meta
            frac = ego_moving_frac(scene)
            meta[sid] = {
                "scene_id": sid,
                "n_frames": m.n_frames,
                "time_of_day": m.time_of_day,
                "ego_moving_frac": round(frac, 3),
            }
            rows.append({**meta[sid], **label_stats(scene, labels_seen)})
            print(f"{sid}: {rows[-1]}", file=sys.stderr)

    missing = (PEDESTRIAN_LABELS | {CYCLIST_LABEL} | VEHICLE_LABELS) - labels_seen
    if missing:
        print(f"label strings never seen: {sorted(missing)}", file=sys.stderr)
    if {"Pedestrian", CYCLIST_LABEL, "Car"} & missing:
        print(
            "core label strings are missing; the devkit's documented labels do not match the data",
            file=sys.stderr,
        )
        return 1

    index_ids, phrase_ids, reasons = select(rows, args.seed)
    index = {
        "dataset": "pandaset",
        "devkit_commit": DEVKIT_COMMIT,
        "built_by": "scripts/build_index.py",
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": args.seed,
        "scenes": [meta[s] for s in index_ids],
        "phrase_tuning_scenes": [meta[s] for s in phrase_ids],
    }
    args.out.write_text(json.dumps(index, indent=2) + "\n")
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    args.stats_out.write_text(
        json.dumps([{**r, "reason": reasons.get(r["scene_id"])} for r in rows], indent=2) + "\n"
    )
    print(f"wrote {args.out} ({len(index_ids)} scenes, {len(phrase_ids)} phrase-tuning) and {args.stats_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
