"""`longtail scenes ...` subcommands."""

import argparse
import sys
from pathlib import Path

from longtail.data.index import INDEX_PATH, load_index, pandaset_root
from longtail.data.pandaset import list_scenes


def register(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    scenes = sub.add_parser("scenes", help="the scene index")
    scenes_sub = scenes.add_subparsers(dest="scenes_command", required=True)
    ls = scenes_sub.add_parser("ls", help="list the indexed scenes")
    ls.add_argument("--index", type=Path, default=INDEX_PATH, help="index file (default: %(default)s)")
    ls.set_defaults(func=_ls)


def _ls(args: argparse.Namespace) -> int:
    if not args.index.is_file():
        print(
            f"No scene index at {args.index}. Download PandaSet with scripts/download_data.sh, "
            "then build the index with scripts/build_index.py.",
            file=sys.stderr,
        )
        return 1
    index = load_index(args.index)
    on_disk = set(list_scenes(pandaset_root()))
    rows = [(s, "index") for s in index["scenes"]] + [
        (s, "phrase-tuning") for s in index["phrase_tuning_scenes"]
    ]
    print(f"{'scene':<8}{'role':<15}{'time':<7}{'ego moving':>11}  on disk")
    for s, role in rows:
        moving = f"{s['ego_moving_frac']:.0%}" if s.get("ego_moving_frac") is not None else "-"
        print(
            f"{s['scene_id']:<8}{role:<15}{s.get('time_of_day') or '-':<7}{moving:>11}  "
            f"{'yes' if s['scene_id'] in on_disk else 'no'}"
        )
    n_disk = sum(s["scene_id"] in on_disk for s, _ in rows)
    n_index, n_phrase = len(index["scenes"]), len(index["phrase_tuning_scenes"])
    print(f"{n_index} indexed, {n_phrase} phrase-tuning, {n_disk} on disk")
    return 0
