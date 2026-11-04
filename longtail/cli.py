"""The `longtail` command. Each area registers its own subcommands."""

# ASSUMPTION A-34: a thin argparse dispatcher; the plan assigns cli.py to Mayank, who may replace it

import argparse
import sys

from longtail import demo as demo_cli
from longtail.agent import cli as agent_cli
from longtail.data import cli as data_cli
from longtail.env import load_dotenv
from longtail.eval import cli as gt_cli
from longtail.explain import cli as explain_cli
from longtail.vision import cli as sam_cli
from longtail.viz import cli as viz_cli


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="longtail")
    sub = parser.add_subparsers(dest="command", required=True)
    data_cli.register(sub)
    agent_cli.register(sub)
    explain_cli.register(sub)
    # ASSUMPTION A-103: `longtail viz` is registered here, in the shared dispatcher (A-34)
    viz_cli.register(sub)
    # ASSUMPTION A-119: `longtail sam` is registered here too, in the shared dispatcher (A-34)
    sam_cli.register(sub)
    # ASSUMPTION A-131: `longtail gt` is registered here too (A-34)
    gt_cli.register(sub)
    # ASSUMPTION A-243: `longtail demo` replays the recorded demo from the cache (A-34)
    demo_cli.register(sub)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
