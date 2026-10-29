"""The `longtail` command. Each area registers its own subcommands."""

# ASSUMPTION A-34: a thin argparse dispatcher; the plan assigns cli.py to Mayank, who may replace it

import argparse
import sys

from longtail.data import cli as data_cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="longtail")
    sub = parser.add_subparsers(dest="command", required=True)
    data_cli.register(sub)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
