"""The evaluation report's results tables, from the results JSON of the ablation and the breaks.

    uv run python scripts/report_results.py                                  # print the tables
    uv run python scripts/report_results.py longtail/eval/runs/all-heldout/results --write docs/eval-report.md

Reads ablation_a.json, ablation_b.json, ablation_c.json and breaks.json from one results directory
(longtail.ablation/v1 and longtail.breaks/v1, A-140). Configuration (d) is never reported: Glimmer was cut
(A-213), so an ablation_d.json, if any, is ignored. A missing configuration is shown as not run.

With --write, the tables replace whatever sits between the results markers in the report (added at the end
of the file the first time). Every number comes with its 95% bootstrap interval over queries (A-133).
"""

import argparse
import json
import sys
from pathlib import Path

CONFIGS = ("a", "b", "c")  # ASSUMPTION A-213: no (d)
NAMES = {
    "a": "(a) Spark Open alone",
    "b": "(b) + SAM 3.1 tracks",
    "c": "(c) + LiDAR verification",
}
BREAK_NAMES = {
    "verifier_off": "Verifier off",
    "calibration_2deg_0.3m": "Calibration 2° and 0.3 m",
    "lidar_offset_200ms": "LiDAR offset 200 ms",
}
JUDGED = ("precision", "catch_rate", "recall")  # what a break must lower (A-141)
START_TAG = "<!-- results:start"
START = START_TAG + " (scripts/report_results.py writes this section) -->"
END = "<!-- results:end -->"
DEFAULT_DIR = Path("longtail/eval/runs/all-heldout/results")


def _share(m: dict | None) -> str:
    if not m or m.get("value") is None:
        return "n/a"
    digits = 3 if 0 < abs(m["value"]) < 0.1 else 2  # 0.025 must not read as 0.03
    return _number(m, "{:." + str(digits) + "f}")


def _number(m: dict | None, fmt: str) -> str:
    if not m or m.get("value") is None:
        return "n/a"
    s = fmt.format(m["value"])
    if m.get("ci95"):
        lo, hi = m["ci95"]
        s += f" [{fmt.format(lo)}, {fmt.format(hi)}]"
    return s


def _p_at_10(metrics: dict) -> str:
    s = _share(metrics.get("p_at_10"))
    returned = metrics.get("returned_top10")
    if s != "n/a" and returned and returned.get("value") is not None:
        s += f"; returned {_number(returned, '{:.1f}')} of 10"
    return s


def load(results_dir: Path) -> tuple[dict[str, dict], dict | None]:
    docs = {}
    for c in CONFIGS:
        p = results_dir / f"ablation_{c}.json"
        if p.is_file():
            doc = json.loads(p.read_text())
            if doc.get("format") != "longtail.ablation/v1":
                raise ValueError(f"{p}: format {doc.get('format')!r}, expected longtail.ablation/v1")
            docs[c] = doc
    p = results_dir / "breaks.json"
    br = json.loads(p.read_text()) if p.is_file() else None
    if br is not None and br.get("format") != "longtail.breaks/v1":
        raise ValueError(f"{p}: format {br.get('format')!r}, expected longtail.breaks/v1")
    if not docs and br is None:
        raise FileNotFoundError(f"no ablation_*.json or breaks.json in {results_dir}")
    return docs, br


def check_consistent(docs: dict[str, dict], br: dict | None) -> list[str]:
    """Problems that make the rows incomparable: different splits, scene sets or thresholds."""
    problems = []
    every = [(f"({c})", d) for c, d in docs.items()] + ([("breaks", br)] if br else [])
    for key in ("split", "thresholds_hash", "gt_config_hash"):
        values = {name: d.get(key) for name, d in every}
        if len(set(values.values())) > 1:
            problems.append(f"{key} differs: {values}")
    scenes = {name: tuple(sorted(d.get("scenes", []))) for name, d in every}
    if len(set(scenes.values())) > 1:
        problems.append("scene sets differ: " + ", ".join(f"{n} {len(s)} scenes" for n, s in scenes.items()))
    return problems


def ablation_table(docs: dict[str, dict]) -> list[str]:
    lines = [
        "| Configuration | Precision | Recall | P@10 | Catch rate | False rejection | Cost per query | "
        "Seconds per query |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in CONFIGS:
        if c not in docs:
            lines.append(f"| {NAMES[c]} | not run | | | | | | |")
            continue
        m = docs[c]["metrics"]
        lines.append(
            f"| {NAMES[c]} | {_share(m.get('precision'))} | {_share(m.get('recall'))} | {_p_at_10(m)} | "
            f"{_share(m.get('catch_rate'))} | {_share(m.get('false_rejection_rate'))} | "
            f"{_number(m.get('cost_per_query_usd'), '${:.3f}')} | "
            f"{_number(m.get('seconds_per_query'), '{:.1f}')} |"
        )
    return lines


def breaks_table(br: dict) -> list[str]:
    base = br["baseline_c"]
    lines = [
        "| Break on (c) | Precision | Recall | Catch rate | P@10 | False rejection | Moved | Result |",
        "|---|---|---|---|---|---|---|---|",
        f"| None (clean (c)) | {_share(base.get('precision'))} | {_share(base.get('recall'))} | "
        f"{_share(base.get('catch_rate'))} | {_share(base.get('p_at_10'))} | "
        f"{_share(base.get('false_rejection_rate'))} | | |",
    ]
    for name, b in br["breaks"].items():
        m = b["metrics"]
        moved = ", ".join(k.replace("_", " ") for k in b["moved"]) or "nothing"
        result = "passes" if b["passed"] else "**hole**: lowered none of " + ", ".join(JUDGED)
        lines.append(
            f"| {BREAK_NAMES.get(name, name)} | {_share(m.get('precision'))} | {_share(m.get('recall'))} | "
            f"{_share(m.get('catch_rate'))} | {_share(m.get('p_at_10'))} | "
            f"{_share(m.get('false_rejection_rate'))} | {moved} | {result} |"
        )
    return lines


def render(docs: dict[str, dict], br: dict | None, source: str) -> str:
    first = next(iter(docs.values()), br)
    n_queries = len(next(iter(docs.values()))["queries"]) if docs else None
    boot = first.get("bootstrap") or {"resamples": 1000, "over": "queries"}
    commits = sorted({d.get("commit", "unknown") for d in [*docs.values(), *([br] if br else [])]})
    out = [
        START,
        "",
        "## Results",
        "",
        f"Split: **{first['split']}**"
        + (f", {n_queries} queries" if n_queries is not None else "")
        + f", {len(first.get('scenes', []))} scene{'' if len(first.get('scenes', [])) == 1 else 's'}. "
        f"Each number has its 95% bootstrap interval in brackets ({boot['resamples']} resamples over "
        f"{boot['over']}, A-133). P@10 divides by 10 and shows how many findings came back per query "
        "(A-134). "
        "Configuration (d) is not run (A-213).",
        "",
        f"From `{source}`, commit {', '.join(f'`{c}`' for c in commits)}, thresholds "
        f"`{first.get('thresholds_hash')}`.",
        "",
    ]
    problems = check_consistent(docs, br)
    if problems:
        out += ["**The rows below are not comparable:** " + "; ".join(problems) + ".", ""]
    if docs:
        out += ["### Ablation", "", *ablation_table(docs), ""]
    if br:
        out += [
            "### Deliberate breaks",
            "",
            "A break passes when it lowers precision, the catch rate or recall of (c); one that lowers "
            "none is a hole in the evaluation (A-141).",
            "",
            *breaks_table(br),
            "",
        ]
        if br.get("holes"):
            out += [f"Holes: {', '.join(br['holes'])}.", ""]
    out.append(END)
    return "\n".join(out) + "\n"


def write_into(report: Path, section: str) -> None:
    text = report.read_text()
    if START_TAG in text and END in text:
        head = text[: text.index(START_TAG)]
        tail = text[text.index(END) + len(END) :].lstrip("\n")
        text = head + section + ("\n" + tail if tail else "")
    else:
        text = text.rstrip("\n") + "\n\n" + section
    report.write_text(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", type=Path, nargs="?", default=DEFAULT_DIR, help=f"default {DEFAULT_DIR}")
    ap.add_argument("--write", type=Path, help="put the tables into this report instead of printing them")
    args = ap.parse_args(argv)
    docs, br = load(args.results)
    section = render(docs, br, str(args.results))
    if args.write:
        write_into(args.write, section)
        print(f"results written into {args.write}")
    else:
        print(section, end="")
    problems = check_consistent(docs, br)
    for p in problems:
        print(f"warning: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
