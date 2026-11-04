"""Write the v1-against-v2 table into docs/eval-report.md, between the v2-results markers.

    PYTHONPATH=. uv run python scripts/report_v2.py

Reads configuration (c) of v1 (results/) and of v2 (results_v2/) for each scene set
and split under longtail/eval/runs/, and fails if a v2 result was not made with the v2 thresholds.
"""

import json
import sys
from pathlib import Path

from report_results import _p_at_10, _share

from longtail.eval import ablation, gt
from longtail.verify.config import CONFIG_PATHS, load_verify_config

RUNS = Path("longtail/eval/runs")
REPORT = Path("docs/eval-report.md")
START, END = "<!-- v2-results:start -->", "<!-- v2-results:end -->"
SETS = [("all-heldout", "Held-out, 102 scenes (headline split)"), ("all-dev", "Dev, 102 scenes"),
        ("index-heldout", "Held-out, 29 index scenes"), ("index-dev", "Dev, 29 index scenes")]  # fmt: skip


def _row(label: str, doc: dict) -> str:
    m = doc["metrics"]
    return (
        f"| {label} | {_share(m['precision'])} | {_share(m['recall'])} | {_p_at_10(m)} | "
        f"{_share(m.get('catch_rate'))} | {_share(m.get('false_rejection_rate'))} |"
    )


def render() -> str:
    v2_hash = ablation.thresholds_hash(load_verify_config(CONFIG_PATHS["v2"]), gt.load_gt_config())
    out = []
    for name, title in SETS:
        v1_path = RUNS / name / "results" / "ablation_c.json"
        v2_path = RUNS / name / "results_v2" / "ablation_c.json"
        if not v2_path.is_file():
            out += [f"**{title}:** v2 not scored.", ""]
            continue
        v1, v2 = json.loads(v1_path.read_text()), json.loads(v2_path.read_text())
        if v2["thresholds_hash"] != v2_hash:
            raise SystemExit(f"{v2_path}: thresholds hash {v2['thresholds_hash']}, v2's is {v2_hash}")
        out += [
            f"**{title}** (configuration (c), {len(v2['scenes'])} scenes; v2 made at `{v2['commit'][:7]}`)",
            "",
            "| Verifier | Precision | Recall | P@10 | Catch rate | False rejection |",
            "|---|---|---|---|---|---|",
            _row("v1 (before)", v1),
            _row("v2 (real-data fixes)", v2),
            "",
        ]
    return "\n".join(out).rstrip()


def main() -> int:
    text = REPORT.read_text()
    if text.count(START) != 1 or text.count(END) != 1:
        print(f"{REPORT}: the v2-results markers are missing", file=sys.stderr)
        return 1
    head, rest = text.split(START)
    _, tail = rest.split(END)
    REPORT.write_text(f"{head}{START}\n{render()}\n{END}{tail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
