"""Cost report from the ledger (plan M8): spend by model and purpose, with cache hits counted."""

from collections import defaultdict
from pathlib import Path
from typing import Any

from longtail.llm.ledger import Ledger


def summarise(path: Path | None = None) -> dict[str, Any]:
    entries = Ledger(path).entries()
    rows: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "cache_hits": 0, "usd": 0.0, "input_tokens": 0, "output_tokens": 0, "frames": 0}
    )
    for e in entries:
        r = rows[(str(e.get("model", "?")), str(e.get("purpose") or e.get("kind") or "?"))]
        r["calls"] += 1
        r["cache_hits"] += bool(e.get("cache_hit"))
        r["usd"] += float(e.get("usd", 0.0))
        r["input_tokens"] += int(e.get("input_tokens") or 0)
        r["output_tokens"] += int(e.get("output_tokens") or 0)
        r["frames"] += int(e.get("frames") or 0) if not e.get("cache_hit") else 0
    table = [{"model": m, "purpose": p, **v} for (m, p), v in sorted(rows.items())]
    return {"entries": len(entries), "total_usd": round(sum(r["usd"] for r in table), 6), "rows": table}


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "| Model | Purpose | Calls | Cache hits | Input tokens | Output tokens | Frames | USD |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summary["rows"]:
        lines.append(
            f"| {r['model']} | {r['purpose']} | {r['calls']} | {r['cache_hits']} | {r['input_tokens']:,} | "
            f"{r['output_tokens']:,} | {r['frames']:,} | ${r['usd']:.4f} |"
        )
    calls = sum(r["calls"] for r in summary["rows"])
    lines.append(f"| **Total** | | {calls} | | | | | **${summary['total_usd']:.4f}** |")
    return "\n".join(lines)
