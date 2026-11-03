"""M8 cost report: totals by model and purpose from the ledger; cache hits cost $0 but are counted."""

from longtail.llm.ledger import Ledger
from longtail.llm.report import markdown, summarise


def test_summary_groups_by_model_and_purpose(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record(
        model="muse-wonder-1.0",
        purpose="planner",
        usd=0.012,
        cache_hit=False,
        input_tokens=2000,
        output_tokens=200,
    )
    ledger.record(
        model="muse-wonder-1.0",
        purpose="planner",
        usd=0.0,
        cache_hit=True,
        input_tokens=2000,
        output_tokens=200,
    )
    ledger.record(model="sam-3.1", kind="sam", usd=0.016, cache_hit=False, frames=80)
    s = summarise(tmp_path / "ledger.jsonl")
    assert s["entries"] == 3 and s["total_usd"] == 0.028
    planner = next(r for r in s["rows"] if r["purpose"] == "planner")
    assert planner["calls"] == 2 and planner["cache_hits"] == 1
    sam = next(r for r in s["rows"] if r["model"] == "sam-3.1")
    assert sam["purpose"] == "sam" and sam["frames"] == 80
    assert "**$0.0280**" in markdown(s)


def test_empty_ledger(tmp_path):
    assert summarise(tmp_path / "none.jsonl") == {"entries": 0, "total_usd": 0, "rows": []}
