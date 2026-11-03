"""First paid calls, one per model, to VERIFY the API details the client assumes (A-203, A-208, A-212, A-218).

    LONGTAIL_ALLOW_PAID=1 uv run python scripts/smoke_live.py [--wonder] [--spark-open]

Each check is skipped unless its key is set. The spend cap is forced to $0.50 for this script, whatever .env
says, so it can never cost more than that. Every call is cached and logged in runs/ledger.jsonl.
"""

import argparse
import os
import sys
import traceback

from longtail.env import load_dotenv  # noqa: E402

load_dotenv()
os.environ["LONGTAIL_SPEND_CAP_USD"] = "0.50"

from longtail.agent.planner import plan  # noqa: E402
from longtail.contracts import ScenarioSpec  # noqa: E402
from longtail.llm.baseten import SPARK_OPEN, spark_open_client  # noqa: E402
from longtail.llm.client import ModelClient  # noqa: E402

QUERY = "a cyclist within 15 m"


def check_wonder() -> bool:
    client = ModelClient.from_env()
    result = plan(QUERY, client)
    print(f"  planner: {result.attempts} attempt(s), ${result.usd:.4f}")
    print(f"  spec: {result.spec.model_dump_json()}")
    assert isinstance(result.spec, ScenarioSpec)
    return True


def check_spark_open() -> bool:
    client = spark_open_client()
    resp = client.respond(
        {
            "model": SPARK_OPEN,
            "input": [{"role": "user", "content": 'Reply with exactly this JSON: {"ok": true}'}],
            "max_output_tokens": 20,
            "temperature": 0,
        },
        purpose="smoke",
    )
    print(f"  reply: {resp.text!r}; usage {resp.usage}; ${resp.usd:.6f}")
    assert resp.text.strip() == '{"ok": true}', (
        "reply is not the exact JSON: check A-226 (reasoning headroom)"
    )
    assert resp.text and resp.usage.get("input_tokens", 0) > 0, "no text or no usage: check A-218"
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wonder", action="store_true")
    ap.add_argument("--spark-open", action="store_true")
    args = ap.parse_args()
    if os.environ.get("LONGTAIL_ALLOW_PAID") != "1":
        print("set LONGTAIL_ALLOW_PAID=1 to make these paid calls (capped at $0.50)")
        return 1
    run_all = not (args.wonder or args.spark_open)
    checks = []
    if (run_all or args.wonder) and os.environ.get("MODEL_API_KEY"):
        checks.append(("Muse Wonder planner (A-203, A-208, A-212)", check_wonder))
    if (run_all or args.spark_open) and os.environ.get("BASETEN_API_KEY"):
        checks.append(("Muse Spark Open on Baseten (A-217, A-218)", check_spark_open))
    if not checks:
        print("no keys set: add MODEL_API_KEY and/or BASETEN_API_KEY to .env")
        return 1
    ok = True
    for name, fn in checks:
        print(name)
        try:
            fn()
            print("  PASS")
        except Exception:
            ok = False
            print("  FAIL")
            traceback.print_exc()
    print("SAM 3.1: run Chethan's scripts/sam_smoke.py (C3) once #9 is merged.")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
