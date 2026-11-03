#!/bin/bash
#SBATCH -J longtail-download
#SBATCH -p short                 # checked 2026-11-03: short is the default CPU partition, 2-day limit
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 03:00:00
#SBATCH -o /scratch/%u/longtail/logs/%x-%j.out
# PandaSet, all 103 sequences (about 44.5 GB), from the Hugging Face mirror (CC BY 4.0), into scratch.
set -euo pipefail
cd /scratch/$USER/longtail
mkdir -p logs data
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen
# Pulls members straight out of the remote zip (no 44.5 GB zip on disk first); resumable, so rerun on a timeout.
# Members are pandaset/<scene>/..., so --data-dir is data, not data/pandaset (A-33).
# The 30 index scenes first (the evaluation can start on them), then the rest; files already there are skipped.
INDEX=$(uv run python -c "import json; print(','.join(s['scene_id'] for s in json.load(open('longtail/data/index.json'))['scenes']))")
uv run python scripts/fetch_pandaset.py --parts full --data-dir data --workers 32 --scenes "$INDEX"
uv run python scripts/fetch_pandaset.py --parts full --data-dir data --workers 32
# longtail/data/index.json is committed (built from the labels of all 103 scenes); do not rebuild it here.
