#!/bin/bash
#SBATCH -J longtail-download
#SBATCH -p short                 # VERIFY with `sinfo -s`
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
uv run python scripts/fetch_pandaset.py --parts full --data-dir data --workers 32
uv run python scripts/build_index.py
