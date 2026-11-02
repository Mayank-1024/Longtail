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
mkdir -p logs data/pandaset
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen
uv run --with huggingface_hub hf download georghess/pandaset --repo-type dataset --local-dir data/hf
unzip -q -n data/hf/pandaset.zip -d data/pandaset
uv run python scripts/build_index.py
