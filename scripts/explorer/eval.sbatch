#!/bin/bash
#SBATCH -J longtail-eval
#SBATCH -p short                 # VERIFY with `sinfo -s`
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH --array=0-2              # configurations (a), (b), (c); (d) is cut with Glimmer
#SBATCH -o /scratch/%u/longtail/logs/%x-%A_%a.out
# Replay only: every model response must already be in cache/ (made on a laptop with the keys).
set -euo pipefail
cd /scratch/$USER/longtail
export PATH="$HOME/.local/bin:$PATH"
export LONGTAIL_DATA_DIR=$PWD/data LONGTAIL_REPLAY=1 LONGTAIL_ALLOW_PAID=0
unset MODEL_API_KEY BASETEN_API_KEY
CONFIGS=(a b c)
uv run longtail gt build longtail/eval/queries/queries.yaml
uv run python -m longtail.eval.ablation --config "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
