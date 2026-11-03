#!/bin/bash
#SBATCH -J longtail-eval
#SBATCH -p short                 # checked 2026-11-03: the default CPU partition, 2-day limit
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 06:00:00
#SBATCH --array=0-102%40         # one task per PandaSet scene (A-233), at most 40 at a time
#SBATCH -o /scratch/%u/longtail/logs/%x-%A_%a.out
# Configurations (c) and (b) on one scene, all 25 queries, saved unscored; merge.sbatch scores them.
# Paid: SAM calls for 5 concepts x 6 cameras, capped at $0.75 per task in its own ledger (A-234).
set -euo pipefail
cd /scratch/$USER/longtail
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR=/scratch/$USER/uv-cache
module load FFmpeg/7.1.1
set -a; source "$HOME/.longtail.env"; set +a   # the keys: chmod 600, in home, never in scratch or git
# Index scenes first, then the rest, so the 30-scene evaluation completes first.
mapfile -t SCENES < <(uv run --no-sync python -c "
import json, os
idx = [s['scene_id'] for s in json.load(open('longtail/data/index.json'))['scenes']]
print('\n'.join(idx + sorted(set(os.listdir('data/pandaset')) - set(idx))))")
SID=${SCENES[$SLURM_ARRAY_TASK_ID]}
mkdir -p runs/ledgers
export LONGTAIL_DATA_DIR=$PWD/data LONGTAIL_ALLOW_PAID=1 LONGTAIL_SPEND_CAP_USD=0.75 \
  LONGTAIL_LEDGER=runs/ledgers/eval-$SID.jsonl LONGTAIL_SAM_WORKERS=6
echo "scene $SID on $(hostname)"
for C in c b; do
  uv run --no-sync python -m longtail.eval.ablation --config $C --queries longtail/eval/queries/queries.yaml \
    --split all --scenes "$SID" --partial results_partial
done
