#!/bin/bash
#SBATCH -J longtail-eval
#SBATCH -p short                 # checked 2026-11-03: the default CPU partition, 2-day limit
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 06:00:00
#SBATCH --array=0-101%40         # one task per usable PandaSet scene (A-233; 054 is left out, A-240)
#SBATCH -o /scratch/%u/longtail/logs/%x-%A_%a.out
# Configurations (c) and (b) and the three breaks on one scene, all 25 queries, saved unscored;
# merge.sbatch scores them.
# Replay only, no keys on the cluster (A-237): every SAM response was made on the laptop by
# scripts/sam_batch.py and copied here with `rsync cache/sam/`. A missing response fails the task loudly.
set -euo pipefail
cd /scratch/$USER/longtail
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR=/scratch/$USER/uv-cache
module load FFmpeg/7.1.1
# Index scenes first, then the rest, so the 30-scene evaluation completes first.
mapfile -t SCENES < <(uv run --no-sync python -c "
import json
from pathlib import Path
from longtail.data.pandaset import list_scenes
idx = [s['scene_id'] for s in json.load(open('longtail/data/index.json'))['scenes']]
print('\n'.join(idx + sorted(set(list_scenes(Path('data/pandaset'))) - set(idx))))")
SID=${SCENES[$SLURM_ARRAY_TASK_ID]}
mkdir -p runs/ledgers
export LONGTAIL_DATA_DIR=$PWD/data LONGTAIL_REPLAY=1 LONGTAIL_ALLOW_PAID=0 \
  LONGTAIL_LEDGER=runs/ledgers/eval-$SID.jsonl LONGTAIL_SAM_WORKERS=6
unset MODEL_API_KEY BASETEN_API_KEY
echo "scene $SID on $(hostname)"
for C in c b; do
  uv run --no-sync python -m longtail.eval.ablation --config $C --queries longtail/eval/queries/queries.yaml \
    --split all --scenes "$SID" --partial results_partial --replay
done
# The deliberate breaks (C8): (c) with the verifier off, the calibration shifted, the LiDAR clock offset.
uv run --no-sync python -m longtail.eval.breaks --queries longtail/eval/queries/queries.yaml \
  --split all --scenes "$SID" --partial results_partial --replay
