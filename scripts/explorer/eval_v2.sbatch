#!/bin/bash
#SBATCH -J longtail-eval-v2
#SBATCH -p short
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH --array=0-101%40         # one task per usable PandaSet scene (A-233; 054 is left out, A-240)
#SBATCH -o /scratch/%u/longtail/logs/%x-%A_%a.out
# The post-freeze verifier (v2, A-244, docs/eval-report.md "v2 (post-freeze, pre-registered)"): configuration
# (c) only, all 25 queries, one scene per task, saved unscored in results_partial_v2/ (v1's results_partial/ is
# left alone); score_v2.sbatch scores them. Replay only, no keys on the cluster (A-237).
set -euo pipefail
cd /scratch/$USER/longtail
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR=/scratch/$USER/uv-cache
module load FFmpeg/7.1.1
mapfile -t SCENES < <(uv run --no-sync python -c "
from pathlib import Path
from longtail.data.index import load_index
from longtail.data.pandaset import list_scenes
idx = [s['scene_id'] for s in load_index()['scenes']]
print('\n'.join(idx + sorted(set(list_scenes(Path('data/pandaset'))) - set(idx))))")
SID=${SCENE:-${SCENES[$SLURM_ARRAY_TASK_ID]}}
mkdir -p runs/ledgers
export LONGTAIL_DATA_DIR=$PWD/data LONGTAIL_REPLAY=1 LONGTAIL_ALLOW_PAID=0 LONGTAIL_VERIFIER=v2 \
  LONGTAIL_LEDGER=runs/ledgers/eval-v2-$SID.jsonl LONGTAIL_SAM_WORKERS=6
unset MODEL_API_KEY BASETEN_API_KEY
echo "scene $SID on $(hostname), verifier $LONGTAIL_VERIFIER"
uv run --no-sync python -m longtail.eval.ablation --config c --queries longtail/eval/queries/queries.yaml \
  --split all --scenes "$SID" --partial results_partial_v2 --replay
