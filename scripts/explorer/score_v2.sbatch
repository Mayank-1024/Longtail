#!/bin/bash
#SBATCH -J longtail-score-v2
#SBATCH -p short
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH -o /scratch/%u/longtail/logs/%x-%A_%a.out
#SBATCH --array=0-3
# Scores the v2 partials (eval_v2.sbatch): configuration (c), {all, index} x {heldout, dev}, into
# longtail/eval/runs/<set>-<split>/results_v2/. Needs gt/ (as score.sbatch). No paid calls.
set -euo pipefail
cd /scratch/$USER/longtail
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR=/scratch/$USER/uv-cache LONGTAIL_DATA_DIR=$PWD/data
export LONGTAIL_ALLOW_PAID=0 LONGTAIL_REPLAY=1 LONGTAIL_VERIFIER=v2
module load FFmpeg/7.1.1
SETS=(all index) SPLITS=(heldout dev)
i=$SLURM_ARRAY_TASK_ID
SET=${SETS[$((i / 2))]} SPLIT=${SPLITS[$((i % 2))]}
if [ $SET = all ]; then
  SCENES=$(uv run --no-sync python -c "from pathlib import Path; from longtail.data.pandaset import list_scenes; print(' '.join(list_scenes(Path('data/pandaset'))))")
else
  SCENES=$(uv run --no-sync python -c "from longtail.data.index import load_index; print(' '.join(s['scene_id'] for s in load_index()['scenes']))")
fi
echo "$SET $SPLIT (c, v2) on $(hostname): $(echo $SCENES | wc -w) scenes"
uv run --no-sync python -m longtail.eval.ablation --config c --queries longtail/eval/queries/queries.yaml \
  --split $SPLIT --scenes $SCENES --merge results_partial_v2 --out longtail/eval/runs/$SET-$SPLIT/results_v2
