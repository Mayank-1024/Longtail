#!/bin/bash
#SBATCH -J longtail-score
#SBATCH -p short
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 05:00:00
#SBATCH -o /scratch/%u/longtail/logs/%x-%A_%a.out
#SBATCH --array=0-15
# One scoring run per task, all in parallel: {all, index} x {heldout, dev} x {c, b, a, breaks}.
# Scoring a run reads every claim's scene labels (about 25 s a scene), so the 16 runs one after another
# (merge.sbatch) take most of a day; side by side they take as long as the slowest one.
# Needs gt/ (merge.sbatch with ONLY_GT=1, or `longtail gt build`) and the partial results. No paid calls.
set -euo pipefail
cd /scratch/$USER/longtail
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR=/scratch/$USER/uv-cache LONGTAIL_DATA_DIR=$PWD/data
export LONGTAIL_ALLOW_PAID=0 LONGTAIL_REPLAY=1
module load FFmpeg/7.1.1
SETS=(all index) SPLITS=(heldout dev) WHATS=(c b a breaks)
i=$SLURM_ARRAY_TASK_ID
SET=${SETS[$((i / 8))]} SPLIT=${SPLITS[$((i / 4 % 2))]} WHAT=${WHATS[$((i % 4))]}
if [ $SET = all ]; then
  SCENES=$(uv run --no-sync python -c "from pathlib import Path; from longtail.data.pandaset import list_scenes; print(' '.join(list_scenes(Path('data/pandaset'))))")
else
  SCENES=$(uv run --no-sync python -c "from longtail.data.index import load_index; print(' '.join(s['scene_id'] for s in load_index()['scenes']))")
fi
OUT=longtail/eval/runs/$SET-$SPLIT/results
Q=longtail/eval/queries/queries.yaml
echo "$SET $SPLIT $WHAT on $(hostname): $(echo $SCENES | wc -w) scenes"
case $WHAT in
  b|c) uv run --no-sync python -m longtail.eval.ablation --config $WHAT --queries $Q --split $SPLIT \
         --scenes $SCENES --merge results_partial --out $OUT ;;
  a)   uv run --no-sync python -m longtail.eval.ablation --config a --queries $Q --split $SPLIT \
         --scenes $SCENES --baseline-dir runs/baseline_a_$SET --out $OUT ;;
  breaks) uv run --no-sync python -m longtail.eval.breaks --queries $Q --split $SPLIT \
         --scenes $SCENES --merge results_partial --out $OUT ;;
esac
