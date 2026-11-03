#!/bin/bash
#SBATCH -J longtail-merge
#SBATCH -p short
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o /scratch/%u/longtail/logs/%x-%j.out
# Ground truth on all 103 scenes, then the scored ablation per split, on all 103 and on the 30-scene index.
# No paid calls: scoring reads the partial results and the labels only.
set -euo pipefail
cd /scratch/$USER/longtail
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR=/scratch/$USER/uv-cache LONGTAIL_DATA_DIR=$PWD/data
export LONGTAIL_ALLOW_PAID=0 LONGTAIL_REPLAY=1
module load FFmpeg/7.1.1
ALL=$(ls data/pandaset | tr '\n' ' ')
INDEX=$(uv run --no-sync python -c "import json; print(' '.join(s['scene_id'] for s in json.load(open('longtail/data/index.json'))['scenes']))")
uv run --no-sync longtail gt build longtail/eval/queries/queries.yaml --scenes $ALL --out gt
CONFIGS=${CONFIGS:-"c b"}
for SET in all103 index30; do
  if [ $SET = all103 ]; then SCENES=$ALL; else SCENES=$INDEX; fi
  for SPLIT in heldout dev; do
    for C in $CONFIGS; do
      uv run --no-sync python -m longtail.eval.ablation --config $C --queries longtail/eval/queries/queries.yaml \
        --split $SPLIT --scenes $SCENES --merge results_partial --out longtail/eval/runs/$SET-$SPLIT/results \
        || echo "merge $SET $SPLIT $C failed"
    done
  done
done
