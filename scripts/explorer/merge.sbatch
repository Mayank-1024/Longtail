#!/bin/bash
#SBATCH -J longtail-merge
#SBATCH -p short
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o /scratch/%u/longtail/logs/%x-%j.out
# Ground truth on every usable scene (102: 054 is left out, A-240), then the scored ablation per split,
# on all of them and on the scene index (29 of its 30: 054 is out).
# No paid calls: scoring reads the partial results and the labels only.
set -euo pipefail
cd /scratch/$USER/longtail
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR=/scratch/$USER/uv-cache LONGTAIL_DATA_DIR=$PWD/data
export LONGTAIL_ALLOW_PAID=0 LONGTAIL_REPLAY=1
module load FFmpeg/7.1.1
ALL=$(uv run --no-sync python -c "from pathlib import Path; from longtail.data.pandaset import list_scenes; print(' '.join(list_scenes(Path('data/pandaset'))))")
INDEX=$(uv run --no-sync python -c "from longtail.data.index import load_index; print(' '.join(s['scene_id'] for s in load_index()['scenes']))")
uv run --no-sync longtail gt build longtail/eval/queries/queries.yaml --scenes $ALL --out gt
CONFIGS=${CONFIGS:-"c b"}
for SET in all index; do
  if [ $SET = all ]; then SCENES=$ALL; else SCENES=$INDEX; fi
  for SPLIT in heldout dev; do
    for C in $CONFIGS; do
      uv run --no-sync python -m longtail.eval.ablation --config $C --queries longtail/eval/queries/queries.yaml \
        --split $SPLIT --scenes $SCENES --merge results_partial --out longtail/eval/runs/$SET-$SPLIT/results \
        || echo "merge $SET $SPLIT $C failed"
    done
    # ASSUMPTION A-241: configuration (a) reads a Spark Open baseline run on the laptop over this same scene set,
    # copied to runs/baseline_a_$SET; scoring it only checks its cited boxes against the LiDAR (no paid calls)
    if [ -f runs/baseline_a_$SET/rankings.json ]; then
      uv run --no-sync python -m longtail.eval.ablation --config a --queries longtail/eval/queries/queries.yaml \
        --split $SPLIT --scenes $SCENES --baseline-dir runs/baseline_a_$SET \
        --out longtail/eval/runs/$SET-$SPLIT/results || echo "config a $SET $SPLIT failed"
    fi
    uv run --no-sync python -m longtail.eval.breaks --queries longtail/eval/queries/queries.yaml \
      --split $SPLIT --scenes $SCENES --merge results_partial --out longtail/eval/runs/$SET-$SPLIT/results \
      || echo "breaks $SET $SPLIT failed"
  done
done
