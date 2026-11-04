#!/bin/bash
#SBATCH -J longtail-hero-v2
#SBATCH -p short
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH --array=0-28             # the 29 index scenes, the ones a search covers
#SBATCH -o /scratch/%u/longtail/logs/%x-%A_%a.out
# The demo's hero spec verified with the v2 verifier, one index scene per task, as pipeline.search_scene does on
# the laptop: runs/<spec>-<scene>-v2/verify_run.json, for the laptop search to resume from. Replay only.
# SPEC=<path to the spec JSON> (default runs/hero_spec.json).
set -euo pipefail
cd /scratch/$USER/longtail
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR=/scratch/$USER/uv-cache
module load FFmpeg/7.1.1
mapfile -t SCENES < <(uv run --no-sync python -c "
from longtail.data.index import load_index
print('\n'.join(s['scene_id'] for s in load_index()['scenes']))")
SID=${SCENES[$SLURM_ARRAY_TASK_ID]}
mkdir -p runs/ledgers
export LONGTAIL_DATA_DIR=$PWD/data LONGTAIL_REPLAY=1 LONGTAIL_ALLOW_PAID=0 LONGTAIL_VERIFIER=v2 \
  LONGTAIL_LEDGER=runs/ledgers/hero-v2-$SID.jsonl LONGTAIL_SAM_WORKERS=6 PYTHONPATH=$PWD
unset MODEL_API_KEY BASETEN_API_KEY
echo "hero spec, scene $SID on $(hostname), verifier $LONGTAIL_VERIFIER"
uv run --no-sync python scripts/search_scene.py "${SPEC:-runs/hero_spec.json}" "$SID"
