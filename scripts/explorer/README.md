# Running the evaluation on Northeastern's Explorer cluster

Only if a full evaluation would take more than about 4 h on a laptop (measure one real scene first). Explorer is
used for CPU work only; every model call is replayed from the cache made on a laptop, so **no API key goes on the
cluster** and nothing is billed from it. Check that Research Computing allows this use first (rchelp@northeastern.edu).

## Steps

1. Log in: https://rc.northeastern.edu/ood/ (Clusters, then Explorer Shell), or SSH per RC's "Connecting To Explorer".
2. Check the partition names and limits, then edit the `#SBATCH -p` lines below to match:
   `sinfo -s` (a CPU partition such as `short`; `gpu`/`gpu-short` are GPU partitions, not needed here).
3. Copy the repo and the model cache up (from the laptop):
   `rsync -a --exclude data --exclude web/node_modules ~/Desktop/longtail/ <user>@<login-host>:/scratch/<user>/longtail/`
   The `cache/` folder must come along: it holds the SAM, planner and card responses the run replays.
4. Download the data on a compute node: `sbatch scripts/explorer/download.sbatch`
5. Run: `sbatch scripts/explorer/eval.sbatch` (one array task per configuration).
6. Copy results back: `rsync -a <user>@<login-host>:/scratch/<user>/longtail/runs/ ~/Desktop/longtail/runs/`

Scratch is purged after a while; this is a one-off run, so keep nothing there you need later.
