#!/usr/bin/bash
export MAX_JOBS=128

WORKDIR="/data/banyuanhao/Mask-GRPO/code"
SETUP_SCRIPT="bash setup_uv.sh"

sudo kubectl exec -it -n slurm slurm-worker-slinky-1 -- srun --cpus-per-task=128 --mem=0 --chdir="$WORKDIR" $SETUP_SCRIPT