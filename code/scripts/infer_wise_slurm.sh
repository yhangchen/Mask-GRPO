#!/usr/bin/env bash
WORKDIR="/data/banyuanhao/Mask-GRPO/code"
SBATCH_FILE="$WORKDIR/scripts/infer_wise.sbatch"

sudo kubectl exec -i -n slurm slurm-controller-0 -- bash -lc \
  "cat > /tmp/infer_wise.sbatch && sbatch --chdir=\"$WORKDIR\" /tmp/infer_wise.sbatch" < "$SBATCH_FILE"
