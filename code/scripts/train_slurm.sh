#!/usr/bin/env bash
WORKDIR="/data/banyuanhao/Mask-GRPO/code"
SBATCH_FILE="$WORKDIR/scripts/train.sbatch"

sudo kubectl exec -i -n slurm slurm-controller-0 -- bash -lc \
  "cat > /tmp/mask_grpo.sbatch && sbatch --chdir=\"$WORKDIR\" /tmp/mask_grpo.sbatch" < "$SBATCH_FILE"
