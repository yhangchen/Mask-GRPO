#!/usr/bin/env bash
WORKDIR="/data/banyuanhao/Mask-GRPO/code"
SBATCH_FILE="$WORKDIR/scripts/infer_tiif.sbatch"

sudo kubectl exec -i -n slurm slurm-controller-0 -- bash -lc \
  "cat > /tmp/infer_tiif.sbatch && sbatch --chdir=\"$WORKDIR\" /tmp/infer_tiif.sbatch" < "$SBATCH_FILE"
