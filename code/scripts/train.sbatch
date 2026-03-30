#!/bin/bash
#SBATCH --job-name=mask_grpo
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH --output=/data/banyuanhao/Mask-GRPO/logs/%j_mask_grpo.out
#SBATCH --error=/data/banyuanhao/Mask-GRPO/logs/%j_mask_grpo.err

set -euo pipefail

# ---------- 网络 ----------
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export LD_LIBRARY_PATH=/usr/lib:/usr/lib64
if [ -f /opt/hpcx/hpcx-init.sh ]; then
  . /opt/hpcx/hpcx-init.sh
fi
ulimit -l unlimited

export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=13

# ---------- git safe.directory (wandb git integration needs this) ----------
git config --global --add safe.directory /data/banyuanhao/Mask-GRPO

# ---------- wandb ----------
export WANDB_NAME="mask_grpo"
export WANDB_PROJECT="mask_grpo"
export WANDB_API_KEY="$(< /data/banyuanhao/flowgrpo/credentials/wandb.key)"
export HF_TOKEN="$(< /data/banyuanhao/flowgrpo/credentials/hf.key)"

# ---------- master / port ----------
export MASTER_PORT=19001
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"

NNODES=$SLURM_NNODES
GPUS_PER_NODE=8
NUM_PROCESSES=$((NNODES * GPUS_PER_NODE))

WORKDIR=/data/banyuanhao/Mask-GRPO/code
VENV=$WORKDIR/.venv

# ---------- 每个节点各跑一个 launcher ----------
srun --export=ALL bash -lc "
  set -euo pipefail
  source $VENV/bin/activate
  cd $WORKDIR

  echo \"HOST=\$(hostname) SLURM_PROCID=\$SLURM_PROCID MASTER=\$MASTER_ADDR:\$MASTER_PORT\"

  $VENV/bin/accelerate launch \
    --num_machines $NNODES \
    --num_processes $NUM_PROCESSES \
    --machine_rank \$SLURM_PROCID \
    --main_process_ip \$MASTER_ADDR \
    --main_process_port \$MASTER_PORT \
    Mask_GRPO_train.py \
    config=configs/Mask_GRPO_train_512x512.yaml \
    mode='t2i' \
    reward_type='iris'
"

# iris ima clip uni 
