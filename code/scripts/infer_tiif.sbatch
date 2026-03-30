#!/bin/bash
#SBATCH --job-name=tiif_infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --exclude=slinky-[2,5]
#SBATCH --export=ALL
#SBATCH --output=/data/banyuanhao/Mask-GRPO/logs/%j_tiif_infer.out
#SBATCH --error=/data/banyuanhao/Mask-GRPO/logs/%j_tiif_infer.err

set -euo pipefail

# ---------- 网络 ----------
export NCCL_SOCKET_IFNAME=eth0
export LD_LIBRARY_PATH=/usr/lib:/usr/lib64
if [ -f /opt/hpcx/hpcx-init.sh ]; then
  . /opt/hpcx/hpcx-init.sh
fi
ulimit -l unlimited

# ---------- git safe.directory ----------
git config --global --add safe.directory /data/banyuanhao/Mask-GRPO

export HF_TOKEN="$(< /data/banyuanhao/flowgrpo/credentials/hf.key)"
export OPENAI_API_KEY="$(< /data/banyuanhao/super-judgmental/credentials/openai.key)"

WORKDIR=/data/banyuanhao/Mask-GRPO/code
VENV=$WORKDIR/.venv
FLOWGRPO_DIR=/data/banyuanhao/flowgrpo
TIIF_BENCH="$FLOWGRPO_DIR/Benchmarks/TIIF-Bench"

# ---------- 推理参数 ----------
RUN_NAME=maskgrpo_iris
CHECKPOINT_NAME=model_0
MODEL_NAME="${RUN_NAME}_${CHECKPOINT_NAME}"  # eval 目录里的模型名

SHOWO_MODEL_PATH="$WORKDIR/showo.pth"
JSONL_DIR="$TIIF_BENCH/data/testmini_prompts"
EVAL_JSONL_DIR="$TIIF_BENCH/data/testmini_eval_prompts"

if [ -n "$CHECKPOINT_NAME" ]; then
  CHECKPOINT="$WORKDIR/outputs/${RUN_NAME}/train/save/${CHECKPOINT_NAME}.pth"
else
  CHECKPOINT=""
  MODEL_NAME="${RUN_NAME}_base"
fi

IMAGE_DIR="/data/banyuanhao/Mask-GRPO_outputs/TIIF/${RUN_NAME}"
EVAL_OUTPUT_DIR="/data/banyuanhao/Mask-GRPO_outputs/TIIF/${RUN_NAME}-results"

SEED=42
GUIDANCE_SCALE=5.0
TIMESTEPS=50
GPUS_PER_NODE=8

export MASTER_PORT=19005
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"

# ---------- 启动推理 + 评测 ----------
srun --export=ALL bash -lc "
  set -euo pipefail
  source $VENV/bin/activate
  cd $WORKDIR

  echo \"[INFO] HOST=\$(hostname) running TIIF-Bench inference with $GPUS_PER_NODE GPUs\"
  echo \"[INFO] RUN_NAME=$RUN_NAME CHECKPOINT_NAME=$CHECKPOINT_NAME MODEL_NAME=$MODEL_NAME\"

  # ===== Step 1: Generate images (8-GPU parallel) =====
  torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    infer_tiif.py \
    --jsonl_dir $JSONL_DIR \
    --save_dir $IMAGE_DIR \
    --model_name $MODEL_NAME \
    --showo_model_path $SHOWO_MODEL_PATH \
    ${CHECKPOINT:+--checkpoint $CHECKPOINT} \
    --seed $SEED \
    --guidance_scale $GUIDANCE_SCALE \
    --timesteps $TIMESTEPS \
    config=configs/Mask_GRPO_train_512x512.yaml

  echo '[INFO] Generation done. Starting TIIF-Bench GPT evaluation...'

  # ===== Step 2: GPT eval =====
  cd $FLOWGRPO_DIR
  python Benchmarks/TIIF-Bench/eval/eval_with_vlm.py \
    --jsonl_dir $EVAL_JSONL_DIR \
    --image_dir $IMAGE_DIR \
    --eval_model $MODEL_NAME \
    --output_dir $EVAL_OUTPUT_DIR \
    --api_key \"\$OPENAI_API_KEY\" \
    --model gpt-4o-2024-05-13

  echo '[INFO] GPT eval done. Summarizing results...'

  # ===== Step 3: Summary =====
  python Benchmarks/TIIF-Bench/eval/summary_results.py \
    --input_dir $EVAL_OUTPUT_DIR

  echo '[INFO] TIIF-Bench evaluation complete.'
"
