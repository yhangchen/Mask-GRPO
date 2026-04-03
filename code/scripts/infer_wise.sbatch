#!/bin/bash
#SBATCH --job-name=wise_infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH --output=/data/banyuanhao/Mask-GRPO/logs/%j_wise_infer.out
#SBATCH --error=/data/banyuanhao/Mask-GRPO/logs/%j_wise_infer.err

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

# ---------- git safe.directory ----------
git config --global --add safe.directory /data/banyuanhao/Mask-GRPO

export HF_TOKEN="$(< /data/banyuanhao/flowgrpo/credentials/hf.key)"

WORKDIR=/data/banyuanhao/Mask-GRPO/code
VENV=$WORKDIR/.venv

# ---------- 推理参数 ----------
RUN_NAME=maskgrpo_iris_0.01
CHECKPOINT_NAME="model_50"

SHOWO_MODEL_PATH="$WORKDIR/showo.pth"
DATA_PATH="/data/banyuanhao/flowgrpo/Benchmarks/WISE/data/wise_eval_data.json"

if [ -n "$CHECKPOINT_NAME" ]; then
  CHECKPOINT="$WORKDIR/outputs/${RUN_NAME}/train/save/${CHECKPOINT_NAME}.pth"
  SAVE_DIR="/data/banyuanhao/Mask-GRPO_outputs/WISE/${RUN_NAME}/${CHECKPOINT_NAME}"
else
  CHECKPOINT=""
  SAVE_DIR="/data/banyuanhao/Mask-GRPO_outpurevts/WISE/base_model"
fi
NUM_GENERATION=3
SEED=42
GUIDANCE_SCALE=5.0
TIMESTEPS=50
GPUS_PER_NODE=8

export MASTER_PORT=19004
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"

# ---------- WISE eval 参数 ----------
export OPENAI_API_KEY="$(< /data/banyuanhao/super-judgmental/credentials/openai.key)"
WISE_DATA_ROOT="/data/banyuanhao/flowgrpo/Benchmarks/WISE/data"
FLOWGRPO_DIR="/data/banyuanhao/flowgrpo"

if [ -n "$CHECKPOINT_NAME" ]; then
  RESULTS_ROOT="/data/banyuanhao/Mask-GRPO_outputs/WISE/${RUN_NAME}-${CHECKPOINT_NAME}-results"
else
  RESULTS_ROOT="/data/banyuanhao/Mask-GRPO_outputs/WISE/base_model-results"
fi

# ---------- 启动推理 ----------
srun --export=ALL bash -lc "
  set -euo pipefail
  source $VENV/bin/activate
  cd $WORKDIR

  echo \"HOST=\$(hostname) running WISE inference with $GPUS_PER_NODE GPUs\"

  # ===== Step 1: Generate images (8-GPU parallel) =====
  torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    infer_wise.py \
    --data_path $DATA_PATH \
    --save_dir $SAVE_DIR \
    --showo_model_path $SHOWO_MODEL_PATH \
    ${CHECKPOINT:+--checkpoint $CHECKPOINT} \
    --num_generation $NUM_GENERATION \
    --seed $SEED \
    --guidance_scale $GUIDANCE_SCALE \
    --timesteps $TIMESTEPS \
    config=configs/Mask_GRPO_train_512x512.yaml

  echo '[INFO] Generation done. Starting WISE GPT evaluation...'

  cd $FLOWGRPO_DIR

  WISE_CATEGORIES=(cultural_common_sense natural_science spatio-temporal_reasoning)

  for CATEGORY in \"\${WISE_CATEGORIES[@]}\"; do
    JSON_PATH=\"${WISE_DATA_ROOT}/\${CATEGORY}.json\"
    OUTPUT_DIR=\"${RESULTS_ROOT}/\${CATEGORY}\"
    RESULT_FULL=\"${RESULTS_ROOT}/\${CATEGORY}_full_results.json\"
    RESULT_SCORES=\"${RESULTS_ROOT}/\${CATEGORY}_scores_results.jsonl\"

    echo \"[INFO] WISE eval \${CATEGORY}\"
    python ./Benchmarks/WISE/gpt_eval.py \
      --json_path \"\$JSON_PATH\" \
      --output_dir \"\$OUTPUT_DIR\" \
      --image_dir \"$SAVE_DIR\" \
      --api_key \"$OPENAI_API_KEY\" \
      --model gpt-4o-2024-05-13 \
      --result_full \"\$RESULT_FULL\" \
      --result_scores \"\$RESULT_SCORES\" \
      --max_workers 96
  done

  echo '[INFO] Running WISE Calculate...'
  if [ -f Benchmarks/WISE/Calculate.py ]; then
    python Benchmarks/WISE/Calculate.py \
      \"${RESULTS_ROOT}/cultural_common_sense_scores_results.jsonl\" \
      \"${RESULTS_ROOT}/natural_science_scores_results.jsonl\" \
      \"${RESULTS_ROOT}/spatio-temporal_reasoning_scores_results.jsonl\" \
      --category all
  fi

  echo '[INFO] WISE evaluation complete.'
"
