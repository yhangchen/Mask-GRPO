export CUDA_VISIBLE_DEVICES=4,5,6,7
export TMPDIR=/tmp/mask_grpo_$$
mkdir -p "$TMPDIR"
accelerate launch --num_processes=4 --num_machines=1 --mixed_precision=bf16 \
--machine_rank=0 \
--main_process_ip=127.0.0.1 \
--main_process_port=29500 Mask_GRPO_train.py config=configs/Mask_GRPO_train_512x512.yaml \
mode='t2i'
