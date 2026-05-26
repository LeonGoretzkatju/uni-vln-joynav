#!/bin/bash
# Dynamic-RoPE training script for Qwen3-VL-2B on TITAN RTX (sm_7.5).
# Key TITAN adjustments vs. single-qwen3-full.sh:
#   * fp16 instead of bf16 (Turing has no native bf16 compute)
#   * SDPA attention instead of flash_attention_2 (flash-attn-2 needs sm_8.0+)
#   * ZeRO-2 with CPU optimizer offload to fit 24GB TITAN RTX cards
#   * Reduced max_pixels / model_max_length to keep activations bounded
set -euo pipefail

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate joynav

export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Force SDPA — flash-attn-2 is not available on Turing (TITAN RTX).
export ATTN_IMPLEMENTATION=sdpa

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=3  # 3x TITAN RTX (GPUs 1,2,3)

# DeepSpeed configuration (ZeRO-2 + CPU optimizer offload for 24GB TITAN RTX)
deepspeed=./scripts/zero2_offload.json

# Model configuration — dynamic-rope variant
llm=/mnt/nas5/xiangchen/vlm_base/Qwen3-VL-2B-Instruct
model_type=qwen3_vl_lm_head_dyna
dataset_type=vln_action

# Training hyperparameters
lr=2e-5
mm_projector_lr=1e-6
batch_size=1
num_train_epochs=1
grad_accum_steps=8
weight_decay=0.01
warmup_ratio=0.03
max_pixels=200704           # 448x448; Turing activations are expensive
max_steps=10                # Short efficiency-validation run; remove for a real train

# Training entry point
entry_file=joynav/train/train_qwen.py

# Dataset configuration
datasets="/mnt/nas5/xiangchen/VLNData/R2R,/mnt/nas5/xiangchen/VLNData/RxR"

# Output configuration
run_name="qwen3_vl_2b_full_dyna_titan"
output_dir=./outputs/${run_name}/validate-dyna-fp16-sdpa-lr_${lr}-mm_lr_${mm_projector_lr}-bs_${batch_size}-ga_${grad_accum_steps}-max_pixels_${max_pixels}

mkdir -p "$output_dir"
script_path="$(realpath "$0")"
cp "$script_path" "$output_dir"

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_type ${model_type} \
    --dataset_type ${dataset_type} \
    --model_name_or_path ${llm} \
    --video_folder ${datasets} \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --fp16 \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_train_epochs} \
    --max_steps ${max_steps} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} \
    --eval_strategy no \
    --save_strategy no \
    --learning_rate ${lr} \
    --mm_projector_lr ${mm_projector_lr} \
    --vision_tower_lr 1e-6 \
    --weight_decay ${weight_decay} \
    --warmup_ratio ${warmup_ratio} \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --run_name ${run_name} \
    --report_to tensorboard"

# Launch training on GPUs 1,2,3
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args} \
         2>&1 | tee -a ${output_dir}/train.log
