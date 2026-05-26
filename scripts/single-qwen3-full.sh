#!/bin/bash
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate joynav
export TOKENIZERS_PARALLELISM=false
export JOYNAV_SF_DEBUG=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=3  # 4x 4090/3090/TITAN GPUs

# DeepSpeed configuration (ZeRO-2 is more efficient for LoRA)
# ZeRO-3 partitions model across GPUs (~2.7GB/card vs ~8GB with ZeRO-2)
# Switch to zero3_offload.json if still OOM
deepspeed=./scripts/zero2.json

# Model configuration
llm=/mnt/nas5/xiangchen/vlm_base/Qwen3-VL-2B-Instruct
model_type=qwen3_vl_lm_head
dataset_type=vln_action

# Training hyperparameters
lr=2e-5
mm_projector_lr=1e-6
batch_size=1
num_train_epochs=1
grad_accum_steps=8
weight_decay=0.01
warmup_ratio=0.03
max_pixels=200704

# Training entry point
entry_file=joynav/train/train_qwen.py

# Dataset configuration
datasets="/mnt/nas5/xiangchen/VLNData/R2R,/mnt/nas5/xiangchen/VLNData/RxR"

# Output configuration
run_name="qwen3_vl_2b_full"
output_dir=./outputs/${run_name}/stage1-r2r+rxr-freeze_vision-lr_${lr}-mm_lr_${mm_projector_lr}-batch_size_${batch_size}-grad_accum_steps_${grad_accum_steps}-epochs_${num_train_epochs}-max_pixels_${max_pixels}

if [ ! -d "$output_dir" ]; then
  mkdir -p $output_dir
fi
script_path="$(realpath "$0")"
cp "$script_path" "$output_dir"

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_type ${model_type} \
    --dataset_type ${dataset_type} \
    --model_name_or_path "${llm}" \
    --video_folder ${datasets} \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 4000 \
    --save_total_limit 6 \
    --learning_rate ${lr} \
    --mm_projector_lr ${mm_projector_lr} \
    --vision_tower_lr 1e-6 \
    --weight_decay ${weight_decay} \
    --warmup_ratio ${warmup_ratio} \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 20 \
    --model_max_length 163840 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to tensorboard"

# Launch training
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args} \
         2>&1 | tee -a ${output_dir}/train.log
