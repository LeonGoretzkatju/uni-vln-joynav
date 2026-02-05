#!/bin/bash

# export CUDA_VISIBLE_DEVICES=1

source /mnt/workspace/users/vln/envs/conda3/bin/activate joynav


export TOKENIZERS_PARALLELISM=false
# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=8

# DeepSpeed configuration
# 注意,由于zero3.json会切分模型参数,会让ActionLatent初始化MultiheadAttention的tensor失败,需要使用zero2.json
deepspeed=./scripts/zero3.json

# Model configuration
llm=/mnt/workspace/users/zhaolin/vln/models/Qwen3-VL-4B-Instruct
# /mnt/workspace/users/heqingrong/dataset/pretrained_models/Qwen/Qwen3-VL-8B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=2e-5
mm_projector_lr=1e-6
batch_size=1
num_train_epochs=1
grad_accum_steps=8
weight_decay=0.01
warmup_ratio=0.03
max_pixels=451584
tune_mm_llm=true


# Training entry point
entry_file=joynav/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets="data/trajectory_data/R2R","data/trajectory_data/RxR"

# Output configuration
run_name="qwen3_vl_action"
output_dir=./outputs/T_0205/stage1-r2r+rxr-freeze_vision-lr_${lr}-batch_size_${batch_size}-grad_accum_steps_${grad_accum_steps}-epochs_${num_train_epochs}-tune_mm_llm_${tune_mm_llm}

if [ ! -d "$output_dir" ]; then
  mkdir -p $output_dir
fi
script_path="$(realpath "$0")"
cp "$script_path" "$output_dir"

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --dataset_type "vln_action" \
    --model_type "qwen3_vl_action" \
    --model_name_or_path "${llm}" \
    --video_folder ${datasets} \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm ${tune_mm_llm} \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --mm_projector_lr ${mm_projector_lr} \
    --vision_tower_lr 1e-6 \
    --weight_decay ${weight_decay} \
    --warmup_ratio ${warmup_ratio} \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 163840 \
    --gradient_checkpointing False \
    --dataloader_num_workers 0 \
    --run_name ${run_name} \
    --report_to tensorboard" \

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args} \
         2>&1 | tee -a ${output_dir}/train.log


