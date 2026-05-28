#!/bin/bash
set -euo pipefail

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate ${CONDA_ENV:-qwenvln}

export TOKENIZERS_PARALLELISM=false
export JOYNAV_SF_DEBUG=${JOYNAV_SF_DEBUG:-0}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-3}

deepspeed=${DEEPSPEED_CONFIG:-./scripts/zero2.json}
llm=${MODEL_PATH:-/mnt/nas5/xiangchen/vlm_base/Qwen3.5-0.8B}
model_type=qwen3_5_dit_head_sf_omega
dataset_type=${DATASET_TYPE:-continuous_vlnn1_action_noninterleave_sf_omega}
omega_mode=${OMEGA_MODE:-text_align_force_qwen}

datasets=${DATASETS:-/mnt/nas5/xiangchen/VLNData/InternData-N1/vln_n1/traj_data}
run_name=${RUN_NAME:-qwen3_5_0_8b_sf_omega_dit_traj}
output_dir=${OUTPUT_DIR:-./outputs/${run_name}}

lr=${LR:-2e-5}
mm_projector_lr=${MM_PROJECTOR_LR:-1e-6}
batch_size=${BATCH_SIZE:-1}
num_train_epochs=${NUM_TRAIN_EPOCHS:-1}
grad_accum_steps=${GRAD_ACCUM_STEPS:-2}
max_pixels=${MAX_PIXELS:-258048}
model_max_length=${MODEL_MAX_LENGTH:-163840}
max_steps=${MAX_STEPS:-"-1"}
save_steps=${SAVE_STEPS:-3000}
save_strategy=${SAVE_STRATEGY:-steps}
gpu_ids=${CUDA_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-1,2,3}}

num_history=${NUM_HISTORY:-8}
action_chunk_size=${ACTION_CHUNK_SIZE:-8}
trajectory_stride=${TRAJECTORY_STRIDE:-3}
sf_alpha=${SF_ALPHA:-0.1}
sf_align_layers=${SF_ALIGN_LAYERS:-18}
sf_teacher_layers=${SF_TEACHER_LAYERS:-23}
sf_geometry_encoder_path=${SF_GEOMETRY_ENCODER_PATH:-/mnt/nas5/xiangchen/vlacode/vggt-omega/facebook/VGGT-Omega/vggt_omega_1b_256_text.pt}
spatial_forcing_image_resolution=${SPATIAL_FORCING_IMAGE_RESOLUTION:-256}

mkdir -p "${output_dir}"
cp "$(realpath "$0")" "$output_dir"

args="
    --deepspeed ${deepspeed} \
    --model_type ${model_type} \
    --dataset_type ${dataset_type} \
    --model_name_or_path ${llm} \
    --video_folder ${datasets} \
    --num_history_frames ${num_history} \
    --interleaved_num_chunks ${INTERLEAVED_NUM_CHUNKS:-4} \
    --action_chunk_size ${action_chunk_size} \
    --action_dim 3 \
    --trajectory_horizon ${action_chunk_size} \
    --trajectory_dim 3 \
    --trajectory_stride ${trajectory_stride} \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --use_lora False \
    --sf_enabled True \
    --sf_alpha ${sf_alpha} \
    --sf_align_layers=${sf_align_layers} \
    --sf_teacher_layers=${sf_teacher_layers} \
    --omega_mode ${omega_mode} \
    --sf_geometry_encoder_path ${sf_geometry_encoder_path} \
    --sf_target_dim 2048 \
    --sf_add_pos_embed False \
    --spatial_forcing_teacher_patch_size 16 \
    --spatial_forcing_image_resolution ${spatial_forcing_image_resolution} \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_train_epochs} \
    --max_steps ${max_steps} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} \
    --eval_strategy no \
    --save_strategy ${save_strategy} \
    --save_steps ${save_steps} \
    --save_total_limit 6 \
    --learning_rate ${lr} \
    --mm_projector_lr ${mm_projector_lr} \
    --vision_tower_lr 1e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --logging_nan_inf_filter False \
    --model_max_length ${model_max_length} \
    --gradient_checkpointing True \
    --ddp_timeout 7200 \
    --dataloader_num_workers 3 \
    --run_name ${run_name} \
    --report_to tensorboard"

CUDA_VISIBLE_DEVICES=${gpu_ids} torchrun --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    joynav/train/train_qwen.py ${args} \
    2>&1 | tee -a ${output_dir}/train.log
