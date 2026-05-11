#!/bin/bash
set -euo pipefail

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate joynav

export TOKENIZERS_PARALLELISM=false
export JOYNAV_SF_DEBUG=${JOYNAV_SF_DEBUG:-0}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}

deepspeed=${DEEPSPEED_CONFIG:-./scripts/zero2.json}

llm=${MODEL_PATH:-/mnt/nas5/xiangchen/vlm_base/Qwen3-VL-2B-Instruct}
model_type=qwen3_vl_lm_head_sf
dataset_type=vln_action_sf
sf_geometry_encoder_path=${SF_GEOMETRY_ENCODER_PATH:-joynav/model/geometry_encoder/depth_anything_v2_vitl.pth}

lr=${LR:-2e-5}
mm_projector_lr=${MM_PROJECTOR_LR:-1e-6}
batch_size=${BATCH_SIZE:-1}
num_train_epochs=${NUM_TRAIN_EPOCHS:-1}
grad_accum_steps=${GRAD_ACCUM_STEPS:-8}
weight_decay=${WEIGHT_DECAY:-0.01}
warmup_ratio=${WARMUP_RATIO:-0.03}
max_pixels=${MAX_PIXELS:-200704}
sf_alpha=${SF_ALPHA:-0.1}
sf_align_layers=${SF_ALIGN_LAYERS:-24}
max_steps=${MAX_STEPS:-"-1"}
model_max_length=${MODEL_MAX_LENGTH:-163840}
precision_arg=${PRECISION_ARG:-"--fp16"}
logging_steps=${LOGGING_STEPS:-10}

num_frames=${NUM_FRAMES:-32}
num_history=${NUM_HISTORY:-8}
action_chunk_num=${ACTION_CHUNK_NUM:-4}

if [ "${max_steps}" != "-1" ]; then
  save_strategy=${SAVE_STRATEGY:-"no"}
else
  save_strategy=${SAVE_STRATEGY:-"steps"}
fi
save_steps=${SAVE_STEPS:-2000}

entry_file=joynav/train/train_qwen.py
datasets=${DATASETS:-"/mnt/nas5/xiangchen/VLNData/R2R,/mnt/nas5/xiangchen/VLNData/RxR"}

run_name=${RUN_NAME:-"qwen3_vl_2b_full_sf_3090"}
output_dir=${OUTPUT_DIR:-./outputs/${run_name}/stage1-r2r+rxr-sf_alpha_${sf_alpha}-layers_${sf_align_layers}-lr_${lr}-mm_lr_${mm_projector_lr}-batch_size_${batch_size}-grad_accum_steps_${grad_accum_steps}-epochs_${num_train_epochs}-max_pixels_${max_pixels}}
if [ "${max_steps}" != "-1" ]; then
  output_dir=${output_dir}-validate_max_steps_${max_steps}-port_${MASTER_PORT}
fi

mkdir -p "${output_dir}"
script_path="$(realpath "$0")"
cp "$script_path" "$output_dir"

args="
    --deepspeed ${deepspeed} \
    --model_type ${model_type} \
    --dataset_type ${dataset_type} \
    --model_name_or_path ${llm} \
    --video_folder ${datasets} \
    --num_frames ${num_frames} \
    --num_history ${num_history} \
    --action_chunk_num ${action_chunk_num} \
    --add_continuous_action False \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --use_lora False \
    --sf_enabled True \
    --sf_alpha ${sf_alpha} \
    --sf_align_layers=${sf_align_layers} \
    --sf_geometry_encoder_path ${sf_geometry_encoder_path} \
    --spatial_forcing_image_size 518 \
    ${precision_arg} \
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
    --weight_decay ${weight_decay} \
    --warmup_ratio ${warmup_ratio} \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps ${logging_steps} \
    --model_max_length ${model_max_length} \
    --gradient_checkpointing True \
    --ddp_timeout 7200 \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to tensorboard"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args} \
         2>&1 | tee -a ${output_dir}/train.log
