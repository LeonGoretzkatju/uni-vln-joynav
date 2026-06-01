#!/bin/bash
set -euo pipefail

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate ${CONDA_ENV:-qwenvln}

export TOKENIZERS_PARALLELISM=false
export JOYNAV_SF_DEBUG=${JOYNAV_SF_DEBUG:-0}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

deepspeed=${DEEPSPEED_CONFIG:-./scripts/zero2.json}
llm=${MODEL_PATH:-/mnt/nas5/xiangchen/vlm_base/Qwen3.5-0.8B}
model_type=qwen3_5_omni_head_sf_omega
dataset_type=${DATASET_TYPE:-omni_waypoint_sf_omega}
omega_mode=${OMEGA_MODE:-text_align_force_qwen}

datasets=${DATASETS:-/mnt/nas5/xiangchen/VLNData/InternData-N1/vln_n1/traj_data,/mnt/nas5/xiangchen/VLNData/R2R,/mnt/nas5/xiangchen/VLNData/RxR}
run_name=${RUN_NAME:-qwen3_5_0_8b_sf_omega_omni}
output_dir=${OUTPUT_DIR:-./outputs/${run_name}}
omni_json=${OMNI_JSON:-${output_dir}/omni_waypoint_train.json}

lr=${LR:-1e-5}
mm_projector_lr=${MM_PROJECTOR_LR:-1e-6}
batch_size=${BATCH_SIZE:-2}
num_train_epochs=${NUM_TRAIN_EPOCHS:-5}
grad_accum_steps=${GRAD_ACCUM_STEPS:-1}
# 258048 matches the SF teacher (spatial_forcing_image_resolution=256, patch=16) under
# omega_mode=text_align_force_qwen, same as the known-good nextdit-traj script. Smaller
# values downscale the Qwen image below the teacher grid and crash the SF token match.
max_pixels=${MAX_PIXELS:-258048}
model_max_length=${MODEL_MAX_LENGTH:-163840}
max_steps=${MAX_STEPS:--1}
save_steps=${SAVE_STEPS:-1000}
save_strategy=${SAVE_STRATEGY:-steps}
gpu_ids=${CUDA_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}

# Precision: bf16 by default (Ampere+ clusters, matches OmniNav torch_dtype=bfloat16).
# Set BF16=false on Turing test GPUs (CC 7.5, weak native bf16) to train in fp16.
precision_flag="--bf16"
case "${BF16:-true}" in
    false|False|0|fp16) precision_flag="--fp16" ;;
esac

num_history=${OMNI_HISTORY_IMAGES:-20}
waypoint_number=${WAYPOINT_NUMBER:-5}
trajectory_stride=${TRAJECTORY_STRIDE:-3}
sf_alpha=${SF_ALPHA:-0.1}
sf_align_layers=${SF_ALIGN_LAYERS:-18}
sf_teacher_layers=${SF_TEACHER_LAYERS:-23}
sf_geometry_encoder_path=${SF_GEOMETRY_ENCODER_PATH:-/mnt/nas5/xiangchen/vlacode/vggt-omega/facebook/VGGT-Omega/vggt_omega_1b_256_text.pt}
spatial_forcing_image_resolution=${SPATIAL_FORCING_IMAGE_RESOLUTION:-256}

mkdir -p "${output_dir}"
cp "$(realpath "$0")" "$output_dir"

if [ "${REGENERATE_OMNI_JSON:-1}" = "1" ] || [ ! -s "${omni_json}" ]; then
    python scripts/data/generate_omni_waypoint_json.py \
        --video-folder "${datasets}" \
        --output "${omni_json}" \
        --max-samples "${OMNI_MAX_SAMPLES:-300}" \
        --num-history-images "${num_history}" \
        --waypoint-number "${waypoint_number}" \
        --trajectory-stride "${trajectory_stride}" \
        --step-scale "${OMNI_STEP_SCALE:-0.3}" \
        2>&1 | tee -a "${output_dir}/generate_omni_json.log"
fi

args="
    --deepspeed ${deepspeed} \
    --model_type ${model_type} \
    --dataset_type ${dataset_type} \
    --model_name_or_path ${llm} \
    --omni_json_path ${omni_json} \
    --waypoint_number ${waypoint_number} \
    --omni_waypoint_number ${waypoint_number} \
    --trajectory_horizon ${waypoint_number} \
    --trajectory_dim 3 \
    --omni_step_scale ${OMNI_STEP_SCALE:-0.3} \
    --omni_norm_method ${OMNI_NORM_METHOD:-min_max_split_arrive} \
    --omni_flow_layers ${OMNI_FLOW_LAYERS:-16} \
    --omni_flow_heads ${OMNI_FLOW_HEADS:-32} \
    --omni_flow_dropout ${OMNI_FLOW_DROPOUT:-0.2} \
    --omni_num_inference_timesteps ${OMNI_NUM_INFERENCE_TIMESTEPS:-10} \
    --omni_noise_beta_alpha ${OMNI_NOISE_BETA_ALPHA:-1.5} \
    --omni_noise_beta_beta ${OMNI_NOISE_BETA_BETA:-1.0} \
    --omni_noise_s ${OMNI_NOISE_S:-0.999} \
    --omni_action_dim ${OMNI_ACTION_DIM:-4} \
    --omni_num_timestep_buckets ${OMNI_NUM_TIMESTEP_BUCKETS:-100} \
    --omni_query_action_layers ${OMNI_QUERY_ACTION_LAYERS:-1} \
    --omni_use_arrive_list ${OMNI_USE_ARRIVE_LIST:-True} \
    --stop_head_loss_weight ${STOP_HEAD_LOSS_WEIGHT:-1.0} \
    --data_flatten False \
    --tune_mm_vision ${TUNE_MM_VISION:-True} \
    --tune_mm_mlp ${TUNE_MM_MLP:-True} \
    --tune_mm_llm ${TUNE_MM_LLM:-True} \
    --model_load_dtype ${MODEL_LOAD_DTYPE:-auto} \
    --use_lora ${USE_LORA:-False} \
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
    ${precision_flag} \
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
    --save_total_limit 3 \
    --learning_rate ${lr} \
    --mm_projector_lr ${mm_projector_lr} \
    --vision_tower_lr 1e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.0 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --logging_nan_inf_filter False \
    --model_max_length ${model_max_length} \
    --gradient_checkpointing True \
    --ddp_timeout 7200 \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS:-0} \
    --run_name ${run_name} \
    --report_to tensorboard"

CUDA_VISIBLE_DEVICES=${gpu_ids} torchrun --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    joynav/train/train_qwen.py ${args} \
    2>&1 | tee -a ${output_dir}/train.log
