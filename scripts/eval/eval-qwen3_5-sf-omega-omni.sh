#!/bin/bash
set -euo pipefail

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate ${CONDA_ENV:-qwenvln}

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-False}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

MODEL_TYPE=${MODEL_TYPE:-qwen3_5_omni_head_sf_omega}
EVALUATOR_TYPE=${EVALUATOR_TYPE:-qwen3_5_omni_head_sf_omega}
MODEL_PATH=${MODEL_PATH:-outputs/qwen3_5_0_8b_sf_omega_omni}
OUTPUT_PATH=${OUTPUT_PATH:-results/r2r/val_unseen/qwen3_5_sf_omega_omni}
# Default to the front-only config to MATCH the front-only training data (the generator
# replicates the front frame into all three current slots), keeping the train/eval input
# distribution aligned. To run OmniNav-style real left/front/right panorama eval, set
# HABITAT_CONFIG_PATH=configs/omni_r2r.yaml — but ONLY for a checkpoint trained on real
# 3-camera current views, otherwise the left/right slots are out-of-distribution.
HABITAT_CONFIG_PATH=${HABITAT_CONFIG_PATH:-configs/vln_r2r.yaml}
EVAL_SPLIT=${EVAL_SPLIT:-val_unseen}
OMEGA_MODE=${OMEGA_MODE:-text_align_force_qwen}
LIMIT=${LIMIT:-2}
ACTION_CHUNK_NUM=${ACTION_CHUNK_NUM:-5}
NUM_FRAMES=${NUM_FRAMES:-24}
NUM_HISTORY=${OMNI_HISTORY_IMAGES:-20}
# Match the training image-token budget (see train script note); keeps eval-time image
# resolution consistent with what the omni head was trained on.
MAX_PIXELS=${MAX_PIXELS:-258048}
SAVE_VIDEO=${SAVE_VIDEO:-0}
SPATIAL_FORCING_IMAGE_RESOLUTION=${SPATIAL_FORCING_IMAGE_RESOLUTION:-256}
SPATIAL_FORCING_TEACHER_PATCH_SIZE=${SPATIAL_FORCING_TEACHER_PATCH_SIZE:-16}
SF_GEOMETRY_ENCODER_PATH=${SF_GEOMETRY_ENCODER_PATH:-/mnt/nas5/xiangchen/vlacode/vggt-omega/facebook/VGGT-Omega/vggt_omega_1b_256_text.pt}

mkdir -p "$OUTPUT_PATH"
cp "$(realpath "$0")" "$OUTPUT_PATH"

extra_args=()
if [ "$SAVE_VIDEO" = "1" ] || [ "$SAVE_VIDEO" = "true" ] || [ "$SAVE_VIDEO" = "True" ]; then
  extra_args+=(--save_video)
fi

torchrun --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    joynav/eval/eval_habitat.py \
    --evaluator_type "$EVALUATOR_TYPE" \
    --model_path "$MODEL_PATH" \
    --model_type "$MODEL_TYPE" \
    --output_path "$OUTPUT_PATH" \
    --habitat_config_path "$HABITAT_CONFIG_PATH" \
    --eval_split "$EVAL_SPLIT" \
    --max_pixels "$MAX_PIXELS" \
    --num_frames "$NUM_FRAMES" \
    --num_history "$NUM_HISTORY" \
    --action_chunk_num "$ACTION_CHUNK_NUM" \
    --trajectory_horizon "$ACTION_CHUNK_NUM" \
    --trajectory_dim 3 \
    --omni_waypoint_number "$ACTION_CHUNK_NUM" \
    --omni_step_scale "${OMNI_STEP_SCALE:-0.3}" \
    --omni_norm_method "${OMNI_NORM_METHOD:-min_max_split_arrive}" \
    --omni_num_inference_timesteps "${OMNI_NUM_INFERENCE_TIMESTEPS:-10}" \
    --omni_use_panorama "${OMNI_USE_PANORAMA:-True}" \
    --stop_threshold "${STOP_THRESHOLD:-0.5}" \
    --replan_every "${REPLAN_EVERY:-1}" \
    --limit "$LIMIT" \
    --omega_mode "$OMEGA_MODE" \
    --spatial_forcing_image_resolution "$SPATIAL_FORCING_IMAGE_RESOLUTION" \
    --spatial_forcing_teacher_patch_size "$SPATIAL_FORCING_TEACHER_PATCH_SIZE" \
    --sf_geometry_encoder_path "$SF_GEOMETRY_ENCODER_PATH" \
    "${extra_args[@]}" \
    2>&1 | tee -a "$OUTPUT_PATH/eval_log.log"
