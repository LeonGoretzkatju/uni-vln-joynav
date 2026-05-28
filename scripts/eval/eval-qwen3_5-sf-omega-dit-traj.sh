#!/bin/bash
set -euo pipefail

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate ${CONDA_ENV:-qwenvln}

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-False}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}

MODEL_TYPE=${MODEL_TYPE:-qwen3_5_dit_head_sf_omega}
EVALUATOR_TYPE=${EVALUATOR_TYPE:-qwen3_5_dit_head_sf_omega}
MODEL_PATH=${MODEL_PATH:-outputs/qwen3_5_0_8b_sf_omega_dit_traj}
OUTPUT_PATH=${OUTPUT_PATH:-results/r2r/val_unseen/qwen3_5_sf_omega_dit_traj}
HABITAT_CONFIG_PATH=${HABITAT_CONFIG_PATH:-configs/vln_r2r.yaml}
EVAL_SPLIT=${EVAL_SPLIT:-val_unseen}
OMEGA_MODE=${OMEGA_MODE:-text_align_force_qwen}
LIMIT=${LIMIT:-0}
ACTION_CHUNK_NUM=${ACTION_CHUNK_NUM:-8}
NUM_FRAMES=${NUM_FRAMES:-24}
NUM_HISTORY=${NUM_HISTORY:-8}
MAX_PIXELS=${MAX_PIXELS:-258048}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
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
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --num_frames "$NUM_FRAMES" \
    --num_history "$NUM_HISTORY" \
    --action_chunk_num "$ACTION_CHUNK_NUM" \
    --trajectory_horizon "$ACTION_CHUNK_NUM" \
    --trajectory_dim 3 \
    --limit "$LIMIT" \
    --omega_mode "$OMEGA_MODE" \
    --spatial_forcing_image_resolution "$SPATIAL_FORCING_IMAGE_RESOLUTION" \
    --spatial_forcing_teacher_patch_size "$SPATIAL_FORCING_TEACHER_PATCH_SIZE" \
    --sf_geometry_encoder_path "$SF_GEOMETRY_ENCODER_PATH" \
    "${extra_args[@]}" \
    2>&1 | tee -a "$OUTPUT_PATH/eval_log.log"
