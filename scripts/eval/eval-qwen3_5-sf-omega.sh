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

MODEL_TYPE=${MODEL_TYPE:-qwen3_5_lm_head_sf_omega}
EVALUATOR_TYPE=${EVALUATOR_TYPE:-qwen3_5_lm_head_sf_omega}
OMEGA_MODE=${OMEGA_MODE:-text_align_force_qwen}

TEXT_ALIGN_CHECKPOINT=/mnt/nas5/xiangchen/vlacode/JD-VLN/outputs/qwen3_5_0_8b_full_sf_omega_bfloat/stage1-r2r+rxr-omega_sf_alpha_0.1-layers_18-teacher_23-lr_5e-6-mm_lr_1e-6-batch_size_1-grad_accum_steps_1-epochs_2-max_pixels_200704/checkpoint-78000
FORCE_QWEN_CHECKPOINT=/mnt/nas5/xiangchen/vlacode/JD-VLN/outputs/qwen3_5_0_8b_full_sf_omega_force_no-interpo/stage1-r2r+rxr-omega_sf_alpha_0.1-layers_18-teacher_23-lr_2e-5-mm_lr_1e-6-batch_size_1-grad_accum_steps_2-epochs_2-max_pixels_258048/checkpoint-40270
OMEGA_512_CHECKPOINT=/mnt/nas5/xiangchen/vlacode/vggt-omega/facebook/VGGT-Omega/vggt_omega_1b_512.pt
OMEGA_TEXT_CHECKPOINT=/mnt/nas5/xiangchen/vlacode/vggt-omega/facebook/VGGT-Omega/vggt_omega_1b_256_text.pt

case "$OMEGA_MODE" in
  text_align)
    DEFAULT_MODEL_PATH=$TEXT_ALIGN_CHECKPOINT
    DEFAULT_OUTPUT_PATH=results/r2r/val_unseen/qwen3_5_sf_omega_text_align
    MAX_PIXELS=${MAX_PIXELS:-258048}
    SPATIAL_FORCING_IMAGE_RESOLUTION=${SPATIAL_FORCING_IMAGE_RESOLUTION:-256}
    SF_GEOMETRY_ENCODER_PATH=${SF_GEOMETRY_ENCODER_PATH:-$OMEGA_TEXT_CHECKPOINT}
    ;;
  text_align_force_qwen)
    DEFAULT_MODEL_PATH=$FORCE_QWEN_CHECKPOINT
    DEFAULT_OUTPUT_PATH=results/r2r/val_unseen/qwen3_5_sf_omega_text_align_force_qwen
    MAX_PIXELS=${MAX_PIXELS:-258048}
    SPATIAL_FORCING_IMAGE_RESOLUTION=${SPATIAL_FORCING_IMAGE_RESOLUTION:-256}
    SF_GEOMETRY_ENCODER_PATH=${SF_GEOMETRY_ENCODER_PATH:-$OMEGA_TEXT_CHECKPOINT}
    ;;
  512_1b)
    DEFAULT_MODEL_PATH=
    DEFAULT_OUTPUT_PATH=results/r2r/val_unseen/qwen3_5_sf_omega_512_1b
    MAX_PIXELS=${MAX_PIXELS:-200704}
    SPATIAL_FORCING_IMAGE_RESOLUTION=${SPATIAL_FORCING_IMAGE_RESOLUTION:-512}
    SF_GEOMETRY_ENCODER_PATH=${SF_GEOMETRY_ENCODER_PATH:-$OMEGA_512_CHECKPOINT}
    ;;
  *)
    echo "Unsupported OMEGA_MODE=$OMEGA_MODE" >&2
    exit 1
    ;;
esac

if [ "$OMEGA_MODE" = "512_1b" ] && [ -z "${MODEL_PATH:-}" ]; then
  echo "OMEGA_MODE=512_1b requires MODEL_PATH because no full 512_1b eval checkpoint is configured." >&2
  exit 1
fi

MODEL_PATH=${MODEL_PATH:-$DEFAULT_MODEL_PATH}
OUTPUT_PATH=${OUTPUT_PATH:-$DEFAULT_OUTPUT_PATH}
HABITAT_CONFIG_PATH=${HABITAT_CONFIG_PATH:-configs/vln_r2r.yaml}
EVAL_SPLIT=${EVAL_SPLIT:-val_unseen}
LIMIT=${LIMIT:-0}
NUM_FRAMES=${NUM_FRAMES:-24}
NUM_HISTORY=${NUM_HISTORY:-6}
ACTION_CHUNK_NUM=${ACTION_CHUNK_NUM:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
USE_CACHE=${USE_CACHE:-0}
SAVE_VIDEO=${SAVE_VIDEO:-1}
SF_TARGET_DIM=${SF_TARGET_DIM:-2048}
SF_TEACHER_LAYERS=${SF_TEACHER_LAYERS:-23}
SF_ALIGN_LAYERS=${SF_ALIGN_LAYERS:-18}
SF_ALPHA=${SF_ALPHA:-0.1}
SPATIAL_FORCING_TEACHER_PATCH_SIZE=${SPATIAL_FORCING_TEACHER_PATCH_SIZE:-16}

resolve_model_path() {
  local path="$1"
  local latest_checkpoint

  if [ -d "$path" ] && [ ! -f "$path/config.json" ]; then
    latest_checkpoint=$(find "$path" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -n 1)
    if [ -n "$latest_checkpoint" ]; then
      echo "$latest_checkpoint"
      return
    fi
  fi

  echo "$path"
}

copy_training_scripts() {
  local source_dir="$1"
  local target_dir="$2"
  local parent_dir
  local sh_files=()

  mapfile -t sh_files < <(find "$source_dir" -maxdepth 1 -name "*.sh" 2>/dev/null || true)
  if [ ${#sh_files[@]} -eq 0 ]; then
    parent_dir=$(dirname "$source_dir")
    mapfile -t sh_files < <(find "$parent_dir" -maxdepth 1 -name "*.sh" 2>/dev/null || true)
  fi

  for file in "${sh_files[@]}"; do
    cp "$file" "$target_dir"
  done
}

MODEL_PATH=$(resolve_model_path "$MODEL_PATH")
if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "Could not find config.json in MODEL_PATH=$MODEL_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_PATH"
copy_training_scripts "$MODEL_PATH" "$OUTPUT_PATH"
cp "$(realpath "$0")" "$OUTPUT_PATH"

extra_args=()
if [ "$USE_CACHE" = "1" ] || [ "$USE_CACHE" = "true" ] || [ "$USE_CACHE" = "True" ]; then
  extra_args+=(--use_cache)
fi
if [ "$SAVE_VIDEO" = "1" ] || [ "$SAVE_VIDEO" = "true" ] || [ "$SAVE_VIDEO" = "True" ]; then
  extra_args+=(--save_video)
fi
if [ -n "${MIN_PIXELS:-}" ]; then
  extra_args+=(--min_pixels "$MIN_PIXELS")
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun --nproc_per_node="$NPROC_PER_NODE" \
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
    --limit "$LIMIT" \
    --omega_mode "$OMEGA_MODE" \
    --spatial_forcing_image_resolution "$SPATIAL_FORCING_IMAGE_RESOLUTION" \
    --spatial_forcing_teacher_patch_size "$SPATIAL_FORCING_TEACHER_PATCH_SIZE" \
    --sf_geometry_encoder_path "$SF_GEOMETRY_ENCODER_PATH" \
    --sf_target_dim "$SF_TARGET_DIM" \
    --sf_teacher_layers "$SF_TEACHER_LAYERS" \
    --sf_align_layers "$SF_ALIGN_LAYERS" \
    --sf_alpha "$SF_ALPHA" \
    "${extra_args[@]}" \
    2>&1 | tee -a "$OUTPUT_PATH/eval_log.log"
