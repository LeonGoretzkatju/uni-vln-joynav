#!/bin/bash
set -euo pipefail

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate joynav

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-False}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-3}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}

MODEL_TYPE=${MODEL_TYPE:-qwen3_vl_lm_head_sf_dyna}
EVALUATOR_TYPE=${EVALUATOR_TYPE:-qwen3_vl_lm_head_sf_dyna}
MODEL_PATH=${MODEL_PATH:-/mnt/nas5/xiangchen/vlacode/JD-VLN/outputs/qwen3_vl_2b_full_sf/stage1-r2r+rxr-sf_alpha_0.1-layers_24-lr_2e-5-mm_lr_1e-6-batch_size_1-grad_accum_steps_8-epochs_1-max_pixels_451584}
OUTPUT_PATH=${OUTPUT_PATH:-results/r2r/val_unseen/qwen3_sf_dyna}

HABITAT_CONFIG_PATH=${HABITAT_CONFIG_PATH:-configs/vln_r2r.yaml}
EVAL_SPLIT=${EVAL_SPLIT:-val_unseen}
MAX_PIXELS=${MAX_PIXELS:-451584}
ACTION_CHUNK_NUM=${ACTION_CHUNK_NUM:-8}
LIMIT=${LIMIT:-0}
USE_CACHE=${USE_CACHE:-1}
SAVE_VIDEO=${SAVE_VIDEO:-0}

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
  echo "Set MODEL_PATH to a Spatial Forcing checkpoint directory." >&2
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
    --action_chunk_num "$ACTION_CHUNK_NUM" \
    --limit "$LIMIT" \
    "${extra_args[@]}" \
    2>&1 | tee -a "$OUTPUT_PATH/eval_log.log"
