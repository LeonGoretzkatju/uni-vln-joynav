#!/bin/bash
# Qwen-VLA Habitat VLN-CE evaluation (arXiv:2605.30280, Section 5.1.3).
# MODEL_PATH must point at a trained qwenvla checkpoint dir (it carries the
# processor, the action-expert architecture, and the baked config.qwenvla_norm).
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

MODEL_TYPE=${MODEL_TYPE:-qwenvla}
EVALUATOR_TYPE=${EVALUATOR_TYPE:-qwenvla}
MODEL_PATH=${MODEL_PATH:-outputs/qwenvla_sft}
OUTPUT_PATH=${OUTPUT_PATH:-results/r2r/val_unseen/qwenvla}
HABITAT_CONFIG_PATH=${HABITAT_CONFIG_PATH:-configs/vln_r2r.yaml}
EVAL_SPLIT=${EVAL_SPLIT:-val_unseen}
LIMIT=${LIMIT:-2}
NUM_HISTORY=${QWENVLA_HISTORY_IMAGES:-8}
ACTION_CHUNK_NUM=${ACTION_CHUNK_NUM:-8}
# Keep eval-time image resolution consistent with training.
MAX_PIXELS=${MAX_PIXELS:-50176}
SAVE_VIDEO=${SAVE_VIDEO:-0}

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
    --num_history "$NUM_HISTORY" \
    --action_chunk_num "$ACTION_CHUNK_NUM" \
    --qwenvla_num_inference_steps "${QWENVLA_NUM_INFERENCE_STEPS:-10}" \
    --control_frequency "${QWENVLA_CONTROL_FREQUENCY:-2.0}" \
    --replan_every "${REPLAN_EVERY:-1}" \
    --stall_distance "${STALL_DISTANCE:-0.05}" \
    --limit "$LIMIT" \
    "${extra_args[@]}" \
    2>&1 | tee -a "$OUTPUT_PATH/eval_log.log"
