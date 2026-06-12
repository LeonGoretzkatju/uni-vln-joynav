#!/bin/bash
# Qwen-VLA training (arXiv:2605.30280): Qwen3.5 backbone + 1.15B DiT
# flow-matching action expert, with the paper's three-stage recipe:
#
#   STAGE=t2a  Stage I  — text-to-action DiT pretraining: VLM frozen, DiT only,
#                          images withheld, Sigmoid-Normal p(tau), no VL loss.
#   STAGE=cpt  Stage II — continued pretraining: both modules unfrozen,
#                          multimodal data, Beta p(tau).
#   STAGE=sft  Stage III— supervised fine-tuning: joint, Beta p(tau),
#                          lambda_vl=0.1, lambda_act=1.0 (paper Section 4.1).
#
# Chain the stages by pointing MODEL_PATH at the previous stage's OUTPUT_DIR:
#   STAGE=t2a OUTPUT_DIR=outputs/qwenvla/t2a bash scripts/train/train-qwenvla.sh
#   STAGE=cpt MODEL_PATH=outputs/qwenvla/t2a OUTPUT_DIR=outputs/qwenvla/cpt ...
#   STAGE=sft MODEL_PATH=outputs/qwenvla/cpt OUTPUT_DIR=outputs/qwenvla/sft ...
#
# Note: the paper builds on Qwen3.5-4B; this environment defaults to
# Qwen3.5-0.8B because only 24 GB GPUs are available. The action expert keeps
# the full reference 1.15B geometry (16 blocks, hidden 1536, SwiGLU 10240).
set -euo pipefail

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate ${CONDA_ENV:-qwenvln}

export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Turing GPUs (TITAN RTX) cannot run flash-attention-2; sdpa is the safe default.
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

STAGE=${STAGE:-sft}
case "${STAGE}" in
    t2a)
        # Stage I: freeze the VLM, train only the DiT, withhold images.
        tune_mm_vision=${TUNE_MM_VISION:-False}
        tune_mm_mlp=${TUNE_MM_MLP:-False}
        tune_mm_llm=${TUNE_MM_LLM:-False}
        time_dist=${QWENVLA_TIME_DIST:-sigmoid_normal}
        lambda_vl=${QWENVLA_LAMBDA_VL:-0.0}
        default_deepspeed=./scripts/zero2.json
        ;;
    cpt)
        # Stage II: unfreeze both modules, ground the action prior in vision.
        tune_mm_vision=${TUNE_MM_VISION:-True}
        tune_mm_mlp=${TUNE_MM_MLP:-True}
        tune_mm_llm=${TUNE_MM_LLM:-True}
        time_dist=${QWENVLA_TIME_DIST:-beta}
        lambda_vl=${QWENVLA_LAMBDA_VL:-0.1}
        default_deepspeed=./scripts/zero2_offload.json
        ;;
    sft)
        # Stage III: joint fine-tuning, lambda_vl=0.1 / lambda_act=1.0.
        tune_mm_vision=${TUNE_MM_VISION:-True}
        tune_mm_mlp=${TUNE_MM_MLP:-True}
        tune_mm_llm=${TUNE_MM_LLM:-True}
        time_dist=${QWENVLA_TIME_DIST:-beta}
        lambda_vl=${QWENVLA_LAMBDA_VL:-0.1}
        default_deepspeed=./scripts/zero2_offload.json
        ;;
    *)
        echo "Unknown STAGE='${STAGE}' (expected t2a|cpt|sft)" >&2
        exit 1
        ;;
esac

deepspeed=${DEEPSPEED_CONFIG:-${default_deepspeed}}
llm=${MODEL_PATH:-/mnt/nas5/xiangchen/vlm_base/Qwen3.5-0.8B}
model_type=qwenvla
dataset_type=qwenvla

datasets=${DATASETS:-/mnt/nas5/xiangchen/VLNData/InternData-N1/vln_n1/traj_data,/mnt/nas5/xiangchen/VLNData/R2R,/mnt/nas5/xiangchen/VLNData/RxR}
run_name=${RUN_NAME:-qwenvla_${STAGE}}
output_dir=${OUTPUT_DIR:-./outputs/${run_name}}
qwenvla_json=${QWENVLA_JSON:-${output_dir}/qwenvla_train.json}

lr=${LR:-1e-5}
action_expert_lr=${ACTION_EXPERT_LR:-${lr}}
mm_projector_lr=${MM_PROJECTOR_LR:-1e-6}
batch_size=${BATCH_SIZE:-2}
num_train_epochs=${NUM_TRAIN_EPOCHS:-5}
grad_accum_steps=${GRAD_ACCUM_STEPS:-1}
max_pixels=${MAX_PIXELS:-50176}
model_max_length=${MODEL_MAX_LENGTH:-32768}
max_steps=${MAX_STEPS:--1}
save_steps=${SAVE_STEPS:-1000}
save_strategy=${SAVE_STRATEGY:-steps}
gpu_ids=${CUDA_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}

# bf16 by default; fp16 full FT is numerically unstable here (loss-scaler underflow).
precision_flag="--bf16"
case "${BF16:-true}" in
    false|False|0|fp16) precision_flag="--fp16" ;;
esac

num_history=${QWENVLA_HISTORY_IMAGES:-8}
action_horizon=${QWENVLA_ACTION_HORIZON:-8}
action_channels=${QWENVLA_ACTION_CHANNELS:-32}

mkdir -p "${output_dir}"
cp "$(realpath "$0")" "$output_dir"

# Generate the dataset when missing, or force with REGENERATE_QWENVLA_JSON=1.
if [ ! -s "${qwenvla_json}" ] || [ "${REGENERATE_QWENVLA_JSON:-0}" = "1" ]; then
    python scripts/data/generate_qwenvla_json.py \
        --video-folder "${datasets}" \
        --output "${qwenvla_json}" \
        --max-samples "${QWENVLA_MAX_SAMPLES:-300}" \
        --num-history-images "${num_history}" \
        --action-horizon "${action_horizon}" \
        --action-channels "${action_channels}" \
        --trajectory-stride "${TRAJECTORY_STRIDE:-3}" \
        2>&1 | tee -a "${output_dir}/generate_qwenvla_json.log"
fi

args="
    --deepspeed ${deepspeed} \
    --model_type ${model_type} \
    --dataset_type ${dataset_type} \
    --model_name_or_path ${llm} \
    --qwenvla_json_path ${qwenvla_json} \
    --qwenvla_stage ${STAGE} \
    --qwenvla_action_horizon ${action_horizon} \
    --qwenvla_action_channels ${action_channels} \
    --qwenvla_dit_hidden ${QWENVLA_DIT_HIDDEN:-1536} \
    --qwenvla_dit_layers ${QWENVLA_DIT_LAYERS:-16} \
    --qwenvla_dit_heads ${QWENVLA_DIT_HEADS:-16} \
    --qwenvla_dit_mlp_dim ${QWENVLA_DIT_MLP_DIM:-10240} \
    --qwenvla_dit_dropout ${QWENVLA_DIT_DROPOUT:-0.0} \
    --qwenvla_num_inference_steps ${QWENVLA_NUM_INFERENCE_STEPS:-10} \
    --qwenvla_time_dist ${time_dist} \
    --qwenvla_beta_alpha ${QWENVLA_BETA_ALPHA:-1.5} \
    --qwenvla_beta_beta ${QWENVLA_BETA_BETA:-1.0} \
    --qwenvla_noise_s ${QWENVLA_NOISE_S:-0.999} \
    --qwenvla_lambda_act ${QWENVLA_LAMBDA_ACT:-1.0} \
    --qwenvla_lambda_vl ${lambda_vl} \
    --data_flatten False \
    --tune_mm_vision ${tune_mm_vision} \
    --tune_mm_mlp ${tune_mm_mlp} \
    --tune_mm_llm ${tune_mm_llm} \
    --model_load_dtype ${MODEL_LOAD_DTYPE:-auto} \
    --use_lora ${USE_LORA:-False} \
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
    --action_expert_lr ${action_expert_lr} \
    --mm_projector_lr ${mm_projector_lr} \
    --vision_tower_lr ${VISION_TOWER_LR:-1e-6} \
    --weight_decay 0.01 \
    --warmup_ratio ${WARMUP_RATIO:-0.0} \
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
