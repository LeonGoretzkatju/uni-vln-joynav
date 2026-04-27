export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))
set -euo pipefail

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate joynav
export TOKENIZERS_PARALLELISM=False

MODEL_TYPE="qwen3_vl_lm_head_dyna"
MODEL_PATH="/mnt/nas5/xiangchen/vlacode/JD-VLN/outputs/qwen3_vl_2b_full/stage1-r2r+rxr-freeze_vision-lr_2e-5-mm_lr_1e-6-batch_size_1-grad_accum_steps_8-epochs_1-max_pixels_200704/checkpoint-27024"

OUTPUT_PATH="results/r2r/val_unseen/qwen3_dyna_final_epoch1"
if [ ! -d "$OUTPUT_PATH" ]; then
  mkdir -p $OUTPUT_PATH
fi

sh_files=($(find "$MODEL_PATH" -maxdepth 1 -name "*.sh"))

if [ ${#sh_files[@]} -gt 0 ]; then
  # 如果有.sh文件，复制到OUTPUT_PATH
  for file in "${sh_files[@]}"; do
    cp "$file" "$OUTPUT_PATH"
  done
else
  # 没有.sh文件，则复制MODEL_PATH上一级目录里的.sh文件到OUTPUT_PATH
  parent_dir=$(dirname "$MODEL_PATH")
  parent_sh_files=($(find "$parent_dir" -maxdepth 1 -name "*.sh"))
  for file in "${parent_sh_files[@]}"; do
    cp "$file" "$OUTPUT_PATH"
  done
fi

script_path="$(realpath "$0")"
cp "$script_path" "$OUTPUT_PATH"

export CUDA_VISIBLE_DEVICES=0,1,2
torchrun --nproc_per_node=3 --master_port=$MASTER_PORT joynav/eval/eval_habitat.py \
    --evaluator_type qwen3_vl_lm_head_dyna \
    --model_path $MODEL_PATH --model_type $MODEL_TYPE --output_path $OUTPUT_PATH \
    --habitat_config_path configs/vln_r2r.yaml \
    --eval_split val_unseen \
    --max_pixels 200704 \
    --action_chunk_num 8 \
    --limit 0 \
    --use_cache \
    2>&1 | tee -a ${OUTPUT_PATH}/eval_log.log


# TODO: Fix bugs for KV Cache used in inference and evaluation.
# torchrun --nproc_per_node=8 --master_port=$MASTER_PORT joynav/eval/eval_habitat.py \
#     --model_path $MODEL_PATH --model_type $MODEL_TYPE --output_path $OUTPUT_PATH \
#     --max_pixels 451584 \
#     --save_video \
#     --use_cache \
#     2>&1 | tee -a ${OUTPUT_PATH}/eval_log.log
