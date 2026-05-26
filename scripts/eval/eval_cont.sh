export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))
source /mnt/workspace/users/vln/envs/conda3/bin/activate joynav
export TOKENIZERS_PARALLELISM=False

NPROC_PER_NODE=1
num_frames=8
MODEL_TYPE="qwen3_vl_dit"
# MODEL_PATH="<your/path/to/the/model/checkpoint>"
# MODEL_PATH="outputs/T_0203/stage1-r2r+rxr-freeze_vision-lr_2e-5-batch_size_1-grad_accum_steps_8-epochs_1-tune_mm_llm_true"

MODEL_PATH="/mnt/workspace/users/dzheng/code/JoyNav/outputs/T_0203/stage1-r2r+rxr-freeze_vision-lr_2e-5-batch_size_1-grad_accum_steps_8-epochs_1-tune_mm_llm_true"

OUTPUT_PATH="results/T_0203/stage1-r2r+rxr-freeze_vision-lr_2e-5-batch_size_1-grad_accum_steps_8-epochs_1-tune_mm_llm_true"
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



# TODO: Fix bugs for KV Cache used in inference and evaluation.
# export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=${NPROC_PER_NODE} --master_port=$MASTER_PORT joynav/eval/eval_habitat.py \
    --evaluator_type qwen3_vl_cont \
    --predict_type discrete \
    --model_path $MODEL_PATH --model_type $MODEL_TYPE --output_path $OUTPUT_PATH \
    --max_pixels 451584 \
    --limit 100 \
    2>&1 | tee -a ${OUTPUT_PATH}/eval_log.log