export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))
source /mnt/workspace/users/vln/envs/conda3/bin/activate joynav
export TOKENIZERS_PARALLELISM=False

MODEL_TYPE="joynav_qwen2_5_vl"
MODEL_PATH="<your/path/to/the/model/checkpoint>"

OUTPUT_PATH="results/r2r/val_unseen/debug"
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

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# torchrun --nproc_per_node=8 --master_port=$MASTER_PORT joynav/eval/eval_habitat.py \
#     --model_path $MODEL_PATH --model_type $MODEL_TYPE --output_path $OUTPUT_PATH \
#     --max_pixels 451584 \
#     --save_video \
#     2>&1 | tee -a ${OUTPUT_PATH}/eval_log.log


# TODO: Fix bugs for KV Cache used in inference and evaluation.
torchrun --nproc_per_node=8 --master_port=$MASTER_PORT joynav/eval/eval_habitat.py \
    --model_path $MODEL_PATH --model_type $MODEL_TYPE --output_path $OUTPUT_PATH \
    --max_pixels 451584 \
    --save_video \
    --use_cache \
    2>&1 | tee -a ${OUTPUT_PATH}/eval_log.log