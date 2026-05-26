#!/bin/bash
export JOB_NAME=$(hostname  | awk -F'-' '{print $1"-"$2}')
export JOB_LOGDIR=/mnt/workspace/users/heqingrong/job-logs/${JOB_NAME=$}
mkdir -p $JOB_LOGDIR

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NCCL_TIMEOUT=1800

RDMA_DEVICES=$(ls /sys/class/infiniband)
if [ -z "$RDMA_DEVICES" ]; then
  log "ERROR: No active RDMA devices found. Exiting script." >&2
  exit 1
fi

# 设置RDMA设备列表 (逗号分隔)
NCCL_IB_HCA=$(echo "$RDMA_DEVICES" | tr '\n' ',' | sed 's/,$//')
export NCCL_IB_HCA
echo "Detected RDMA devices: $NCCL_IB_HCA"

# 获取GID_INDEX
NCCL_IB_GID_INDEX=""
output=$(show_gids | grep v2)
# 遍历每一行
while IFS= read -r line; do
  ipv4=$(echo "$line" | awk '{print $5}')
  # 检查IPv4地址是否有值（不是空字符串且不是全0地址）
  if [[ -n "$ipv4" && "$ipv4" != "0000:0000:0000:0000:0000:ffff:0000:0000" && "$ipv4" =~ [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+ ]]; then
    NCCL_IB_GID_INDEX=$(echo "$line" | awk '{print $3}')
    # 找到第一个匹配项后立即退出循环
    break
  fi
done <<<"$output"

export NCCL_SOCKET_IFNAME="eth0"

# 处理分布式训练需要使用到的相关环境变量
export NCCL_IB_HCA=${NCCL_IB_HCA}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}
export NCCL_NET_GDR_LEVEL=2
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}

# ==== 多节点配置 ====
# 总节点数（根据实际情况修改）
NUM_NODES=${PET_NNODES}
# 当前节点索引（0-based，每个节点启动时需指定不同值）
NODE_RANK=${PET_NODE_RANK}
# 主节点IP地址（确保所有节点可访问）
MASTER_ADDR=${PET_MASTER_ADDR}  # 替换为实际主节点IP
# 主节点端口（保持与代码中一致）
MASTER_PORT=${PET_MASTER_PORT}


# 每个节点使用的GPU（根据节点实际GPU数量调整）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS_PER_NODE=8  # 每个节点的GPU数量
NUM_GPUS=$NUM_GPUS_PER_NODE  # 补充定义：单节点的GPU数（用于条件判断）
TOTAL_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))  # 总GPU数量（多节点）





source /mnt/workspace/users/vln/envs/conda3/bin/activate joynav
cd /mnt/workspace/users/heqingrong/coding/JoyNav/

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=/mnt/workspace/users/heqingrong/dataset/pretrained_models/Qwen/Qwen3-VL-8B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=2e-5
mm_projector_lr=5e-6
batch_size=4
num_train_epochs=3
grad_accum_steps=1
weight_decay=0.01
warmup_ratio=0.03
max_pixels=451584

# Training entry point
entry_file=joynav/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets="data/trajectory_data/R2R","data/trajectory_data/RxR"

# Output configuration
run_name="joynav_qwen3_vl"
output_dir=./outputs/cluster-nnodes_${NUM_NODES}/${run_name}/stage1-r2r+rxr-freeze_vision-lr_${lr}-mm_lr_${mm_projector_lr}-batch_size_${batch_size}-grad_accum_steps_${grad_accum_steps}-epochs_${num_train_epochs}-max_pixels_${max_pixels}-weight_decay_${weight_decay}-warmup_ratio_${warmup_ratio}

if [ ! -d "$output_dir" ]; then
  mkdir -p $output_dir
fi
script_path="$(realpath "$0")"
cp "$script_path" "$output_dir"

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --video_folder ${datasets} \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --mm_projector_lr ${mm_projector_lr} \
    --vision_tower_lr 1e-6 \
    --weight_decay ${weight_decay} \
    --warmup_ratio ${warmup_ratio} \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 163840 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to tensorboard"

# Launch training
env IS_TORCHRUN=1 torchrun --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
         ${entry_file} ${args} \
         2>&1 | tee -a ${output_dir}/training_log_${NODE_RANK}.log


