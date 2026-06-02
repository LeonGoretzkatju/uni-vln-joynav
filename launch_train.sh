#!/bin/bash
CONDA_ENV=qwenvln \
MODEL_PATH=/data1/xiangchen/Qwen/Qwen3.5-0.8B \
SF_GEOMETRY_ENCODER_PATH=/data0/lujia/models/VGGT-Omega/vggt_omega_1b_256_text.pt \
DATASETS=/data0/lujia/VLNData/InternData-N1/vln_n1/traj_data,/data1/xiangchen/traj_data/R2R,/data1/xiangchen/traj_data/RxR \
OUTPUT_DIR=/data0/xiangchen/outputs/omni_run \
OMNI_JSON=/data1/xiangchen/traj_data/omni_waypoint_train.json \
REGENERATE_OMNI_JSON=0 \
OMNI_MAX_SAMPLES=200000 \
OMNI_HISTORY_IMAGES=20 \
NPROC_PER_NODE=2 CUDA_GPU_IDS=1,3 \
BATCH_SIZE=3 NUM_TRAIN_EPOCHS=5 LR=2e-5 \
bash scripts/train/train-qwen3_5-sf-omega-omni.sh