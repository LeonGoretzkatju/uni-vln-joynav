# JoyNav

## 数据集
```bash
ln -s /mnt/workspace/users/heqingrong/dataset/VLN_dataset_processed/StreamVLN/data data
```
```
data
├── datasets
│   ├── envdrop    # R2R_VLNCE_v1-3_preprocessed/envdrop
│   ├── r2r        # R2R_VLNCE_v1
│   ├── rxr        # RxR_VLNCE_v0
│   └── scalevln
├── scene_datasets
│   ├── hm3d
│   └── mp3d
├── trajectory_data
│   ├── R2R
│   │   ├── annotations.json
│   │   └── images
│   └── RxR
│       ├── annotations.json
│       └── images
└── trajectory_data_generated
    ├── envdrop
    │   └── 20251021_2016
    └── scalevln
        └── 20251021_1440
```

## 训练
### 单机多卡（r2r+rxr）
#### qwen3_vl
```bash
bash scripts/single-qwen3.sh
```

#### qwen2_5_vl
```bash
bash scripts/single-qwen2_5.sh
```

### 多机多卡（r2r+rxr）：joybuilder起训练任务
#### qwen3_vl
```bash
bash <absolute/path/to>/scripts/cluster-qwen3.sh
```

#### qwen2_5_vl
```bash
bash <absolute/path/to>/scripts/cluster-qwen2_5.sh
```

## 仿真评测
G2暂不支持gpu渲染，需要设置 configs/vln_r2r.yaml 的 gpu_device_id: -1

#### qwen3_vl
```bash
bash scripts/eval-qwen3.sh
```

#### qwen2_5_vl
```bash
bash scripts/eval-qwen2_5.sh
```

## TODO
- [ ] Fix bugs for KV Cache used in inference and evaluation.

## Bugs
```bash 
# kill all gpu-pids
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 
```