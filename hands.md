# hands.md — Running Omni Mode End-to-End on a Cluster

This is the operational runbook for the `omni` mode (OmniNav VLM + flow-matching waypoint
action head on Qwen3.5-VL + VGGT-Omega spatial forcing). It covers the full pipeline:

```
generate annotation JSON  →  train  →  evaluate in Habitat
   (Step 1)                   (Step 2)        (Step 3)
```

Design background lives in `omni.md`. This file is the "what do I run" guide.

---

## 0. The three registered components

| Role | ID | Where |
|---|---|---|
| Model | `qwen3_5_omni_head_sf_omega` | `joynav/model/qwen3_5_trajectory_heads.py` |
| Dataset | `omni_waypoint_sf_omega` | `joynav/dataset/omni_waypoint_dataset.py` |
| Evaluator | `qwen3_5_omni_head_sf_omega` | `joynav/eval/qwen3_5_omega_trajectory_head_evaluator.py` |

Entry points:
- `scripts/data/generate_omni_waypoint_json.py` — annotation generator
- `scripts/train/train-qwen3_5-sf-omega-omni.sh` — training (auto-runs the generator)
- `scripts/eval/eval-qwen3_5-sf-omega-omni.sh` — Habitat evaluation

---

## 1. Prerequisites

### 1.1 Conda environment
The env is named `qwenvln` (Python 3.10, torch 2.5.1+cu121). Every script does
`conda activate ${CONDA_ENV:-qwenvln}` itself; override with `CONDA_ENV=<name>` if your
cluster uses a different name. Verify once:

```bash
conda run -n qwenvln python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

### 1.2 Assets to stage on the cluster (adjust all paths to your mounts)

| Asset | Default path (edit per cluster) | Used by |
|---|---|---|
| Base VLM | `/mnt/nas5/xiangchen/vlm_base/Qwen3.5-0.8B` | train (`MODEL_PATH`) |
| VGGT-Omega SF teacher (~5.4 GB) | `/mnt/nas5/.../VGGT-Omega/vggt_omega_1b_256_text.pt` | train + eval (`SF_GEOMETRY_ENCODER_PATH`) |
| Train data roots (3) | `…/InternData-N1/vln_n1/traj_data`, `…/R2R`, `…/RxR` | generate (`DATASETS`) |
| Habitat scenes (MP3D) | `/mnt/nas5/xiangchen/data/scene_datasets/mp3d` | eval (`configs/vln_r2r.yaml: scenes_dir`) |
| R2R VLN-CE episodes | `…/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz` | eval (`configs/vln_r2r.yaml: data_path`) |

> Train and eval use **different** data. Training reads RGB-frame datasets (InternData-N1 /
> R2R / RxR with `annotations.json`); evaluation reads the Habitat MP3D scenes + VLN-CE
> episode files referenced by `configs/vln_r2r.yaml` (or `configs/vln_rxr*.yaml`).

### 1.3 Expected train-data layout (the generator checks every image exists)

- **InternData-N1 (VLNN1)** root — `annotations.json` whose records have
  `id`, `path`, `chunk_id`, `continuous_actions` (`{step: [[0,0,0],[x,y,yaw],…]}`),
  `stop_flags`, `instructions`. Frames at:
  `<root>/<path>/videos/<chunk_id>/observation.images.rgb/episode_{id:06d}_{frame:03d}.jpg`
- **R2R / RxR** roots — `annotations.json` whose records have `video`, `actions`
  (discrete `0=stop,1=fwd,2=left,3=right`), `instructions`. Frames at:
  `<root>/<video>/rgb/*.jpg`

The generator auto-detects source per root: `continuous_actions` → VLNN1, `actions` → R2R/RxR.

### 1.4 Pre-flight sanity check (1 minute, no GPU)

```bash
cd <repo>
conda run -n qwenvln python -m py_compile \
  joynav/model/qwen3_5_trajectory_heads.py \
  joynav/dataset/omni_waypoint_dataset.py \
  joynav/eval/qwen3_5_omega_trajectory_head_evaluator.py \
  scripts/data/generate_omni_waypoint_json.py
conda run -n qwenvln python -c "import joynav.model, joynav.dataset; from joynav.utils.registry import get_component; \
print(get_component('model','qwen3_5_omni_head_sf_omega').__name__)"
```

---

## 2. Step 1 — Generate the annotation JSON

You can run the generator **standalone** (recommended for a real run, so you build the
dataset once and reuse it) or let the train script run it automatically.

### 2.1 Standalone (recommended for cluster)

```bash
conda run -n qwenvln python scripts/data/generate_omni_waypoint_json.py \
  --video-folder "/path/InternData-N1/vln_n1/traj_data,/path/R2R,/path/RxR" \
  --output /path/data/omni_waypoint_train.json \
  --max-samples 200000 \
  --num-history-images 20 \
  --waypoint-number 5 \
  --trajectory-stride 3 \
  --step-scale 0.3
```

Generator flags:

| Flag | Default | Meaning |
|---|---|---|
| `--video-folder` | (required) | Comma-separated roots. Append `%N` to subsample a root, e.g. `…/R2R%10` = 10 %. |
| `--annotations-file` | `annotations.json` | Per-root annotation filename. |
| `--output` | (required) | Output JSON path. A sidecar `*.norm.json` is also written. |
| `--max-samples` | `300` | **Raise this for real runs** (300 is a smoke size). Split evenly across roots. |
| `--num-history-images` | `20` | History frames per sample. **Must equal train & eval `OMNI_HISTORY_IMAGES`.** |
| `--waypoint-number` | `5` | Future waypoints (OmniNav = 5). |
| `--trajectory-stride` | `3` | VLNN1 frame stride. |
| `--r2r-step-stride` | `4` | R2R/RxR step stride. |
| `--r2r-forward-step` / `--r2r-turn-angle` | `0.25` / `15` | R2R action→metric geometry (match the Habitat sim config). |
| `--step-scale` | `0.3` | Waypoints stored ÷ this; eval multiplies back by it. **Keep 0.3.** |
| `--seed` | `42` | Sampling seed. |

### 2.2 What a record looks like (OmniNav contract)

Each record: `{"messages":[user, assistant], "images":[ <num_history> history + 3 current ]}`.
With `--num-history-images 20` that is **23 images** and **23 `<image>` tokens** in the prompt
(history block + `leftside/frontside/rightside`), ending in `<|NAV|>\nOutput the waypoint`.
The assistant message carries `gt_waypoints` (`[5,3]`, absolute & ÷step_scale),
`gt_heading_angles` (`[5]`), `arrive`, `arrive_list` (`[5]`), `step_scale`, and a `norm`
(5×5 over `[dx, dy, dsin, dcos, arrive]`). The model uses the first 4 channels for the flow
head and predicts arrival with a separate head.

> **Normalization is automatic.** The train script bakes the dataset's `norm` into the
> checkpoint `config.omni_norm`, and eval reads it back. You never pass a norm file to eval.

Validate a freshly generated file:

```bash
conda run -n qwenvln python -c "
import json,numpy as np; d=json.load(open('/path/data/omni_waypoint_train.json'))
print('records',len(d),'images/rec',{len(r['images']) for r in d})
a=d[0]['messages'][1]; print('wp',np.array(a['gt_waypoints']).shape,'norm',np.array(a['norm']['min']).shape)
print('prompt tail:',repr(d[0]['messages'][0]['content'][-40:]))"
```

---

## 3. Step 2 — Train

### 3.1 Canonical single-node, multi-GPU run

```bash
cd <repo>
CONDA_ENV=qwenvln \
MODEL_PATH=/path/Qwen3.5-0.8B \
SF_GEOMETRY_ENCODER_PATH=/path/VGGT-Omega/vggt_omega_1b_256_text.pt \
DATASETS=/path/InternData-N1/vln_n1/traj_data,/path/R2R,/path/RxR \
OUTPUT_DIR=/path/outputs/omni_run \
OMNI_JSON=/path/data/omni_waypoint_train.json \
REGENERATE_OMNI_JSON=0 \
OMNI_MAX_SAMPLES=200000 \
OMNI_HISTORY_IMAGES=20 \
NPROC_PER_NODE=8 CUDA_GPU_IDS=0,1,2,3,4,5,6,7 \
BATCH_SIZE=2 NUM_TRAIN_EPOCHS=5 LR=1e-5 \
bash scripts/train/train-qwen3_5-sf-omega-omni.sh
```

- Set `REGENERATE_OMNI_JSON=0` and point `OMNI_JSON` at the file from Step 1 to reuse it.
  (Default `REGENERATE_OMNI_JSON=1` regenerates into `${OUTPUT_DIR}/omni_waypoint_train.json`
  every run — fine for the first run, wasteful afterwards.)
- The output checkpoint (with `config.omni_norm` + omni architecture) lands in `OUTPUT_DIR`.
- The script `cp`s itself into `OUTPUT_DIR` and tees logs to `${OUTPUT_DIR}/train.log`.

### 3.2 Training env vars

| Var | Default | Notes |
|---|---|---|
| `MODEL_PATH` | Qwen3.5-0.8B | Base VLM. |
| `DATASETS` | 3 roots | Passed to the generator. |
| `OMNI_JSON` / `OMNI_MAX_SAMPLES` / `REGENERATE_OMNI_JSON` | …/omni_waypoint_train.json / 300 / 1 | See Step 1. |
| `OMNI_HISTORY_IMAGES` | 20 | History frames. **Must match Step 1 and eval.** |
| `NPROC_PER_NODE` / `CUDA_GPU_IDS` | 1 / 0 | GPUs per node / device list. |
| `DEEPSPEED_CONFIG` | `./scripts/zero2.json` | Use `./scripts/zero3.json` for large backbones / tight VRAM. |
| `BATCH_SIZE` / `GRAD_ACCUM_STEPS` | 2 / 1 | Effective batch = `BATCH_SIZE × GRAD_ACCUM_STEPS × num_GPUs`. |
| `NUM_TRAIN_EPOCHS` / `MAX_STEPS` | 5 / -1 | `MAX_STEPS=-1` → epochs drive length. Set `MAX_STEPS` for a quick run. |
| `LR` / `MM_PROJECTOR_LR` | 1e-5 / 1e-6 | OmniNav LR is 1e-5. Consider scaling LR up if effective batch ≫ 2. |
| `MAX_PIXELS` | **258048** | **Do not lower with `OMEGA_MODE=text_align_force_qwen`** — see §5. |
| `MODEL_MAX_LENGTH` | 163840 | Upper bound; not a truncation in practice. |
| `BF16` | true | bf16 on Ampere+/H100. Set `BF16=false` for fp16 on older GPUs. |
| `TUNE_MM_VISION` / `TUNE_MM_MLP` / `TUNE_MM_LLM` | True / True / True | Full FT (OmniNav `freeze_*=false`). |
| `OMEGA_MODE` | `text_align_force_qwen` | Spatial-forcing mode. `text_align` tolerates small `MAX_PIXELS`. |
| `SF_ALPHA` / `SF_ALIGN_LAYERS` / `SF_TEACHER_LAYERS` | 0.1 / 18 / 23 | SF loss weight + alignment/teacher layers. |
| `SPATIAL_FORCING_IMAGE_RESOLUTION` | 256 | SF teacher input res (pairs with `MAX_PIXELS=258048`). |
| `SAVE_STEPS` / `SAVE_STRATEGY` | 1000 / steps | Checkpoint cadence (`save_total_limit=3`). |
| `USE_LORA` | False | LoRA path (prefer `zero2`). |

The omni head hyperparameters are passed explicitly and match OmniNav defaults:
`--omni_action_dim 4 --omni_num_timestep_buckets 100 --omni_query_action_layers 1
--omni_use_arrive_list True --omni_flow_layers 16 --omni_flow_heads 32 --omni_flow_dropout 0.2
--omni_norm_method min_max_split_arrive --omni_num_inference_timesteps 10
--omni_noise_beta_alpha 1.5 --omni_noise_beta_beta 1.0 --omni_noise_s 0.999 --omni_step_scale 0.3`.
Override any with the matching `OMNI_*` env var if you must — but train and eval must agree.

### 3.3 Multi-node

The script's `torchrun` line is single-node (`--nproc_per_node --master_addr --master_port`).
For multiple nodes, launch the same command on each node with the rendezvous flags added, e.g.
edit the `torchrun` invocation to:

```bash
torchrun --nnodes=$NNODES --node_rank=$NODE_RANK \
         --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
         joynav/train/train_qwen.py ${args}
```

and export `NNODES`, `NODE_RANK`, a shared `MASTER_ADDR`/`MASTER_PORT` per node (or follow the
existing `scripts/cluster-qwen3.sh` / `joybuilder` pattern used elsewhere in this repo).

### 3.4 Healthy-run signals
`train.log` shows decreasing/finite `loss`, `learning_rate: 1e-05`, periodic checkpoint saves.
`${OUTPUT_DIR}/config.json` must contain `omni_norm` and `omni_norm_method=min_max_split_arrive`.
(`grad_norm: 0` in the log is a benign DeepSpeed-ZeRO logging artifact, not a stuck model.)

---

## 4. Step 3 — Evaluate in Habitat

### 4.1 Canonical run (R2R val_unseen, multi-GPU sharded)

```bash
cd <repo>
CONDA_ENV=qwenvln \
MODEL_PATH=/path/outputs/omni_run \
SF_GEOMETRY_ENCODER_PATH=/path/VGGT-Omega/vggt_omega_1b_256_text.pt \
HABITAT_CONFIG_PATH=configs/vln_r2r.yaml \
EVAL_SPLIT=val_unseen \
OMNI_HISTORY_IMAGES=20 \
MAX_PIXELS=258048 \
LIMIT=-1 \
NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUTPUT_PATH=/path/results/omni_run/r2r_val_unseen \
bash scripts/eval/eval-qwen3_5-sf-omega-omni.sh
```

- `MODEL_PATH` = the trained checkpoint dir (it holds the processor, `config.omni_norm`, and
  the omni architecture, so the evaluator rebuilds the exact trained model).
- **`OMNI_HISTORY_IMAGES` must equal the training value** (not enforced by the checkpoint).
- `LIMIT=-1` (or `0`) evaluates the whole split; `LIMIT=N` shuffles (seed 42) and takes N
  episodes for a quick check. Eval shards episodes across `NPROC_PER_NODE` GPUs automatically.
- RxR: use `HABITAT_CONFIG_PATH=configs/vln_rxr-512s-hfov_90.yaml` (and the matching split). A
  higher-res R2R variant `configs/vln_r2r-512s-hfov_90.yaml` is also available.

### 4.2 Eval env vars

| Var | Default | Notes |
|---|---|---|
| `MODEL_PATH` | outputs/qwen3_5_0_8b_sf_omega_omni | Trained checkpoint. |
| `HABITAT_CONFIG_PATH` | `configs/vln_r2r.yaml` | R2R or RxR config. |
| `EVAL_SPLIT` | `val_unseen` | `val_unseen` / `val_seen` / `train`. |
| `LIMIT` | 2 | `-1`/`0` = full split. |
| `OMNI_HISTORY_IMAGES` / `MAX_PIXELS` | 20 / 258048 | **Match training.** |
| `ACTION_CHUNK_NUM` | 5 | Discrete actions emitted per plan. |
| `STOP_THRESHOLD` | 0.5 | STOP when **all** arrival logits > this (OmniNav `cnt==5`). |
| `REPLAN_EVERY` | 1 | Receding-horizon: actions executed before re-planning. |
| `OMNI_NUM_INFERENCE_TIMESTEPS` | 10 | Euler steps for flow sampling. |
| `OMEGA_MODE` | `text_align_force_qwen` | Keep consistent with training. |
| `NPROC_PER_NODE` / `CUDA_VISIBLE_DEVICES` | 1 / 0 | Parallel eval across GPUs. |
| `SAVE_VIDEO` | 0 | `1` saves top-down nav videos. |

### 4.3 Habitat rendering
`configs/vln_r2r.yaml` ships with `habitat_sim_v0.gpu_device_id: -1` (CPU rendering — slow but
portable, used on G2 hosts). On a cluster with EGL/GPU rendering, set that field to a valid GPU
id for a large speedup. For multi-GPU eval with GPU rendering, give each rank its own render GPU
(or keep `-1` for simplicity and correctness).

### 4.4 Reading results
Results stream to `${OUTPUT_PATH}/result.json` (one JSON line per episode + a final summary
line). Metrics: `success`, `spl`, `os` (oracle success), `ne` (navigation error, meters),
`ndtw`. The run is resumable — re-running skips episodes already in `result.json`.
Set `JOYNAV_OMNI_EVAL_DEBUG=1` to print per-step `wp_pred` / `arrive` / `action_seq`.

---

## 5. Critical consistency rules (read this)

1. **`MAX_PIXELS` ↔ SF resolution.** With `OMEGA_MODE=text_align_force_qwen` and
   `SPATIAL_FORCING_IMAGE_RESOLUTION=256`, keep `MAX_PIXELS=258048`. Lowering it downsizes the
   Qwen image below the teacher grid and **crashes** with
   `Spatial Forcing target/image-token mismatch`. (Only `OMEGA_MODE=text_align` interpolates the
   targets and tolerates small `MAX_PIXELS`.)
2. **`OMNI_HISTORY_IMAGES` must be identical** across generate → train → eval. It sets the image
   count (`history + 3`) and the prompt's `<image>` count; a mismatch shifts the input
   distribution the head was trained on.
3. **`--step-scale` / `omni_step_scale` = 0.3 everywhere.** The generator stores waypoints ÷0.3
   and eval multiplies by 0.3.
4. **Don't change omni head hyperparameters between train and eval.** Architecture
   (`omni_flow_layers/heads`, `omni_query_action_layers`, `omni_action_dim`,
   `omni_use_arrive_list`) is loaded from the checkpoint config; eval only overrides
   inference-time knobs you pass on its CLI.

---

## 6. Memory tuning (when OOM)

Lower these in order, **keeping rule §5.1 in mind**:
- `BATCH_SIZE=1`, raise `GRAD_ACCUM_STEPS` to preserve effective batch.
- `DEEPSPEED_CONFIG=./scripts/zero3.json` (or `zero3_offload.json` as a last resort).
- `OMNI_HISTORY_IMAGES` down (e.g. 12/8) — **regenerate the JSON and eval with the same value.**
- Freeze parts: `TUNE_MM_VISION=False` (and/or `TUNE_MM_LLM=False`) to shrink optimizer state.
- Switch to `OMEGA_MODE=text_align` to also allow a smaller `MAX_PIXELS` (changes the SF path).
- `gradient_checkpointing` is already on.

---

## 7. Quick smoke (validate the wiring before a big run)

This is the exact recipe that passed on a 24 GB Turing test GPU (fp16):

```bash
# tiny data
conda run -n qwenvln python scripts/data/generate_omni_waypoint_json.py \
  --video-folder "$DATASETS" --output outputs/omni_smoke/omni_waypoint_train.json \
  --max-samples 6 --num-history-images 4 --waypoint-number 5 --step-scale 0.3

# 10-step train (frozen LLM/vision to keep it light)
OMNI_HISTORY_IMAGES=4 BATCH_SIZE=1 MAX_STEPS=10 SAVE_STEPS=10 \
TUNE_MM_VISION=False TUNE_MM_LLM=False BF16=false \
REGENERATE_OMNI_JSON=0 CUDA_GPU_IDS=2 OUTPUT_DIR=outputs/omni_smoke \
bash scripts/train/train-qwen3_5-sf-omega-omni.sh

# 2-episode eval
MODEL_PATH=outputs/omni_smoke LIMIT=2 OMNI_HISTORY_IMAGES=4 \
CUDA_VISIBLE_DEVICES=2 OUTPUT_PATH=outputs/omni_smoke/eval2 \
bash scripts/eval/eval-qwen3_5-sf-omega-omni.sh
```

Smoke metrics are meaningless (toy checkpoint); you're checking that every stage runs and
`outputs/omni_smoke/eval2/result.json` appears. For the real cluster run use full
`MAX_PIXELS=258048`, `OMNI_HISTORY_IMAGES=20`, `BATCH_SIZE=2`, all tune flags `True`, bf16.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Spatial Forcing target/image-token mismatch: N vs M` | `MAX_PIXELS` too small for `text_align_force_qwen`. Use `258048` (rule §5.1) or switch to `OMEGA_MODE=text_align`. |
| Eval waypoints are garbage / NaN | `config.omni_norm` missing from the checkpoint (training didn't bake it), or `OMNI_NORM_METHOD` differs from training. Confirm `omni_norm` in `${MODEL_PATH}/config.json`. |
| Agent never stops / wanders | Expected from an undertrained checkpoint (arrive head init bias −2.0). Needs real training, not a fix. |
| bf16 errors / very slow on older GPUs | `BF16=false` (fp16). bf16 wants Ampere+. |
| Generator writes 0 records | Image files missing at the expected layout (§1.3), or wrong `--annotations-file`. Check the `generate_omni_json.log`. |
| Habitat can't find scene / episodes | Fix `scenes_dir` / `data_path` in `configs/vln_r2r.yaml` to your mounts; ensure the split's `.json.gz` exists. |
| Free wedged GPUs after a crash | `nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9` |
