# JD-VLN continuous trajectory VLN memory

This file records the `qwenvln` environment repair and the continuous trajectory
VLN smoke/evaluation run completed for future Codex threads.

## Goal status

Completed in `/mnt/nas5/xiangchen/vlacode/JD-VLN`.

The conda environment `qwenvln` exists at:

```bash
/opt/conda/envs/qwenvln
```

JD-VLN training and evaluation run non-interactively under this env with:

```bash
CUDA_VISIBLE_DEVICES=1,2
```

The user machine has CUDA 12.4 tooling, but PyTorch in `qwenvln` is the
`cu121` build because the observed NVIDIA driver only advertises a CUDA 12.2
runtime. This worked for all requested smoke runs.

## Important `qwenvln` library versions

Verified by imports under `conda run -n qwenvln`:

```text
torch 2.5.1+cu121
torch CUDA runtime 12.1
transformers 5.9.0
qwen-vl-utils 0.0.14
accelerate 1.13.0
peft 0.19.1
diffusers 0.38.0
deepspeed 0.19.0
av 17.0.1
decord 0.6.0
numpy 1.26.4
habitat-lab 0.2.4
habitat-baselines 0.2.4
habitat-sim 0.2.4
```

Habitat-Sim is wired through a local CPython 3.10 build by:

```text
/opt/conda/envs/qwenvln/lib/python3.10/site-packages/habitat_sim_v024_local_build.pth
```

That `.pth` points to:

```text
/mnt/nas5/xiangchen/vlacode/deps/habitat-sim-v0.2.4/build/lib.linux-x86_64-cpython-310
/mnt/nas5/xiangchen/vlacode/deps/habitat-sim-v0.2.4/build/deps/magnum-bindings/src/python
```

Current Habitat-Sim caveat:

```text
habitat_sim.__version__ == 0.2.4
habitat_sim.cuda_enabled == False
habitat_sim.built_with_bullet == False
```

`pip check` caveat:

```text
decord 0.6.0 is not supported on this platform
```

`import decord` succeeds despite that metadata warning.

Qwen3.5-VL notes:

- Local base model used by the scripts: `/mnt/nas5/xiangchen/vlm_base/Qwen3.5-0.8B`.
- `qwen_vl_utils` imports successfully.
- The Qwen3.5 fast linear attention path was not used because optional
  `flash-linear-attention`/`causal-conv1d` style packages were not installed.
  The torch fallback path completed training and evaluation.

## Repo modifications made for this goal

### `scripts/train/train_dit.sh`

Changed this requested entrypoint into a small wrapper around the continuous
Qwen3.5 DiT trajectory script:

```bash
#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${script_dir}/train-qwen3_5-sf-omega-dit-traj.sh" "$@"
```

Reason: the old file was a hardcoded discrete Qwen3/8-GPU script. The user
requested `scripts/train/train_dit.sh` specifically, and this wrapper makes that
path run the continuous trajectory DiT workflow.

### `joynav/model/action_latent/module_utils.py`

Fixed `CategorySpecificLinear.forward` bias broadcasting:

```python
return torch.matmul(x, selected_W) + selected_b.unsqueeze(-2)
```

Reason: the action tensors are `[B, T, D]`; category bias must broadcast across
the trajectory time dimension.

### `joynav/model/action_latent/modeling_action_latent.py`

Stability and initialization fixes:

- Imported `os` and `CategorySpecificLinear`.
- Added `_init_weights` handling for `CategorySpecificLinear`.
- Added `_init_weights` handling for `nn.MultiheadAttention`.
- Clamped sampled flow times from the beta distribution into `[0, 1]`.
- Upcast action flow MSE and mask denominator to fp32 and clamp the denominator
  with `clamp_min(1.0)`.
- Added nonfinite/debug tensor statistics controlled by
  `JOYNAV_ACTION_LATENT_DEBUG=1`, and automatically printed on nonfinite loss.

Reason: DiT training initially completed with `nan` loss. Debug showed finite
Qwen hidden states and finite targets but nonfinite ActionLatent internal
features, so the custom action modules needed robust initialization and loss
guarding.

### `joynav/model/qwen3_5_trajectory_heads.py`

Trajectory model fixes:

- Imported `os`.
- Added `Qwen35OmegaTrajectoryMixin.post_update_model()`.
- If loading from a base checkpoint reports missing `action_head.*` or
  `action_latent.*` keys, the mixin reinitializes those custom modules after
  Hugging Face loading.
- Added DiT action-loss and spatial-forcing debug prints controlled by
  `JOYNAV_TRAJ_DEBUG=1`, and automatically printed on nonfinite loss.

Reason: with base Qwen3.5 checkpoints, the custom trajectory modules are missing
from the checkpoint. Without explicit post-load initialization, DiT parameters
could materialize uninitialized and produce NaNs.

### Existing dirty file note

`tests/test_qwen3_5_support.py` was already modified before this work started.
Do not assume its current diff belongs to the environment/trajectory fix without
checking history or asking the user.

## Continuous trajectory VLN model structure

Main implementation file:

```text
joynav/model/qwen3_5_trajectory_heads.py
```

The continuous trajectory models extend:

```text
JoyNav_Qwen3_5OmegaSpatialForcingForCausalLM
```

with:

```text
Qwen35OmegaTrajectoryMixin
```

Shared behavior:

- The Qwen3.5-VL/Omega spatial-forcing backbone is run with
  `output_hidden_states=True`.
- Final hidden states are read from `outputs.hidden_states[-1]`.
- `select_mask` identifies the action token positions to supervise.
- `continuous_actions` are normalized to `[selected_action_tokens, horizon, 3]`.
- Default trajectory horizon is `8`.
- Trajectory dimension is `3`: ego-centric `x`, `y`, `yaw`.
- `trajectory_stride` in the train scripts defaults to `3`.
- `action_head_loss_weight` defaults to `1.0`.
- `propagate_action_head_grad` defaults to `True`.
- Spatial forcing loss is still added during training when enabled by the base
  Omega spatial-forcing model.

Important shared arguments live in `Qwen35OmegaTrajectoryArguments`:

```text
trajectory_horizon=8
trajectory_dim=3
action_latent_layers=8
action_latent_dim=1536
action_latent_heads=16
action_num_inference_timesteps=4
nextdit_dim=384
nextdit_layers=12
nextdit_heads=6
nextdit_kv_heads=6
nextdit_num_inference_steps=10
nextdit_num_sample_trajs=1
nextdit_guidance_scale=1.0
```

### MLP trajectory model

Class:

```text
JoyNav_Qwen3_5OmegaMLPForCausalLM
```

Head:

```text
TrajectoryMLP
```

Flow:

```text
selected Qwen hidden states
  -> Linear(hidden, hidden)
  -> SiLU
  -> Linear(hidden, hidden)
  -> SiLU
  -> Linear(hidden, horizon * 3)
  -> reshape to [N, horizon, 3]
```

Loss:

```text
trajectory_mse_loss(pred_actions, target_actions)
```

`trajectory_mse_loss` uses standard MSE for `x,y` and wraps yaw error with:

```python
atan2(sin(delta_yaw), cos(delta_yaw))
```

### DiT trajectory model

Class:

```text
JoyNav_Qwen3_5OmegaDiTForCausalLM
```

Head:

```text
TrajectoryDiTHead -> ActionLatent
```

`ActionLatent` implementation:

```text
joynav/model/action_latent/modeling_action_latent.py
```

Core flow:

```text
selected Qwen hidden states
  -> ActionLatent.vl_encoder
  -> PerceiverNet conditioning

target continuous trajectory
  -> beta-sampled flow time t
  -> interpolate noise/actions into noisy trajectory
  -> MultiEmbodimentActionEncoder
  -> position embeddings
  -> PerceiverNet cross-attention with VL states and timestep
  -> CategorySpecificMLP decoder
  -> predicted velocity
```

Training loss:

```text
masked fp32 MSE(predicted_velocity, actions - noise)
```

Inference:

```text
sample random trajectory noise
repeat num_inference_timesteps times
  -> encode current actions and timestep
  -> PerceiverNet
  -> decode velocity
  -> Euler update actions = actions + dt * velocity
return action_pred
```

Default DiT/ActionLatent inference steps: `4`.

### NextDiT trajectory model

Class:

```text
JoyNav_Qwen3_5OmegaNextDiTForCausalLM
```

Head:

```text
TrajectoryNextDiTHead
```

Core modules:

```text
cond_projector: Qwen hidden size -> nextdit_dim
action_encoder: trajectory dim 3 -> nextdit_dim
SinusoidalPositionalEncoding
NextDiTCrossAttn
action_decoder: nextdit_dim -> trajectory dim 3
FlowMatchEulerDiscreteScheduler
```

Default NextDiT shape:

```text
dim=384
layers=12
heads=6
kv_heads=6
num_inference_steps=10
num_sample_trajs=1
guidance_scale=1.0
```

Training loss:

```text
FlowMatchEulerDiscreteScheduler timestep/sigma sample
noisy_trajectory = (1 - sigma) * target_actions + sigma * noise
target velocity = noise - target_actions
fp32 MSE(predicted_velocity, target_velocity)
```

Inference:

```text
classifier-free guidance with zero condition and real condition
FlowMatchEulerDiscreteScheduler denoising for num_inference_steps
average over num_sample_trajs
return [B, horizon, 3]
```

## Training smoke experiments

All three smoke runs used `qwenvln`, GPUs `1,2`, `NPROC_PER_NODE=2`, and
`MAX_STEPS=20`.

### NextDiT 20-step smoke

Command:

```bash
CONDA_ENV=qwenvln CUDA_VISIBLE_DEVICES=1,2 CUDA_GPU_IDS=1,2 NPROC_PER_NODE=2 MAX_STEPS=20 SAVE_STEPS=20 SAVE_STRATEGY=steps GRAD_ACCUM_STEPS=1 RUN_NAME=goal_nextdit_20step OUTPUT_DIR=outputs/goal_qwenvln_20260529/nextdit_20step bash scripts/train/train-qwen3_5-sf-omega-nextdit-traj.sh
```

Result:

```text
global_step: 20
max_steps: 20
train_loss: 1.0848169058561326
```

Artifacts:

```text
outputs/goal_qwenvln_20260529/nextdit_20step
outputs/goal_qwenvln_20260529/nextdit_20step/train.log
outputs/goal_qwenvln_20260529/nextdit_20step/trainer_state.json
outputs/goal_qwenvln_20260529/nextdit_20step/checkpoint-20
```

### DiT 20-step smoke

Command:

```bash
CONDA_ENV=qwenvln CUDA_VISIBLE_DEVICES=1,2 CUDA_GPU_IDS=1,2 NPROC_PER_NODE=2 MAX_STEPS=20 SAVE_STEPS=20 SAVE_STRATEGY=steps GRAD_ACCUM_STEPS=1 NUM_HISTORY=2 RUN_NAME=goal_dit_20step_initfix OUTPUT_DIR=outputs/goal_qwenvln_20260529/dit_20step_initfix bash scripts/train/train_dit.sh
```

Result:

```text
global_step: 20
max_steps: 20
train_loss: 1.227314856648445
```

Artifacts:

```text
outputs/goal_qwenvln_20260529/dit_20step_initfix
outputs/goal_qwenvln_20260529/dit_20step_initfix/train.log
outputs/goal_qwenvln_20260529/dit_20step_initfix/trainer_state.json
outputs/goal_qwenvln_20260529/dit_20step_initfix/checkpoint-20
```

Older failed/debug DiT directories may also exist:

```text
outputs/goal_qwenvln_20260529/dit_20step
outputs/goal_qwenvln_20260529/dit_20step_stable
outputs/goal_qwenvln_20260529/dit_debug_1step
outputs/goal_qwenvln_20260529/dit_debug_1step_initfix
```

Use `dit_20step_initfix` as the successful DiT smoke checkpoint.

### MLP 20-step smoke

Command:

```bash
CONDA_ENV=qwenvln CUDA_VISIBLE_DEVICES=1,2 CUDA_GPU_IDS=1,2 NPROC_PER_NODE=2 MAX_STEPS=20 SAVE_STEPS=20 SAVE_STRATEGY=steps GRAD_ACCUM_STEPS=1 NUM_HISTORY=2 RUN_NAME=goal_mlp_20step OUTPUT_DIR=outputs/goal_qwenvln_20260529/mlp_20step bash scripts/train/train-qwen3_5-sf-omega-mlp-traj.sh
```

Result:

```text
global_step: 20
max_steps: 20
train_loss: 0.13924702629446983
```

Artifacts:

```text
outputs/goal_qwenvln_20260529/mlp_20step
outputs/goal_qwenvln_20260529/mlp_20step/train.log
outputs/goal_qwenvln_20260529/mlp_20step/trainer_state.json
outputs/goal_qwenvln_20260529/mlp_20step/checkpoint-20
```

## Evaluation experiments

The project evaluation scripts do not expose an epoch loop. They expose
`LIMIT`, which limits evaluation episodes. For the requested "10 epochs", the
closest project-specific equivalent used here was:

```bash
LIMIT=10
```

All three evaluations used `qwenvln`, GPUs `1,2`, `NPROC_PER_NODE=2`, and
`NUM_HISTORY=2`.

### NextDiT eval, 10 episodes

Command:

```bash
CONDA_ENV=qwenvln CUDA_VISIBLE_DEVICES=1,2 NPROC_PER_NODE=2 LIMIT=10 NUM_HISTORY=2 MODEL_PATH=outputs/goal_qwenvln_20260529/nextdit_20step OUTPUT_PATH=results/goal_qwenvln_20260529/nextdit_eval_limit10 bash scripts/eval/eval-qwen3_5-sf-omega-nextdit-traj.sh
```

Aggregate result:

```json
{"sucs_all": 0.0, "spls_all": 0.0, "oss_all": 0.0, "nes_all": 9.237483024597168, "ndtws_all": 0.0, "length": 10}
```

Artifacts:

```text
results/goal_qwenvln_20260529/nextdit_eval_limit10/result.json
results/goal_qwenvln_20260529/nextdit_eval_limit10/eval_log.log
```

### DiT eval, 10 episodes

Command:

```bash
CONDA_ENV=qwenvln CUDA_VISIBLE_DEVICES=1,2 NPROC_PER_NODE=2 LIMIT=10 NUM_HISTORY=2 MODEL_PATH=outputs/goal_qwenvln_20260529/dit_20step_initfix OUTPUT_PATH=results/goal_qwenvln_20260529/dit_eval_limit10 bash scripts/eval/eval-qwen3_5-sf-omega-dit-traj.sh
```

Aggregate result:

```json
{"sucs_all": 0.0, "spls_all": 0.0, "oss_all": 0.30000001192092896, "nes_all": 8.017817497253418, "ndtws_all": 0.0, "length": 10}
```

Artifacts:

```text
results/goal_qwenvln_20260529/dit_eval_limit10/result.json
results/goal_qwenvln_20260529/dit_eval_limit10/eval_log.log
```

### MLP eval, 10 episodes

Command:

```bash
CONDA_ENV=qwenvln CUDA_VISIBLE_DEVICES=1,2 NPROC_PER_NODE=2 LIMIT=10 NUM_HISTORY=2 MODEL_PATH=outputs/goal_qwenvln_20260529/mlp_20step OUTPUT_PATH=results/goal_qwenvln_20260529/mlp_eval_limit10 bash scripts/eval/eval-qwen3_5-sf-omega-mlp-traj.sh
```

Aggregate result:

```json
{"sucs_all": 0.0, "spls_all": 0.0, "oss_all": 0.0, "nes_all": 7.567929267883301, "ndtws_all": 0.0, "length": 10}
```

Artifacts:

```text
results/goal_qwenvln_20260529/mlp_eval_limit10/result.json
results/goal_qwenvln_20260529/mlp_eval_limit10/eval_log.log
```

Each `result.json` had 11 JSON objects: 10 per-episode JSON objects plus the
final aggregate JSON object. The files do not end with a newline, so `wc -l`
reports 10 even though `read_text().splitlines()` returns 11 objects.

## Verification performed

Environment import/version check:

```bash
CUDA_VISIBLE_DEVICES=1,2 conda run -n qwenvln python -c "import torch, habitat_sim, habitat, habitat_baselines, transformers, qwen_vl_utils; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Focused regression test:

```bash
CUDA_VISIBLE_DEVICES=1,2 conda run -n qwenvln python -m unittest tests.test_qwen3_5_support.Qwen35SupportTest.test_trajectory_dit_head_uses_action_latent_flow_contract
```

Result:

```text
Ran 1 test in 0.018s
OK
```

Whitespace check:

```bash
git diff --check
```

Result: no output, exit code 0.

Known test caveat:

```bash
python -m unittest tests.test_qwen3_5_support
```

had one existing failure unrelated to the runtime fixes: the test expected
`checkpoint-36000` in `scripts/eval/eval-qwen3_5-sf-omega.sh`, while the actual
script uses `checkpoint-40270`.

## Future operation notes

- Prefer `conda run -n qwenvln ...` for non-interactive commands.
- Use `CUDA_VISIBLE_DEVICES=1,2` and `NPROC_PER_NODE=2` for matching the tested
  setup.
- For DiT trajectory runs, use the successful checkpoint/output directory
  `outputs/goal_qwenvln_20260529/dit_20step_initfix`, not the older NaN/debug
  directories.
- Evaluation logs are verbose because decoded prompt windows are printed at
  many action steps. Summarize aggregate metrics from the final JSON line rather
  than pasting full logs.
- If DiT NaNs recur, rerun with:

```bash
JOYNAV_ACTION_LATENT_DEBUG=1 JOYNAV_TRAJ_DEBUG=1
```

and inspect whether selected Qwen hidden states, target actions, ActionLatent
features, model output, predicted actions, and loss are finite.

## Learnable STOP fix (2026-05-31)

Root problem found: the continuous-trajectory heads had **no STOP mechanism**, so
SR was structurally 0 (the agent passes goals but never stops). Three gaps, all
now fixed end-to-end:

1. Data (`joynav/dataset/vlnn1_annotation_utils.py::build_continuous_actions`):
   previously dropped the final `future_len` frames (`max_start = T-(future_len+1)`),
   so the goal-approach/arrival region was never trained and there was no stop label.
   Now emits chunks for the **full** episode (padding the future with the last pose ->
   zero-motion "arrived" target) and returns `stop_flags[str(step)]` (1.0 when
   `last_index - step <= stop_window`, default `stop_window = future_len`). Return is
   now a 3-tuple `(continuous_actions, stop_flags, action_results)`; `ActionChunkResult`
   gained a `stop` field. `scripts/data/create-vlnn1-annotations.py` writes `stop_flags`,
   adds `--stop-window`, and bumps schema to `vlnn1_ego_xyz_yaw_stop_v1`.
   **Annotations must be regenerated** (old files lack the goal region/labels); the
   dataset has a geometry fallback so old files don't crash. Regenerated InternData-N1
   annotations: 740 chunks (was 636), 15.5% stop-positive, schema `..._stop_v1`. The v1
   files are backed up next to them as `annotations.json.v1.bak` / `annotations_meta.json.v1.bak`.

2. Dataset (`joynav/dataset/continuous_vlnn1_action_dataset.py`): `prepare_sources`
   (interleaved + non-interleaved) attach `stop_targets`; `_get_item` adds them (with
   `_derive_stop_targets` fallback); `ContinuousVLNN1ActionCollator` concatenates
   `stop_targets` in the same selection order as `continuous_actions`.

3. Model (`joynav/model/qwen3_5_trajectory_heads.py`): added a shared
   `stop_head = Linear(hidden, 1)` in `Qwen35OmegaTrajectoryMixin` (built in all three
   MLP/DiT/NextDiT `__init__`s, reinit in `post_update_model`), BCE-with-logits stop
   loss (`stop_head_loss_weight`, `stop_pos_weight` args; bias init -2 for low initial
   stop-prob), and `predict_action` now exposes `outputs.stop_logit`.

4. Eval (`joynav/eval/qwen3_5_omega_trajectory_head_evaluator.py`): learned STOP via
   `_should_stop` (`sigmoid(stop_logit) > stop_threshold`, default 0.5), **receding-horizon**
   execution (`replan_every`, default 2), and each re-plan is a **fresh single-turn prompt**
   (instruction + sparse history + current frame) matching non-interleaved training; the
   evaluator prompt wording was aligned to the dataset prompt. Args wired through the
   train/eval `*-traj.sh` scripts (`STOP_HEAD_LOSS_WEIGHT`/`STOP_POS_WEIGHT`,
   `STOP_THRESHOLD`/`REPLAN_EVERY`).

Verification (env `qwenvln`, GPUs 1,2): 75 unit tests pass except one pre-existing
stale assertion (`checkpoint-36000` vs actual `40270` in the unrelated discrete
`eval-qwen3_5-sf-omega.sh`); MLP 20-step smoke train finite (`train_loss≈1.59`, stop
loss included, no NaN); 10-ep eval runs end-to-end with no crash. At the 20-step
checkpoint STOP does not fire at threshold 0.5 (head is massively undertrained: ~3
stop-positive samples seen) so all episodes hit the 300-step cap (SR 0, same NE≈7.5 as
before) — i.e. no spurious early stop; a low-threshold (0.3) 3-ep run terminates
episodes at step 1, confirming the decode->STOP->terminate wiring. Real SR gains require
a proper (non-smoke) training run.

Note: the older `qwen3_vl` discrete path (`joynav/dataset/vln_action_dataset.py::transform_action_chunk`)
already used a continuous format **with** an `is_stop` dim `[x,y,cosθ,sinθ,is_stop]`; the
VLNN1 3-dim `[x,y,yaw]` reformulation dropped it — that is the regression this fix
reverses, and it is a ready template for Phase 2 (R2R/RxR discrete->continuous co-training,
still pending).
