# Versatile Qwen-based Continuous Trajectory VLN

**Continuous-trajectory Vision-and-Language Navigation on Qwen3.5-VL with VGGT-Omega
spatial forcing.**

We turns a VLM into a navigation policy that regresses a *continuous* ego-centric
path `[x, y, yaw]` instead of emitting discrete tokens. One backbone — Qwen3.5-VL with
3D spatial grounding — feeds a **pluggable trajectory head**, so you can switch between a
plain MLP, two diffusion/flow-matching DiTs, and an OmniNav-style waypoint head by
swapping a single script.

```
     instruction  +  sparse front history  +  current front frame
                              │
                              ▼
              ┌────────────────────────────────┐
              │           Qwen3.5-VL            │ ◀── VGGT-Omega spatial forcing
              │   (output_hidden_states=True)   │     aux 3D-geometry loss, train-only,
              └───────────────┬────────────────┘     no inference-time cost
                       hidden_states[-1]
                  (action-token positions selected)
                              ▼
              ┌────────────────────────────────┐
              │    pluggable trajectory head    │
              │   MLP │ DiT │ NextDiT │ Omni    │
              └───────────────┬────────────────┘
                              ▼
            continuous trajectory  +  learnable STOP
              ego [x, y, yaw] × horizon
                              ▼
                Habitat discrete actions  (eval, receding-horizon)
```

---

## 1 · Backbone — Qwen3.5-VL + VGGT-Omega spatial forcing

The VLM consumes a normal navigation prompt with `N` sparse front-history frames plus the
current front frame, and runs with `output_hidden_states=True`. The **last hidden state
`hidden_states[-1]` is the conditioning sequence** for every head; only action-token
positions are selected and supervised.

**Spatial forcing** is what makes the backbone spatially aware. During training it aligns
an intermediate Qwen layer (default `18`) to 3D-geometry targets produced by a frozen
**VGGT-Omega** teacher (layer `23`, resolution `256`) as an auxiliary loss (`SF_ALPHA=0.1`).
The policy learns geometric structure **without any extra inference cost**. Two modes:

| `OMEGA_MODE` | behavior |
|---|---|
| `text_align` | interpolates targets; tolerant of small `MAX_PIXELS` |
| `text_align_force_qwen` *(default)* | fixed teacher grid; **keep `MAX_PIXELS=258048`** (smaller crashes on a target/token mismatch) |

A shared `stop_head` (`Linear(hidden, 1)`, BCE) gives the agent a **learnable STOP**, and
evaluation replans on a receding horizon. All heads extend
`JoyNav_Qwen3_5OmegaSpatialForcingForCausalLM` through `Qwen35OmegaTrajectoryMixin`
(`joynav/model/qwen3_5_trajectory_heads.py`). Shared defaults: `trajectory_horizon=8`,
`trajectory_dim=3` (ego `x, y, yaw`).

---

## 2 · Versatile continuous trajectory interface

Four interchangeable heads share the backbone, data path, and STOP head — only the head
module and its objective change. Pick one by choosing its script.

| Head | Mechanism | Training target | Inference |
|------|-----------|-----------------|-----------|
| **MLP** | 3-layer SiLU MLP, direct regression | yaw-wrapped trajectory MSE | 1 forward pass |
| **DiT** | `ActionLatent` flow head, PerceiverNet cross-attn, beta-sampled flow time | masked MSE( v, `actions − noise` ) | 4 Euler steps |
| **NextDiT** | cross-attn DiT + `FlowMatchEulerDiscreteScheduler`, classifier-free guidance | MSE( v, `noise − target` ) | 10 Euler steps |
| **Omni** | OmniNav flow head + arrival action-former | flow MSE (4-D) + arrival BCE | 10 Euler steps |

Registry keys: `qwen3_5_{mlp,dit,nextdit,omni}_head_sf_omega`.

**Flow matching (DiT / NextDiT / Omni).** A noisy trajectory is interpolated between
Gaussian noise and the target; the head predicts the **velocity field** conditioned on the
VLM hidden states and the flow timestep; inference integrates that velocity with a handful
of Euler steps. This is the shared engine behind all three diffusion-style heads — they
differ only in conditioning architecture, scheduler, and step count.

**Omni** additionally mirrors OmniNav end-to-end: a 5-step `[dx, dy, dsin, dcos, arrive]`
delta target, `min_max_split_arrive` normalization over the first four channels, a separate
arrival head for STOP, and a cumulative-sum decode into absolute waypoints returning
`(wp_pred, arrive_pred, sin, cos)`. A LoRA variant is available. See **`omni.md`** for the
full contract.

---

## 3 · Quickstart

Environment: conda `qwenvln` (torch `2.5.1+cu121`, transformers `5.9.0`, habitat `0.2.4`).
Base model: `Qwen3.5-0.8B`.

```bash
# Train — swap mlp / dit / nextdit / omni in the script name
CONDA_ENV=qwenvln CUDA_VISIBLE_DEVICES=1,2 NPROC_PER_NODE=2 \
  OUTPUT_DIR=outputs/run_dit \
  bash scripts/train/train-qwen3_5-sf-omega-dit-traj.sh

# Evaluate — LIMIT = number of Habitat episodes
CONDA_ENV=qwenvln CUDA_VISIBLE_DEVICES=1,2 NPROC_PER_NODE=2 LIMIT=10 \
  MODEL_PATH=outputs/run_dit OUTPUT_PATH=results/run_dit \
  bash scripts/eval/eval-qwen3_5-sf-omega-dit-traj.sh
```

`scripts/train/train_dit.sh` is a convenience wrapper for the DiT script. The Omni LoRA
path (`*-omni-lora.sh`) requires merging the adapter into the base model before eval.

**Common knobs** (env vars, same across train scripts):

| var | default | meaning |
|---|---|---|
| `NUM_HISTORY` | `8` | sparse front-history frames |
| `MAX_PIXELS` | `258048` | image budget — do not lower under `text_align_force_qwen` |
| `OMEGA_MODE` | `text_align_force_qwen` | spatial-forcing mode |
| `BATCH_SIZE` / `GRAD_ACCUM_STEPS` | `1` / `2` | per-GPU batch / accumulation |
| `LR` | `2e-5` | learning rate |
| `DEEPSPEED_CONFIG` | `./scripts/zero2.json` | ZeRO stage (use `zero3.json` for ≤24 GB at high history) |

**Memory.** Use **bf16** for full fine-tuning — fp16 underflows the DeepSpeed loss scaler.
History=12 fits full FT on 2×24 GB (~15 GB/GPU); history=20 needs ≥40 GB, ZeRO-3/offload,
or a frozen vision tower (`TUNE_MM_VISION=False`).

---

## Repo map

| path | contents |
|---|---|
| `joynav/model/qwen3_5_trajectory_heads.py` | backbone mixin + MLP/DiT/NextDiT/Omni heads |
| `joynav/model/action_latent/` | DiT `ActionLatent` flow-matching modules |
| `joynav/dataset/` | continuous VLN datasets (InternData-N1, R2R, RxR) |
| `joynav/eval/` | Habitat trajectory evaluators |
| `scripts/{train,eval,data}/` | entrypoints |
| `summary.md` · `omni.md` | detailed design & operation notes |
