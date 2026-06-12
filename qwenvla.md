# Qwen-VLA Mode Design

`qwenvla` is the JD-VLN reproduction of **Qwen-VLA** (arXiv:2605.30280,
https://qwen.ai/blog?id=qwenvla, https://github.com/QwenLM/Qwen-VLA): a Qwen3.5-VL
backbone with a **1.15B single-stream DiT flow-matching action decoder**, trained with the
paper's progressive recipe — Stage I **T2A** (text-to-action DiT pretraining), Stage II
**CPT** (continued pretraining), Stage III **SFT** (supervised fine-tuning). The paper's
Stage IV (RL) is out of scope here.

## Registered Components

| Role | ID | Where |
|---|---|---|
| Model | `qwenvla` | `joynav/model/qwenvla.py` (`JoyNav_QwenVLAForCausalLM`) |
| Dataset | `qwenvla` | `joynav/dataset/qwenvla_dataset.py` (`QwenVLADataset`) |
| Evaluator | `qwenvla` | `joynav/eval/qwenvla_evaluator.py` (`QwenVLAEvaluator`) |

Entry points:

```text
scripts/data/generate_qwenvla_json.py
scripts/train/train-qwenvla.sh      (STAGE=t2a|cpt|sft)
scripts/eval/eval-qwenvla.sh
```

## Architecture (paper Section 2.2)

`JoyNav_QwenVLAForCausalLM` extends `JoyNav_Qwen3_5ForCausalLM` (plain Qwen3.5 wrapper —
no spatial forcing; the paper has none) with `QwenVLAActionExpert`:

- **Single-stream joint attention**: the (projected) VLM final hidden states and the noisy
  action chunk are concatenated into ONE sequence and processed by 16 DiT blocks with
  joint self-attention, **AdaLN-Zero timestep conditioning**, and **multi-section RoPE
  aligned with the backbone** (sections scaled from the backbone's `mrope_section`;
  VLM-token positions come from `Qwen3_5Model.get_rope_index`, action tokens continue
  sequentially).
- **Exact 1.15B geometry**, solved from the paper's parameter table and verified in code:

  | Component | Paper | Ours |
  |---|---|---|
  | 16 DiT blocks (hidden 1536, attn 4h², AdaLN-Zero 6h², SwiGLU MLP 10240) | 70.8M each / 1.13B | 70.79M / 1.133B |
  | Action projection MLPs (K→h, h→K, shared zero-padding design) | 4.9M | 4.82M |
  | VLM→DiT linear (2560→1536 at the reference 4B backbone) | 3.9M | 3.93M |
  | Timestep embedding (sinusoidal 256 → MLP h) | 2.8M | 2.76M |
  | Output AdaLN modulation (Linear(h, 2h)) | 4.7M | 4.72M |
  | **Total** | **≈1.15B** | **1.149B** |

- **Backbone**: the paper uses Qwen3.5-4B. This environment defaults to
  **Qwen3.5-0.8B** (only 24 GB GPUs available — explicitly sanctioned substitution); only
  the VLM→DiT linear input dim follows the actual backbone.

## Unified Action Representation (Section 2.4)

- Targets are `Y ∈ R^{H×K}` (fixed horizon H, fixed channel dim K=32) with a validity
  mask `M ∈ {0,1}^{H×K}`; a control mode occupies the leading `c ≤ K` channels
  (zero-padding projection — the paper's preferred design, Section 5.2.2/Table 10).
- **Navigation (VLN)**: `(Δx, Δy, Δθ)` per waypoint (`c=3`), **8 waypoints per chunk**
  (Section 4.1). The generator converts the ego trajectories (VLNN1 continuous actions;
  R2R/RxR discrete actions integrated by `discrete_actions_to_ego_trajectory`) into
  per-waypoint relative displacements.
- **Per-dataset quantile normalization (eq. 5)**: `ã = 2(a − q01)/(q99 − q01) − 1`,
  clipped to [−1,1]. Stats are computed over the generated nav corpus, stored per record,
  and baked into the checkpoint as `config.qwenvla_norm` by `train_qwen.py` — eval reads
  them back automatically.

## Prompts (Sections 2.3, 3.2.1)

Embodiment-aware prompt conditioning, instantiated for VLN:

```
The robot is a wheeled navigation robot with mobile base. The control convention is
planar waypoints (delta x, delta y, delta yaw) in meters and radians. The control
frequency is 2.0 Hz. Please predict the next 8 control actions to execute the following
task: {instruction}
# Historical visual observations: <|ego_start|><image><|ego_end|>...
# Current visual observation: <|ego_start|><image><|ego_end|>
```

Every image is wrapped in view boundary tokens `<|ego_start|>…<|ego_end|>` (registered as
additional special tokens). Each record also carries a text-only `t2a_content` prompt
(the first paragraph) used by Stage I.

## Flow Matching (Sections 2.5, 5.2.1)

- Interpolant `Y_τ = (1−τ)Y0 + τY1`, `Y1 ~ N(0,I)`; the expert predicts the velocity
  `(Y1 − Y0)`.
- **Two-level masked averaging loss** (eqs. 1–2): per-channel masked MSE over time, then a
  uniform mean over active channels — padding never influences the gradient and every
  control dimension contributes equally.
- **Timestep distribution**: Sigmoid-Normal (logit-normal, peaks at intermediate noise) at
  **T2A**; Beta toward the clean end (`τ = (1 − Beta(1.5, 1.0)) · 0.999`) at **CPT/SFT** —
  the paper's best ablation combination (71.1%).
- **Inference**: Euler integration from τ=1 to τ=0, 10 steps by default
  (`qwenvla_num_inference_steps`).

## Three-Stage Training (Section 3.1)

`scripts/train/train-qwenvla.sh` selects the stage with `STAGE=`:

| Stage | Frozen | Images | p(τ) | λ_vl | λ_act |
|---|---|---|---|---|---|
| `t2a` | VLM (vision+mlp+llm) — only the DiT trains | **withheld** (text + embodiment prompt only) | Sigmoid-Normal | 0.0 | 1.0 |
| `cpt` | nothing (both modules unfrozen) | yes | Beta | 0.1 | 1.0 |
| `sft` | nothing | yes | Beta | 0.1 (paper Section 4.1) | 1.0 |

Stages chain via `MODEL_PATH` (each starts from the previous stage's `OUTPUT_DIR`).
`--action_expert_lr` gives the DiT its own LR group (paper: separate group-wise
cosine schedules for backbone and action decoder); both default to `LR`.

## Evaluation (Section 5.1.3)

`QwenVLAEvaluator` mirrors the paper's VLN-CE protocol: predicted `(Δx, Δy, Δθ)` deltas
are cumsum-integrated into ego waypoints and executed via a **sliding-window waypoint
action** — the deltas convert to discrete VLN-CE actions, only `REPLAN_EVERY` execute
before re-planning. STOP fires when the predicted trajectory stalls (total displacement
below `STALL_DISTANCE`; trained delta targets go to zero at the goal). The prompt and the
checkpoint's `qwenvla_norm`/architecture come from training.

## Smoke Recipe (2× TITAN RTX 24 GB, GPUs 2,3)

```bash
# data (shared by all stages)
conda run -n qwenvln python scripts/data/generate_qwenvla_json.py \
  --video-folder "$DATASETS" --output outputs/qwenvla_smoke/qwenvla_train.json \
  --max-samples 64 --num-history-images 4 --action-horizon 8

# Stage I  T2A   (20 steps)
STAGE=t2a QWENVLA_JSON=outputs/qwenvla_smoke/qwenvla_train.json \
QWENVLA_HISTORY_IMAGES=4 MAX_PIXELS=12544 BATCH_SIZE=1 MAX_STEPS=20 SAVE_STEPS=20 \
NPROC_PER_NODE=2 CUDA_GPU_IDS=2,3 OUTPUT_DIR=outputs/qwenvla_smoke/t2a \
bash scripts/train/train-qwenvla.sh

# Stage II CPT   (20 steps, from the T2A checkpoint)
STAGE=cpt MODEL_PATH=outputs/qwenvla_smoke/t2a OUTPUT_DIR=outputs/qwenvla_smoke/cpt \
QWENVLA_JSON=... QWENVLA_HISTORY_IMAGES=4 MAX_PIXELS=12544 BATCH_SIZE=1 \
MAX_STEPS=20 SAVE_STEPS=20 NPROC_PER_NODE=2 CUDA_GPU_IDS=2,3 \
bash scripts/train/train-qwenvla.sh

# Stage III SFT  (20 steps, from the CPT checkpoint)
STAGE=sft MODEL_PATH=outputs/qwenvla_smoke/cpt OUTPUT_DIR=outputs/qwenvla_smoke/sft ...

# 3-episode eval of the SFT checkpoint
MODEL_PATH=outputs/qwenvla_smoke/sft LIMIT=3 QWENVLA_HISTORY_IMAGES=4 MAX_PIXELS=12544 \
NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=2,3 OUTPUT_PATH=outputs/qwenvla_smoke/sft/eval3 \
bash scripts/eval/eval-qwenvla.sh
```

Memory notes (24 GB Turing): T2A trains only the 1.15B expert → `zero2` is enough.
CPT/SFT train backbone+expert (~2B trainable, fp32 master under DeepSpeed) →
`zero2_offload` is the script default for those stages. `ATTN_IMPLEMENTATION=sdpa` is the
default (no flash-attn on Turing); bf16 by default.

## Code Review 2026-06-12 (correctness pass)

A full train+eval review found and fixed:

1. **`set_model` lm_head freeze no-op** (`joynav/train/train_qwen.py`): `model.lm_head.requires_grad = ...`
   sets an attribute on the Module, not its parameters (pre-existing repo bug). Now iterates
   `lm_head.parameters()`, so T2A freezes the VLM completely (verified: after `set_model`
   with t2a flags, exactly the 160 `action_expert.*` tensors are trainable, 0 backbone).
2. **Prompt centralization**: the embodiment prompt + vision block were duplicated in the
   generator and evaluator. Now both import `build_qwenvla_prompt` from
   `joynav/dataset/qwenvla_dataset.py` (single source of truth); byte-parity against the
   generated training JSON verified for all records.
3. **Loud one-time RoPE fallback warning**: if `get_rope_index` ever fails, the expert falls
   back to sequential positions — now warns loudly instead of silently, since a one-sided
   fallback would shift the train/eval input distribution. Verified the fallback does NOT
   fire in either training or evaluation.
4. **All-masked-labels NaN guard**: CE over zero unmasked targets is 0/0; `use_vl_loss` now
   also requires `(labels != -100).any()`.
5. **T2A fallback** for records without `t2a_content` (`strip_qwenvla_vision_block`).
6. **`--control_frequency` passthrough** in `eval-qwenvla.sh` (prompt-parity knob).
7. Earlier fix (worth remembering): never register non-persistent buffers in custom modules —
   transformers 5.x meta-device loading materializes them as uninitialized memory
   (this produced an all-NaN run); the expert RoPE computes `inv_freq` on the fly.

Verification performed:
- `tests/test_qwenvla.py` (16 tests, all pass): paper parameter table pinned (1.149B total,
  70.79M/block, 3.9/4.9/2.8/4.7M components), two-level masked loss hand-checked against
  eqs. 1–2, Euler integration recovers Y0 given the true velocity, tau distributions
  (Beta→clean end, Sigmoid-Normal→intermediate), quantile-norm round trip, delta↔waypoint
  round trip + yaw wrap, prompt parity/T2A strip, and the train script's stage contracts.
- Checkpoint-diff freezing audit: T2A backbone identical to base (the only diffs are 18
  `linear_attn.norm.weight` tensors changed by exactly one bf16 ULP — load-time fp32→bf16
  rounding, no training involved, affects every mode equally); CPT/SFT move both backbone
  (357/474 tensors; the rest are vision-tower params whose lr-1e-6 updates round away in
  bf16) and all 160 expert tensors; expert finite in all three checkpoints.
- Existing repo suites: 75/77 pass; the 2 failures are pre-existing stale assertions about
  scripts this work never touched (`checkpoint-36000` vs `40270`, documented in summary.md,
  and a `NUM_HISTORY:-2` expectation against the nextdit script).
- Fresh 2-step T2A after the fixes: finite losses, backbone frozen; fresh 1-episode eval on
  the SFT checkpoint: runs end-to-end with the centralized prompt, no RoPE fallback.

## Compatibility Choices (deviations and why)

- **Qwen3.5-0.8B instead of 4B** — 24 GB GPU constraint (explicitly allowed). The expert
  keeps the full reference 1.15B geometry.
- **T2A data** — the paper pretrains T2A on language-action manipulation corpora with
  full-sequence prediction; JD-VLN only has navigation data, so T2A here runs on the same
  nav chunks with images withheld. The stage mechanics (frozen VLM, no images,
  Sigmoid-Normal p(τ), DiT-only updates) are faithful.
- **CPT data** — the paper's CPT mixes 8 heterogeneous data families (74% manipulation);
  here it is the navigation corpus (+ the tiny LM loss on assistant tokens as the L_vl
  placeholder; real VL data can be added by extending the generator).
- **Stop behavior** — the paper does not describe an explicit VLN stop head; stop emerges
  from the trajectory itself (stall detection on near-zero predicted displacement).
- **RL (Stage IV)** is not implemented.
