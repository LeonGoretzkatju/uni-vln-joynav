# JD-VLN Project Memory

Last updated: 2026-05-28 UTC.

This file is intended as carry-over context for a future session. It summarizes the JD-VLN repository structure, the Qwen3.5-VL + Spatial Forcing + VGGT-Omega workflow, the relevant inheritance relationships, model inputs/outputs, and the recent evaluation prompt fix.

## Repository Structure

Top-level layout:

- `README.md`: basic project setup, dataset symlink expectations, training/eval examples.
- `configs/`: Habitat VLN evaluation configs. Main file in current work: `configs/vln_r2r.yaml`.
- `habitat-lab/`: vendored Habitat-Lab/Habitat-Baselines code used for simulator/evaluation.
- `joynav/`: main project package.
- `scripts/`: training and eval shell entrypoints plus DeepSpeed configs.
- `tests/`: unit/regression tests.
- `vggt_omega/`: local VGGT-Omega implementation used as frozen geometry teacher.
- `outputs/`: training outputs and smoke eval outputs.
- `results/`: persistent eval results.

Important `joynav/` subpackages:

- `joynav/model/`: model wrappers registered by string name.
- `joynav/dataset/`: supervised and VLN datasets, collators, image/video preprocessing.
- `joynav/eval/`: Habitat evaluators and eval entrypoint.
- `joynav/train/`: training entrypoint and trainer helpers.
- `joynav/utils/registry.py`: central registry used by scripts to map names to model/dataset/evaluator classes.

## Registry And Entrypoints

The code uses registry names rather than direct imports in scripts.

Model registration is in `joynav/model/__init__.py`.

Key model names:

- `qwen3_5_lm_head`: `JoyNav_Qwen3_5ForCausalLM`
- `qwen3_5_lm_head_sf`: `JoyNav_Qwen3_5SpatialForcingForCausalLM`
- `qwen3_5_lm_head_sf_omega`: `JoyNav_Qwen3_5OmegaSpatialForcingForCausalLM`

Dataset registration is in `joynav/dataset/__init__.py`.

Key dataset names:

- `vln_action`: normal VLN action dataset.
- `vln_action_interleave`: same class as `vln_action`.
- `vln_action_sf`: Spatial Forcing dataset using non-Omega geometry teacher.
- `vln_action_sf_omega`: `VLNActionOmegaSpatialForcingDataset`, used for Qwen3.5 + VGGT-Omega training.

Evaluator registration is in `joynav/eval/eval_habitat.py`.

Key evaluator names:

- `qwen3_vl_lm_head`: shared Qwen-VL LM-head evaluator.
- `qwen3_vl_lm_head_sf`: Qwen3-VL Spatial Forcing evaluator.
- `qwen3_5_lm_head_sf_omega`: Qwen3.5-VL VGGT-Omega Spatial Forcing evaluator.

Training entrypoint:

- `joynav/train/train_qwen.py`
- Shell scripts call this through `torchrun`.
- It parses component args, loads model by `model_type`, builds dataset by `dataset_type`, creates `Trainer`, trains, saves processor and model.

Evaluation entrypoint:

- `joynav/eval/eval_habitat.py`
- Shell scripts call this through `torchrun`.
- It parses evaluator args, loads model/processor, builds evaluator by `evaluator_type`, runs Habitat eval, gathers distributed results, appends aggregate row to `result.json`.

## Qwen3.5-VL Model Inheritance

Current Qwen3.5-Omega class chain:

```text
BaseModel
  + transformers.Qwen3_5ForConditionalGeneration
    -> JoyNav_Qwen3_5ForCausalLM
       -> JoyNav_Qwen3_5SpatialForcingForCausalLM
          -> JoyNav_Qwen3_5OmegaSpatialForcingForCausalLM
```

Files:

- `joynav/model/qwen3_5_lm_head.py`
- `joynav/model/qwen3_5_lm_head_sf.py`
- `joynav/model/qwen3_5_lm_head_sf_omega.py`
- `joynav/model/spatial_forcing.py`
- `joynav/model/geometry_encoder/vggt_omega_encoder.py`

`JoyNav_Qwen3_5ForCausalLM`:

- Wraps Transformers `Qwen3_5ForConditionalGeneration`.
- Fixes Qwen3.5 position handling:
  - If `mm_token_type_ids` exists, clears `position_ids`.
  - Otherwise expands Qwen multimodal position ids through `_expand_qwen3_5_position_ids`.
- Overrides `prepare_inputs_for_generation` so `mm_token_type_ids` is preserved during generation when pixel values are present.

`JoyNav_Qwen3_5SpatialForcingForCausalLM`:

- Adds Spatial Forcing training loss.
- Adds `SpatialForcingProjector`, a two-layer MLP that maps Qwen visual hidden states to geometry-teacher feature dimension.
- During training only, if `labels`, `input_ids`, `image_grid_thw`, and `sf_image_tensors` exist:
  - Forces `output_hidden_states=True`.
  - Builds frozen geometry targets.
  - Selects hidden states at `sf_align_layers`.
  - Extracts hidden states at Qwen image-token positions.
  - Projects them through `spatial_forcing_projector`.
  - Adds `sf_alpha * cosine_alignment_loss(projected_tokens, target_tokens)` to LM loss.
- Exposes `outputs.spatial_forcing_loss`.

`JoyNav_Qwen3_5OmegaSpatialForcingForCausalLM`:

- Replaces the default Depth Anything teacher with VGGT-Omega.
- Uses `VGGTOmegaEncoder`.
- Key args:
  - `omega_mode`: `512_1b`, `text_align`, or `text_align_force_qwen`.
  - `sf_geometry_encoder_path`
  - `sf_target_dim=2048`
  - `sf_teacher_layers=23`
  - `sf_align_layers=18`
  - `sf_alpha=0.1`
  - `sf_add_pos_embed=False` in Omega training/eval scripts.
- In `text_align_force_qwen`, Omega features are returned directly after reshape. In the other modes, Omega features are resized to Qwen image-token grid with `resize_spatial_features_to_grid`.

## VGGT-Omega Teacher

Core file: `joynav/model/geometry_encoder/vggt_omega_encoder.py`.

Local implementation: `vggt_omega/`.

Default checkpoints:

- `512_1b`: `/mnt/nas5/xiangchen/vlacode/vggt-omega/facebook/VGGT-Omega/vggt_omega_1b_512.pt`
- `text_align` and `text_align_force_qwen`: `/mnt/nas5/xiangchen/vlacode/vggt-omega/facebook/VGGT-Omega/vggt_omega_1b_256_text.pt`

Modes:

- `512_1b`
  - Resolution 512.
  - Uses aggregator checkpoint without text-alignment head.
- `text_align`
  - Resolution 256.
  - Uses text-aligned Omega checkpoint.
  - Qwen image preprocessing remains normal; Omega targets are resized to Qwen visual-token grid.
- `text_align_force_qwen`
  - Resolution 256.
  - Uses text-aligned Omega checkpoint.
  - Forces Qwen images to Omega-compatible direct alignment shape so Qwen image-token count matches Omega patch-token count.
  - Important helper: `prepare_qwen_images_for_omega_direct` in `joynav/dataset/vln_action_omega_spatial_forcing_dataset.py`.

`VGGTOmegaEncoder.encode(images)`:

- Input shape can be `[frames, C, H, W]` or `[batch, frames, C, H, W]`.
- Runs `vggt_omega.models.aggregator.Aggregator`.
- Selects cached patch tokens from specified teacher layer(s).
- Returns flattened per-image patch features with final dim `2 * embed_dim`, currently 2048.

## Training Workflow For Qwen3.5-VL + VGGT-Omega

Common training scripts:

- `scripts/single-qwen3_5-0_8b-sf-omega.sh`
- `scripts/single-qwen3_5-0_8b-sf-omega-force-qwen.sh`

The checkpoint path that was explicitly referenced by the user for matching hyperparameters:

```text
/mnt/nas5/xiangchen/vlacode/JD-VLN/outputs/qwen3_5_0_8b_full_sf_omega_bfloat/stage1-r2r+rxr-omega_sf_alpha_0.1-layers_18-teacher_23-lr_5e-6-mm_lr_1e-6-batch_size_1-grad_accum_steps_1-epochs_2-max_pixels_200704/single-qwen3_5-0_8b-sf-omega.sh
```

Training defaults from that script:

- `model_type=qwen3_5_lm_head_sf_omega`
- `dataset_type=vln_action_sf_omega`
- `omega_mode=${OMEGA_MODE:-text_align}`
- `NUM_FRAMES=24`
- `NUM_HISTORY=6`
- `ACTION_CHUNK_NUM=4`
- `SF_ALPHA=0.1`
- `SF_ALIGN_LAYERS=18`
- `SF_TEACHER_LAYERS=23`
- `sf_target_dim=2048`
- `sf_add_pos_embed=False`
- `spatial_forcing_teacher_patch_size=16`
- `spatial_forcing_image_resolution=256` for text-align modes.
- `MAX_PIXELS=200704` for `text_align`, `258048` for `text_align_force_qwen`.

Data flow in training:

1. `train_qwen.py` parses args and loads the model through registry.
2. `train_qwen.py` maps Qwen3.5 model type to `qwen3vl` for dataset preprocessing.
3. `VLNActionOmegaSpatialForcingDataset` loads VLN navigation episodes from R2R/RxR folders.
4. It builds one sample per 24-step training window.
5. It creates interleaved conversation turns:
   - First user turn:
     - instruction
     - optional sparse history images if the window starts after step 0
     - current observation image
   - Assistant turn:
     - `<|action|>` plus one action chunk of length up to 4.
   - Later user turns:
     - exactly one current observation image.
   - Later assistant turns:
     - `<|action|>` plus next action chunk.
6. `lazy_supervised_dataset._build_messages` converts `<image>` placeholders into Qwen message content blocks.
7. `processor.apply_chat_template(..., tokenize=True, return_dict=True, return_tensors="pt")` creates Qwen model inputs.
8. Labels are masked so only assistant spans contribute to LM loss.
9. `VLNActionOmegaSpatialForcingDataset` also loads `sf_image_tensors` through VGGT-Omega preprocessing.
10. In `text_align_force_qwen`, the dataset also replaces Qwen input images with direct Omega-aligned resized PIL images before normal Qwen tokenization.
11. `VLNActionOmegaSpatialForcingCollator` forwards `sf_image_tensors` in the batch.
12. Model forward computes normal LM loss plus optional Spatial Forcing loss.

Training model inputs:

- `input_ids`
- `labels`
- `attention_mask`
- `position_ids`
- `pixel_values`
- `image_grid_thw`
- optional `mm_token_type_ids`
- `sf_image_tensors` for Spatial Forcing

Training model outputs:

- Standard Qwen CausalLM output, including `loss` and `logits`.
- If Spatial Forcing is active, `loss` includes `sf_alpha * spatial_forcing_loss`.
- `outputs.spatial_forcing_loss` is attached for debugging/inspection.

## Evaluation Workflow For Qwen3.5-VL + VGGT-Omega

Main eval script:

- `scripts/eval/eval-qwen3_5-sf-omega.sh`

Current intended defaults:

- `CONDA_ENV=qwenvln`
- `EVALUATOR_TYPE=qwen3_5_lm_head_sf_omega`
- `MODEL_TYPE=qwen3_5_lm_head_sf_omega`
- `OMEGA_MODE=text_align_force_qwen`
- `USE_CACHE=0`
- `NUM_FRAMES=24`
- `NUM_HISTORY=6`
- `ACTION_CHUNK_NUM=4`
- `TEXT_ALIGN_CHECKPOINT=.../checkpoint-78000`
- `FORCE_QWEN_CHECKPOINT=.../checkpoint-36000`
- CUDA devices are runtime-overridable. The user explicitly said GPU IDs can be changed freely based on current situation.

Eval entrypoint:

```bash
CONDA_ENV=qwenvln CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 OMEGA_MODE=text_align_force_qwen LIMIT=10 USE_CACHE=0 OUTPUT_PATH=outputs/eval_smoke/... bash scripts/eval/eval-qwen3_5-sf-omega.sh
```

`eval_habitat.py` flow:

1. Parses evaluator args through `HfArgumentParser`.
2. Loads processor from `args.model_path`.
3. Updates image processor pixel limits.
4. Loads model class from registry.
5. For `qwen3_5_lm_head_sf_omega`, `build_model_config` loads checkpoint config and only applies explicit CLI overrides for Omega fields.
6. Creates `Qwen3_5OmegaSpatialForcingEvaluator`.
7. Runs distributed Habitat evaluation.
8. Appends one JSON row per episode and one aggregate JSON row to `result.json`.

Evaluator inheritance:

```text
BaseEvaluator
  -> QwenVLLMHeadEvaluator
     -> Qwen3VLLMHeadEvaluator   # compatibility name for older Qwen3 scripts
     -> Qwen3_5OmegaSpatialForcingEvaluator
```

Evaluation model input:

- Built by processor from Qwen messages:
  - `input_ids`
  - `attention_mask`
  - `pixel_values`
  - `image_grid_thw`
  - possibly `mm_token_type_ids`
- No `labels`.
- No `sf_image_tensors`; the model is used as normal action-generating LM at inference.

Evaluation model output:

- `model.generate(...)` returns token sequences.
- Generated tokens are decoded with `skip_special_tokens=False`.
- Terminators such as `<|im_end|>` and EOS are trimmed.
- `<|action|>` must remain in text for future dialogue and action parsing.
- `parse_actions` extracts STOP/MOVE/TURN action tokens from generated text.
- At most `ACTION_CHUNK_NUM=4` actions are executed before the next model generation.

Habitat action mapping:

- STOP -> 0
- MOVE_FORWARD -> 1
- TURN_LEFT -> 2
- TURN_RIGHT -> 3
- The prompt and generated text use action glyphs internally. If editing exact prompts, inspect the existing source instead of rewriting from memory.

## Interleaved Dialogue State Machine In Evaluation

Core file:

- `joynav/eval/qwen3_vl_lm_head_evaluator.py`

Qwen3.5-Omega-specific override:

- `joynav/eval/qwen3_5_lm_head_sf_omega_evaluator.py`

No-cache evaluation maintains a `source` dict:

```python
source = {
    "image": [],
    "conversations": [],
}
```

At each model generation:

1. Append current RGB PIL image to `rgb_list`.
2. If `previous_assistant_text is None`, build a new first user turn:
   - instruction prompt
   - if `step_id != 0`, sparse history observations:
     - `num_history=6`
     - history ids sampled from prior environment steps
   - one current image prompt
3. Else append:
   - previous assistant output as a `gpt` turn
   - a new `human` turn with exactly one current image token
4. `prepare_processor_source` optionally transforms images for `text_align_force_qwen`.
5. `_build_messages` converts conversations to Qwen message format.
6. Prompt mode is chosen.
7. Processor inputs are built.
8. `model.generate` produces assistant action text.
9. The generated assistant text is saved as `previous_assistant_text` for the next turn.
10. Parsed actions are executed in Habitat.
11. When `step_id % num_frames == 0`, reset `source` and `previous_assistant_text` so the next generation starts a new 24-step dialogue window with sparse history.

## Recent Fix: Qwen3.5-Omega Early STOP After Interleaved Dialogue

Observed problem:

- After earlier interleaved-dialogue evaluation changes, Qwen3.5-Omega often generated STOP very early.
- Example investigation target:
  - `results/r2r/val_unseen/qwen3_5_sf_omega_text_align/result.json`
- Previous non-interleaved eval continued for more steps.

Root cause:

- Evaluation used `processor.apply_chat_template(..., add_generation_prompt=True)` at every generation turn.
- For Qwen3.5, that inserts the official final-generation assistant prompt, including an empty `<think>` block.
- Training interleaved samples do not put that final-round generation prompt before every intermediate assistant action turn.
- Training-style intermediate assistant turns look like:
  - previous turns end with `<|im_end|>`
  - new generation starts at plain `<|im_start|>assistant\n`
- Only the final generation context should use the official Qwen final prompt behavior.

Fix implemented:

- Added prompt modes in `qwen3_vl_lm_head_evaluator.py`:
  - `PROMPT_MODE_QWEN_FINAL`
  - `PROMPT_MODE_TRAINING_INTERMEDIATE`
- Default evaluator keeps `PROMPT_MODE_QWEN_FINAL`.
- Qwen3.5-Omega evaluator overrides `get_generation_prompt_mode(step_id)`:
  - if `step_id % num_frames >= num_frames - action_chunk_num`, use `PROMPT_MODE_QWEN_FINAL`
  - else use `PROMPT_MODE_TRAINING_INTERMEDIATE`
- Intermediate prompt rendering:
  - `apply_chat_template(messages, tokenize=False, add_generation_prompt=False)`
  - append plain `<|im_start|>assistant\n`
  - call processor directly with text/images/videos.
- Final prompt rendering:
  - use official `processor.apply_chat_template(..., add_generation_prompt=True)`.

Why this matches Qwen3.5:

- It still uses the official Qwen `apply_chat_template` format for final generation.
- It avoids inserting the Qwen final-round empty thinking block before intermediate action turns.
- Qwen docs say the thinking block is a final-round generation construct, not something to put before every assistant turn in a prefilled multi-turn dialogue.

Related docs:

- Transformers Qwen3.5 docs: `https://huggingface.co/docs/transformers/model_doc/qwen3_5`
- Qwen concepts/docs: `https://qwen.readthedocs.io/en/latest/getting_started/concepts.html`

Result of the fix:

- Immediate early STOP was removed without oracle STOP suppression.
- Smoke runs showed normal multi-step rollouts and successful STOPs near goals.

## What Was Done Recently

Files changed during the Qwen3.5-Omega interleaved-eval fix:

- `joynav/eval/qwen3_vl_lm_head_evaluator.py`
  - Added prompt mode constants.
  - Added `get_generation_prompt_mode`, `render_generation_prompt_text`, and `build_processor_inputs`.
  - Updated no-cache eval input building to use prompt mode.
  - Added decoded prompt logging with prompt mode.
  - Added `should_save_navigation_video`.
  - Changed video saving so directories/videos are created only when `--save_video` is enabled and episode success is exactly 1.0.
- `joynav/eval/qwen3_5_lm_head_sf_omega_evaluator.py`
  - Forces no-cache evaluation.
  - Keeps direct-image preprocessing hook for `text_align_force_qwen`.
  - Overrides prompt mode so intermediate chunks match training while final chunk uses official Qwen final prompt.
- `scripts/eval/eval-qwen3_5-sf-omega.sh`
  - Passes `--num_frames`, `--num_history`, and `--action_chunk_num`.
  - Defaults to `USE_CACHE=0`.
  - Defaults to `OMEGA_MODE=text_align_force_qwen`.
  - Uses `checkpoint-36000` for force-Qwen and `checkpoint-78000` for text-align.
  - Supports `SAVE_VIDEO=1` -> `--save_video`.
  - CUDA visible devices are overrideable at runtime.
- `tests/test_qwen3_5_support.py`
  - Added tests for:
    - intermediate prompt without Qwen final `<think>` wrapper
    - subsequent interleaved assistant/user turn preservation
    - final chunk using official Qwen final prompt
    - script defaults/contracts
    - successful-only video saving

Verification that was run:

```bash
conda run -n qwenvln python -m unittest tests.test_vln_action_interleaving tests.test_qwen3_5_support
bash -n scripts/eval/eval-qwen3_5-sf-omega.sh
git diff --check
```

Result:

- Unit tests passed: 50 tests.
- Shell syntax check passed.
- Diff whitespace check passed.

Smoke evals:

1. Text-align mode:

```text
outputs/eval_smoke/qwen3_5_sf_omega_promptfix_text_align_20260526_154145/result.json
```

- 10 episode rows plus aggregate.
- Success rate: 0.4.
- Mean steps: 53.
- Max steps: 93.

2. Force-Qwen mode:

```text
outputs/eval_smoke/qwen3_5_sf_omega_interleaved_force_qwen_20260526_154820/result.json
```

- 10 episode rows plus aggregate.
- Success rate: 0.4.
- Mean steps: 81.
- Max steps: 182.
- Force-Qwen decoded prompts showed image token count `image_pad * 252`, consistent with direct alignment for resized Qwen images in that run.

## Navigation Video Saving

User requested:

- Add function to save navigation video only when success rate is 1.0.
- Use `--save_video` as args.

Implemented behavior:

- CLI/eval arg exists: `save_video`.
- Shell script supports `SAVE_VIDEO=1` and adds `--save_video`.
- Evaluator records frames during rollout, but only writes video if:

```python
self.save_video and float(metrics.get("success", 0.0)) == 1.0
```

Video output path:

```text
<output_path>/vis_<epoch>/<scene_id>/<episode_id>.mp4
```

The directory is only created for successful episodes when video saving is enabled.

## Current Local Worktree Notes

At the time this file was written, `git status --short` showed:

```text
 M configs/vln_r2r.yaml
 M scripts/eval/eval-qwen3_5-sf-omega.sh
```

Observed diffs:

- `configs/vln_r2r.yaml`
  - `habitat.environment.max_episode_steps` changed from 5000 to 300.
  - This may be a local smoke/debug setting. Be careful before assuming it is intended for full eval.
- `scripts/eval/eval-qwen3_5-sf-omega.sh`
  - `CUDA_VISIBLE_DEVICES` default changed from `0,3` to `2,3`.
  - User explicitly allowed freely changing CUDA IDs based on current situation.

Do not blindly revert these; confirm intent if they matter.

## Important Files For Future Work

Training:

- `joynav/train/train_qwen.py`
- `joynav/train/argument.py`
- `scripts/single-qwen3_5-0_8b-sf-omega.sh`
- `scripts/single-qwen3_5-0_8b-sf-omega-force-qwen.sh`

Datasets/preprocessing:

- `joynav/dataset/lazy_supervised_dataset.py`
- `joynav/dataset/vln_action_dataset.py`
- `joynav/dataset/vln_action_omega_spatial_forcing_dataset.py`
- `joynav/dataset/vln_action_dataset_args.py`

Models:

- `joynav/model/qwen3_5_lm_head.py`
- `joynav/model/qwen3_5_lm_head_sf.py`
- `joynav/model/qwen3_5_lm_head_sf_omega.py`
- `joynav/model/spatial_forcing.py`
- `joynav/model/geometry_encoder/vggt_omega_encoder.py`
- `joynav/model/geometry_encoder/factory.py`

Evaluation:

- `joynav/eval/eval_habitat.py`
- `joynav/eval/qwen3_vl_lm_head_evaluator.py`
- `joynav/eval/qwen3_5_lm_head_sf_omega_evaluator.py`
- `scripts/eval/eval-qwen3_5-sf-omega.sh`
- `configs/vln_r2r.yaml`

Tests:

- `tests/test_qwen3_5_support.py`
- `tests/test_vln_action_interleaving.py`

Outputs/results to inspect:

- `results/r2r/val_unseen/qwen3_5_sf_omega_text_align/result.json`
- `results/r2r/val_unseen/qwen3_5_sf_omega_text_align_force_qwen/result.json`
- `outputs/eval_smoke/`

## Common Commands

Run unit tests in required conda env:

```bash
conda run -n qwenvln python -m unittest tests.test_vln_action_interleaving tests.test_qwen3_5_support
```

Syntax check eval script:

```bash
bash -n scripts/eval/eval-qwen3_5-sf-omega.sh
```

Run 10-episode force-Qwen smoke:

```bash
CONDA_ENV=qwenvln CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 OMEGA_MODE=text_align_force_qwen LIMIT=10 USE_CACHE=0 OUTPUT_PATH=outputs/eval_smoke/qwen3_5_sf_omega_interleaved_force_qwen bash scripts/eval/eval-qwen3_5-sf-omega.sh
```

Run with successful-only video saving:

```bash
CONDA_ENV=qwenvln CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 OMEGA_MODE=text_align_force_qwen LIMIT=10 USE_CACHE=0 SAVE_VIDEO=1 OUTPUT_PATH=outputs/eval_smoke/qwen3_5_sf_omega_video_success_only bash scripts/eval/eval-qwen3_5-sf-omega.sh
```

Summarize a `result.json`:

```bash
python - <<'PY'
import json, pathlib, statistics
p = pathlib.Path("outputs/eval_smoke/qwen3_5_sf_omega_interleaved_force_qwen_20260526_154820/result.json")
rows = [json.loads(l) for l in p.open() if l.strip()]
eps = [r for r in rows if "steps" in r]
steps = [int(r["steps"]) for r in eps]
print("rows", len(rows), "episodes", len(eps), "aggregate", rows[-1] if rows and "steps" not in rows[-1] else None)
print("success_sum", sum(float(r.get("success", 0)) for r in eps))
print("mean_steps", statistics.mean(steps), "median_steps", statistics.median(steps), "min", min(steps), "max", max(steps))
PY
```

Check decoded prompt modes in a log:

```bash
rg -n "prompt_mode|decoded input_ids|llm_outputs" outputs/eval_smoke/<run>/eval_log.log
```

## Gotchas

- Do not use Qwen3-VL docs/classes as proof for Qwen3.5 behavior unless the shared base file is the reason. Qwen3.5-Omega uses `qwen3_vl_lm_head_evaluator.py` only because it is the shared evaluator base.
- For Qwen3.5 prompt formatting, prefer actual local Qwen3.5 processor behavior plus official Qwen/Transformers docs.
- Do not enable KV cache for Qwen3.5-Omega interleaved eval unless implementing and testing cache-aware dialogue splicing. Current correct path is no-cache full prompt rebuild.
- In `text_align_force_qwen`, keep source images raw in dialogue state, then apply `prepare_qwen_images_for_omega_direct` only in `prepare_processor_source`. This prevents corrupting future history/state while still aligning processor input shapes.
- Do not suppress STOP by oracle rules. The fix should preserve generated model behavior and only correct prompt semantics.
- The `result.json` aggregate row is appended as JSON without a trailing newline in `eval_habitat.py`; handle parsing line-by-line carefully.
- Full Habitat eval can resume/skip episodes already present in `result.json`, based on scene id, episode id, and instruction.
