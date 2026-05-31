import inspect
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor

from joynav.dataset.lazy_supervised_dataset import preprocess_qwen_visual
from joynav.dataset.vln_action_spatial_forcing_dataset import resize_image_to_qwen_grid
from joynav.model.spatial_forcing import SpatialForcingProjector, cosine_alignment_loss
from joynav.utils.registry import get_component


class Qwen35SupportTest(unittest.TestCase):
    def test_qwen3_5_assistant_labels_are_not_all_masked(self):
        processor = AutoProcessor.from_pretrained("/mnt/nas5/xiangchen/vlm_base/Qwen3.5-0.8B", use_fast=False)
        sample = {
            "conversations": [
                {"from": "human", "value": "Navigate to the chair."},
                {"from": "gpt", "value": "STOP"},
            ]
        }

        result = preprocess_qwen_visual([sample], processor)
        label_ids = result["labels"][0]
        visible_label_ids = [
            token_id if token_id != -100 else processor.tokenizer.pad_token_id
            for token_id in label_ids.tolist()
        ]
        decoded_labels = processor.tokenizer.decode(visible_label_ids, skip_special_tokens=False)

        self.assertIn("STOP", decoded_labels)

    def test_qwen3_5_model_types_are_registered(self):
        import joynav.model  # noqa: F401

        self.assertEqual(get_component("model", "qwen3_5_lm_head").__name__, "JoyNav_Qwen3_5ForCausalLM")
        self.assertEqual(
            get_component("model", "qwen3_5_lm_head_sf").__name__,
            "JoyNav_Qwen3_5SpatialForcingForCausalLM",
        )
        self.assertEqual(
            get_component("model", "qwen3_5_lm_head_sf_omega").__name__,
            "JoyNav_Qwen3_5OmegaSpatialForcingForCausalLM",
        )

    def test_qwen3_5_action_script_contract(self):
        text = Path("scripts/single-qwen3_5-0_8b.sh").read_text()

        self.assertIn("conda activate ${CONDA_ENV:-qwenvln}", text)
        self.assertIn("llm=${MODEL_PATH:-/mnt/nas5/xiangchen/vlm_base/Qwen3.5-0.8B}", text)
        self.assertIn("model_type=qwen3_5_lm_head", text)
        self.assertIn("dataset_type=vln_action", text)
        self.assertIn('datasets=${DATASETS:-"/mnt/nas5/xiangchen/VLNData/R2R,/mnt/nas5/xiangchen/VLNData/RxR"}', text)
        self.assertIn("--tune_mm_mlp True", text)
        self.assertIn("--tune_mm_llm True", text)
        self.assertIn("--use_lora False", text)

    def test_qwen3_5_spatial_forcing_script_contract(self):
        text = Path("scripts/single-qwen3_5-0_8b-sf.sh").read_text()

        self.assertIn("conda activate ${CONDA_ENV:-qwenvln}", text)
        self.assertIn("llm=${MODEL_PATH:-/mnt/nas5/xiangchen/vlm_base/Qwen3.5-0.8B}", text)
        self.assertIn("model_type=qwen3_5_lm_head_sf", text)
        self.assertIn("dataset_type=vln_action_sf", text)
        self.assertIn("--sf_enabled True", text)
        self.assertIn("--sf_align_layers=${sf_align_layers}", text)
        self.assertIn("sf_align_layers=${SF_ALIGN_LAYERS:-18}", text)
        self.assertIn("--spatial_forcing_image_size 518", text)

    def test_qwen3_5_omega_spatial_forcing_script_contract(self):
        text = Path("scripts/single-qwen3_5-0_8b-sf-omega.sh").read_text()

        self.assertIn("conda activate ${CONDA_ENV:-qwenvln}", text)
        self.assertIn("model_type=qwen3_5_lm_head_sf_omega", text)
        self.assertIn("dataset_type=vln_action_sf_omega", text)
        self.assertIn("omega_mode=${OMEGA_MODE:-text_align}", text)
        self.assertIn("text_align)", text)
        self.assertIn(
            "vggt_omega_1b_512.pt",
            text,
        )
        self.assertIn("vggt_omega_1b_256_text.pt", text)
        self.assertIn("--sf_target_dim 2048", text)
        self.assertIn("--sf_teacher_layers=${sf_teacher_layers}", text)
        self.assertIn("--omega_mode ${omega_mode}", text)
        self.assertNotIn("--model_load_dtype", text)
        self.assertIn("gpu_ids=${CUDA_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-1,2,3}}", text)
        self.assertIn("logging_nan_inf_filter=${LOGGING_NAN_INF_FILTER:-False}", text)
        self.assertIn("--logging_nan_inf_filter ${logging_nan_inf_filter}", text)
        self.assertIn("--spatial_forcing_teacher_patch_size 16", text)
        self.assertIn(
            "spatial_forcing_image_resolution=${SPATIAL_FORCING_IMAGE_RESOLUTION:-${default_spatial_forcing_image_resolution}}",
            text,
        )
        self.assertIn("--spatial_forcing_image_resolution ${spatial_forcing_image_resolution}", text)

    def test_qwen3_5_preprocess_keeps_mm_token_type_ids(self):
        processor = AutoProcessor.from_pretrained("/mnt/nas5/xiangchen/vlm_base/Qwen3.5-0.8B", use_fast=False)
        sample = {
            "image": "/mnt/nas5/xiangchen/VLNData/R2R/images/17DRP5sb8fy_r2r_001803/rgb/001.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nNavigate to the chair."},
                {"from": "gpt", "value": "STOP"},
            ],
        }

        result = preprocess_qwen_visual([sample], processor)

        self.assertIn("mm_token_type_ids", result)
        self.assertEqual(result["mm_token_type_ids"].shape, result["input_ids"].shape)
        self.assertGreater(int((result["mm_token_type_ids"] == 1).sum()), 0)

    def test_spatial_forcing_teacher_resize_matches_qwen_grid(self):
        image = Image.new("RGB", (640, 480), (255, 0, 0))
        grid_thw = torch.tensor([1, 30, 40])

        resized = resize_image_to_qwen_grid(image, grid_thw, patch_size=16)

        self.assertEqual(resized.size[0] % 14, 0)
        self.assertEqual(resized.size[1] % 14, 0)
        self.assertLess(abs((resized.size[0] / resized.size[1]) - (640 / 480)), 0.03)

    def test_omega_teacher_resize_uses_patch_size_16(self):
        image = Image.new("RGB", (640, 480), (255, 0, 0))
        grid_thw = torch.tensor([1, 30, 40])

        resized = resize_image_to_qwen_grid(image, grid_thw, patch_size=16, teacher_patch_size=16)

        self.assertEqual(resized.size, (640, 480))
        self.assertEqual(resized.size[0] % 16, 0)
        self.assertEqual(resized.size[1] % 16, 0)

    def test_vggt_omega_balanced_512_preprocess_shape(self):
        from vggt_omega.utils.load_fn import load_and_preprocess_images

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "image.jpg"
            Image.new("RGB", (640, 480), (255, 0, 0)).save(image_path)

            images = load_and_preprocess_images([str(image_path)], image_resolution=512, patch_size=16)

        self.assertEqual(images.shape, (1, 3, 448, 592))

    def test_vggt_omega_balanced_256_preprocess_shape(self):
        from vggt_omega.utils.load_fn import load_and_preprocess_images

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "image.jpg"
            Image.new("RGB", (640, 480), (255, 0, 0)).save(image_path)

            images = load_and_preprocess_images([str(image_path)], image_resolution=256, patch_size=16)

        self.assertEqual(images.shape, (1, 3, 224, 288))

    def test_vggt_omega_mode_config(self):
        from joynav.model.geometry_encoder.vggt_omega_encoder import resolve_vggt_omega_mode

        mode_512 = resolve_vggt_omega_mode("512_1b")
        mode_text = resolve_vggt_omega_mode("text_align")
        mode_force = resolve_vggt_omega_mode("text_align_force_qwen")

        self.assertEqual(mode_512.image_resolution, 512)
        self.assertFalse(mode_512.enable_alignment)
        self.assertTrue(mode_512.checkpoint_path.endswith("vggt_omega_1b_512.pt"))
        self.assertEqual(mode_text.image_resolution, 256)
        self.assertTrue(mode_text.enable_alignment)
        self.assertTrue(mode_text.checkpoint_path.endswith("vggt_omega_1b_256_text.pt"))
        self.assertEqual(mode_force.image_resolution, 256)
        self.assertTrue(mode_force.enable_alignment)
        self.assertTrue(mode_force.checkpoint_path.endswith("vggt_omega_1b_256_text.pt"))

    def test_text_align_force_qwen_resizes_qwen_grid_to_omega_patch_grid(self):
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from joynav.dataset.vln_action_omega_spatial_forcing_dataset import load_qwen_images_for_omega_direct

        image_path = "/mnt/nas5/xiangchen/VLNData/R2R/images/17DRP5sb8fy_r2r_001803/rgb/001.jpg"
        omega_images = load_and_preprocess_images([image_path], image_resolution=256, patch_size=16)
        qwen_images = load_qwen_images_for_omega_direct([image_path], omega_images.shape[-2:], spatial_merge_size=2)

        processor = AutoProcessor.from_pretrained("/mnt/nas5/xiangchen/vlm_base/Qwen3.5-0.8B", use_fast=False)
        processor.image_processor.max_pixels = 258048
        processor.image_processor.size["longest_edge"] = 258048
        sample = {
            "image": qwen_images,
            "conversations": [
                {"from": "human", "value": "<image>\nNavigate to the chair."},
                {"from": "gpt", "value": "STOP"},
            ],
        }

        result = preprocess_qwen_visual([sample], processor)

        self.assertEqual(tuple(omega_images.shape[-2:]), (224, 288))
        self.assertEqual(qwen_images[0].size, (576, 448))
        self.assertEqual(result["image_grid_thw"].tolist(), [[1, 28, 36]])
        self.assertEqual(int((result["mm_token_type_ids"] == 1).sum()), 14 * 18)

    def test_update_processor_pixels_sets_qwen3_5_size_dict(self):
        from joynav.dataset.lazy_supervised_dataset import update_processor_pixels

        processor = AutoProcessor.from_pretrained("/mnt/nas5/xiangchen/vlm_base/Qwen3.5-0.8B", use_fast=False)
        data_args = type(
            "DataArgs",
            (),
            {
                "min_pixels": 65536,
                "max_pixels": 258048,
                "video_min_pixels": 4096,
                "video_max_pixels": 25165824,
                "video_min_frames": 4,
                "video_max_frames": 8,
                "video_fps": 2.0,
            },
        )()

        update_processor_pixels(processor, data_args)

        self.assertEqual(processor.image_processor.size["shortest_edge"], 65536)
        self.assertEqual(processor.image_processor.size["longest_edge"], 258048)

    def test_qwen3_5_omega_force_qwen_script_contract(self):
        text = Path("scripts/single-qwen3_5-0_8b-sf-omega-force-qwen.sh").read_text()

        self.assertIn("omega_mode=${OMEGA_MODE:-text_align_force_qwen}", text)
        self.assertIn("text_align_force_qwen)", text)
        self.assertIn("max_pixels=${MAX_PIXELS:-258048}", text)
        self.assertIn("gpu_ids=${CUDA_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-1,2,3}}", text)
        self.assertIn("--omega_mode ${omega_mode}", text)

    def test_qwen3_5_omega_eval_script_contract(self):
        text = Path("scripts/eval/eval-qwen3_5-sf-omega.sh").read_text()

        self.assertIn("conda activate ${CONDA_ENV:-qwenvln}", text)
        self.assertRegex(text, r"export CUDA_VISIBLE_DEVICES=\$\{CUDA_VISIBLE_DEVICES:-[0-9,]+\}")
        self.assertIn("NPROC_PER_NODE=${NPROC_PER_NODE:-2}", text)
        self.assertIn("MODEL_TYPE=${MODEL_TYPE:-qwen3_5_lm_head_sf_omega}", text)
        self.assertIn("EVALUATOR_TYPE=${EVALUATOR_TYPE:-qwen3_5_lm_head_sf_omega}", text)
        self.assertRegex(text, r"(?m)^OMEGA_MODE=\$\{OMEGA_MODE:-text_align_force_qwen\}$")
        self.assertIn("vggt_omega_1b_512.pt", text)
        self.assertIn("vggt_omega_1b_256_text.pt", text)
        self.assertIn("checkpoint-78000", text)
        self.assertIn("checkpoint-36000", text)
        self.assertNotIn("checkpoint-24000", text)
        self.assertIn("MAX_PIXELS=${MAX_PIXELS:-258048}", text)
        self.assertIn("NUM_FRAMES=${NUM_FRAMES:-24}", text)
        self.assertIn("NUM_HISTORY=${NUM_HISTORY:-6}", text)
        self.assertIn("ACTION_CHUNK_NUM=${ACTION_CHUNK_NUM:-4}", text)
        self.assertIn("USE_CACHE=${USE_CACHE:-0}", text)
        self.assertIn("--omega_mode \"$OMEGA_MODE\"", text)
        self.assertIn("--num_frames \"$NUM_FRAMES\"", text)
        self.assertIn("--num_history \"$NUM_HISTORY\"", text)
        self.assertIn("--action_chunk_num \"$ACTION_CHUNK_NUM\"", text)
        self.assertIn("--spatial_forcing_image_resolution \"$SPATIAL_FORCING_IMAGE_RESOLUTION\"", text)
        self.assertIn("--limit \"$LIMIT\"", text)

    def test_qwen3_5_omega_evaluator_registered_for_habitat_eval(self):
        text = Path("joynav/eval/eval_habitat.py").read_text()

        self.assertIn("Qwen3_5OmegaSpatialForcingEvaluator", text)
        self.assertIn(
            "register_component('evaluator', 'qwen3_5_lm_head_sf_omega', Qwen3_5OmegaSpatialForcingEvaluator)",
            text,
        )
        self.assertIn("OMEGA_MODEL_CONFIG_FIELDS", text)
        self.assertIn("model_config = build_model_config(args)", text)
        self.assertIn("config=model_config", text)
        self.assertIn("get_explicit_cli_fields", text)

    def test_eval_uses_checkpoint_omega_config_without_cli_override(self):
        from types import SimpleNamespace
        from joynav.eval.eval_habitat import build_model_config

        args = SimpleNamespace(
            model_path="/mnt/nas5/xiangchen/vlacode/JD-VLN/outputs/qwen3_5_0_8b_full_sf_omega_force_no-interpo/stage1-r2r+rxr-omega_sf_alpha_0.1-layers_18-teacher_23-lr_2e-5-mm_lr_1e-6-batch_size_1-grad_accum_steps_2-epochs_2-max_pixels_258048/checkpoint-36000",
            model_type="qwen3_5_lm_head_sf_omega",
            omega_mode="text_align",
            sf_geometry_encoder_path="",
            sf_target_dim=2048,
            sf_teacher_layers="23",
            sf_align_layers="18",
            sf_alpha=0.1,
            sf_add_pos_embed=False,
        )

        config = build_model_config(args, explicit_fields=set())

        self.assertEqual(config.omega_mode, "text_align_force_qwen")
        self.assertEqual(args.omega_mode, "text_align_force_qwen")

    def test_eval_respects_explicit_omega_cli_override(self):
        from types import SimpleNamespace
        from joynav.eval.eval_habitat import build_model_config

        args = SimpleNamespace(
            model_path="/mnt/nas5/xiangchen/vlacode/JD-VLN/outputs/qwen3_5_0_8b_full_sf_omega_bfloat/stage1-r2r+rxr-omega_sf_alpha_0.1-layers_18-teacher_23-lr_5e-6-mm_lr_1e-6-batch_size_1-grad_accum_steps_1-epochs_2-max_pixels_200704/checkpoint-78000",
            model_type="qwen3_5_lm_head_sf_omega",
            omega_mode="text_align_force_qwen",
            sf_geometry_encoder_path="/mnt/nas5/xiangchen/vlacode/vggt-omega/facebook/VGGT-Omega/vggt_omega_1b_256_text.pt",
            sf_target_dim=2048,
            sf_teacher_layers="23",
            sf_align_layers="18",
            sf_alpha=0.1,
            sf_add_pos_embed=False,
        )

        config = build_model_config(args, explicit_fields={"omega_mode", "sf_geometry_encoder_path"})

        self.assertEqual(config.omega_mode, "text_align_force_qwen")
        self.assertEqual(args.omega_mode, "text_align_force_qwen")

    def test_qwen3_5_omega_evaluator_uses_neutral_qwen_vln_base(self):
        text = Path("joynav/eval/qwen3_5_lm_head_sf_omega_evaluator.py").read_text()

        self.assertIn("QwenVLLMHeadEvaluator", text)
        self.assertIn("class Qwen3_5OmegaSpatialForcingEvaluator(QwenVLLMHeadEvaluator)", text)
        self.assertNotIn("class Qwen3_5OmegaSpatialForcingEvaluator(Qwen3VLLMHeadEvaluator)", text)

    def test_eval_uses_requested_max_new_tokens(self):
        text = Path("joynav/eval/qwen3_vl_lm_head_evaluator.py").read_text()

        self.assertIn("max_new_tokens=self.args.max_new_tokens", text)
        self.assertNotIn("generate(**inputs, max_new_tokens=128", text)
        self.assertNotIn("return self.prepare_inputs_no_cache(step_id, episode, rgb_list)", text)

    def test_eval_build_messages_reuses_training_qwen_message_builder(self):
        text = Path("joynav/eval/qwen3_vl_lm_head_evaluator.py").read_text()

        self.assertIn("_build_messages as build_training_messages", text)
        self.assertIn("return build_training_messages(item, base_path)", text)

    def test_eval_build_messages_matches_training_schema_and_official_qwen_format(self):
        from joynav.dataset.lazy_supervised_dataset import _build_messages as build_training_messages
        from joynav.eval.qwen3_vl_lm_head_evaluator import build_messages

        images = [
            Image.new("RGB", (16, 16), (255, 0, 0)),
            Image.new("RGB", (16, 16), (0, 255, 0)),
        ]
        source = {
            "image": images,
            "conversations": [
                {"from": "human", "value": "first <image> then <image>"},
                {"from": "gpt", "value": "<|action|>↑→"},
                {"from": "human", "value": "again"},
            ],
        }

        eval_messages = build_messages(source)
        training_messages = build_training_messages(source, Path(""))

        self.assertEqual(eval_messages, training_messages)
        self.assertEqual([message["role"] for message in eval_messages], ["user", "assistant", "user"])
        self.assertEqual(eval_messages[0]["content"][0], {"type": "text", "text": "first"})
        self.assertEqual(eval_messages[0]["content"][1]["type"], "image")
        self.assertIs(eval_messages[0]["content"][1]["image"], images[0])
        self.assertEqual(eval_messages[0]["content"][2], {"type": "text", "text": "then"})
        self.assertEqual(eval_messages[0]["content"][3]["type"], "image")
        self.assertIs(eval_messages[0]["content"][3]["image"], images[1])
        self.assertEqual(eval_messages[1]["content"], [{"type": "text", "text": "<|action|>↑→"}])

    def test_qwen_omega_eval_first_interleaved_turn_uses_sparse_history(self):
        from joynav.eval.qwen3_vl_lm_head_evaluator import DEFAULT_IMAGE_TOKEN, QwenVLLMHeadEvaluator

        evaluator = QwenVLLMHeadEvaluator.__new__(QwenVLLMHeadEvaluator)
        evaluator.conversation = [
            {
                "from": "human",
                "value": (
                    "You are an autonomous navigation assistant. Your task is to <instruction>. "
                    "Devise an action sequence."
                ),
            }
        ]
        evaluator.conjunctions = ["observe "]
        evaluator.num_history = 6
        rgb_list = [object() for _ in range(25)]
        episode = SimpleNamespace(instruction=SimpleNamespace(instruction_text="Walk to the sofa."))

        source, history_ids = evaluator.build_interleaved_source(
            {"image": [], "conversations": []},
            previous_assistant_text=None,
            step_id=24,
            episode=episode,
            rgb_list=rgb_list,
        )

        self.assertEqual(history_ids, [0, 4, 8, 12, 16, 20])
        self.assertEqual(source["image"], [rgb_list[i] for i in history_ids] + [rgb_list[24]])
        self.assertEqual([turn["from"] for turn in source["conversations"]], ["human"])
        first_turn = source["conversations"][0]["value"]
        self.assertIn("Walk to the sofa.", first_turn)
        self.assertIn("These are your historical observations", first_turn)
        self.assertTrue(first_turn.endswith(f"observe {DEFAULT_IMAGE_TOKEN}."))
        self.assertEqual(first_turn.count(DEFAULT_IMAGE_TOKEN), len(source["image"]))

    def test_qwen_omega_eval_subsequent_turn_appends_assistant_and_one_image(self):
        from joynav.eval.qwen3_vl_lm_head_evaluator import DEFAULT_IMAGE_TOKEN, QwenVLLMHeadEvaluator

        evaluator = QwenVLLMHeadEvaluator.__new__(QwenVLLMHeadEvaluator)
        evaluator.conjunctions = ["observe "]
        evaluator.num_history = 6
        first_image = object()
        rgb_list = [first_image] + [object() for _ in range(4)]
        source = {
            "image": [first_image],
            "conversations": [{"from": "human", "value": f"observe {DEFAULT_IMAGE_TOKEN}."}],
        }

        source, history_ids = evaluator.build_interleaved_source(
            source,
            previous_assistant_text="<|action|>↑→",
            step_id=4,
            episode=SimpleNamespace(instruction=SimpleNamespace(instruction_text="Walk.")),
            rgb_list=rgb_list,
        )

        self.assertEqual(history_ids, [])
        self.assertEqual(source["image"], [first_image, rgb_list[4]])
        self.assertEqual([turn["from"] for turn in source["conversations"]], ["human", "gpt", "human"])
        self.assertEqual(source["conversations"][1]["value"], "<|action|>↑→")
        self.assertEqual(source["conversations"][2]["value"], f"observe {DEFAULT_IMAGE_TOKEN}.")
        self.assertEqual(
            sum(turn["value"].count(DEFAULT_IMAGE_TOKEN) for turn in source["conversations"]),
            len(source["image"]),
        )

    def test_qwen_omega_eval_interleaved_source_resets_on_num_frames(self):
        from joynav.eval.qwen3_vl_lm_head_evaluator import QwenVLLMHeadEvaluator

        evaluator = QwenVLLMHeadEvaluator.__new__(QwenVLLMHeadEvaluator)
        evaluator.num_frames = 24

        self.assertFalse(evaluator.should_reset_interleaved_source(0))
        self.assertFalse(evaluator.should_reset_interleaved_source(23))
        self.assertTrue(evaluator.should_reset_interleaved_source(24))
        self.assertFalse(evaluator.should_reset_interleaved_source(25))

    def test_qwen3_5_omega_eval_args_match_training_window_defaults(self):
        from joynav.eval.qwen3_5_lm_head_sf_omega_evaluator import (
            Qwen3_5OmegaSpatialForcingEvaluatorArguments,
        )

        args = Qwen3_5OmegaSpatialForcingEvaluatorArguments()

        self.assertEqual(args.num_frames, 24)
        self.assertEqual(args.num_history, 6)
        self.assertEqual(args.action_chunk_num, 4)

    def test_qwen3_5_omega_evaluator_forces_no_cache(self):
        text = Path("joynav/eval/qwen3_5_lm_head_sf_omega_evaluator.py").read_text()

        self.assertIn("self.use_cache = False", text)
        self.assertIn("self.args.use_cache = False", text)

    def test_qwen3_5_omega_eval_uses_training_prompt_for_intermediate_chunk(self):
        from joynav.eval.qwen3_vl_lm_head_evaluator import DEFAULT_IMAGE_TOKEN
        from joynav.eval.qwen3_5_lm_head_sf_omega_evaluator import Qwen3_5OmegaSpatialForcingEvaluator

        processor = AutoProcessor.from_pretrained("/mnt/nas5/xiangchen/vlm_base/Qwen3.5-0.8B", use_fast=False)
        evaluator = Qwen3_5OmegaSpatialForcingEvaluator.__new__(Qwen3_5OmegaSpatialForcingEvaluator)
        evaluator.processor = processor
        evaluator.model = SimpleNamespace(device=torch.device("cpu"))
        evaluator.args = SimpleNamespace(omega_mode="text_align")
        evaluator.conversation = [
            {
                "from": "human",
                "value": (
                    "You are an autonomous navigation assistant. Your task is to <instruction>. "
                    "Devise an action sequence."
                ),
            }
        ]
        evaluator.conjunctions = ["observe "]
        evaluator.num_frames = 24
        evaluator.num_history = 6
        evaluator.action_chunk_num = 4
        rgb_list = [Image.new("RGB", (16, 16), (255, 0, 0))]
        episode = SimpleNamespace(episode_id="0", instruction=SimpleNamespace(instruction_text="Walk."))

        inputs, source = evaluator.prepare_inputs_no_cache(
            {"image": [], "conversations": []},
            previous_assistant_text=None,
            step_id=0,
            episode=episode,
            rgb_list=rgb_list,
        )

        decoded = evaluator.decode_input_ids(inputs["input_ids"])
        self.assertEqual(source["image"], rgb_list)
        self.assertTrue(decoded.endswith("<|im_start|>assistant\n"))
        self.assertIn(f"observe<|vision_start|>", decoded)
        self.assertNotIn("<|im_start|>assistant\n<think>\n\n</think>\n\n", decoded)
        self.assertEqual(source["conversations"][0]["value"].count(DEFAULT_IMAGE_TOKEN), 1)

    def test_qwen3_5_omega_eval_subsequent_intermediate_prompt_preserves_action_turn(self):
        from joynav.eval.qwen3_vl_lm_head_evaluator import DEFAULT_IMAGE_TOKEN
        from joynav.eval.qwen3_5_lm_head_sf_omega_evaluator import Qwen3_5OmegaSpatialForcingEvaluator

        processor = AutoProcessor.from_pretrained("/mnt/nas5/xiangchen/vlm_base/Qwen3.5-0.8B", use_fast=False)
        evaluator = Qwen3_5OmegaSpatialForcingEvaluator.__new__(Qwen3_5OmegaSpatialForcingEvaluator)
        evaluator.processor = processor
        evaluator.model = SimpleNamespace(device=torch.device("cpu"))
        evaluator.args = SimpleNamespace(omega_mode="text_align")
        evaluator.conjunctions = ["observe "]
        evaluator.num_frames = 24
        evaluator.num_history = 6
        evaluator.action_chunk_num = 4
        images = [Image.new("RGB", (16, 16), (i, 0, 0)) for i in range(5)]
        source = {
            "image": [images[0]],
            "conversations": [{"from": "human", "value": f"observe {DEFAULT_IMAGE_TOKEN}."}],
        }
        episode = SimpleNamespace(episode_id="0", instruction=SimpleNamespace(instruction_text="Walk."))

        inputs, source = evaluator.prepare_inputs_no_cache(
            source,
            previous_assistant_text="<|action|>↑↑↑↑",
            step_id=4,
            episode=episode,
            rgb_list=images,
        )

        decoded = evaluator.decode_input_ids(inputs["input_ids"])
        self.assertEqual(len(source["image"]), 2)
        self.assertEqual([turn["from"] for turn in source["conversations"]], ["human", "gpt", "human"])
        self.assertEqual(source["conversations"][1]["value"], "<|action|>↑↑↑↑")
        self.assertEqual(source["conversations"][2]["value"].count(DEFAULT_IMAGE_TOKEN), 1)
        self.assertIn("<|im_start|>assistant\n<|action|>↑↑↑↑<|im_end|>", decoded)
        self.assertTrue(decoded.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("<|im_start|>assistant\n<think>\n\n</think>\n\n", decoded)

    def test_qwen3_5_omega_eval_uses_official_qwen_final_prompt_for_last_chunk(self):
        from joynav.eval.qwen3_vl_lm_head_evaluator import DEFAULT_IMAGE_TOKEN
        from joynav.eval.qwen3_5_lm_head_sf_omega_evaluator import Qwen3_5OmegaSpatialForcingEvaluator

        processor = AutoProcessor.from_pretrained("/mnt/nas5/xiangchen/vlm_base/Qwen3.5-0.8B", use_fast=False)
        evaluator = Qwen3_5OmegaSpatialForcingEvaluator.__new__(Qwen3_5OmegaSpatialForcingEvaluator)
        evaluator.processor = processor
        evaluator.model = SimpleNamespace(device=torch.device("cpu"))
        evaluator.args = SimpleNamespace(omega_mode="text_align")
        evaluator.conjunctions = ["observe "]
        evaluator.num_frames = 24
        evaluator.num_history = 6
        evaluator.action_chunk_num = 4
        images = [Image.new("RGB", (16, 16), (i, 0, 0)) for i in range(21)]
        source = {
            "image": [images[0]],
            "conversations": [{"from": "human", "value": f"observe {DEFAULT_IMAGE_TOKEN}."}],
        }
        episode = SimpleNamespace(episode_id="0", instruction=SimpleNamespace(instruction_text="Walk."))

        inputs, _ = evaluator.prepare_inputs_no_cache(
            source,
            previous_assistant_text="<|action|>↑↑↑↑",
            step_id=20,
            episode=episode,
            rgb_list=images,
        )

        decoded = evaluator.decode_input_ids(inputs["input_ids"])
        self.assertTrue(decoded.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))

    def test_qwen_eval_saves_navigation_video_only_for_successful_episodes(self):
        from joynav.eval.qwen3_vl_lm_head_evaluator import QwenVLLMHeadEvaluator

        evaluator = QwenVLLMHeadEvaluator.__new__(QwenVLLMHeadEvaluator)
        evaluator.save_video = True

        self.assertTrue(evaluator.should_save_navigation_video({"success": 1.0}))
        self.assertTrue(evaluator.should_save_navigation_video({"success": True}))
        self.assertFalse(evaluator.should_save_navigation_video({"success": 0.0}))
        self.assertFalse(evaluator.should_save_navigation_video({"success": False}))
        self.assertFalse(evaluator.should_save_navigation_video({}))

        evaluator.save_video = False
        self.assertFalse(evaluator.should_save_navigation_video({"success": 1.0}))

    def test_qwen_omega_eval_decodes_generated_action_token_for_future_turn(self):
        from joynav.eval.qwen3_vl_lm_head_evaluator import QwenVLLMHeadEvaluator

        evaluator = QwenVLLMHeadEvaluator.__new__(QwenVLLMHeadEvaluator)
        evaluator.processor = SimpleNamespace(
            tokenizer=SimpleNamespace(
                eos_token="<|endoftext|>",
                decode=lambda token_ids, skip_special_tokens=False: "<|action|>↑→<|im_end|>\nignored",
            )
        )

        self.assertEqual(evaluator.decode_generated_text([1, 2, 3]), "<|action|>↑→")

    def test_eval_force_qwen_preprocess_keeps_interleaved_source_images_raw(self):
        from joynav.eval.qwen3_5_lm_head_sf_omega_evaluator import Qwen3_5OmegaSpatialForcingEvaluator

        evaluator = Qwen3_5OmegaSpatialForcingEvaluator.__new__(Qwen3_5OmegaSpatialForcingEvaluator)
        evaluator.args = SimpleNamespace(
            omega_mode="text_align_force_qwen",
            spatial_forcing_image_resolution=256,
            spatial_forcing_teacher_patch_size=16,
        )
        evaluator.processor = SimpleNamespace(image_processor=SimpleNamespace(merge_size=2))
        images = [
            Image.new("RGB", (640, 480), (255, 0, 0)),
            Image.new("RGB", (480, 640), (0, 255, 0)),
        ]
        source = {"image": images, "conversations": []}

        processor_source = evaluator.prepare_processor_source(source)

        self.assertIs(source["image"][0], images[0])
        self.assertEqual(source["image"][0].size, (640, 480))
        self.assertEqual(source["image"][1].size, (480, 640))
        self.assertIsNot(processor_source, source)
        self.assertIsNot(processor_source["image"][0], images[0])
        self.assertEqual(processor_source["image"][0].size, processor_source["image"][1].size)

    def test_qwen3_5_generation_accepts_mm_token_type_ids(self):
        text = Path("joynav/model/qwen3_5_lm_head.py").read_text()

        self.assertIn("def prepare_inputs_for_generation", text)
        self.assertIn("mm_token_type_ids=None", text)
        self.assertIn('model_inputs["mm_token_type_ids"] = mm_token_type_ids', text)

    def test_eval_force_qwen_preprocess_accepts_pil_images(self):
        from joynav.dataset.vln_action_omega_spatial_forcing_dataset import (
            prepare_qwen_images_for_omega_direct,
        )

        image = Image.new("RGB", (640, 480), (255, 0, 0))
        qwen_images = prepare_qwen_images_for_omega_direct([image], (224, 288), spatial_merge_size=2)

        self.assertEqual(qwen_images[0].size, (576, 448))
        self.assertIsNot(qwen_images[0], image)

    def test_eval_force_qwen_preprocess_uses_common_omega_shape_for_history(self):
        from joynav.dataset.vln_action_omega_spatial_forcing_dataset import (
            prepare_qwen_images_for_omega_direct,
        )
        from vggt_omega.utils.load_fn import _balanced_target_shape

        images = [
            Image.new("RGB", (640, 480), (255, 0, 0)),
            Image.new("RGB", (480, 640), (0, 255, 0)),
        ]
        expected_shapes = [
            _balanced_target_shape(image.size[1] / image.size[0], 256, 16)
            for image in images
        ]
        expected_h = max(shape[0] for shape in expected_shapes)
        expected_w = max(shape[1] for shape in expected_shapes)

        qwen_images = prepare_qwen_images_for_omega_direct(
            images,
            spatial_merge_size=2,
            image_resolution=256,
            patch_size=16,
        )

        self.assertEqual([image.size for image in qwen_images], [(expected_w * 2, expected_h * 2)] * 2)

    def test_text_align_force_qwen_directly_flattens_omega_patch_tokens(self):
        from joynav.model.qwen3_5_lm_head_sf_omega import JoyNav_Qwen3_5OmegaSpatialForcingForCausalLM

        class FakeEncoder:
            patch_size = 16

            def __init__(self):
                self.features = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)

            def encode(self, image_sequence, teacher_layer_spec):
                return self.features

        fake_self = type("FakeModel", (), {"omega_mode": "text_align_force_qwen", "sf_teacher_layers": "23"})()
        encoder = FakeEncoder()

        target = JoyNav_Qwen3_5OmegaSpatialForcingForCausalLM._encode_omega_sequence(
            fake_self,
            encoder=encoder,
            image_sequence=torch.zeros(2, 3, 32, 48),
            image_grid_thw=torch.tensor([[1, 8, 10], [1, 8, 10]]),
            dtype=torch.float32,
            spatial_merge_size=2,
        )

        self.assertTrue(torch.equal(target, encoder.features.reshape(-1, 4)))

    def test_vggt_omega_patch_token_selection_shape(self):
        from joynav.model.geometry_encoder.vggt_omega_encoder import select_vggt_omega_patch_tokens

        cached_tokens = [None] * 24
        cached_tokens[23] = torch.randn(1, 2, 17 + 12, 2048)

        patch_tokens = select_vggt_omega_patch_tokens(
            aggregated_tokens_list=cached_tokens,
            patch_token_start=17,
            teacher_layer_spec="23",
            source_hw=(3, 4),
        )

        self.assertEqual(patch_tokens.shape, (2, 12, 2048))

    def test_vggt_omega_package_is_vendored(self):
        from vggt_omega.models.aggregator import Aggregator

        self.assertEqual(Aggregator.__name__, "Aggregator")

    def test_spatial_forcing_uses_hidden_states_not_forward_hooks(self):
        for path in [
            Path("joynav/model/qwen3_vl_lm_head_sf.py"),
            Path("joynav/model/qwen3_5_lm_head_sf.py"),
        ]:
            text = path.read_text()
            self.assertNotIn("register_forward_hook", text)
            self.assertIn("output_hidden_states", text)

    def test_qwen3_wrappers_do_not_claim_loss_kwargs_support(self):
        import joynav.model  # noqa: F401

        def trainer_accepts_loss_kwargs(model_cls):
            if hasattr(model_cls, "accepts_loss_kwargs"):
                return model_cls.accepts_loss_kwargs
            parameters = inspect.signature(model_cls.forward).parameters.values()
            return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)

        for model_type in [
            "qwen3_5_lm_head",
            "qwen3_5_lm_head_sf",
            "qwen3_5_lm_head_sf_omega",
            "qwen3_vl_lm_head",
            "qwen3_vl_lm_head_sf",
        ]:
            self.assertFalse(trainer_accepts_loss_kwargs(get_component("model", model_type)))

    def test_spatial_forcing_projector_maps_visual_dim_to_depth_dim(self):
        projector = SpatialForcingProjector(input_dim=1024, target_dim=768, hidden_dim=1536)
        visual_tokens = torch.randn(2, 5, 1024)
        target_tokens = torch.randn(2, 5, 768)

        projected_tokens = projector(visual_tokens)
        loss = cosine_alignment_loss(projected_tokens, target_tokens)

        self.assertEqual(projected_tokens.shape, target_tokens.shape)
        self.assertTrue(torch.isfinite(loss))

    def test_spatial_forcing_cosine_loss_upcasts_low_precision_inputs(self):
        projected_tokens = torch.randn(2, 5, 32, dtype=torch.bfloat16)
        target_tokens = torch.randn(2, 5, 32, dtype=torch.bfloat16)

        loss = cosine_alignment_loss(projected_tokens, target_tokens)

        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))

    def test_trainable_parameter_printer_supports_plain_torch_modules(self):
        sys.path.insert(0, str(Path("joynav/train").resolve()))
        from joynav.train.train_qwen import print_trainable_parameters

        module = torch.nn.Linear(2, 1)
        print_trainable_parameters(module)

    def test_full_finetune_loads_weights_in_fp32_under_bf16_training(self):
        sys.path.insert(0, str(Path("joynav/train").resolve()))
        from joynav.train.train_qwen import resolve_model_load_dtype

        training_args = type("TrainingArgs", (), {"bf16": True})()
        full_finetune_args = type(
            "ModelArgs",
            (),
            {"model_load_dtype": "auto", "tune_mm_llm": True, "use_lora": False},
        )()
        lora_args = type(
            "ModelArgs",
            (),
            {"model_load_dtype": "auto", "tune_mm_llm": True, "use_lora": True},
        )()

        self.assertEqual(resolve_model_load_dtype(training_args, full_finetune_args), torch.float32)
        self.assertEqual(resolve_model_load_dtype(training_args, lora_args), torch.bfloat16)

    def test_trainable_low_precision_parameters_are_promoted_to_fp32(self):
        sys.path.insert(0, str(Path("joynav/train").resolve()))
        from joynav.train.train_qwen import promote_trainable_parameters_to_fp32

        module = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)).to(torch.bfloat16)
        for param in module[0].parameters():
            param.requires_grad = False

        promote_trainable_parameters_to_fp32(module)

        self.assertEqual(module[0].weight.dtype, torch.bfloat16)
        self.assertEqual(module[1].weight.dtype, torch.float32)

    def test_use_cache_is_disabled_on_nested_qwen_text_config(self):
        sys.path.insert(0, str(Path("joynav/train").resolve()))
        from joynav.train.train_qwen import set_model_use_cache

        text_config = type("TextConfig", (), {"use_cache": True})()
        config = type("Config", (), {"use_cache": True, "text_config": text_config})()
        model = type("Model", (), {"config": config})()

        set_model_use_cache(model, False)

        self.assertFalse(model.config.use_cache)
        self.assertFalse(model.config.text_config.use_cache)

    def test_vlnn1_relative_pose_is_ego_centric_ros_xy(self):
        from joynav.dataset.vlnn1_annotation_utils import build_continuous_actions

        transforms = torch.eye(4, dtype=torch.float64).repeat(3, 1, 1).numpy()
        transforms[1, :3, 3] = [0.0, 1.0, 0.0]  # Blender +Y -> ROS +X
        transforms[2, :3, 3] = [1.0, 0.0, 0.0]  # Blender +X -> ROS -Y
        frame_indices = torch.tensor([0, 3, 6]).numpy()

        actions, stop_flags, _ = build_continuous_actions(
            transforms=transforms,
            frame_indices=frame_indices,
            step_stride=1,
            action_chunk_size=3,
        )

        # Goal-approach region is now emitted (one chunk per frame), each padded
        # to a fixed (future_len + 1) length.
        self.assertEqual(sorted(actions), ["0", "1", "2"])
        self.assertEqual(actions["0"][0], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(actions["0"][1][0], 1.0, places=6)
        self.assertAlmostEqual(actions["0"][1][1], 0.0, places=6)
        self.assertAlmostEqual(actions["0"][2][0], 0.0, places=6)
        self.assertAlmostEqual(actions["0"][2][1], -1.0, places=6)
        # Final chunk is a stop chunk and is padded to the full horizon.
        self.assertEqual(stop_flags["2"], 1.0)
        self.assertEqual(len(actions["2"]), 3)

    def test_trajectory_mse_wraps_yaw_at_pi_boundary(self):
        from joynav.model.qwen3_5_trajectory_heads import trajectory_mse_loss

        pred = torch.tensor([[[0.0, 0.0, -3.0415926]]])
        target = torch.tensor([[[0.0, 0.0, 3.0415926]]])

        loss = trajectory_mse_loss(pred, target)

        self.assertLess(float(loss), 0.02)

    def test_trajectory_dit_head_uses_action_latent_flow_contract(self):
        from joynav.model.action_latent.modeling_action_latent import ActionLatent_Config
        from joynav.model.qwen3_5_trajectory_heads import (
            TrajectoryDiTHead,
            select_action_token_features,
        )

        config = ActionLatent_Config(
            time_channel=8,
            time_embedding_dim=8,
            latent_dim=8,
            vl_input_dim=4,
            heads=2,
            layers=1,
            output_dim=8,
            action_dim=3,
            action_hidden_dim=8,
            vl_heads=2,
            vl_self_layers=1,
            max_seq_len=8,
            action_horizon=2,
        )
        hidden_states = torch.randn(2, 5, 4)
        select_mask = torch.zeros(2, 5, dtype=torch.bool)
        select_mask[0, 1] = True
        select_mask[1, 4] = True
        target = torch.randn(2, 2, 3)

        selected = select_action_token_features(hidden_states, select_mask)
        head = TrajectoryDiTHead(config)
        loss = head.flow_matching_loss(selected, target)
        pred = head(selected)

        self.assertEqual(selected.shape, (2, 4))
        self.assertEqual(pred.shape, target.shape)
        self.assertTrue(torch.isfinite(loss))

    def test_qwen3_5_dit_trajectory_forward_uses_action_latent_flow_loss(self):
        import inspect

        from joynav.model.qwen3_5_trajectory_heads import JoyNav_Qwen3_5OmegaDiTForCausalLM

        source = inspect.getsource(JoyNav_Qwen3_5OmegaDiTForCausalLM.forward)

        self.assertIn("selected_features = select_action_token_features(features, select_mask)", source)
        self.assertIn("self.action_head.flow_matching_loss(selected_features, target_actions)", source)
        self.assertNotIn("trajectory_mse_loss(pred_actions, target_actions)", source)

    def test_nextdit_trajectory_modules_import(self):
        from diffusers import FlowMatchEulerDiscreteScheduler
        from joynav.model.nextdit.nextdit_crossattn_traj import NextDiTCrossAttn, NextDiTCrossAttnConfig
        from joynav.model.nextdit.nextdit_traj import LuminaNextDiT2DModel

        self.assertEqual(FlowMatchEulerDiscreteScheduler.__name__, "FlowMatchEulerDiscreteScheduler")
        self.assertEqual(NextDiTCrossAttn.__name__, "NextDiTCrossAttn")
        self.assertEqual(NextDiTCrossAttnConfig.model_type, "nextdit-crossattn")
        self.assertEqual(LuminaNextDiT2DModel.__name__, "LuminaNextDiT2DModel")

    def test_trajectory_nextdit_head_forward_and_loss_are_finite(self):
        from joynav.model.qwen3_5_trajectory_heads import TrajectoryNextDiTHead

        head = TrajectoryNextDiTHead(
            input_dim=4,
            action_horizon=8,
            action_dim=3,
            dim=8,
            layers=1,
            heads=2,
            kv_heads=2,
            num_inference_steps=2,
        )
        selected_features = torch.randn(2, 4)
        target = torch.randn(2, 8, 3)

        pred = head(selected_features)
        loss = head.flow_matching_loss(selected_features, target)

        self.assertEqual(pred.shape, (2, 8, 3))
        self.assertTrue(torch.isfinite(loss))

    def test_nextdit_loss_accepts_flattened_interleaved_targets(self):
        from joynav.model.qwen3_5_trajectory_heads import Qwen35OmegaTrajectoryMixin, TrajectoryNextDiTHead

        head = TrajectoryNextDiTHead(
            input_dim=4,
            action_horizon=8,
            action_dim=3,
            dim=8,
            layers=1,
            heads=2,
            kv_heads=2,
            num_inference_steps=2,
        )
        mixin = Qwen35OmegaTrajectoryMixin()
        selected_features = torch.randn(6, 4)
        target = mixin._normalize_target_actions(torch.randn(2, 3, 8, 3), selected_count=6)

        loss = head.flow_matching_loss(selected_features, target)

        self.assertEqual(target.shape, (6, 8, 3))
        self.assertTrue(torch.isfinite(loss))

    def test_vlnn1_stop_flags_mark_goal_region_and_pad(self):
        from joynav.dataset.vlnn1_annotation_utils import build_continuous_actions

        num_frames = 12
        transforms = np.tile(np.eye(4), (num_frames, 1, 1))
        for i in range(num_frames):
            transforms[i, :3, 3] = [0.0, float(i), 0.0]  # Blender +Y -> ROS +X (forward)
        frame_indices = np.arange(num_frames)

        actions, stop_flags, results = build_continuous_actions(
            transforms=transforms,
            frame_indices=frame_indices,
            step_stride=1,
            action_chunk_size=5,  # future_len = 4
        )

        # Full episode is emitted, each chunk padded to a fixed (future_len + 1, 3).
        self.assertEqual(len(actions), num_frames)
        self.assertTrue(all(np.asarray(chunk).shape == (5, 3) for chunk in actions.values()))
        # stop_window defaults to future_len (=4): final 5 steps (7..11) are stop chunks.
        self.assertEqual(stop_flags["0"], 0.0)
        self.assertEqual(stop_flags["6"], 0.0)
        self.assertEqual(stop_flags["7"], 1.0)
        self.assertEqual(stop_flags["11"], 1.0)
        self.assertEqual(sum(int(value) for value in stop_flags.values()), 5)
        # Goal chunk's future motion is padded to ~zero ("arrived / stay-put").
        self.assertTrue(np.allclose(np.asarray(actions["11"])[1:], 0.0))
        self.assertEqual(results[-1].stop, 1.0)

    def test_continuous_dataset_stop_target_helpers(self):
        from joynav.dataset.continuous_vlnn1_action_dataset import ContinuousVLNN1ActionDataset

        # Annotation stop_flags are read aligned to the requested steps.
        item = {"stop_flags": {"0": 0.0, "3": 1.0, "6": 1.0}}
        flags = ContinuousVLNN1ActionDataset._get_stop_flags(None, item, [0, 3, 6])
        self.assertTrue(np.array_equal(flags, np.array([0.0, 1.0, 1.0], dtype=np.float32)))
        # Pre-stop-schema annotations fall back to geometry derivation.
        self.assertIsNone(ContinuousVLNN1ActionDataset._get_stop_flags(None, {}, [0]))

        stay = torch.zeros(8, 3)
        moving = torch.zeros(8, 3)
        moving[:, 0] = torch.linspace(0.25, 2.0, 8)
        self.assertEqual(ContinuousVLNN1ActionDataset._derive_stop_targets(stay).tolist(), [1.0])
        self.assertEqual(ContinuousVLNN1ActionDataset._derive_stop_targets(moving).tolist(), [0.0])
        chunked = torch.stack([stay, moving], dim=0)  # [2, 8, 3]
        self.assertEqual(ContinuousVLNN1ActionDataset._derive_stop_targets(chunked).tolist(), [1.0, 0.0])

    def test_trajectory_stop_head_loss_is_finite_and_weighted(self):
        from joynav.model.qwen3_5_trajectory_heads import Qwen35OmegaTrajectoryMixin

        mixin = Qwen35OmegaTrajectoryMixin()
        mixin.config = SimpleNamespace(stop_pos_weight=1.0, stop_head_loss_weight=2.0)
        mixin._build_stop_head(4)

        # Negative bias keeps the initial stop probability low (no stop at step 0).
        self.assertLess(torch.sigmoid(mixin.stop_head.bias).item(), 0.5)

        features = torch.randn(3, 4)
        targets = torch.tensor([1.0, 0.0, 1.0])
        stop_loss = mixin._compute_stop_loss(features, targets)
        self.assertTrue(torch.isfinite(stop_loss))
        self.assertEqual(stop_loss.dim(), 0)

        outputs = SimpleNamespace(loss=torch.tensor(1.0))
        mixin._apply_stop_loss(outputs, features, targets)
        self.assertGreater(float(outputs.loss), 1.0)  # weighted stop loss added
        self.assertIsNotNone(outputs.stop_loss)

        # Without stop_targets the loss is untouched and stop_loss is None.
        passthrough = SimpleNamespace(loss=torch.tensor(1.0))
        mixin._apply_stop_loss(passthrough, features, None)
        self.assertEqual(float(passthrough.loss), 1.0)
        self.assertIsNone(passthrough.stop_loss)

    def test_trajectory_heads_wire_stop_head(self):
        from joynav.model.qwen3_5_trajectory_heads import (
            JoyNav_Qwen3_5OmegaDiTForCausalLM,
            JoyNav_Qwen3_5OmegaMLPForCausalLM,
            JoyNav_Qwen3_5OmegaNextDiTForCausalLM,
        )

        for cls in (
            JoyNav_Qwen3_5OmegaMLPForCausalLM,
            JoyNav_Qwen3_5OmegaDiTForCausalLM,
            JoyNav_Qwen3_5OmegaNextDiTForCausalLM,
        ):
            init_src = inspect.getsource(cls.__init__)
            forward_src = inspect.getsource(cls.forward)
            predict_src = inspect.getsource(cls.predict_action)
            self.assertIn("self._build_stop_head(hidden_size)", init_src)
            self.assertIn("stop_targets=None", forward_src)
            self.assertIn("self._apply_stop_loss(outputs, selected_features, stop_targets)", forward_src)
            self.assertIn("outputs.stop_logit = self._predict_stop_logit(", predict_src)

    def test_trajectory_evaluator_stop_and_receding_horizon_contract(self):
        text = Path("joynav/eval/qwen3_5_omega_trajectory_head_evaluator.py").read_text()

        self.assertIn("stop_threshold: float = field(default=0.5", text)
        self.assertIn("replan_every: int = field(default=2", text)
        self.assertIn("def _should_stop(self, outputs)", text)
        self.assertIn("torch.sigmoid(stop_logit[0]).item()", text)
        self.assertIn("if self._should_stop(outputs):", text)
        self.assertIn("replan_every = max(int(self.args.replan_every), 1)", text)

    def test_action_latent_can_initialize_on_meta_device(self):
        from joynav.model.action_latent.modeling_action_latent import ActionLatent, ActionLatent_Config

        config = ActionLatent_Config(
            time_channel=8,
            time_embedding_dim=16,
            latent_dim=16,
            vl_input_dim=16,
            heads=4,
            layers=1,
            output_dim=16,
            action_dim=3,
            action_hidden_dim=16,
            vl_heads=4,
            vl_self_layers=1,
            max_seq_len=8,
            action_horizon=2,
        )

        with torch.device("meta"):
            module = ActionLatent(config)

        self.assertEqual(module.action_horizon, 2)

    def test_action_latent_can_freeze_projector_without_state_encoder(self):
        from joynav.model.action_latent.modeling_action_latent import ActionLatent, ActionLatent_Config

        config = ActionLatent_Config(
            time_channel=8,
            time_embedding_dim=16,
            latent_dim=16,
            vl_input_dim=16,
            heads=4,
            layers=1,
            output_dim=16,
            action_dim=3,
            action_hidden_dim=16,
            vl_heads=4,
            vl_self_layers=1,
            max_seq_len=8,
            action_horizon=2,
            tune_projector=False,
        )

        module = ActionLatent(config)
        module.train()
        module.set_frozen_modules_to_eval_mode()

        self.assertFalse(any(p.requires_grad for p in module.action_encoder.parameters()))
        self.assertFalse(any(p.requires_grad for p in module.action_decoder.parameters()))
        self.assertFalse(module.position_embedding.weight.requires_grad)

    def test_qwen3_5_continuous_omega_model_types_are_registered(self):
        import joynav.model  # noqa: F401
        import joynav.dataset  # noqa: F401

        self.assertEqual(
            get_component("model", "qwen3_5_mlp_head_sf_omega").__name__,
            "JoyNav_Qwen3_5OmegaMLPForCausalLM",
        )
        self.assertEqual(
            get_component("model", "qwen3_5_dit_head_sf_omega").__name__,
            "JoyNav_Qwen3_5OmegaDiTForCausalLM",
        )
        self.assertEqual(
            get_component("model", "qwen3_5_nextdit_head_sf_omega").__name__,
            "JoyNav_Qwen3_5OmegaNextDiTForCausalLM",
        )
        self.assertEqual(
            get_component("dataset", "continuous_vlnn1_action_noninterleave_sf_omega").__name__,
            "ContinuousVLNN1ActionOmegaSpatialForcingDataset",
        )
        self.assertEqual(
            get_component("dataset", "continuous_vlnn1_action_interleave_sf_omega").__name__,
            "ContinuousVLNN1ActionInterleavedOmegaSpatialForcingDataset",
        )

    def test_qwen3_5_trajectory_selects_multiple_action_tokens(self):
        from joynav.model.qwen3_5_trajectory_heads import select_action_token_features

        hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
        select_mask = torch.zeros(2, 5, dtype=torch.bool)
        select_mask[0, [1, 3]] = True
        select_mask[1, [2]] = True

        selected = select_action_token_features(hidden, select_mask)

        self.assertEqual(selected.shape, (3, 3))
        self.assertTrue(torch.equal(selected[0], hidden[0, 1]))
        self.assertTrue(torch.equal(selected[1], hidden[0, 3]))
        self.assertTrue(torch.equal(selected[2], hidden[1, 2]))

    def test_qwen3_5_trajectory_targets_flatten_interleaved_chunks(self):
        from joynav.model.qwen3_5_trajectory_heads import Qwen35OmegaTrajectoryMixin

        mixin = Qwen35OmegaTrajectoryMixin()
        targets = torch.zeros(2, 3, 8, 3)

        flattened = mixin._normalize_target_actions(targets, selected_count=6)

        self.assertEqual(flattened.shape, (6, 8, 3))

    def test_qwen3_5_continuous_train_args_have_no_parser_collisions(self):
        from transformers import HfArgumentParser

        from joynav.dataset.continuous_vlnn1_action_dataset_args import ContinuousVLNN1ActionDatasetArguments
        from joynav.model.qwen3_5_trajectory_heads import Qwen35OmegaTrajectoryArguments

        parser = HfArgumentParser((ContinuousVLNN1ActionDatasetArguments, Qwen35OmegaTrajectoryArguments))
        data_args, model_args = parser.parse_args_into_dataclasses(
            args=[
                "--video_folder",
                "/tmp/vlnn1",
                "--action_chunk_size",
                "8",
                "--trajectory_horizon",
                "8",
                "--omega_mode",
                "text_align_force_qwen",
            ]
        )

        self.assertEqual(data_args.action_chunk_size, 8)
        self.assertEqual(model_args.trajectory_horizon, 8)
        self.assertEqual(model_args.omega_mode, "text_align_force_qwen")

    def test_qwen3_5_continuous_eval_args_accept_trajectory_overrides(self):
        import sys

        import joynav.eval.eval_habitat as eval_habitat

        old_argv = sys.argv
        try:
            sys.argv = [
                "prog",
                "--evaluator_type",
                "qwen3_5_nextdit_head_sf_omega",
                "--model_type",
                "qwen3_5_nextdit_head_sf_omega",
                "--model_path",
                "/tmp/model",
                "--output_path",
                "/tmp/out",
                "--trajectory_horizon",
                "9",
                "--trajectory_dim",
                "3",
                "--nextdit_dim",
                "8",
                "--nextdit_layers",
                "1",
                "--nextdit_heads",
                "2",
                "--nextdit_kv_heads",
                "2",
                "--nextdit_num_inference_steps",
                "2",
            ]
            args = eval_habitat.parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.trajectory_horizon, 9)
        self.assertEqual(args.trajectory_dim, 3)
        self.assertEqual(args.nextdit_dim, 8)
        self.assertEqual(args.nextdit_num_inference_steps, 2)

    def test_vlnn1_continuous_interleaved_sources_are_multi_turn(self):
        from joynav.dataset.continuous_vlnn1_action_dataset import ContinuousVLNN1ActionInterleavedDataset

        dataset = ContinuousVLNN1ActionInterleavedDataset.__new__(ContinuousVLNN1ActionInterleavedDataset)
        dataset.num_history_frames = 2
        dataset.action_chunk_size = 8
        dataset.action_dim = 3
        dataset.trajectory_stride = 3
        dataset.image_type = "rgb"
        dataset.episodes_per_chunk = 1000
        dataset.action_token = "<|action|>"
        dataset.interleaved_num_chunks = 2
        dataset.conversations = [
            {"from": "human", "value": "You are an autonomous navigation assistant. Your task is to <instruction>."},
            {"from": "gpt", "value": "<|action|>"},
        ]
        dataset.list_data_dict = [(0, 0, 0)]
        chunk = [[0.0, 0.0, 0.0]] + [[float(i), 0.0, 0.0] for i in range(8)]
        dataset.nav_data = [
            {
                "video_folder": "/tmp/vlnn1",
                "path": "matterport3d_d435i/scene",
                "id": 7,
                "chunk_id": "chunk-000",
                "instructions": ["go"],
                "continuous_actions": {"0": chunk, "4": chunk, "8": chunk},
            }
        ]

        source = dataset.prepare_sources(0)

        self.assertEqual(np.asarray(source["continuous_actions"]).shape, (2, 8, 3))
        self.assertEqual(len(source["image"]), 2)
        self.assertEqual(sum(turn["from"] == "gpt" for turn in source["conversations"]), 2)
        self.assertTrue(all(turn["value"] == "<|action|>" for turn in source["conversations"] if turn["from"] == "gpt"))
        self.assertIn("episode_000007_000.jpg", source["image"][0])
        self.assertIn("episode_000007_012.jpg", source["image"][1])

    def test_vlnn1_continuous_collator_preserves_qwen3_5_fields(self):
        from joynav.dataset.continuous_vlnn1_action_dataset import ContinuousVLNN1ActionCollator

        tokenizer = type("Tokenizer", (), {"pad_token_id": 0, "model_max_length": 16})()
        collator = ContinuousVLNN1ActionCollator(tokenizer)
        instances = []
        for seq_len in [4, 6]:
            instances.append(
                {
                    "input_ids": torch.ones(1, seq_len, dtype=torch.long),
                    "labels": torch.ones(1, seq_len, dtype=torch.long),
                    "position_ids": torch.ones(4, 1, seq_len, dtype=torch.long),
                    "mm_token_type_ids": torch.ones(1, seq_len, dtype=torch.long),
                    "pixel_values": torch.ones(1, 3),
                    "image_grid_thw": torch.tensor([[1, 2, 2]]),
                    "pixel_values_videos": None,
                    "video_grid_thw": None,
                    "continuous_actions": torch.ones(8, 3),
                    "select_mask": torch.zeros(seq_len, dtype=torch.bool),
                }
            )
            instances[-1]["select_mask"][-1] = True

        batch = collator(instances)

        self.assertEqual(batch["input_ids"].shape, (2, 6))
        self.assertEqual(batch["position_ids"].shape, (4, 2, 6))
        self.assertEqual(batch["mm_token_type_ids"].shape, (2, 6))
        self.assertEqual(batch["continuous_actions"].shape, (2, 8, 3))
        self.assertEqual(batch["select_mask"].shape, (2, 6))
        self.assertEqual(int(batch["select_mask"].sum()), 2)

    def test_vlnn1_continuous_collator_flattens_interleaved_action_chunks(self):
        from joynav.dataset.continuous_vlnn1_action_dataset import ContinuousVLNN1ActionCollator

        tokenizer = type("Tokenizer", (), {"pad_token_id": 0, "model_max_length": 16})()
        collator = ContinuousVLNN1ActionCollator(tokenizer)
        instances = []
        for seq_len, chunks in [(5, 2), (7, 3)]:
            select_mask = torch.zeros(seq_len, dtype=torch.bool)
            select_mask[:chunks] = True
            instances.append(
                {
                    "input_ids": torch.ones(1, seq_len, dtype=torch.long),
                    "labels": torch.ones(1, seq_len, dtype=torch.long),
                    "position_ids": torch.ones(4, 1, seq_len, dtype=torch.long),
                    "mm_token_type_ids": torch.ones(1, seq_len, dtype=torch.long),
                    "pixel_values": torch.ones(1, 3),
                    "image_grid_thw": torch.tensor([[1, 2, 2]]),
                    "continuous_actions": torch.ones(chunks, 8, 3),
                    "select_mask": select_mask,
                }
            )

        batch = collator(instances)

        self.assertEqual(batch["continuous_actions"].shape, (5, 8, 3))
        self.assertEqual(int(batch["select_mask"].sum()), 5)

    def test_trajectory_to_discrete_uses_step_size_and_stall_stop(self):
        from joynav.eval.qwen3_5_omega_trajectory_head_evaluator import trajectory_to_discrete_actions_3d

        forward = torch.tensor([[[0.25, 0.0, 0.0], [0.50, 0.0, 0.0]]])
        stalled = torch.zeros(1, 8, 3)

        self.assertEqual(
            trajectory_to_discrete_actions_3d(forward, forward_step=0.25, turn_angle_deg=15.0)[0],
            [1, 1],
        )
        self.assertEqual(
            trajectory_to_discrete_actions_3d(stalled, forward_step=0.25, turn_angle_deg=15.0)[0],
            [0],
        )

    def test_trajectory_to_discrete_sanitizes_nonfinite_predictions(self):
        from joynav.eval.qwen3_5_omega_trajectory_head_evaluator import trajectory_to_discrete_actions_3d

        nonfinite = torch.tensor(
            [[[float("nan"), 0.0, float("nan")], [float("inf"), float("-inf"), 0.0]]],
            dtype=torch.float32,
        )

        self.assertEqual(
            trajectory_to_discrete_actions_3d(nonfinite, forward_step=0.25, turn_angle_deg=15.0)[0],
            [0],
        )

    def test_qwen3_5_continuous_omega_script_contracts(self):
        train_mlp = Path("scripts/train/train-qwen3_5-sf-omega-mlp-traj.sh").read_text()
        train_dit = Path("scripts/train/train-qwen3_5-sf-omega-dit-traj.sh").read_text()
        train_nextdit = Path("scripts/train/train-qwen3_5-sf-omega-nextdit-traj.sh").read_text()
        eval_mlp = Path("scripts/eval/eval-qwen3_5-sf-omega-mlp-traj.sh").read_text()
        eval_dit = Path("scripts/eval/eval-qwen3_5-sf-omega-dit-traj.sh").read_text()
        eval_nextdit = Path("scripts/eval/eval-qwen3_5-sf-omega-nextdit-traj.sh").read_text()

        self.assertIn("conda activate ${CONDA_ENV:-qwenvln}", train_mlp)
        self.assertIn("model_type=qwen3_5_mlp_head_sf_omega", train_mlp)
        self.assertIn("dataset_type=${DATASET_TYPE:-continuous_vlnn1_action_noninterleave_sf_omega}", train_mlp)
        self.assertIn("--action_chunk_size ${action_chunk_size}", train_mlp)
        self.assertIn("--trajectory_horizon ${action_chunk_size}", train_mlp)
        self.assertIn("model_type=qwen3_5_dit_head_sf_omega", train_dit)
        self.assertIn("--action_chunk_size ${action_chunk_size}", train_dit)
        self.assertIn("--trajectory_horizon ${action_chunk_size}", train_dit)
        self.assertIn("MODEL_TYPE=${MODEL_TYPE:-qwen3_5_mlp_head_sf_omega}", eval_mlp)
        self.assertIn("EVALUATOR_TYPE=${EVALUATOR_TYPE:-qwen3_5_mlp_head_sf_omega}", eval_mlp)
        self.assertIn("--trajectory_horizon \"$ACTION_CHUNK_NUM\"", eval_mlp)
        self.assertIn("MODEL_TYPE=${MODEL_TYPE:-qwen3_5_dit_head_sf_omega}", eval_dit)
        self.assertIn("EVALUATOR_TYPE=${EVALUATOR_TYPE:-qwen3_5_dit_head_sf_omega}", eval_dit)
        self.assertIn("--trajectory_horizon \"$ACTION_CHUNK_NUM\"", eval_dit)
        self.assertIn("model_type=qwen3_5_nextdit_head_sf_omega", train_nextdit)
        self.assertIn("gpu_ids=${CUDA_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-2,3}}", train_nextdit)
        self.assertIn("NPROC_PER_NODE=${NPROC_PER_NODE:-2}", train_nextdit)
        self.assertIn("num_history=${NUM_HISTORY:-2}", train_nextdit)
        self.assertIn("--nextdit_dim ${nextdit_dim}", train_nextdit)
        self.assertIn("--nextdit_num_inference_steps ${nextdit_num_inference_steps}", train_nextdit)
        self.assertIn("MODEL_TYPE=${MODEL_TYPE:-qwen3_5_nextdit_head_sf_omega}", eval_nextdit)
        self.assertIn("EVALUATOR_TYPE=${EVALUATOR_TYPE:-qwen3_5_nextdit_head_sf_omega}", eval_nextdit)
        self.assertIn("NUM_HISTORY=${NUM_HISTORY:-2}", eval_nextdit)
        self.assertIn("ACTION_CHUNK_NUM=${ACTION_CHUNK_NUM:-8}", eval_nextdit)
        self.assertIn("--nextdit_guidance_scale \"$NEXTDIT_GUIDANCE_SCALE\"", eval_nextdit)


if __name__ == "__main__":
    unittest.main()
