import inspect
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

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
        self.assertIn("export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}", text)
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


if __name__ == "__main__":
    unittest.main()
