import inspect
from pathlib import Path
import sys
import tempfile
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
