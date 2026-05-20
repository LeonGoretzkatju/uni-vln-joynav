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
        self.assertIn(
            "vggt_omega_1b_512.pt",
            text,
        )
        self.assertIn("--sf_target_dim 2048", text)
        self.assertIn("--sf_teacher_layers=${sf_teacher_layers}", text)
        self.assertIn("--spatial_forcing_teacher_patch_size 16", text)
        self.assertIn("spatial_forcing_image_resolution=${SPATIAL_FORCING_IMAGE_RESOLUTION:-512}", text)
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

    def test_spatial_forcing_projector_maps_visual_dim_to_depth_dim(self):
        projector = SpatialForcingProjector(input_dim=1024, target_dim=768, hidden_dim=1536)
        visual_tokens = torch.randn(2, 5, 1024)
        target_tokens = torch.randn(2, 5, 768)

        projected_tokens = projector(visual_tokens)
        loss = cosine_alignment_loss(projected_tokens, target_tokens)

        self.assertEqual(projected_tokens.shape, target_tokens.shape)
        self.assertTrue(torch.isfinite(loss))

    def test_trainable_parameter_printer_supports_plain_torch_modules(self):
        sys.path.insert(0, str(Path("joynav/train").resolve()))
        from joynav.train.train_qwen import print_trainable_parameters

        module = torch.nn.Linear(2, 1)
        print_trainable_parameters(module)


if __name__ == "__main__":
    unittest.main()
