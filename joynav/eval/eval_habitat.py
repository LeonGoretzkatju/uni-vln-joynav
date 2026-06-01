import argparse
import json
import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoProcessor

# Import for Habitat registry side effects — do not remove
from joynav.eval.habitat_extensions import measures

# Import models to register them
import joynav.model
from joynav.utils.dist import *
from joynav.utils.registry import get_component, register_component
from joynav.eval.base_evaluator import BaseEvaluatorArguments

# Register evaluators
from joynav.eval.habitat_vln_evaluator import StreamVLNEvaluator
from joynav.eval.qwen3_vl_dit_head_evaluator import Qwen3VLDiTEvaluator
from joynav.eval.qwen3_vl_mlp_head_evaluator import Qwen3VLMLPEvaluator
from joynav.eval.qwen3_vl_lm_head_evaluator import Qwen3VLLMHeadEvaluator
from joynav.eval.qwen3_vl_lm_head_dyna_evaluator import Qwen3VLLMDynamicRopeEvaluator
from joynav.eval.qwen3_vl_lm_head_sf_evaluator import Qwen3VLSpatialForcingEvaluator
from joynav.eval.qwen3_vl_lm_head_sf_dyna_evaluator import Qwen3VLSpatialForcingDynamicRopeEvaluator
from joynav.eval.qwen3_5_lm_head_sf_omega_evaluator import Qwen3_5OmegaSpatialForcingEvaluator
from joynav.eval.qwen3_5_omega_trajectory_head_evaluator import (
    Qwen3_5OmegaDiTTrajectoryEvaluator,
    Qwen3_5OmegaMLPTrajectoryEvaluator,
    Qwen3_5OmegaNextDiTTrajectoryEvaluator,
    Qwen3_5OmegaOmniEvaluator,
)

register_component('evaluator', 'streamvln', StreamVLNEvaluator)
register_component('evaluator', 'qwen3_vl_dit_head', Qwen3VLDiTEvaluator)
register_component('evaluator', 'qwen3_vl_mlp_head', Qwen3VLMLPEvaluator)
register_component('evaluator', 'qwen3_vl_lm_head', Qwen3VLLMHeadEvaluator)
register_component('evaluator', 'qwen3_vl_lm_head_dyna', Qwen3VLLMDynamicRopeEvaluator)
register_component('evaluator', 'qwen3_vl_lm_head_sf', Qwen3VLSpatialForcingEvaluator)
register_component('evaluator', 'qwen3_vl_lm_head_sf_dyna', Qwen3VLSpatialForcingDynamicRopeEvaluator)
register_component('evaluator', 'qwen3_5_lm_head_sf_omega', Qwen3_5OmegaSpatialForcingEvaluator)
register_component('evaluator', 'qwen3_5_mlp_head_sf_omega', Qwen3_5OmegaMLPTrajectoryEvaluator)
register_component('evaluator', 'qwen3_5_dit_head_sf_omega', Qwen3_5OmegaDiTTrajectoryEvaluator)
register_component('evaluator', 'qwen3_5_nextdit_head_sf_omega', Qwen3_5OmegaNextDiTTrajectoryEvaluator)
register_component('evaluator', 'qwen3_5_omni_head_sf_omega', Qwen3_5OmegaOmniEvaluator)

OMEGA_MODEL_CONFIG_FIELDS = (
    "omega_mode",
    "sf_geometry_encoder_path",
    "sf_target_dim",
    "sf_teacher_layers",
    "sf_align_layers",
    "sf_alpha",
    "sf_add_pos_embed",
)

OMEGA_MODEL_TYPES = {
    "qwen3_5_lm_head_sf_omega",
    "qwen3_5_mlp_head_sf_omega",
    "qwen3_5_dit_head_sf_omega",
    "qwen3_5_nextdit_head_sf_omega",
    "qwen3_5_omni_head_sf_omega",
}

TRAJECTORY_MODEL_CONFIG_FIELDS = (
    "trajectory_horizon",
    "trajectory_dim",
    "action_head_hidden_dim",
    "action_head_loss_weight",
    "propagate_action_head_grad",
    "action_latent_config",
    "action_latent_layers",
    "action_latent_dim",
    "action_latent_heads",
    "action_num_inference_timesteps",
    "nextdit_dim",
    "nextdit_layers",
    "nextdit_heads",
    "nextdit_kv_heads",
    "nextdit_num_inference_steps",
    "nextdit_num_sample_trajs",
    "nextdit_guidance_scale",
    "omni_waypoint_number",
    "omni_action_dim",
    "omni_step_scale",
    "omni_norm_method",
    "omni_coord_scale",
    "omni_flow_hidden_dim",
    "omni_flow_layers",
    "omni_flow_heads",
    "omni_flow_dropout",
    "omni_num_inference_timesteps",
    "omni_num_timestep_buckets",
    "omni_noise_beta_alpha",
    "omni_noise_beta_beta",
    "omni_noise_s",
    "omni_query_action_layers",
    "omni_use_arrive_list",
)


def get_explicit_cli_fields(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    fields = set()
    for arg in argv:
        if arg.startswith("--"):
            fields.add(arg[2:].split("=", 1)[0].replace("-", "_"))
    return fields

def parse_args():
    """Two-stage argument parsing."""
    # Stage 1: Parse selector to determine evaluator type
    base_parser = transformers.HfArgumentParser(BaseEvaluatorArguments)
    selector, remaining = base_parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    
    # Stage 2: Get evaluator-specific args class and parse full arguments
    evaluator_class = get_component('evaluator', selector.evaluator_type)
    evaluator_args_class = evaluator_class.get_argument_class()
    
    full_parser = transformers.HfArgumentParser(evaluator_args_class)
    eval_args = full_parser.parse_args_into_dataclasses()[0]
    
    return eval_args


def rank0_print(*args):
    if get_rank() == 0:
        print(f"Rank {get_rank()}: ", *args)
    else:
        print(*args)

def build_model_config(args, explicit_fields=None):
    config = AutoConfig.from_pretrained(args.model_path)
    if args.model_type in OMEGA_MODEL_TYPES:
        explicit_fields = get_explicit_cli_fields() if explicit_fields is None else explicit_fields
        config_fields = list(OMEGA_MODEL_CONFIG_FIELDS)
        if args.model_type != "qwen3_5_lm_head_sf_omega":
            config_fields += list(TRAJECTORY_MODEL_CONFIG_FIELDS)
        for field_name in config_fields:
            if field_name in explicit_fields and hasattr(args, field_name):
                setattr(config, field_name, getattr(args, field_name))
        for field_name in config_fields:
            if hasattr(config, field_name) and hasattr(args, field_name):
                setattr(args, field_name, getattr(config, field_name))
    return config

def update_processor_pixels(processor, data_args):
    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        if data_args.min_pixels is not None:
            ip.min_pixels = data_args.min_pixels
        if data_args.max_pixels is not None:
            ip.max_pixels = data_args.max_pixels
        rank0_print(f"✅ Updated image_processor min_pixels to {data_args.min_pixels}")
        rank0_print(f"✅ Updated image_processor max_pixels to {data_args.max_pixels}")

    if hasattr(ip, "size") and hasattr(ip.size, "__setitem__"):
        if data_args.min_pixels is not None:
            ip.size["shortest_edge"] = data_args.min_pixels
        if data_args.max_pixels is not None:
            ip.size["longest_edge"] = data_args.max_pixels
        rank0_print(
            f"✅ Updated image_processor size['shortest_edge'] to {data_args.min_pixels}"
        )
        rank0_print(
            f"✅ Updated image_processor size['longest_edge'] to {data_args.max_pixels}"
        )

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")
    return processor

def main():
    # Parse arguments using two-stage parsing
    args = parse_args()

    init_distributed_mode(args)
    local_rank = args.local_rank
    np.random.seed(local_rank)

    rank0_print(f"Evaluator type: {args.evaluator_type}")
    rank0_print(f"Model type: {args.model_type}")

    # * 1. Load model using registry
    model_class = get_component('model', args.model_type)
    rank0_print(f"Loading model from {args.model_path}")
    
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = 'left'
    update_processor_pixels(processor, args)

    device = torch.device(f"cuda:{local_rank}")
    major_cc, _ = torch.cuda.get_device_capability(device)
    eval_dtype = torch.bfloat16 if major_cc >= 8 and torch.cuda.is_bf16_supported() else torch.float16
    attn_implementation = os.environ.get("ATTN_IMPLEMENTATION", "sdpa")
    rank0_print(f"Evaluation dtype: {eval_dtype}, attention: {attn_implementation}")

    model_config = build_model_config(args)
    model = model_class.from_pretrained(
        args.model_path,
        dtype=eval_dtype,
        attn_implementation=attn_implementation,
        device_map={"": device},
        config=model_config,
    )
    if hasattr(model, "post_update_model"):
        model.post_update_model()
    model.eval()
    world_size = get_world_size()

    # * 2. Initialize evaluator using registry
    evaluator_class = get_component('evaluator', args.evaluator_type)
    evaluator = evaluator_class(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        processor=processor,
        epoch=0,
        args=args,
    )

    # * 3. do eval
    sucs, spls, oss, nes, ndtws, ep_num = evaluator.eval_action(idx=get_rank())
    ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]

    # import ipdb; ipdb.set_trace()
    dist.all_gather(ep_num_all, ep_num)
    sucs_all = [torch.zeros(ep_num_all[i], dtype=sucs.dtype).to(sucs.device) for i in range(world_size)]
    spls_all = [torch.zeros(ep_num_all[i], dtype=spls.dtype).to(spls.device) for i in range(world_size)]
    oss_all = [torch.zeros(ep_num_all[i], dtype=oss.dtype).to(oss.device) for i in range(world_size)]
    nes_all = [torch.zeros(ep_num_all[i], dtype=nes.dtype).to(nes.device) for i in range(world_size)]
    ndtws_all = [torch.zeros(ep_num_all[i], dtype=ndtws.dtype).to(ndtws.device) for i in range(world_size)]
    dist.barrier()
    dist.all_gather(sucs_all, sucs)
    dist.all_gather(spls_all, spls)
    dist.all_gather(oss_all, oss)
    dist.all_gather(nes_all, nes)
    dist.all_gather(ndtws_all, ndtws)

    sucs_all = torch.cat(sucs_all, dim=0)
    spls_all = torch.cat(spls_all, dim=0)
    oss_all = torch.cat(oss_all, dim=0)
    nes_all = torch.cat(nes_all, dim=0)
    ndtws_all = torch.cat(ndtws_all, dim=0)
    result_all = {
        "sucs_all": (sum(sucs_all) / len(sucs_all)).item(),
        "spls_all": (sum(spls_all) / len(spls_all)).item(),
        "oss_all": (sum(oss_all) / len(oss_all)).item(),
        "nes_all": (sum(nes_all) / len(nes_all)).item(),
        "ndtws_all": (sum(ndtws_all) / len(ndtws_all)).item(),
        'length': len(sucs_all),
    }

    print(result_all)
    if get_rank() == 0:
        with open(os.path.join(args.output_path, f'result.json'), 'a') as f:
            f.write(json.dumps(result_all))


if __name__ == '__main__':
    main()
