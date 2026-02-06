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
from transformers import AutoProcessor

# Import for Habitat registry side effects — do not remove
from joynav.eval.habitat_extensions import measures

# Import models to register them
import joynav.model
from joynav.utils.dist import *
from joynav.utils.registry import get_component, register_component
from joynav.eval.base_evaluator import BaseEvaluatorArguments

# Register evaluators
from joynav.eval.habitat_vln_evaluator import StreamVLNEvaluator
from joynav.eval.qwen3_vl_dit_evaluator import Qwen3VLDiTEvaluator
from joynav.eval.qwen3_vl_cont_evaluator import Qwen3VLContEvaluator
register_component('evaluator', 'streamvln', StreamVLNEvaluator)
register_component('evaluator', 'qwen3_vl_dit', Qwen3VLDiTEvaluator)
register_component('evaluator', 'qwen3_vl_cont', Qwen3VLContEvaluator)


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

    if hasattr(ip, "size") and isinstance(ip.size, dict):
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
    model = model_class.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map={"": device},
    )
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
    sucs, spls, oss, nes, ep_num = evaluator.eval_action(idx=get_rank())
    ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]

    # import ipdb; ipdb.set_trace()
    dist.all_gather(ep_num_all, ep_num)
    sucs_all = [torch.zeros(ep_num_all[i], dtype=sucs.dtype).to(sucs.device) for i in range(world_size)]
    spls_all = [torch.zeros(ep_num_all[i], dtype=spls.dtype).to(spls.device) for i in range(world_size)]
    oss_all = [torch.zeros(ep_num_all[i], dtype=oss.dtype).to(oss.device) for i in range(world_size)]
    nes_all = [torch.zeros(ep_num_all[i], dtype=nes.dtype).to(nes.device) for i in range(world_size)]
    dist.barrier()
    dist.all_gather(sucs_all, sucs)
    dist.all_gather(spls_all, spls)
    dist.all_gather(oss_all, oss)
    dist.all_gather(nes_all, nes)

    sucs_all = torch.cat(sucs_all, dim=0)
    spls_all = torch.cat(spls_all, dim=0)
    oss_all = torch.cat(oss_all, dim=0)
    nes_all = torch.cat(nes_all, dim=0)
    result_all = {
        "sucs_all": (sum(sucs_all) / len(sucs_all)).item(),
        "spls_all": (sum(spls_all) / len(spls_all)).item(),
        "oss_all": (sum(oss_all) / len(oss_all)).item(),
        "nes_all": (sum(nes_all) / len(nes_all)).item(),
        'length': len(sucs_all),
    }

    print(result_all)
    if get_rank() == 0:
        with open(os.path.join(args.output_path, f'result.json'), 'a') as f:
            f.write(json.dumps(result_all))


if __name__ == '__main__':
    main()
