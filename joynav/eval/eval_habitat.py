import argparse
import json
import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
import numpy as np
import torch
from transformers import AutoProcessor

# Import for Habitat registry side effects — do not remove
from joynav.eval.habitat_extensions import measures
from joynav.eval.habitat_vln_evaluator import VLNEvaluator
from joynav.model.joynav_qwen3_vl import JoyNav_Qwen3VLForCausalLM
from joynav.model.joynav_qwen2_5_vl import JoyNav_Qwen2_5_VLForCausalLM
from joynav.utils.dist import *


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate JoyNav on Habitat')
    parser.add_argument("--model_type", default='joynav_qwen3_vl', type=str, help="model type: joynav_qwen3_vl / joynav_qwen2_5_vl")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default='configs/vln_r2r.yaml')
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--output_path", type=str, default='./results/r2r/val_unseen')  #!
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--min_pixels", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=392*392)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--use_cache", action="store_true", default=False)

    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int, help='rank')
    parser.add_argument('--gpu', default=0, type=int, help='gpu')
    parser.add_argument('--port', default='2333')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    return parser.parse_args()

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
    args = parse_args()

    init_distributed_mode(args)
    local_rank = args.local_rank
    np.random.seed(local_rank)

    # * 1. Load model and tokenizer.
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = 'left'
    update_processor_pixels(processor, args)

    device = torch.device(f"cuda:{local_rank}")
    if args.model_type == 'joynav_qwen2_5_vl':
        model = JoyNav_Qwen2_5_VLForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            device_map={"": device},
        )
    elif args.model_type == 'joynav_qwen3_vl':
        model = JoyNav_Qwen3VLForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            device_map={"": device},
        )
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    model.eval()
    world_size = get_world_size()

    # * 2. initialize evaluator
    evaluator = VLNEvaluator(
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
