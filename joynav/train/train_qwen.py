# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from transformers import (
    AutoProcessor,
    Trainer,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
import joynav.model
import joynav.dataset
from joynav.utils.registry import get_component, parse_component_args
from joynav.train.argument import TrainingArguments
from trainer import replace_qwen2_vl_attention_class

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def make_supervised_data_module(processor, data_args) -> dict:
    """Make dataset and collator for supervised fine-tuning."""
    
    data_class = get_component('dataset', data_args.dataset_type)
    train_dataset = data_class(processor, data_args=data_args)
    data_collator = data_class.create_collator(processor.tokenizer)

    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    trainer.accelerator.wait_for_everyone()

    # PEFT model: save only LoRA adapters
    try:
        from peft import PeftModel
        if isinstance(trainer.model, PeftModel):
            if trainer.args.should_save:
                trainer.model.save_pretrained(output_dir)
            return
    except ImportError:
        pass

    if trainer.deepspeed:
        torch.cuda.synchronize()
        print(f"Saving model checkpoint with local_rank {trainer.args.local_rank}")
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if hasattr(model, 'model') and hasattr(model.model, 'geometry_encoder'):
        # Always train the merger
        for n, p in model.model.geometry_encoder.merger.named_parameters():
            p.requires_grad = True

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def print_trainable_parameters(module):
    if hasattr(module, "print_trainable_parameters"):
        module.print_trainable_parameters()
        return

    trainable_params = 0
    all_param = 0
    for _, param in module.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    trainable_percent = 100 * trainable_params / all_param if all_param else 0
    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {trainable_percent:.4f}"
    )


def promote_trainable_parameters_to_fp32(module):
    for param in module.parameters():
        if param.requires_grad and param.dtype in (torch.float16, torch.bfloat16):
            param.data = param.data.float()


def set_model_use_cache(model, use_cache: bool):
    config = getattr(model, "config", None)
    if config is None:
        return

    config.use_cache = use_cache
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        text_config.use_cache = use_cache


def resolve_model_load_dtype(training_args, model_args):
    requested_dtype = str(getattr(model_args, "model_load_dtype", "auto")).lower()
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if requested_dtype != "auto":
        if requested_dtype not in dtype_map:
            raise ValueError(f"Unsupported model_load_dtype: {requested_dtype}")
        return dtype_map[requested_dtype]

    if training_args.bf16 and (
        getattr(model_args, "use_lora", False) or not getattr(model_args, "tune_mm_llm", False)
    ):
        return torch.bfloat16
    return torch.float32


def train(attn_implementation="flash_attention_2"):
    global local_rank

    training_args, data_args, model_args = parse_component_args(
        additional_args_classes=(TrainingArguments,), component_types=['dataset', 'model']
    )

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    model_class = get_component('model', data_args.model_type)
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        dtype=resolve_model_load_dtype(training_args, model_args),
        model_args=model_args
    )
    model.post_update_model()

    # Apply LoRA if enabled
    if model_args.use_lora:
        from peft import LoraConfig, get_peft_model

        target_modules = [m.strip() for m in model_args.lora_target_modules.split(",")]
        modules_to_save = [m.strip() for m in model_args.lora_modules_to_save.split(",") if m.strip()]

        # Keep merger fully trainable if tune_mm_mlp is set
        if model_args.tune_mm_mlp and "merger" not in modules_to_save:
            modules_to_save.append("merger")
        # Keep action_head fully trainable if present
        if hasattr(model, "action_head") and "action_head" not in modules_to_save:
            modules_to_save.append("action_head")
        if hasattr(model, "action_latent") and "action_latent" not in modules_to_save:
            modules_to_save.append("action_latent")
        if getattr(model, "spatial_forcing_projector", None) is not None and "spatial_forcing_projector" not in modules_to_save:
            modules_to_save.append("spatial_forcing_projector")

        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save if modules_to_save else None,
        )
        model = get_peft_model(model, lora_config)
        rank0_print("LoRA applied. Trainable parameters:")
        model.print_trainable_parameters()

    if "qwen3_5" in data_args.model_type:
        data_args.model_type = "qwen3vl"
    elif "qwen3_vl" in data_args.model_type:
        data_args.model_type = "qwen3vl"
    elif "qwen2_5_vl" in data_args.model_type:
        data_args.model_type = "qwen2.5vl"
    else:
        data_args.model_type = "qwen2vl"

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    set_model_use_cache(model, False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if hasattr(model_args, "omega_mode"):
        data_args.omega_mode = model_args.omega_mode
    
    data_module = make_supervised_data_module(processor, data_args=data_args)

    # Resize model embeddings if tokenizer vocabulary has been extended
    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    tokenizer_vocab_size = len(processor.tokenizer)
    
    if tokenizer_vocab_size > model_vocab_size:
        rank0_print(f"Resizing model embeddings from {model_vocab_size} to {tokenizer_vocab_size}")
        model.resize_token_embeddings(tokenizer_vocab_size)

    if not model_args.use_lora:
        set_model(model_args, model)

    promote_trainable_parameters_to_fp32(model)

    if torch.distributed.get_rank() == 0:
        if model_args.use_lora:
            model.print_trainable_parameters()
        else:
            print_trainable_parameters(model.visual)
            print_trainable_parameters(model.model)
    
    trainer = Trainer(
        model=model, processing_class=processor, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    processor.save_pretrained(training_args.output_dir)

    set_model_use_cache(model, True)

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    # ATTN_IMPLEMENTATION env var lets Turing-class GPUs (TITAN RTX, 20xx)
    # fall back to "sdpa"/"eager" since flash-attn-2 needs sm_8.0+.
    train(attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2"))
