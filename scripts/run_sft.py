#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Supervised fine-tuning script for decoder language models.
"""

import logging
import os
import random
import sys
from typing import Any, Dict, List, Union

import datasets
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed, DataCollatorForLanguageModeling

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    SFTConfig,
    apply_chat_template,
    decontaminate_humaneval,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
from trl import SFTTrainer, setup_chat_format, DataCollatorForCompletionOnlyLM


logger = logging.getLogger(__name__)

os.environ["TMPDIR"] = "/tmp"
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
os.environ["WANDB_PROJECT"] = "arc-improve"


class DataCollatorForAssistantOnlyLM(DataCollatorForLanguageModeling):
    # ref: https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/finetuning/datasets/custom_dataset.py
    def __init__(
        self,
        tokenizer,
        mlm: bool = False,
        ignore_index: int = -100,
        padding_free: bool = False,
    ):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.ignore_index = ignore_index
        self.padding_free = padding_free
        self.eot_token_id = 128009  # <|eot_id|>
        
        # Get system and user token IDs for role detection
        self.system_user_tokens = (
            tokenizer.encode("system")[-1], 
            tokenizer.encode("user")[-1]
        )
        
        # Get assistant header template for masking
        self.assistant_header = tokenizer.encode(
            "<|start_header_id|>assistant<|end_header_id|>",
            add_special_tokens=False
        )
        
    def _mask_header_template(self, labels: torch.Tensor, template: List[int]) -> torch.Tensor:
        """Mask all occurrences of the template sequence in labels tensor."""
        for i in range(len(labels) - len(template)):
            if labels[i:i+len(template)].tolist() == template:
                labels[i:i+len(template)] = self.ignore_index
        return labels
    
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)
        
        for i in range(len(examples)):
            labels = batch["labels"][i]
            input_ids = batch["input_ids"][i]
            
            # Mask BOS token
            labels[0] = self.ignore_index
            
            # Find EOT positions
            eot_positions = (input_ids == self.eot_token_id).nonzero().squeeze(-1)
            
            # Process sections between EOTs
            last_idx = 1
            for pos in eot_positions:
                pos = pos.item()
                # Check role token after last EOT
                if last_idx + 1 < len(input_ids):
                    role_token = input_ids[last_idx + 1]
                    if role_token in self.system_user_tokens:
                        # Mask system/user sections
                        labels[last_idx:pos+1] = self.ignore_index
                last_idx = pos + 1
            
            # Mask assistant headers
            labels = self._mask_header_template(labels, self.assistant_header)
            batch["labels"][i] = labels

        if self.padding_free:  # https://huggingface.co/blog/packing-with-FA2
            attn_mask = batch.pop("attention_mask")
            batch["input_ids"] = batch["input_ids"][attn_mask.bool()].unsqueeze(0)
            batch["position_ids"] = attn_mask.cumsum(1)[attn_mask.bool()].unsqueeze(0) - 1
            batch["labels"] = batch["labels"][attn_mask.bool()].unsqueeze(0)
            batch["labels"][batch["position_ids"] == 0] = self.ignore_index
            
        return batch

def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label"],
    )
    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, data_args)
    tokenizer.pad_token_id = 128004  # "<|finetune_right_pad_id|>"

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path
    # For ChatML we need to add special tokens and resize the embedding layer
    if "<|im_start|>" in tokenizer.chat_template and "gemma-tokenizer-chatml" not in tokenizer.name_or_path:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
        model, tokenizer = setup_chat_format(model, tokenizer)
        model_kwargs = None

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "sft",
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Applying chat template",
    )
    
    collator = DataCollatorForAssistantOnlyLM(tokenizer=tokenizer, padding_free=True)
    
    # response_template = "<|start_header_id|>assistant<|end_header_id|>"
    # response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    # collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, padding_free=True)

    ##########################
    # Decontaminate benchmarks
    ##########################
    # num_raw_train_samples = len(raw_datasets["train"])
    # raw_datasets = raw_datasets.filter(decontaminate_humaneval, batched=True, batch_size=10_000, num_proc=1)
    # num_filtered_train_samples = num_raw_train_samples - len(raw_datasets["train"])
    # logger.info(
    #     f"Decontaminated {num_filtered_train_samples} ({num_filtered_train_samples/num_raw_train_samples * 100:.2f}%) samples from the training set."
    # )

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]
    logger.info(f"Train dataset: length {len(train_dataset)}")
    logger.info(f"Eval dataset: length {len(eval_dataset)}")

    with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
        for index in random.sample(range(len(raw_datasets["train"])), 3):
            logger.info(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")

    ########################
    # Initialize the Trainer
    ########################
    trainer = SFTTrainer(
        model=model,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=training_args.max_seq_length,
        tokenizer=tokenizer,
        packing=data_args.use_packing,
        data_collator=collator if data_args.use_packing else None,
        peft_config=get_peft_config(model_args),
        dataset_kwargs=training_args.dataset_kwargs,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        # "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    main()
