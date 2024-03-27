#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
import glob
import random
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List
import datetime

import datasets
import evaluate
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.distributed as dist

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    OPTForCausalLM,
    PreTrainedTokenizerFast,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint, seed_worker
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.generic import PaddingStrategy
from transformers.utils.versions import require_version
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils.import_utils import is_datasets_available



from tokenizers import (
        Tokenizer,
        processors
)
from aim.hugging_face import AimCallback


logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    vocab_size:  Optional[int] = field(
        default=None, metadata={"help": "Extended vocabulary size if needed."}
    )
    max_position_embeddings:  Optional[int] = field(
        default=None, metadata={"help": "Maximal sequence length of the input."}
    )
    dropout:  Optional[float] = field(
        default=None, metadata={"help": "User defined dropout."}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    fine_tune_from: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint file path."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


@dataclass
class CustomTrainingArguments(TrainingArguments):
    train_with_sample_size: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "Sample dataset to train."
            )
        },
    )
    aim_exp_name: str = field(
        default="",
        metadata={
            "help": (
                "The name of the experiment, racking by aim."
            )
        },
    )
    aim_repo_dir: str = field(
        default="./",
        metadata={
            "help": (
                "The repo for saving aim files."
            )
        },
    )
    train_with_min_loss_seq: bool = field(
        default=False, 
        metadata={
            "help": (
                "Whether to run training by minimizing only the smiles with minimum loss for one molecule."
                )
            }
        )


class CustomPreTrainedTokenizerFast(PreTrainedTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # overriding the method
    def _pad(
        self,
        encoded_inputs,
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                    )
                # by overriding add dynamic padding for labels also     
                if "labels" in encoded_inputs:
                    encoded_inputs["labels"] = (
                        encoded_inputs["labels"] + [self.pad_token_id] * difference
                    )

                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]

                # by overriding add dynamic padding for labels also   
                if "labels" in encoded_inputs:
                    encoded_inputs["labels"] = (
                        encoded_inputs["labels"] + [self.pad_token_id] * difference
                    )

                encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        return encoded_inputs    


def molecularLoss(loss, batch_size, seq_len):
    block_size = 8

    # Shape (bs, seq_len) 
    loss_molecular = loss.detach().clone().view(batch_size, -1)    

    # Shape (bs), sum loss across the seq_len
    loss_molecular = loss_molecular.sum(-1)

    # Shape (batch_size//block_size, block_size) 
    loss_molecular = loss_molecular.view(-1, block_size)

    # Shape (batch_size//block_size)
    non_zero_idexes = loss_molecular.min(dim=1, keepdim=False)[1]

    # Creating a mask for loss
    mask = torch.zeros(batch_size//block_size, block_size)
    mask[torch.arange(mask.size(0)), non_zero_idexes] = 1

    # Shape loss.shape
    mask_unsqueezed = torch.repeat_interleave(mask.view(-1), seq_len)
    mask_unsqueezed = mask_unsqueezed.to(loss.device)

    # Mask loss
    loss_masked = loss * mask_unsqueezed

    return loss_masked
     

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            # don't shuffle !!!
            dataloader_params["sampler"] = self._get_eval_sampler(train_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


class CustomOPTForCausalLM(OPTForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # overriding the method
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")

            # Shape (bs x seq_len)
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

            batch_size = labels.shape[0]
            seq_len = shift_logits.shape[1]

            # Shape (bs x seq_len)
            loss = molecularLoss(loss, batch_size, seq_len)

            loss = loss.mean()

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.gradient_checkpointing_kwargs={"use_reentrant": False}

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()     

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Output file 
    # if os.path.exists(training_args.output_dir):
    #     print(f"{training_args.output_dir} already exists")
    #     sys.exit(1)

    # Initialize aim_callback
    aim_callback = None
    if training_args.aim_exp_name:
        aim_callback = AimCallback(repo=training_args.aim_repo_dir, experiment=training_args.aim_exp_name)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model's config
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name)

        # Change some config parameters
        config.vocab_size = model_args.vocab_size or config.vocab_size 
        config.dropout = model_args.dropout if model_args.dropout is not None else config.dropout
        config.max_position_embeddings = model_args.max_position_embeddings or config.max_position_embeddings
        config.use_cache = False
        logger.info(f"Overriding config: {config}") 

    if model_args.tokenizer_name:
        tokenizer = Tokenizer.from_file(model_args.tokenizer_name)

        # convert to Transformer's tokenizer
        tokenizer = CustomPreTrainedTokenizerFast(tokenizer_object=tokenizer)
        tokenizer.pad_token = "<pad>"

        if len(tokenizer.get_vocab()) != config.vocab_size:
            i = len(tokenizer.get_vocab())

            while i < config.vocab_size:
                symbol = "madeupword{:03d}".format(i)
            
                tokenizer.add_tokens(symbol)
                i += 1
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )     

    # Dataset
    if os.path.exists(f"{data_args.dataset_name}_tokenized"):
        logger.info(f"Loading local already tokenized data from {data_args.dataset_name}_tokenized ...")
        tokenized_datasets = load_from_disk(
            f"{data_args.dataset_name}_tokenized", 
            keep_in_memory=None
            )
    else:
        data_files = {}
        dataset_args = {
            "cache_dir": os.path.dirname(data_args.dataset_name)
        }

        if data_args.dataset_name:
            data_files["train"] = glob.glob(f"{data_args.dataset_name}/train/00/*.jsonl")
            data_files["valid"] = glob.glob(f"{data_args.dataset_name}/valid/00/*.jsonl")    
        else:
            raise ValueError(
                "Dataset name is required."
            ) 
        logger.info(f"Loading local data from {data_args.dataset_name} ...")
        raw_datasets = load_dataset("json", data_files=data_files, **dataset_args)
        logger.info(raw_datasets["train"], len(raw_datasets["train"]))

        # For debugging
        if training_args.train_with_sample_size:
            logger.info(f"Training with a sample of {training_args.train_with_sample_size}")
            raw_datasets["train"] = raw_datasets["train"].select(range(training_args.train_with_sample_size))
            logger.info("Train example", raw_datasets["train"][0])

            valid_size = min(len(raw_datasets["valid"]), training_args.train_with_sample_size)
            raw_datasets["valid"] = raw_datasets["valid"].select(range(valid_size))
            logger.info("Valid example", raw_datasets["valid"][0]) 

        # Preprocessing the datasets.
        # First we tokenize all the texts.
        if training_args.do_train:
            column_names = list(raw_datasets["train"].features)
        else:
            column_names = list(raw_datasets["valid"].features)

        text_column_name = "text" if "text" in column_names else column_names[0]

        # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
        tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                batch_encodings = tokenizer(
                    examples[text_column_name],
                    return_token_type_ids=False
                    )
            
                # Just make copy of the same list, shift will be applied during the cross-entropy loss
                batch_encodings["labels"] = batch_encodings["input_ids"][:]

            return batch_encodings
            

        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                    load_from_cache_file=False,
                    keep_in_memory=True
                )

        tokenized_datasets.save_to_disk(f"{data_args.dataset_name}_tokenized")           

    train_dataset, eval_dataset = tokenized_datasets["train"], tokenized_datasets["valid"]

    # Log a few random samples from the training set
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")


    data_collator_with_padding = DataCollatorWithPadding(
        tokenizer=tokenizer, 
        padding=True
        )

    if training_args.train_with_min_loss_seq:
        logger.info(f"Train with Custom Loss")  
        model = CustomOPTForCausalLM(config)
    else:  
        logger.info(f"Train with Regular Loss")  
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)


    if model_args.fine_tune_from:
        logger.info(f"Fine-tuning from path {model_args.fine_tune_from}")
        state_dict = torch.load(model_args.fine_tune_from, map_location="cpu")
        model.load_state_dict(state_dict, False)
        del state_dict

    logger.info(model)    
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Training new model from scratch - Total size={n_params} or {n_params/2**20:.2f}M params")

    if training_args.do_eval:
        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                # In case the model returns more than the prediction logits
                logits = logits[0]
            return logits
        
        @torch.no_grad()
        def compute_perplexity(eval_preds, base=2):
            logits, labels = eval_preds
            logits = torch.FloatTensor(logits)
            labels = torch.LongTensor(labels)

            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

            # pad_mask = labels != 1
            # loss = loss * pad_mask  # ignore pad tokens
            # comp_perp = base ** (loss.sum() / pad_mask.sum() / math.log(2))
            comp_perp = base ** (loss.mean() / math.log(2))
            return {"ppl": comp_perp.item()}

    # Initialize our Trainer     
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator_with_padding,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            compute_metrics=compute_perplexity if training_args.do_eval else None,
            callbacks=[aim_callback],
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint:
            logger.info(f"Resumed from checkpoint: {training_args.resume_from_checkpoint}")
            checkpoint = training_args.resume_from_checkpoint
        else:
            logger.info("Training new model from scratch")    
            
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()    


if __name__ == "__main__":
    main()
