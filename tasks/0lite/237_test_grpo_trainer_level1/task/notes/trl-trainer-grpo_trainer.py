# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import copy
import inspect
import os
import re
import textwrap
from collections import defaultdict, deque
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Union

import datasets
import torch
import torch.utils.data
import transformers
from accelerate import logging
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_flash_attn_2_available, is_peft_available, is_rich_available

from ..data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template, prepare_multimodal_messages
from ..extras.profiling import profiling_context, profiling_decorator
from ..extras.vllm_client import VLLMClient
from ..import_utils import is_liger_kernel_available, is_vllm_available
from ..models import prepare_deepspeed, prepare_fsdp, prepare_peft_model, unwrap_model_for_generation
from ..models.utils import _ForwardRedirection
from .callbacks import SyncRefModelCallback
from .grpo_config import GRPOConfig
from .utils import (
    RepeatSampler,
    disable_dropout_in_model,
    entropy_from_logits,
    generate_model_card,
    get_comet_experiment_url,
    identity,
    nanmax,
    nanmin,
    nanstd,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
    shuffle_sequence_dict,
    split_pixel_values_by_grid,
    split_tensor_dict,
    truncate_with_protected_tokens,
    unsplit_pixel_values_by_grid,
)


if is_peft_available():
    from peft import PeftConfig, PeftModel

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_wandb_available():
    import wandb


logger = logging.get_logger(__name__)

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class GRPOTrainer(Trainer):
    """

        Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
        paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
        Models](https://huggingface.co/papers/2402.03300).

        Example:

        ```python
        from datasets import load_dataset
        from trl import GRPOTrainer

        dataset = load_dataset("trl-lib/tldr", split="train")


        def reward_func(completions, **kwargs):
            # Dummy reward function that rewards completions with more unique letters.
            return [float(len(set(completion))) for completion in completions]


        trainer = GRPOTrainer(
            model="Qwen/Qwen2-0.5B-Instruct",
            reward_funcs=reward_func,
            train_dataset=dataset,
        )

        trainer.train()
        ```

        Args:
            model (`Union[str, PreTrainedModel]`):
                Model to be trained. Can be either:

                - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
                  path to a *directory* containing model weights saved using
                  [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                  using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
                  `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
            reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
                Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
                functions with the prompts and completions and sum the rewards. Can be either:

                - A single reward function, such as:
                    - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                    path to a *directory* containing model weights saved using
                    [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                    using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                    keyword arguments in `args.model_init_kwargs`.
                    - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                    - A custom reward function: The function is provided with the prompts and the generated completions,
                      plus any additional columns in the dataset. It should return a list of rewards. Custom reward
                      functions can also return `None` when the reward is not applicable to those samples. This is useful
                      for multi-task training where different reward functions apply to different types of samples. When a
                      reward function returns `None` for a sample, that reward function is excluded from the reward
                      calculation for that sample. For more details, see [Using a custom reward
                      function](#using-a-custom-reward-function).

                      The trainer's state is also passed to the reward function. The trainer's state is an instance of
                      [`~transformers.TrainerState`] and can be accessed by accessing the `trainer_state` argument to the
                      reward function's signature.
                - A list of reward functions, where each item can independently be any of the above types. Mixing different
                types within the list (e.g., a string model ID and a custom reward function) is allowed.
            args ([`GRPOConfig`], *optional*):
                Configuration for this trainer. If `None`, a default configuration is used.
            train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
                Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
                ignored. The format of the samples can be either:

                - [Standard](dataset_formats#standard): Each sample contains plain text.
                - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
                  and content).
            eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
                Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
            processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
                Processing class used to process the data. The padding side must be set to "left". If `None`, the
                processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
                padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
                `tokenizer.eos_token` will be used as the default.
            reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*):
                Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

                - A single processing class: Used when `reward_funcs` contains only one reward function.
                - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
                If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
                `None`, the tokenizer for the model is automatically loaded using
                [`~transformers.AutoTokenizer.from_pretrained`]. For elements in `reward_funcs` that are custom reward
                functions (not [`~transformers.PreTrainedModel`]), the corresponding entries in `reward_processing_classes`
                are ignored.
            callbacks (list of [`~transformers.TrainerCallback`], *optional*):
                List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
                in [here](https://huggingface.co/docs/transformers/main_classes/callback).

                If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
                method.
            optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
                A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
                model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
            peft_config ([`~peft.PeftConfig`], *optional*):
                PEFT configuration used to wrap the model. If `None`, the model is not wrapped.

    """

    _tag_names = ['trl', 'grpo']

    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            args: Optional[GRPOConfig] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
            processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
            reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
            peft_config: Optional['PeftConfig'] = None
        ):
        raise NotImplementedError('This function has been masked for testing')

    def _set_signature_columns_if_needed(self):
        raise NotImplementedError('This function has been masked for testing')

    def get_train_dataloader(self):
        raise NotImplementedError('This function has been masked for testing')

    def _get_train_sampler(
            self,
            dataset: Optional[Dataset] = None
        ) -> Sampler:
        raise NotImplementedError('This function has been masked for testing')

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        raise NotImplementedError('This function has been masked for testing')

    @profiling_decorator
    def _get_last_hidden_state(
            self,
            unwrapped_model,
            input_ids,
            attention_mask,
            logits_to_keep,
            pixel_values = None,
            image_grid_thw = None,
            pixel_attention_mask = None,
            image_sizes = None
        ):
        raise NotImplementedError('This function has been masked for testing')

    def get_high_entropy_mask(
            self,
            entropies: torch.Tensor,
            mask: torch.Tensor,
            threshold: float
        ) -> torch.Tensor:
        """

                Returns a binary mask identifying tokens whose entropy exceeds a given quantile threshold.

                Args:
                    entropies (`torch.Tensor`):
                        Tensor of shape (batch_size, seq_len) with per-token entropy values.
                    mask (`torch.Tensor`):
                        Binary mask of the same shape as `entropies`, where `1` indicates valid tokens and `0` padding.
                    threshold (`float`):
                        Quantile threshold between `0.0` and `1.0` to select high-entropy tokens.

                Returns:
                    `torch.Tensor`:
                        Boolean mask of shape (batch_size, seq_len), where `True` indicates tokens with entropy >= threshold
                        and `False` otherwise.

        """
        raise NotImplementedError('This function has been masked for testing')

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
            self,
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            batch_size = None,
            compute_entropy = False,
            pixel_values = None,
            image_grid_thw = None,
            pixel_attention_mask = None,
            image_sizes = None
        ) -> dict[str, Optional[torch.Tensor]]:
        """
        Compute log-probs and (optionally) entropies for each token.
        """
        raise NotImplementedError('This function has been masked for testing')

    def _fix_param_name_to_vllm(
            self,
            name,
            extra_prefixes: Optional[list[str]] = None
        ):
        raise NotImplementedError('This function has been masked for testing')

    def _sync_fsdp1_params_to_vllm(
            self,
            module: nn.Module,
            prefix: str = '',
            visited = None
        ):
        """
        Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM.
        """
        raise NotImplementedError('This function has been masked for testing')

    def _sync_fsdp2_params_to_vllm(self, module: nn.Module):
        raise NotImplementedError('This function has been masked for testing')

    @profiling_decorator
    def _move_model_to_vllm(self):
        raise NotImplementedError('This function has been masked for testing')

    @profiling_decorator
    def _prepare_inputs(
            self,
            generation_batch: dict[str, Union[torch.Tensor, Any]]
        ) -> dict[str, Union[torch.Tensor, Any]]:
        raise NotImplementedError('This function has been masked for testing')

    @profiling_decorator
    def _calculate_rewards(
            self,
            inputs,
            prompts,
            completions,
            completion_ids_list
        ):
        raise NotImplementedError('This function has been masked for testing')

    def _generate_and_score_completions(
            self,
            inputs: list[dict[str, Union[torch.Tensor, Any]]]
        ) -> dict[str, Union[torch.Tensor, Any]]:
        raise NotImplementedError('This function has been masked for testing')

    def compute_liger_loss(self, unwrapped_model, inputs):
        raise NotImplementedError('This function has been masked for testing')

    @profiling_decorator
    def compute_loss(
            self,
            model,
            inputs,
            return_outputs = False,
            num_items_in_batch = None
        ):
        raise NotImplementedError('This function has been masked for testing')

    def _compute_loss(self, model, inputs):
        raise NotImplementedError('This function has been masked for testing')

    def prediction_step(
            self,
            model,
            inputs,
            prediction_loss_only,
            ignore_keys: Optional[list[str]] = None
        ):
        raise NotImplementedError('This function has been masked for testing')

    def log(
            self,
            logs: dict[str, float],
            start_time: Optional[float] = None
        ) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def _save_checkpoint(self, model, trial):
        raise NotImplementedError('This function has been masked for testing')

    def create_model_card(
            self,
            model_name: Optional[str] = None,
            dataset_name: Optional[str] = None,
            tags: Union[str, list[str], None] = None
        ):
        """

                Creates a draft of a model card using the information available to the `Trainer`.

                Args:
                    model_name (`str`, *optional*):
                        Name of the model.
                    dataset_name (`str`, *optional*):
                        Name of the dataset used for training.
                    tags (`str`, `list[str]`, *optional*):
                        Tags to be associated with the model card.

        """
        raise NotImplementedError('This function has been masked for testing')