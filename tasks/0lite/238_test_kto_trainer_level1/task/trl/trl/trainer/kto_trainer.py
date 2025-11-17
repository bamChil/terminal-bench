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

import inspect
import os
import random
import textwrap
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState, logging
from accelerate.utils import tqdm
from datasets import Dataset, concatenate_datasets
from torch import autocast
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoModelForCausalLM,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    is_comet_available,
    is_wandb_available,
)
from transformers.trainer_utils import EvalLoopOutput, has_length
from transformers.utils import is_peft_available

from ..data_utils import maybe_apply_chat_template, maybe_extract_prompt, maybe_unpair_preference_dataset
from ..import_utils import is_liger_kernel_available
from ..models import create_reference_model, prepare_deepspeed
from .kto_config import KTOConfig
from .utils import (
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    generate_model_card,
    get_comet_experiment_url,
    log_table_to_comet_experiment,
    pad_to_length,
    peft_module_casting_to_bf16,
    selective_log_softmax,
)


if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearKTOLoss

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


logger = logging.get_logger(__name__)

RUNNING_NAME = "running.pt"


def _get_kl_dataset(
        batch: dict[str, list[Any]]
    ) -> dict[str, list[Any]]:
    """

        Creates mismatched pairs of prompts and completions for the KL dataset by adding a +1 offset to the order of
        completions. For best results, the mismatched outputs y' used to estimate the KL term for a batch should be the
        same set as the matched outputs y used to estimate the rewards in that batch, just paired with different x.

    """
    raise NotImplementedError('This function has been masked for testing')


def _tokenize(
        batch: dict[str, list[Any]],
        tokenizer: 'PreTrainedTokenizer'
    ) -> dict[str, list[Any]]:
    """
    Tokenize a batch from a KTO specific dataset.
    """
    raise NotImplementedError('This function has been masked for testing')


def _process_tokens(
        example: dict[str, Any],
        model: 'PreTrainedModel' = None,
        **kwargs
    ) -> dict:
    """
    Process tokens of a KTO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation in case the prompt +
        completion responses is/are too long. First we truncate the prompt; if we're still too long, we truncate the
        completion.

        We also create the labels for the completion responses, which are of length equal to the sum of the length of the
        prompt and the completion response, with label_pad_token_id for the prompt tokens.

    """
    raise NotImplementedError('This function has been masked for testing')


class KTOTrainer(Trainer):
    """

        Initialize KTOTrainer.

        Args:
            model (`transformers.PreTrainedModel`):
                The model to train, preferably an `AutoModelForSequenceClassification`.
            ref_model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation
                and loss. If no reference model is provided, the trainer will create a reference model with the same
                architecture as the model to be optimized.
            args (`KTOConfig`):
                The arguments to use for training.
            train_dataset (`datasets.Dataset`):
                The dataset to use for training.
            eval_dataset (`datasets.Dataset`):
                The dataset to use for evaluation.
            processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.BaseImageProcessor`], [`~transformers.FeatureExtractionMixin`] or [`~transformers.ProcessorMixin`], *optional*):
                Processing class used to process the data. If provided, will be used to automatically process the inputs
                for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
                reuse the fine-tuned model.
            data_collator (`transformers.DataCollator`, *optional*):
                The data collator to use for training. If None is specified, the default data collator
                (`DPODataCollatorWithPadding`) will be used which will pad the sequences to the maximum length of the
                sequences in the batch, given a dataset of paired sequences.
            model_init (`Callable[[], transformers.PreTrainedModel]`):
                The model initializer to use for training. If None is specified, the default model initializer will be
                used.
            callbacks (`list[transformers.TrainerCallback]`):
                The callbacks to use for training.
            optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
                The optimizer and scheduler to use for training.
            preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
                The function to use to preprocess the logits before computing the metrics.
            peft_config (`dict`, defaults to `None`):
                The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in
                a PEFT model.
            compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
                The function to use to compute the metrics. Must take a `EvalPrediction` and return a dictionary string to
                metric values.
            model_adapter_name (`str`, defaults to `None`):
                Name of the train target PEFT adapter, when using LoRA with multiple adapters.
            ref_adapter_name (`str`, defaults to `None`):
                Name of the reference PEFT adapter, when using LoRA with multiple adapters.

    """

    _tag_names = ['trl', 'kto']

    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module, str] = None,
            ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
            args: KTOConfig = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
            processing_class: Optional[Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]] = None,
            data_collator: Optional[DataCollator] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            peft_config: Optional[dict] = None,
            compute_metrics: Optional[Callable[[EvalLoopOutput], dict]] = None,
            model_adapter_name: Optional[str] = None,
            ref_adapter_name: Optional[str] = None
        ):
        raise NotImplementedError('This function has been masked for testing')

    @contextmanager
    def null_ref_context(self):
        """
        Context manager for handling null reference model (that is, peft adapter manipulation).
        """
        raise NotImplementedError('This function has been masked for testing')

    def get_train_dataloader(self) -> DataLoader:
        """

                Returns the training [`~torch.utils.data.DataLoader`].

                Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.

        """
        raise NotImplementedError('This function has been masked for testing')

    def get_eval_dataloader(
            self,
            eval_dataset: Optional[Dataset] = None
        ) -> DataLoader:
        """

                Returns the evaluation [`~torch.utils.data.DataLoader`].

                Subclass of transformers.src.transformers.trainer.get_eval_dataloader to precompute `ref_log_probs`.

                Args:
                    eval_dataset (`torch.utils.data.Dataset`, *optional*):
                        If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                        by the `model.forward()` method are automatically removed. It must implement `__len__`.

        """
        raise NotImplementedError('This function has been masked for testing')

    def compute_reference_log_probs(self, padded_batch: dict) -> dict:
        """
        Computes log probabilities of the reference model for a single padded batch of a KTO specific dataset.
        """
        raise NotImplementedError('This function has been masked for testing')

    @staticmethod
    def get_batch_logps(
            logits: torch.FloatTensor,
            labels: torch.LongTensor,
            average_log_prob: bool = False,
            label_pad_token_id: int = -100,
            is_encoder_decoder: bool = False
        ) -> torch.FloatTensor:
        """
        Compute the log probabilities of the given labels under the given logits.

                Args:
                    logits:
                        Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
                    labels:
                        Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are
                        ignored. Shape: (batch_size, sequence_length)
                    average_log_prob:
                        If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the
                        log probabilities of the (non-masked) tokens.
                    label_pad_token_id:
                        The label value to ignore when computing log probabilities.
                    is_encoder_decoder:
                        Whether the model is an encoder-decoder model. If True, the labels are not shifted and the logits are
                        assumed to already be aligned with the labels. If False, the labels are shifted to the right by one
                        position, and the logits are assumed to be aligned with the shifted labels.

                Returns:
                    A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the
                    given logits.

        """
        raise NotImplementedError('This function has been masked for testing')

    def forward(
            self,
            model: nn.Module,
            batch: dict[str, Union[list, torch.LongTensor]]
        ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        raise NotImplementedError('This function has been masked for testing')

    def kto_loss(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            policy_KL_logps: torch.FloatTensor,
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps: torch.FloatTensor,
            reference_KL_logps: torch.FloatTensor
        ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the KTO loss for a batch of policy and reference model log probabilities.

                Args:
                    policy_chosen_logps:
                        Log probabilities of the policy model for the chosen responses. Shape: (num(chosen) in batch_size,)
                    policy_rejected_logps:
                        Log probabilities of the policy model for the rejected responses. Shape: (num(rejected) in batch_size,)
                    policy_KL_logps: Log probabilities of the policy model for the KL responses. Shape: (batch_size,)
                    reference_chosen_logps:
                        Log probabilities of the reference model for the chosen responses. Shape: (num(chosen) in batch_size,)
                    reference_rejected_logps:
                        Log probabilities of the reference model for the rejected responses. Shape: (num(rejected) in
                        batch_size,)
                    reference_KL_logps: Log probabilities of the reference model for the KL responses. Shape: (batch_size,)

                Returns:
                    A tuple of four tensors: (losses, chosen_rewards, rejected_rewards, KL). The losses tensor contains the KTO
                    loss for each example in the batch. The chosen_rewards and rejected_rewards tensors contain the rewards for
                    the chosen and rejected responses, respectively. The KL tensor contains the detached KL divergence estimate
                    between the policy and reference models.

        """
        raise NotImplementedError('This function has been masked for testing')

    def _compute_kl_logps(self, model, batch):
        """
        Compute KL log probabilities for a given batch.
        """
        raise NotImplementedError('This function has been masked for testing')

    def _compute_loss_liger(self, model, batch):
        """

                Compute the KTO loss using the Liger-Kernel's LigerFusedLinearKTOLoss.

                Args:
                    model:
                        The policy model used for generating log probabilities and outputs. It could be an encoder-decoder
                        model or a regular language model.
                    batch: A dictionary containing the input data and labels for the batch.

                Returns:
                    A dictionary containing the following keys:
                        - "loss": The computed KTO loss for the batch.
                        - "chosen_logits_sum": Sum of the logits for the chosen responses from the policy model.
                        - "rejected_logits_sum": Sum of the logits for the rejected responses from the policy model.
                        - "chosen_logps": Log probabilities of the chosen responses from the policy model.
                        - "rejected_logps": Log probabilities of the rejected responses from the policy model.
                        - "chosen_rewards": Rewards for the chosen responses.
                        - "rejected_rewards": Rewards for the rejected responses.
                        - "kl": The KL divergence between the policy and reference models (detached).

                    If auxiliary loss is enabled, the dictionary will also include:
                        - "aux_loss": The auxiliary loss from the model outputs.

        """
        raise NotImplementedError('This function has been masked for testing')

    def get_batch_loss_metrics(
            self,
            model,
            batch: dict[str, Union[list, torch.LongTensor]]
        ):
        """
        Compute the KTO loss and other metrics for the given batch of inputs for train or test.
        """
        raise NotImplementedError('This function has been masked for testing')

    def compute_loss(
            self,
            model: Union[PreTrainedModel, nn.Module],
            inputs: dict[str, Union[torch.Tensor, Any]],
            return_outputs = False,
            num_items_in_batch = None
        ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        raise NotImplementedError('This function has been masked for testing')

    def store_metrics(
            self,
            metrics: dict[str, float],
            train_eval: Literal['train', 'eval'] = 'train'
        ) -> None:
        raise NotImplementedError('This function has been masked for testing')

    def _get_train_sampler(
            self,
            dataset: Optional[Dataset] = None
        ) -> Optional[torch.utils.data.Sampler]:
        raise NotImplementedError('This function has been masked for testing')

    def generate_from_model_and_ref(
            self,
            model,
            batch: dict[str, torch.LongTensor]
        ) -> tuple[str, str]:
        """
        Generate samples from the model and reference model for the given batch of inputs.
        """
        raise NotImplementedError('This function has been masked for testing')

    def prediction_step(
            self,
            model: Union[PreTrainedModel, nn.Module],
            inputs: dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[list[str]] = None
        ):
        raise NotImplementedError('This function has been masked for testing')

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[list[str]] = None,
            metric_key_prefix: str = 'eval'
        ) -> EvalLoopOutput:
        """

                Overriding built-in evaluation loop to store metrics for each batch. Prediction/evaluation loop, shared by
                `Trainer.evaluate()` and `Trainer.predict()`.

                Works both with or without labels.

        """
        raise NotImplementedError('This function has been masked for testing')

    def log(
            self,
            logs: dict[str, float],
            start_time: Optional[float] = None
        ) -> None:
        """

                Log `logs` on the various objects watching training, including stored metrics.

                Args:
                    logs (`dict[str, float]`):
                        The values to log.
                    start_time (`float`, *optional*):
                        Start time of the training.

        """
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