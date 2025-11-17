# coding=utf-8
# Copyright 2023 The HuggingFace Inc. & Google team. All rights reserved.
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
"""Pix2Struct modeling file"""

import math
from typing import Optional, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, EncoderDecoderCache
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    auto_docstring,
    is_torch_flex_attn_available,
    is_torch_fx_proxy,
    is_torchdynamo_compiling,
    logging,
)
from ...utils.deprecation import deprecate_kwarg
from .configuration_pix2struct import Pix2StructConfig, Pix2StructTextConfig, Pix2StructVisionConfig


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from ...integrations.flex_attention import make_flex_block_causal_mask


logger = logging.get_logger(__name__)

# General docstring


# Adapted from transformers.models.t5.modeling_t5.T5LayerNorm with T5->Pix2Struct


try:
    from apex.normalization import FusedRMSNorm

    Pix2StructLayerNorm = FusedRMSNorm  # noqa

    logger.info("Discovered apex.normalization.FusedRMSNorm - will use it instead of Pix2StructLayerNorm")
except ImportError:
    # using the normal Pix2StructLayerNorm
    pass
except Exception:
    logger.warning("Discovered apex but it failed to load, falling back to Pix2StructLayerNorm")
    pass






# Copied from transformers.models.t5.modeling_t5.T5DenseGatedActDense with T5DenseGatedActDense->Pix2StructVisionMlp,T5Config->Pix2StructVisionConfig,config.d_model->config.hidden_size,dropout_rate->dropout_rate








@auto_docstring
class Pix2StructVisionModel(Pix2StructPreTrainedModel):

    config = "# Type: Pix2StructVisionConfig"
    main_input_name = "flattened_patches"
    supports_gradient_checkpointing = True
    _no_split_modules = ['Pix2StructVisionLayer']

    def __init__(
            self,
            config: Pix2StructVisionConfig
        ):
        raise NotImplementedError('This function has been masked for testing')

    def get_input_embeddings(self):
        raise NotImplementedError('This function has been masked for testing')

    def _prune_heads(
            self,
            heads_to_prune: dict[int, list[int]]
        ) -> None:
        """

                Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
                class PreTrainedModel

        """
        raise NotImplementedError('This function has been masked for testing')

    @auto_docstring
    def forward(
            self,
            flattened_patches: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None
        ) -> Union[tuple, BaseModelOutputWithPooling]:
        """

                flattened_patches (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_channels x patch_height x patch_width)`):
                    Flattened and padded pixel values. These values can be obtained using [`AutoImageProcessor`]. See
                    [`Pix2StructVisionImageProcessor.__call__`] for details. Check the [original
                    paper](https://huggingface.co/papers/2210.03347) (figure 5) for more details.

                Example:

                ```python
                >>> import requests
                >>> from PIL import Image
                >>> from transformers import AutoProcessor, Pix2StructVisionModel

                >>> image_processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
                >>> model = Pix2StructVisionModel.from_pretrained("google/pix2struct-textcaps-base")

                >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
                >>> image = Image.open(requests.get(url, stream=True).raw)

                >>> inputs = image_processor(images=image, return_tensors="pt")
                >>> with torch.no_grad():
                ...     outputs = model(**inputs)

                >>> last_hidden_states = outputs.last_hidden_state
                >>> list(last_hidden_states.shape)
                [1, 2048, 768]
                ```

        """
        raise NotImplementedError('This function has been masked for testing')


# Copied from transformers.models.t5.modeling_t5.T5DenseGatedActDense with T5->Pix2StructText,d_model->hidden_size






# Copied from transformers.models.t5.modeling_t5.T5LayerSelfAttention with T5LayerNorm->Pix2StructLayerNorm,T5Attention->Pix2StructTextAttention,T5LayerSelfAttention->Pix2StructTextLayerSelfAttention,self.SelfAttention->self.attention,config.d_model->config.hidden_size


# Copied from transformers.models.t5.modeling_t5.T5LayerCrossAttention with T5LayerNorm->Pix2StructLayerNorm,T5Attention->Pix2StructTextAttention,T5LayerCrossAttention->Pix2StructTextLayerCrossAttention,self.EncDecAttention->self.attention,config.d_model->config.hidden_size




@auto_docstring(custom_intro='\n    The standalone text decoder of Pix2Struct\n    ')
class Pix2StructTextModel(Pix2StructPreTrainedModel):

    config = "# Type: Pix2StructTextConfig"
    _no_split_modules = ['Pix2StructTextBlock']
    _tied_weights_keys = ['lm_head.weight']
    supports_gradient_checkpointing = True

    def __init__(self, config):
        raise NotImplementedError('This function has been masked for testing')

    def set_input_embeddings(self, new_embeddings):
        raise NotImplementedError('This function has been masked for testing')

    @auto_docstring
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Cache] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            labels: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs
        ) -> Union[tuple[torch.FloatTensor, ...], CausalLMOutputWithCrossAttentions]:
        """

                input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                    Indices of input sequence tokens in the vocabulary. Pix2StructText is a model with relative position
                    embeddings so you should be able to pad the inputs on both the right and the left.

                    Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                    [`PreTrainedTokenizer.__call__`] for detail.

                    [What are input IDs?](../glossary#input-ids)

                    To know more on how to prepare `input_ids` for pretraining take a look a [Pix2StructText
                    Training](./t5#training).
                cross_attn_head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                    Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
                    `[0, 1]`:

                    - 1 indicates the head is **not masked**,
                    - 0 indicates the head is **masked**.

                Example:

                ```python
                >>> from transformers import AutoProcessor, Pix2StructTextModel

                >>> processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
                >>> model = Pix2StructTextModel.from_pretrained("google/pix2struct-textcaps-base")

                >>> inputs = processor(text="Hello, my dog is cute", return_tensors="pt")
                >>> outputs = model(**inputs)
                >>> loss = outputs.loss
                ```

        """
        raise NotImplementedError('This function has been masked for testing')

    def _update_causal_mask(
            self,
            attention_mask: Union[torch.Tensor, 'BlockMask'],
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values: Cache,
            output_attentions: bool = False
        ):
        raise NotImplementedError('This function has been masked for testing')

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask: torch.Tensor,
            sequence_length: int,
            target_length: int,
            dtype: torch.dtype,
            cache_position: torch.Tensor,
            batch_size: int,
            **kwargs
        ):
        """

                Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
                `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

                Args:
                    attention_mask (`torch.Tensor`):
                        A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                        `(batch_size, 1, query_length, key_value_length)`.
                    sequence_length (`int`):
                        The sequence length being processed.
                    target_length (`int`):
                        The target length: when generating with static cache, the mask should be as long as the static cache,
                        to account for the 0 padding, the part of the cache that is not filled yet.
                    dtype (`torch.dtype`):
                        The dtype to use for the 4D attention mask.
                    cache_position (`torch.Tensor`):
                        Indices depicting the position of the input sequence tokens in the sequence.
                    batch_size (`torch.Tensor`):
                        Batch size.

        """
        raise NotImplementedError('This function has been masked for testing')


@auto_docstring(custom_intro='\n    A conditional generation model with a language modeling head. Can be used for sequence generation tasks.\n    ')
class Pix2StructForConditionalGeneration(Pix2StructPreTrainedModel, GenerationMixin):

    config = "# Type: Pix2StructConfig"
    main_input_name = "flattened_patches"
    _tied_weights_keys = ['decoder.lm_head.weight']

    def __init__(
            self,
            config: Pix2StructConfig
        ):
        raise NotImplementedError('This function has been masked for testing')

    def get_input_embeddings(self):
        raise NotImplementedError('This function has been masked for testing')

    def set_input_embeddings(self, new_embeddings):
        raise NotImplementedError('This function has been masked for testing')

    def get_output_embeddings(self) -> nn.Module:
        raise NotImplementedError('This function has been masked for testing')

    def set_output_embeddings(self, new_embeddings):
        raise NotImplementedError('This function has been masked for testing')

    def get_encoder(self):
        raise NotImplementedError('This function has been masked for testing')

    @auto_docstring
    def forward(
            self,
            flattened_patches: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[tuple[tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Cache] = None,
            labels: Optional[torch.LongTensor] = None,
            decoder_inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None
        ) -> Union[tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        """

                flattened_patches (`torch.FloatTensor` of shape `(batch_size, seq_length, hidden_size)`):
                    Flattened pixel patches. the `hidden_size` is obtained by the following formula: `hidden_size` =
                    `num_channels` * `patch_size` * `patch_size`

                    The process of flattening the pixel patches is done by `Pix2StructProcessor`.
                decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
                    Indices of decoder input sequence tokens in the vocabulary.

                    Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                    [`PreTrainedTokenizer.__call__`] for details.

                    [What are decoder input IDs?](../glossary#decoder-input-ids)

                    Pix2StructText uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If
                    `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
                    `past_key_values`).

                    To know more on how to prepare `decoder_input_ids` for pretraining take a look at [Pix2StructText
                    Training](./t5#training).
                decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
                    Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
                    be used by default.
                decoder_head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                    Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0,
                    1]`:

                    - 1 indicates the head is **not masked**,
                    - 0 indicates the head is **masked**.
                cross_attn_head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                    Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
                    `[0, 1]`:

                    - 1 indicates the head is **not masked**,
                    - 0 indicates the head is **masked**.
                labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                    Labels for computing the masked language modeling loss for the decoder.

                Example:

                Inference:

                ```python
                >>> from PIL import Image
                >>> import requests
                >>> from transformers import AutoProcessor, Pix2StructForConditionalGeneration

                >>> processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
                >>> model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-textcaps-base")

                >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
                >>> image = Image.open(requests.get(url, stream=True).raw)

                >>> inputs = processor(images=image, return_tensors="pt")

                >>> # autoregressive generation
                >>> generated_ids = model.generate(**inputs, max_new_tokens=50)
                >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                >>> print(generated_text)
                A stop sign is on a street corner.

                >>> # conditional generation
                >>> text = "A picture of"
                >>> inputs = processor(text=text, images=image, return_tensors="pt", add_special_tokens=False)

                >>> generated_ids = model.generate(**inputs, max_new_tokens=50)
                >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                >>> print(generated_text)
                A picture of a stop sign with a red stop sign
                ```

                Training:

                ```python
                >>> from PIL import Image
                >>> import requests
                >>> from transformers import AutoProcessor, Pix2StructForConditionalGeneration

                >>> processor = AutoProcessor.from_pretrained("google/pix2struct-base")
                >>> model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-base")

                >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
                >>> image = Image.open(requests.get(url, stream=True).raw)
                >>> text = "A stop sign is on the street corner."

                >>> inputs = processor(images=image, return_tensors="pt")
                >>> labels = processor(text=text, return_tensors="pt").input_ids

                >>> # forward pass
                >>> outputs = model(**inputs, labels=labels)
                >>> loss = outputs.loss
                >>> print(f"{loss.item():.5f}")
                5.94282
                ```
        """
        raise NotImplementedError('This function has been masked for testing')


__all__ = [
    "Pix2StructPreTrainedModel",
    "Pix2StructForConditionalGeneration",
    "Pix2StructVisionModel",
    "Pix2StructTextModel",
]