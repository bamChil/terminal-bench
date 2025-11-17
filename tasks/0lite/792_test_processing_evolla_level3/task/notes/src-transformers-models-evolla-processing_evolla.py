# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
Processor class for EVOLLA.
"""

import os
from typing import Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import (
    ProcessorMixin,
)
from ..auto import AutoTokenizer


PROTEIN_VALID_KEYS = ["aa_seq", "foldseek", "msa"]


class EvollaProcessor(ProcessorMixin):
    """

        Constructs a EVOLLA processor which wraps a LLama tokenizer and SaProt tokenizer (EsmTokenizer) into a single processor.

        [`EvollaProcessor`] offers all the functionalities of [`EsmTokenizer`] and [`LlamaTokenizerFast`]. See the
        docstring of [`~EvollaProcessor.__call__`] and [`~EvollaProcessor.decode`] for more information.

        Args:
            protein_tokenizer (`EsmTokenizer`):
                An instance of [`EsmTokenizer`]. The protein tokenizer is a required input.
            tokenizer (`LlamaTokenizerFast`, *optional*):
                An instance of [`LlamaTokenizerFast`]. The tokenizer is a required input.
            protein_max_length (`int`, *optional*, defaults to 1024):
                The maximum length of the sequence to be generated.
            text_max_length (`int`, *optional*, defaults to 512):
                The maximum length of the text to be generated.

    """

    attributes = ['protein_tokenizer', 'tokenizer']
    valid_kwargs = ['sequence_max_length']
    protein_tokenizer_class = "AutoTokenizer"
    tokenizer_class = "AutoTokenizer"
    protein_tokenizer_dir_name = "protein_tokenizer"

    def __init__(
            self,
            protein_tokenizer,
            tokenizer = None,
            protein_max_length = 1024,
            text_max_length = 512,
            **kwargs
        ):
        raise NotImplementedError('This function has been masked for testing')

    def process_proteins(
            self,
            proteins,
            protein_max_length = 1024
        ):
        raise NotImplementedError('This function has been masked for testing')

    def process_text(
            self,
            texts,
            text_max_length: int = 512
        ):
        raise NotImplementedError('This function has been masked for testing')

    def __call__(
            self,
            proteins: Optional[Union[list[dict], dict]] = None,
            messages_list: Optional[Union[list[list[dict]], list[dict]]] = None,
            protein_max_length: Optional[int] = None,
            text_max_length: Optional[int] = None,
            **kwargs
        ):
        """
        This method takes batched or non-batched proteins and messages_list and converts them into format that can be used by
                the model.

                Args:
                    proteins (`Union[List[dict], dict]`):
                        A list of dictionaries or a single dictionary containing the following keys:
                            - `"aa_seq"` (`str`) -- The amino acid sequence of the protein.
                            - `"foldseek"` (`str`) -- The foldseek string of the protein.
                    messages_list (`Union[List[List[dict]], List[dict]]`):
                        A list of lists of dictionaries or a list of dictionaries containing the following keys:
                            - `"role"` (`str`) -- The role of the message.
                            - `"content"` (`str`) -- The content of the message.
                    protein_max_length (`int`, *optional*, defaults to 1024):
                        The maximum length of the sequence to be generated.
                    text_max_length (`int`, *optional*, defaults to 512):
                        The maximum length of the text.

                Return:
                    a dict with following keys:
                        - `protein_input_ids` (`torch.Tensor` of shape `(batch_size, sequence_length)`) -- The input IDs for the protein sequence.
                        - `protein_attention_mask` (`torch.Tensor` of shape `(batch_size, sequence_length)`) -- The attention mask for the protein sequence.
                        - `text_input_ids` (`torch.Tensor` of shape `(batch_size, sequence_length)`) -- The input IDs for the text sequence.
                        - `text_attention_mask` (`torch.Tensor` of shape `(batch_size, sequence_length)`) -- The attention mask for the text sequence.

        """
        raise NotImplementedError('This function has been masked for testing')

    def batch_decode(self, *args, **kwargs):
        raise NotImplementedError('This function has been masked for testing')

    def decode(self, *args, **kwargs):
        raise NotImplementedError('This function has been masked for testing')

    def protein_batch_decode(self, *args, **kwargs):
        raise NotImplementedError('This function has been masked for testing')

    def protein_decode(self, *args, **kwargs):
        raise NotImplementedError('This function has been masked for testing')

    def save_pretrained(self, save_directory, **kwargs):
        raise NotImplementedError('This function has been masked for testing')

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path,
            **kwargs
        ):
        raise NotImplementedError('This function has been masked for testing')


__all__ = ["EvollaProcessor"]