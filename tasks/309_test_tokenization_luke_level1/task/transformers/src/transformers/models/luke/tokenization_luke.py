# coding=utf-8
# Copyright Studio-Ouisa and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for LUKE."""

import itertools
import json
import os
from collections.abc import Mapping
from functools import lru_cache
from typing import Optional, Union

import numpy as np
import regex as re

from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    AddedToken,
    BatchEncoding,
    EncodedInput,
    PaddingStrategy,
    TensorType,
    TextInput,
    TextInputPair,
    TruncationStrategy,
    to_py_obj,
)
from ...utils import add_end_docstrings, is_tf_tensor, is_torch_tensor, logging


logger = logging.get_logger(__name__)

EntitySpan = tuple[int, int]
EntitySpanInput = list[EntitySpan]
Entity = str
EntityInput = list[Entity]

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "entity_vocab_file": "entity_vocab.json",
}


ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING = r"""
            return_token_type_ids (`bool`, *optional*):
                Whether to return token type IDs. If left to the default, will return the token type IDs according to
                the specific tokenizer's default, defined by the `return_outputs` attribute.

                [What are token type IDs?](../glossary#token-type-ids)
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the `return_outputs` attribute.

                [What are attention masks?](../glossary#attention-mask)
            return_overflowing_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
                of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead
                of returning overflowing tokens.
            return_special_tokens_mask (`bool`, *optional*, defaults to `False`):
                Whether or not to return special tokens mask information.
            return_offsets_mapping (`bool`, *optional*, defaults to `False`):
                Whether or not to return `(char_start, char_end)` for each token.

                This is only available on fast tokenizers inheriting from [`PreTrainedTokenizerFast`], if using
                Python's tokenizer, this method will raise `NotImplementedError`.
            return_length  (`bool`, *optional*, defaults to `False`):
                Whether or not to return the lengths of the encoded inputs.
            verbose (`bool`, *optional*, defaults to `True`):
                Whether or not to print more information and warnings.
            **kwargs: passed to the `self.tokenize()` method

        Return:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.

              [What are input IDs?](../glossary#input-ids)

            - **token_type_ids** -- List of token type ids to be fed to a model (when `return_token_type_ids=True` or
              if *"token_type_ids"* is in `self.model_input_names`).

              [What are token type IDs?](../glossary#token-type-ids)

            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).

              [What are attention masks?](../glossary#attention-mask)

            - **entity_ids** -- List of entity ids to be fed to a model.

              [What are input IDs?](../glossary#input-ids)

            - **entity_position_ids** -- List of entity positions in the input sequence to be fed to a model.

            - **entity_token_type_ids** -- List of entity token type ids to be fed to a model (when
              `return_token_type_ids=True` or if *"entity_token_type_ids"* is in `self.model_input_names`).

              [What are token type IDs?](../glossary#token-type-ids)

            - **entity_attention_mask** -- List of indices specifying which entities should be attended to by the model
              (when `return_attention_mask=True` or if *"entity_attention_mask"* is in `self.model_input_names`).

              [What are attention masks?](../glossary#attention-mask)

            - **entity_start_positions** -- List of the start positions of entities in the word token sequence (when
              `task="entity_span_classification"`).
            - **entity_end_positions** -- List of the end positions of entities in the word token sequence (when
              `task="entity_span_classification"`).
            - **overflowing_tokens** -- List of overflowing tokens sequences (when a `max_length` is specified and
              `return_overflowing_tokens=True`).
            - **num_truncated_tokens** -- Number of tokens truncated (when a `max_length` is specified and
              `return_overflowing_tokens=True`).
            - **special_tokens_mask** -- List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
              regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
            - **length** -- The length of the inputs (when `return_length=True`)

"""




# Copied from transformers.models.roberta.tokenization_roberta.get_pairs


class LukeTokenizer(PreTrainedTokenizer):
    """

        Constructs a LUKE tokenizer, derived from the GPT-2 tokenizer, using byte-level Byte-Pair-Encoding.

        This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
        be encoded differently whether it is at the beginning of the sentence (without space) or not:

        ```python
        >>> from transformers import LukeTokenizer

        >>> tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")
        >>> tokenizer("Hello world")["input_ids"]
        [0, 31414, 232, 2]

        >>> tokenizer(" Hello world")["input_ids"]
        [0, 20920, 232, 2]
        ```

        You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
        call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

        <Tip>

        When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

        </Tip>

        This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
        this superclass for more information regarding those methods. It also creates entity sequences, namely
        `entity_ids`, `entity_attention_mask`, `entity_token_type_ids`, and `entity_position_ids` to be used by the LUKE
        model.

        Args:
            vocab_file (`str`):
                Path to the vocabulary file.
            merges_file (`str`):
                Path to the merges file.
            entity_vocab_file (`str`):
                Path to the entity vocabulary file.
            task (`str`, *optional*):
                Task for which you want to prepare sequences. One of `"entity_classification"`,
                `"entity_pair_classification"`, or `"entity_span_classification"`. If you specify this argument, the entity
                sequence is automatically created based on the given entity span(s).
            max_entity_length (`int`, *optional*, defaults to 32):
                The maximum length of `entity_ids`.
            max_mention_length (`int`, *optional*, defaults to 30):
                The maximum number of tokens inside an entity span.
            entity_token_1 (`str`, *optional*, defaults to `<ent>`):
                The special token used to represent an entity span in a word token sequence. This token is only used when
                `task` is set to `"entity_classification"` or `"entity_pair_classification"`.
            entity_token_2 (`str`, *optional*, defaults to `<ent2>`):
                The special token used to represent an entity span in a word token sequence. This token is only used when
                `task` is set to `"entity_pair_classification"`.
            errors (`str`, *optional*, defaults to `"replace"`):
                Paradigm to follow when decoding bytes to UTF-8. See
                [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
            bos_token (`str`, *optional*, defaults to `"<s>"`):
                The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

                <Tip>

                When building a sequence using special tokens, this is not the token that is used for the beginning of
                sequence. The token used is the `cls_token`.

                </Tip>

            eos_token (`str`, *optional*, defaults to `"</s>"`):
                The end of sequence token.

                <Tip>

                When building a sequence using special tokens, this is not the token that is used for the end of sequence.
                The token used is the `sep_token`.

                </Tip>

            sep_token (`str`, *optional*, defaults to `"</s>"`):
                The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
                sequence classification or for a text and a question for question answering. It is also used as the last
                token of a sequence built with special tokens.
            cls_token (`str`, *optional*, defaults to `"<s>"`):
                The classifier token which is used when doing sequence classification (classification of the whole sequence
                instead of per-token classification). It is the first token of the sequence when built with special tokens.
            unk_token (`str`, *optional*, defaults to `"<unk>"`):
                The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
                token instead.
            pad_token (`str`, *optional*, defaults to `"<pad>"`):
                The token used for padding, for example when batching sequences of different lengths.
            mask_token (`str`, *optional*, defaults to `"<mask>"`):
                The token used for masking values. This is the token used when training this model with masked language
                modeling. This is the token which the model will try to predict.
            add_prefix_space (`bool`, *optional*, defaults to `False`):
                Whether or not to add an initial space to the input. This allows to treat the leading word just as any
                other word. (LUKE tokenizer detect beginning of words by the preceding space).

    """

    vocab_files_names = "VOCAB_FILES_NAMES"
    model_input_names = ['input_ids', 'attention_mask']

    def __init__(
            self,
            vocab_file,
            merges_file,
            entity_vocab_file,
            task = None,
            max_entity_length = 32,
            max_mention_length = 30,
            entity_token_1 = '<ent>',
            entity_token_2 = '<ent2>',
            entity_unk_token = '[UNK]',
            entity_pad_token = '[PAD]',
            entity_mask_token = '[MASK]',
            entity_mask2_token = '[MASK2]',
            errors = 'replace',
            bos_token = '<s>',
            eos_token = '</s>',
            sep_token = '</s>',
            cls_token = '<s>',
            unk_token = '<unk>',
            pad_token = '<pad>',
            mask_token = '<mask>',
            add_prefix_space = False,
            **kwargs
        ):
        raise NotImplementedError('This function has been masked for testing')

    @property
    def vocab_size(self):
        raise NotImplementedError('This function has been masked for testing')

    def get_vocab(self):
        raise NotImplementedError('This function has been masked for testing')

    def bpe(self, token):
        raise NotImplementedError('This function has been masked for testing')

    def _tokenize(self, text):
        """
        Tokenize a string.
        """
        raise NotImplementedError('This function has been masked for testing')

    def _convert_token_to_id(self, token):
        """
        Converts a token (str) in an id using the vocab.
        """
        raise NotImplementedError('This function has been masked for testing')

    def _convert_id_to_token(self, index):
        """
        Converts an index (integer) in a token (str) using the vocab.
        """
        raise NotImplementedError('This function has been masked for testing')

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (string) in a single string.
        """
        raise NotImplementedError('This function has been masked for testing')

    def build_inputs_with_special_tokens(
            self,
            token_ids_0: list[int],
            token_ids_1: Optional[list[int]] = None
        ) -> list[int]:
        """

                Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
                adding special tokens. A LUKE sequence has the following format:

                - single sequence: `<s> X </s>`
                - pair of sequences: `<s> A </s></s> B </s>`

                Args:
                    token_ids_0 (`list[int]`):
                        List of IDs to which the special tokens will be added.
                    token_ids_1 (`list[int]`, *optional*):
                        Optional second list of IDs for sequence pairs.

                Returns:
                    `list[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

        """
        raise NotImplementedError('This function has been masked for testing')

    def get_special_tokens_mask(
            self,
            token_ids_0: list[int],
            token_ids_1: Optional[list[int]] = None,
            already_has_special_tokens: bool = False
        ) -> list[int]:
        """

                Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
                special tokens using the tokenizer `prepare_for_model` method.

                Args:
                    token_ids_0 (`list[int]`):
                        List of IDs.
                    token_ids_1 (`list[int]`, *optional*):
                        Optional second list of IDs for sequence pairs.
                    already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                        Whether or not the token list is already formatted with special tokens for the model.

                Returns:
                    `list[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.

        """
        raise NotImplementedError('This function has been masked for testing')

    def create_token_type_ids_from_sequences(
            self,
            token_ids_0: list[int],
            token_ids_1: Optional[list[int]] = None
        ) -> list[int]:
        """

                Create a mask from the two sequences passed to be used in a sequence-pair classification task. LUKE does not
                make use of token type ids, therefore a list of zeros is returned.

                Args:
                    token_ids_0 (`list[int]`):
                        List of IDs.
                    token_ids_1 (`list[int]`, *optional*):
                        Optional second list of IDs for sequence pairs.

                Returns:
                    `list[int]`: List of zeros.

        """
        raise NotImplementedError('This function has been masked for testing')

    def prepare_for_tokenization(
            self,
            text,
            is_split_into_words = False,
            **kwargs
        ):
        raise NotImplementedError('This function has been masked for testing')

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def __call__(
            self,
            text: Union[TextInput, list[TextInput]],
            text_pair: Optional[Union[TextInput, list[TextInput]]] = None,
            entity_spans: Optional[Union[EntitySpanInput, list[EntitySpanInput]]] = None,
            entity_spans_pair: Optional[Union[EntitySpanInput, list[EntitySpanInput]]] = None,
            entities: Optional[Union[EntityInput, list[EntityInput]]] = None,
            entities_pair: Optional[Union[EntityInput, list[EntityInput]]] = None,
            add_special_tokens: bool = True,
            padding: Union[bool, str, PaddingStrategy] = False,
            truncation: Union[bool, str, TruncationStrategy] = None,
            max_length: Optional[int] = None,
            max_entity_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: Optional[bool] = False,
            pad_to_multiple_of: Optional[int] = None,
            padding_side: Optional[str] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs
        ) -> BatchEncoding:
        """

                Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
                sequences, depending on the task you want to prepare them for.

                Args:
                    text (`str`, `list[str]`, `list[list[str]]`):
                        The sequence or batch of sequences to be encoded. Each sequence must be a string. Note that this
                        tokenizer does not support tokenization based on pretokenized strings.
                    text_pair (`str`, `list[str]`, `list[list[str]]`):
                        The sequence or batch of sequences to be encoded. Each sequence must be a string. Note that this
                        tokenizer does not support tokenization based on pretokenized strings.
                    entity_spans (`list[tuple[int, int]]`, `list[list[tuple[int, int]]]`, *optional*):
                        The sequence or batch of sequences of entity spans to be encoded. Each sequence consists of tuples each
                        with two integers denoting character-based start and end positions of entities. If you specify
                        `"entity_classification"` or `"entity_pair_classification"` as the `task` argument in the constructor,
                        the length of each sequence must be 1 or 2, respectively. If you specify `entities`, the length of each
                        sequence must be equal to the length of each sequence of `entities`.
                    entity_spans_pair (`list[tuple[int, int]]`, `list[list[tuple[int, int]]]`, *optional*):
                        The sequence or batch of sequences of entity spans to be encoded. Each sequence consists of tuples each
                        with two integers denoting character-based start and end positions of entities. If you specify the
                        `task` argument in the constructor, this argument is ignored. If you specify `entities_pair`, the
                        length of each sequence must be equal to the length of each sequence of `entities_pair`.
                    entities (`list[str]`, `list[list[str]]`, *optional*):
                        The sequence or batch of sequences of entities to be encoded. Each sequence consists of strings
                        representing entities, i.e., special entities (e.g., [MASK]) or entity titles of Wikipedia (e.g., Los
                        Angeles). This argument is ignored if you specify the `task` argument in the constructor. The length of
                        each sequence must be equal to the length of each sequence of `entity_spans`. If you specify
                        `entity_spans` without specifying this argument, the entity sequence or the batch of entity sequences
                        is automatically constructed by filling it with the [MASK] entity.
                    entities_pair (`list[str]`, `list[list[str]]`, *optional*):
                        The sequence or batch of sequences of entities to be encoded. Each sequence consists of strings
                        representing entities, i.e., special entities (e.g., [MASK]) or entity titles of Wikipedia (e.g., Los
                        Angeles). This argument is ignored if you specify the `task` argument in the constructor. The length of
                        each sequence must be equal to the length of each sequence of `entity_spans_pair`. If you specify
                        `entity_spans_pair` without specifying this argument, the entity sequence or the batch of entity
                        sequences is automatically constructed by filling it with the [MASK] entity.
                    max_entity_length (`int`, *optional*):
                        The maximum length of `entity_ids`.

        """
        raise NotImplementedError('This function has been masked for testing')

    def _encode_plus(
            self,
            text: Union[TextInput],
            text_pair: Optional[Union[TextInput]] = None,
            entity_spans: Optional[EntitySpanInput] = None,
            entity_spans_pair: Optional[EntitySpanInput] = None,
            entities: Optional[EntityInput] = None,
            entities_pair: Optional[EntityInput] = None,
            add_special_tokens: bool = True,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
            max_length: Optional[int] = None,
            max_entity_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: Optional[bool] = False,
            pad_to_multiple_of: Optional[int] = None,
            padding_side: Optional[str] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs
        ) -> BatchEncoding:
        raise NotImplementedError('This function has been masked for testing')

    def _batch_encode_plus(
            self,
            batch_text_or_text_pairs: Union[list[TextInput], list[TextInputPair]],
            batch_entity_spans_or_entity_spans_pairs: Optional[Union[list[EntitySpanInput], list[tuple[EntitySpanInput, EntitySpanInput]]]] = None,
            batch_entities_or_entities_pairs: Optional[Union[list[EntityInput], list[tuple[EntityInput, EntityInput]]]] = None,
            add_special_tokens: bool = True,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
            max_length: Optional[int] = None,
            max_entity_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: Optional[bool] = False,
            pad_to_multiple_of: Optional[int] = None,
            padding_side: Optional[str] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs
        ) -> BatchEncoding:
        raise NotImplementedError('This function has been masked for testing')

    def _check_entity_input_format(
            self,
            entities: Optional[EntityInput],
            entity_spans: Optional[EntitySpanInput]
        ):
        raise NotImplementedError('This function has been masked for testing')

    def _create_input_sequence(
            self,
            text: Union[TextInput],
            text_pair: Optional[Union[TextInput]] = None,
            entities: Optional[EntityInput] = None,
            entities_pair: Optional[EntityInput] = None,
            entity_spans: Optional[EntitySpanInput] = None,
            entity_spans_pair: Optional[EntitySpanInput] = None,
            **kwargs
        ) -> tuple[list, list, list, list, list, list]:
        raise NotImplementedError('This function has been masked for testing')

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def _batch_prepare_for_model(
            self,
            batch_ids_pairs: list[tuple[list[int], None]],
            batch_entity_ids_pairs: list[tuple[Optional[list[int]], Optional[list[int]]]],
            batch_entity_token_spans_pairs: list[tuple[Optional[list[tuple[int, int]]], Optional[list[tuple[int, int]]]]],
            add_special_tokens: bool = True,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
            max_length: Optional[int] = None,
            max_entity_length: Optional[int] = None,
            stride: int = 0,
            pad_to_multiple_of: Optional[int] = None,
            padding_side: Optional[str] = None,
            return_tensors: Optional[str] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_length: bool = False,
            verbose: bool = True
        ) -> BatchEncoding:
        """

                Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
                adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
                manages a moving window (with user defined stride) for overflowing tokens


                Args:
                    batch_ids_pairs: list of tokenized input ids or input ids pairs
                    batch_entity_ids_pairs: list of entity ids or entity ids pairs
                    batch_entity_token_spans_pairs: list of entity spans or entity spans pairs
                    max_entity_length: The maximum length of the entity sequence.

        """
        raise NotImplementedError('This function has been masked for testing')

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def prepare_for_model(
            self,
            ids: list[int],
            pair_ids: Optional[list[int]] = None,
            entity_ids: Optional[list[int]] = None,
            pair_entity_ids: Optional[list[int]] = None,
            entity_token_spans: Optional[list[tuple[int, int]]] = None,
            pair_entity_token_spans: Optional[list[tuple[int, int]]] = None,
            add_special_tokens: bool = True,
            padding: Union[bool, str, PaddingStrategy] = False,
            truncation: Union[bool, str, TruncationStrategy] = None,
            max_length: Optional[int] = None,
            max_entity_length: Optional[int] = None,
            stride: int = 0,
            pad_to_multiple_of: Optional[int] = None,
            padding_side: Optional[str] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            prepend_batch_axis: bool = False,
            **kwargs
        ) -> BatchEncoding:
        """

                Prepares a sequence of input id, entity id and entity span, or a pair of sequences of inputs ids, entity ids,
                entity spans so that it can be used by the model. It adds special tokens, truncates sequences if overflowing
                while taking into account the special tokens and manages a moving window (with user defined stride) for
                overflowing tokens. Please Note, for *pair_ids* different than `None` and *truncation_strategy = longest_first*
                or `True`, it is not possible to return overflowing tokens. Such a combination of arguments will raise an
                error.

                Args:
                    ids (`list[int]`):
                        Tokenized input ids of the first sequence.
                    pair_ids (`list[int]`, *optional*):
                        Tokenized input ids of the second sequence.
                    entity_ids (`list[int]`, *optional*):
                        Entity ids of the first sequence.
                    pair_entity_ids (`list[int]`, *optional*):
                        Entity ids of the second sequence.
                    entity_token_spans (`list[tuple[int, int]]`, *optional*):
                        Entity spans of the first sequence.
                    pair_entity_token_spans (`list[tuple[int, int]]`, *optional*):
                        Entity spans of the second sequence.
                    max_entity_length (`int`, *optional*):
                        The maximum length of the entity sequence.

        """
        raise NotImplementedError('This function has been masked for testing')

    def pad(
            self,
            encoded_inputs: Union[BatchEncoding, list[BatchEncoding], dict[str, EncodedInput], dict[str, list[EncodedInput]], list[dict[str, EncodedInput]]],
            padding: Union[bool, str, PaddingStrategy] = True,
            max_length: Optional[int] = None,
            max_entity_length: Optional[int] = None,
            pad_to_multiple_of: Optional[int] = None,
            padding_side: Optional[str] = None,
            return_attention_mask: Optional[bool] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            verbose: bool = True
        ) -> BatchEncoding:
        """

                Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
                in the batch. Padding side (left/right) padding token ids are defined at the tokenizer level (with
                `self.padding_side`, `self.pad_token_id` and `self.pad_token_type_id`) .. note:: If the `encoded_inputs` passed
                are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the result will use the same type unless
                you provide a different tensor type with `return_tensors`. In the case of PyTorch tensors, you will lose the
                specific device of your tensors however.

                Args:
                    encoded_inputs ([`BatchEncoding`], list of [`BatchEncoding`], `dict[str, list[int]]`, `dict[str, list[list[int]]` or `list[dict[str, list[int]]]`):
                        Tokenized inputs. Can represent one input ([`BatchEncoding`] or `dict[str, list[int]]`) or a batch of
                        tokenized inputs (list of [`BatchEncoding`], *dict[str, list[list[int]]]* or *list[dict[str,
                        list[int]]]*) so you can use this method during preprocessing as well as in a PyTorch Dataloader
                        collate function. Instead of `list[int]` you can have tensors (numpy arrays, PyTorch tensors or
                        TensorFlow tensors), see the note above for the return type.
                    padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                         Select a strategy to pad the returned sequences (according to the model's padding side and padding
                         index) among:

                        - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                          sequence if provided).
                        - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                          acceptable input length for the model if that argument is not provided.
                        - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                          lengths).
                    max_length (`int`, *optional*):
                        Maximum length of the returned list and optionally padding length (see above).
                    max_entity_length (`int`, *optional*):
                        The maximum length of the entity sequence.
                    pad_to_multiple_of (`int`, *optional*):
                        If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                        the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
                    padding_side:
                        The side on which the model should have padding applied. Should be selected between ['right', 'left'].
                        Default value is picked from the class attribute of the same name.
                    return_attention_mask (`bool`, *optional*):
                        Whether to return the attention mask. If left to the default, will return the attention mask according
                        to the specific tokenizer's default, defined by the `return_outputs` attribute. [What are attention
                        masks?](../glossary#attention-mask)
                    return_tensors (`str` or [`~utils.TensorType`], *optional*):
                        If set, will return tensors instead of list of python integers. Acceptable values are:

                        - `'tf'`: Return TensorFlow `tf.constant` objects.
                        - `'pt'`: Return PyTorch `torch.Tensor` objects.
                        - `'np'`: Return Numpy `np.ndarray` objects.
                    verbose (`bool`, *optional*, defaults to `True`):
                        Whether or not to print more information and warnings.

        """
        raise NotImplementedError('This function has been masked for testing')

    def _pad(
            self,
            encoded_inputs: Union[dict[str, EncodedInput], BatchEncoding],
            max_length: Optional[int] = None,
            max_entity_length: Optional[int] = None,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            pad_to_multiple_of: Optional[int] = None,
            padding_side: Optional[str] = None,
            return_attention_mask: Optional[bool] = None
        ) -> dict:
        """

                Pad encoded inputs (on left/right and up to predefined length or max length in the batch)


                Args:
                    encoded_inputs:
                        Dictionary of tokenized inputs (`list[int]`) or batch of tokenized inputs (`list[list[int]]`).
                    max_length: maximum length of the returned list and optionally padding length (see below).
                        Will truncate by taking into account the special tokens.
                    max_entity_length: The maximum length of the entity sequence.
                    padding_strategy: PaddingStrategy to use for padding.


                        - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                        - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                        - PaddingStrategy.DO_NOT_PAD: Do not pad
                        The tokenizer padding sides are defined in self.padding_side:


                            - 'left': pads on the left of the sequences
                            - 'right': pads on the right of the sequences
                    pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                        This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                        `>= 7.5` (Volta).
                    padding_side:
                        The side on which the model should have padding applied. Should be selected between ['right', 'left'].
                        Default value is picked from the class attribute of the same name.
                    return_attention_mask:
                        (optional) Set to False to avoid returning attention mask (default: set to model specifics)

        """
        raise NotImplementedError('This function has been masked for testing')

    def save_vocabulary(
            self,
            save_directory: str,
            filename_prefix: Optional[str] = None
        ) -> tuple[str]:
        raise NotImplementedError('This function has been masked for testing')


__all__ = ["LukeTokenizer"]