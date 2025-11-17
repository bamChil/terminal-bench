# coding=utf-8
# Copyright 2020 The Allen Institute for AI team and The HuggingFace Inc. team.
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

import json
import os
from functools import lru_cache
from typing import Optional

import regex as re

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}




# Copied from transformers.models.roberta.tokenization_roberta.get_pairs


# Copied from transformers.models.roberta.tokenization_roberta.RobertaTokenizer with FacebookAI/roberta-base->allenai/longformer-base-4096, RoBERTa->Longformer all-casing, RobertaTokenizer->LongformerTokenizer
class LongformerTokenizer(PreTrainedTokenizer):
    """

        Constructs a Longformer tokenizer, derived from the GPT-2 tokenizer, using byte-level Byte-Pair-Encoding.

        This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
        be encoded differently whether it is at the beginning of the sentence (without space) or not:

        ```python
        >>> from transformers import LongformerTokenizer

        >>> tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
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
        this superclass for more information regarding those methods.

        Args:
            vocab_file (`str`):
                Path to the vocabulary file.
            merges_file (`str`):
                Path to the merges file.
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
                other word. (Longformer tokenizer detect beginning of words by the preceding space).

    """

    vocab_files_names = "VOCAB_FILES_NAMES"
    model_input_names = ['input_ids', 'attention_mask']

    def __init__(
            self,
            vocab_file,
            merges_file,
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

    def save_vocabulary(
            self,
            save_directory: str,
            filename_prefix: Optional[str] = None
        ) -> tuple[str]:
        raise NotImplementedError('This function has been masked for testing')

    def build_inputs_with_special_tokens(
            self,
            token_ids_0: list[int],
            token_ids_1: Optional[list[int]] = None
        ) -> list[int]:
        """

                Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
                adding special tokens. A Longformer sequence has the following format:

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

                Create a mask from the two sequences passed to be used in a sequence-pair classification task. Longformer does not
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


__all__ = ["LongformerTokenizer"]