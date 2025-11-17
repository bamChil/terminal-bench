# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes."""

import collections
import copy
import os
import unicodedata
from typing import Any, Optional

from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import is_sentencepiece_available, is_sudachi_projection_available, logging


if is_sentencepiece_available():
    import sentencepiece as spm
else:
    spm = None

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "spm_file": "spiece.model"}

SPIECE_UNDERLINE = "▁"


# Copied from transformers.models.bert.tokenization_bert.load_vocab


# Copied from transformers.models.bert.tokenization_bert.whitespace_tokenize


class BertJapaneseTokenizer(PreTrainedTokenizer):
    """

        Construct a BERT tokenizer for Japanese text.

        This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer
        to: this superclass for more information regarding those methods.

        Args:
            vocab_file (`str`):
                Path to a one-wordpiece-per-line vocabulary file.
            spm_file (`str`, *optional*):
                Path to [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm or .model
                extension) that contains the vocabulary.
            do_lower_case (`bool`, *optional*, defaults to `True`):
                Whether to lower case the input. Only has an effect when do_basic_tokenize=True.
            do_word_tokenize (`bool`, *optional*, defaults to `True`):
                Whether to do word tokenization.
            do_subword_tokenize (`bool`, *optional*, defaults to `True`):
                Whether to do subword tokenization.
            word_tokenizer_type (`str`, *optional*, defaults to `"basic"`):
                Type of word tokenizer. Choose from ["basic", "mecab", "sudachi", "jumanpp"].
            subword_tokenizer_type (`str`, *optional*, defaults to `"wordpiece"`):
                Type of subword tokenizer. Choose from ["wordpiece", "character", "sentencepiece",].
            mecab_kwargs (`dict`, *optional*):
                Dictionary passed to the `MecabTokenizer` constructor.
            sudachi_kwargs (`dict`, *optional*):
                Dictionary passed to the `SudachiTokenizer` constructor.
            jumanpp_kwargs (`dict`, *optional*):
                Dictionary passed to the `JumanppTokenizer` constructor.

    """

    vocab_files_names = "VOCAB_FILES_NAMES"

    def __init__(
            self,
            vocab_file,
            spm_file = None,
            do_lower_case = False,
            do_word_tokenize = True,
            do_subword_tokenize = True,
            word_tokenizer_type = 'basic',
            subword_tokenizer_type = 'wordpiece',
            never_split = None,
            unk_token = '[UNK]',
            sep_token = '[SEP]',
            pad_token = '[PAD]',
            cls_token = '[CLS]',
            mask_token = '[MASK]',
            mecab_kwargs = None,
            sudachi_kwargs = None,
            jumanpp_kwargs = None,
            **kwargs
        ):
        raise NotImplementedError('This function has been masked for testing')

    @property
    def do_lower_case(self):
        raise NotImplementedError('This function has been masked for testing')

    def __getstate__(self):
        raise NotImplementedError('This function has been masked for testing')

    def __setstate__(self, state):
        raise NotImplementedError('This function has been masked for testing')

    def _tokenize(self, text):
        raise NotImplementedError('This function has been masked for testing')

    @property
    def vocab_size(self):
        raise NotImplementedError('This function has been masked for testing')

    def get_vocab(self):
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
                adding special tokens. A BERT sequence has the following format:

                - single sequence: `[CLS] X [SEP]`
                - pair of sequences: `[CLS] A [SEP] B [SEP]`

                Args:
                    token_ids_0 (`List[int]`):
                        List of IDs to which the special tokens will be added.
                    token_ids_1 (`List[int]`, *optional*):
                        Optional second list of IDs for sequence pairs.

                Returns:
                    `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

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
                    token_ids_0 (`List[int]`):
                        List of IDs.
                    token_ids_1 (`List[int]`, *optional*):
                        Optional second list of IDs for sequence pairs.
                    already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                        Whether or not the token list is already formatted with special tokens for the model.

                Returns:
                    `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.

        """
        raise NotImplementedError('This function has been masked for testing')

    def save_vocabulary(
            self,
            save_directory: str,
            filename_prefix: Optional[str] = None
        ) -> tuple[str]:
        raise NotImplementedError('This function has been masked for testing')


class MecabTokenizer:
    """
    Runs basic tokenization with MeCab morphological parser.
    """

    def __init__(
            self,
            do_lower_case = False,
            never_split = None,
            normalize_text = True,
            mecab_dic: Optional[str] = 'unidic_lite',
            mecab_option: Optional[str] = None
        ):
        """

                Constructs a MecabTokenizer.

                Args:
                    **do_lower_case**: (*optional*) boolean (default True)
                        Whether to lowercase the input.
                    **never_split**: (*optional*) list of str
                        Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                        [`PreTrainedTokenizer.tokenize`]) List of tokens not to split.
                    **normalize_text**: (*optional*) boolean (default True)
                        Whether to apply unicode normalization to text before tokenization.
                    **mecab_dic**: (*optional*) string (default "ipadic")
                        Name of dictionary to be used for MeCab initialization. If you are using a system-installed dictionary,
                        set this option to `None` and modify *mecab_option*.
                    **mecab_option**: (*optional*) string
                        String passed to MeCab constructor.

        """
        raise NotImplementedError('This function has been masked for testing')

    def tokenize(
            self,
            text,
            never_split = None,
            **kwargs
        ):
        """
        Tokenizes a piece of text.
        """
        raise NotImplementedError('This function has been masked for testing')


class SudachiTokenizer:
    """
    Runs basic tokenization with Sudachi morphological parser.
    """

    def __init__(
            self,
            do_lower_case = False,
            never_split = None,
            normalize_text = True,
            trim_whitespace = False,
            sudachi_split_mode = 'A',
            sudachi_config_path = None,
            sudachi_resource_dir = None,
            sudachi_dict_type = 'core',
            sudachi_projection = None
        ):
        """

                Constructs a SudachiTokenizer.

                Args:
                    **do_lower_case**: (*optional*) boolean (default True)
                        Whether to lowercase the input.
                    **never_split**: (*optional*) list of str
                        Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                        [`PreTrainedTokenizer.tokenize`]) List of tokens not to split.
                    **normalize_text**: (*optional*) boolean (default True)
                        Whether to apply unicode normalization to text before tokenization.
                    **trim_whitespace**: (*optional*) boolean (default False)
                        Whether to trim all whitespace, tab, newline from tokens.
                    **sudachi_split_mode**: (*optional*) string
                        Split mode of sudachi, choose from `["A", "B", "C"]`.
                    **sudachi_config_path**: (*optional*) string
                    **sudachi_resource_dir**: (*optional*) string
                    **sudachi_dict_type**: (*optional*) string
                        dict type of sudachi, choose from `["small", "core", "full"]`.
                    **sudachi_projection**: (*optional*) string
                        Word projection mode of sudachi, choose from `["surface", "normalized", "reading", "dictionary", "dictionary_and_surface", "normalized_and_surface", "normalized_nouns"]`.

        """
        raise NotImplementedError('This function has been masked for testing')

    def tokenize(
            self,
            text,
            never_split = None,
            **kwargs
        ):
        """
        Tokenizes a piece of text.
        """
        raise NotImplementedError('This function has been masked for testing')


class JumanppTokenizer:
    """
    Runs basic tokenization with jumanpp morphological parser.
    """

    def __init__(
            self,
            do_lower_case = False,
            never_split = None,
            normalize_text = True,
            trim_whitespace = False
        ):
        """

                Constructs a JumanppTokenizer.

                Args:
                    **do_lower_case**: (*optional*) boolean (default True)
                        Whether to lowercase the input.
                    **never_split**: (*optional*) list of str
                        Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                        [`PreTrainedTokenizer.tokenize`]) List of tokens not to split.
                    **normalize_text**: (*optional*) boolean (default True)
                        Whether to apply unicode normalization to text before tokenization.
                    **trim_whitespace**: (*optional*) boolean (default False)
                        Whether to trim all whitespace, tab, newline from tokens.

        """
        raise NotImplementedError('This function has been masked for testing')

    def tokenize(
            self,
            text,
            never_split = None,
            **kwargs
        ):
        """
        Tokenizes a piece of text.
        """
        raise NotImplementedError('This function has been masked for testing')


class CharacterTokenizer:
    """
    Runs Character tokenization.
    """

    def __init__(
            self,
            vocab,
            unk_token,
            normalize_text = True
        ):
        """

                Constructs a CharacterTokenizer.

                Args:
                    **vocab**:
                        Vocabulary object.
                    **unk_token**: str
                        A special symbol for out-of-vocabulary token.
                    **normalize_text**: (`optional`) boolean (default True)
                        Whether to apply unicode normalization to text before tokenization.

        """
        raise NotImplementedError('This function has been masked for testing')

    def tokenize(self, text):
        """

                Tokenizes a piece of text into characters.

                For example, `input = "apple""` will return as output `["a", "p", "p", "l", "e"]`.

                Args:
                    text: A single token or whitespace separated tokens.
                        This should have already been passed through *BasicTokenizer*.

                Returns:
                    A list of characters.

        """
        raise NotImplementedError('This function has been masked for testing')


# Copied from transformers.models.bert.tokenization_bert.BasicTokenizer


# Copied from transformers.models.bert.tokenization_bert.WordpieceTokenizer
class WordpieceTokenizer:
    """
    Runs WordPiece tokenization.
    """

    def __init__(
            self,
            vocab,
            unk_token,
            max_input_chars_per_word = 100
        ):
        raise NotImplementedError('This function has been masked for testing')

    def tokenize(self, text):
        """

                Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
                tokenization using the given vocabulary.

                For example, `input = "unaffable"` will return as output `["un", "##aff", "##able"]`.

                Args:
                    text: A single token or whitespace separated tokens. This should have
                        already been passed through *BasicTokenizer*.

                Returns:
                    A list of wordpiece tokens.

        """
        raise NotImplementedError('This function has been masked for testing')




__all__ = ["BertJapaneseTokenizer", "CharacterTokenizer", "MecabTokenizer"]