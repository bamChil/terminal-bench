## Task
**Task: Implement a Japanese Text Tokenizer for BERT Models**

**Core Functionalities:**
- Build a comprehensive Japanese text tokenization system that converts raw Japanese text into tokens suitable for BERT model processing
- Support multiple Japanese morphological analyzers (MeCab, Sudachi, Juman++) and subword tokenization methods (WordPiece, SentencePiece, Character-level)
- Handle BERT-specific requirements including special tokens ([CLS], [SEP], [MASK], etc.) and sequence formatting

**Main Features & Requirements:**
- Flexible two-stage tokenization pipeline: word-level tokenization followed by subword tokenization
- Configurable tokenization strategies with support for different Japanese linguistic tools
- Vocabulary management with token-to-ID mapping and unknown token handling
- Serialization support for saving/loading tokenizer state
- Integration with pre-trained tokenizer framework and BERT input formatting

**Key Challenges:**
- Handle Japanese language complexities requiring specialized morphological analysis
- Manage dependencies on external Japanese NLP libraries that may not be serializable
- Balance between different tokenization granularities (word vs. subword vs. character level)
- Ensure compatibility between vocabulary files, SentencePiece models, and tokenization methods
- Maintain consistent special token handling across different tokenization approaches

**NOTE**: 
- This test comes from the `transformers` library, and we have given you the content of this code repository under `/testbed/`, and you need to complete based on this code repository and supplement the files we specify. Remember, all your changes must be in this codebase, and changes that are not in this codebase will not be discovered and tested by us.
- What's more, you need to install `pytest, pytest-timeout, pytest-json-report` in your environment, otherwise our tests won't run and you'll get **ZERO POINTS**!

Your available resources are listed below:
- `/testbed/ace_bench/task/black_links.txt`: Prohibited URLs (all other web resources are allowed)


## Precautions
- You may need to install some of the libraries to support you in accomplishing our task, some of the packages are already pre-installed in your environment, you can check them out yourself via `pip list` etc. For standard installs, just run `pip install <package>`. There's no need to add `--index-url`, the domestic mirrors are already set up unless you have special requirements.
- Please note that when running `pip install <package>`, you should not include the `--force-reinstall` flag, as it may cause pre-installed packages to be reinstalled.
- **IMPORTANT**: While you can install libraries using pip, you should never access the actual implementations in the libraries you install, as the tasks we give you originate from github, and if you look at the contents of the libraries, it could result in you being awarded 0 points directly for alleged cheating. Specifically, you cannot read any files under `/usr/local/lib/python3.x` and its subfolders (here python3.x means any version of python).
- **IMPORTANT**: Your installed python library may contain a real implementation of the task, and you are prohibited from directly calling the library's interface of the same name and pretending to package it as your answer, which will also be detected and awarded 0 points.
- **CRITICAL REQUIREMENT**: After completing the task, pytest will be used to test your implementation. **YOU MUST**:
    - Build proper code hierarchy with correct import relationships shown in **Test Description** (I will give you this later)
    - Match the exact interface shown in the **Interface Description** (I will give you this later)
- I will tell you details about **CRITICAL REQUIREMENT** below.

Your final deliverable should be code under the `/testbed/` directory, and after completing the codebase, we will evaluate your completion and it is important that you complete our tasks with integrity and precision
The final structure is like below, note that  your codebase's structure should match import structure in **Test Description**, which I will tell you later.
```
/testbed                   # all your work should be put into this codebase and match the specific dir structure
├── dir1/
│   ├── file1.py
│   ├── ...
├── dir2/
```

## Test and Interface Descriptions

The **Test Description** will tell you the position of the function or class which we're testing should satisfy.
This means that when you generate some files and complete the functionality we want to test in the files, you need to put these files in the specified directory, otherwise our tests won't be able to import your generated.
For example, if the **Test Description** show you this:
```python
from transformers.models.bert_japanese.tokenization_bert_japanese import BertJapaneseTokenizer
```
This means that we will test one function/class: BertJapaneseTokenizer.
And the defination and implementation of class BertJapaneseTokenizer should be in `/testbed/src/transformers/models/bert_japanese/tokenization_bert_japanese.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
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
        """
        Initialize a BERT tokenizer for Japanese text processing.

        This constructor sets up a comprehensive tokenization pipeline that can handle both word-level
        and subword-level tokenization using various Japanese morphological analyzers and tokenization
        strategies. The tokenizer inherits from PreTrainedTokenizer and provides flexible configuration
        options for different Japanese text processing needs.

        Parameters:
            vocab_file (str): Path to a one-wordpiece-per-line vocabulary file. Required when using
                wordpiece or character subword tokenization.
            spm_file (str, optional): Path to SentencePiece model file (typically .spm or .model extension)
                containing the vocabulary. Required when subword_tokenizer_type is 'sentencepiece'.
                Defaults to None.
            do_lower_case (bool, optional): Whether to convert input text to lowercase during tokenization.
                Only effective when do_word_tokenize is True. Defaults to False.
            do_word_tokenize (bool, optional): Whether to perform word-level tokenization before subword
                tokenization. Defaults to True.
            do_subword_tokenize (bool, optional): Whether to perform subword tokenization after word
                tokenization. Defaults to True.
            word_tokenizer_type (str, optional): Type of word tokenizer to use. Must be one of:
                'basic', 'mecab', 'sudachi', 'jumanpp'. Defaults to 'basic'.
            subword_tokenizer_type (str, optional): Type of subword tokenizer to use. Must be one of:
                'wordpiece', 'character', 'sentencepiece'. Defaults to 'wordpiece'.
            never_split (list of str, optional): List of tokens that should never be split during
                tokenization. Defaults to None.
            unk_token (str, optional): Special token to represent unknown/out-of-vocabulary words.
                Defaults to '[UNK]'.
            sep_token (str, optional): Special token used to separate sequences. Defaults to '[SEP]'.
            pad_token (str, optional): Special token used for padding sequences to equal length.
                Defaults to '[PAD]'.
            cls_token (str, optional): Special token used at the beginning of sequences for classification
                tasks. Defaults to '[CLS]'.
            mask_token (str, optional): Special token used for masked language modeling tasks.
                Defaults to '[MASK]'.
            mecab_kwargs (dict, optional): Additional keyword arguments passed to MecabTokenizer
                constructor when word_tokenizer_type is 'mecab'. Defaults to None.
            sudachi_kwargs (dict, optional): Additional keyword arguments passed to SudachiTokenizer
                constructor when word_tokenizer_type is 'sudachi'. Defaults to None.
            jumanpp_kwargs (dict, optional): Additional keyword arguments passed to JumanppTokenizer
                constructor when word_tokenizer_type is 'jumanpp'. Defaults to None.
            **kwargs: Additional keyword arguments passed to the parent PreTrainedTokenizer class.

        Raises:
            ValueError: If the specified vocab_file or spm_file does not exist, or if invalid
                word_tokenizer_type or subword_tokenizer_type values are provided.
            ModuleNotFoundError: If required dependencies for specific tokenizers (fugashi for MeCab,
                sudachipy for Sudachi, rhoknp for Juman++, sentencepiece for SentencePiece) are not
                installed.

        Notes:
            - When using 'sentencepiece' as subword_tokenizer_type, spm_file must be provided and
              vocab_file is not used.
            - For morphological analyzers (MeCab, Sudachi, Juman++), additional system dependencies
              and dictionaries may need to be installed separately.
            - The tokenizer supports serialization but morphological analyzer instances are recreated
              during deserialization to handle non-serializable components.
            - Special tokens are automatically handled and can be accessed through properties like
              cls_token_id, sep_token_id, etc.
        """
        <your code>
...
```

In order to implement this functionality, some additional libraries etc. are often required, I don't restrict you to any libraries, you need to think about what dependencies you might need and fetch and install and call them yourself. The only thing is that you **MUST** fulfill the input/output format described by this interface, otherwise the test will not pass and you will get zero points for this feature.
And note that there may be not only one **Interface Description**, you should match all **Interface Description {n}**

### Test Description 1
Below is **Test Description 1**
```python
from transformers.models.bert_japanese.tokenization_bert_japanese import BertJapaneseTokenizer
from transformers.models.bert_japanese.tokenization_bert_japanese import CharacterTokenizer
from transformers.models.bert_japanese.tokenization_bert_japanese import JumanppTokenizer
from transformers.models.bert_japanese.tokenization_bert_japanese import MecabTokenizer
from transformers.models.bert_japanese.tokenization_bert_japanese import SudachiTokenizer
from transformers.models.bert_japanese.tokenization_bert_japanese import WordpieceTokenizer
```

### Interface Description 1
Below is **Interface Description 1** for file: src-transformers-models-bert_japanese-tokenization_bert_japanese.py

This file contains 6 top-level interface(s) that need to be implemented.

```python
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
        """
        Initialize a BERT tokenizer for Japanese text processing.

        This constructor sets up a comprehensive tokenization pipeline that can handle both word-level
        and subword-level tokenization using various Japanese morphological analyzers and tokenization
        strategies. The tokenizer inherits from PreTrainedTokenizer and provides flexible configuration
        options for different Japanese text processing needs.

        Parameters:
            vocab_file (str): Path to a one-wordpiece-per-line vocabulary file. Required when using
                wordpiece or character subword tokenization.
            spm_file (str, optional): Path to SentencePiece model file (typically .spm or .model extension)
                containing the vocabulary. Required when subword_tokenizer_type is 'sentencepiece'.
                Defaults to None.
            do_lower_case (bool, optional): Whether to convert input text to lowercase during tokenization.
                Only effective when do_word_tokenize is True. Defaults to False.
            do_word_tokenize (bool, optional): Whether to perform word-level tokenization before subword
                tokenization. Defaults to True.
            do_subword_tokenize (bool, optional): Whether to perform subword tokenization after word
                tokenization. Defaults to True.
            word_tokenizer_type (str, optional): Type of word tokenizer to use. Must be one of:
                'basic', 'mecab', 'sudachi', 'jumanpp'. Defaults to 'basic'.
            subword_tokenizer_type (str, optional): Type of subword tokenizer to use. Must be one of:
                'wordpiece', 'character', 'sentencepiece'. Defaults to 'wordpiece'.
            never_split (list of str, optional): List of tokens that should never be split during
                tokenization. Defaults to None.
            unk_token (str, optional): Special token to represent unknown/out-of-vocabulary words.
                Defaults to '[UNK]'.
            sep_token (str, optional): Special token used to separate sequences. Defaults to '[SEP]'.
            pad_token (str, optional): Special token used for padding sequences to equal length.
                Defaults to '[PAD]'.
            cls_token (str, optional): Special token used at the beginning of sequences for classification
                tasks. Defaults to '[CLS]'.
            mask_token (str, optional): Special token used for masked language modeling tasks.
                Defaults to '[MASK]'.
            mecab_kwargs (dict, optional): Additional keyword arguments passed to MecabTokenizer
                constructor when word_tokenizer_type is 'mecab'. Defaults to None.
            sudachi_kwargs (dict, optional): Additional keyword arguments passed to SudachiTokenizer
                constructor when word_tokenizer_type is 'sudachi'. Defaults to None.
            jumanpp_kwargs (dict, optional): Additional keyword arguments passed to JumanppTokenizer
                constructor when word_tokenizer_type is 'jumanpp'. Defaults to None.
            **kwargs: Additional keyword arguments passed to the parent PreTrainedTokenizer class.

        Raises:
            ValueError: If the specified vocab_file or spm_file does not exist, or if invalid
                word_tokenizer_type or subword_tokenizer_type values are provided.
            ModuleNotFoundError: If required dependencies for specific tokenizers (fugashi for MeCab,
                sudachipy for Sudachi, rhoknp for Juman++, sentencepiece for SentencePiece) are not
                installed.

        Notes:
            - When using 'sentencepiece' as subword_tokenizer_type, spm_file must be provided and
              vocab_file is not used.
            - For morphological analyzers (MeCab, Sudachi, Juman++), additional system dependencies
              and dictionaries may need to be installed separately.
            - The tokenizer supports serialization but morphological analyzer instances are recreated
              during deserialization to handle non-serializable components.
            - Special tokens are automatically handled and can be accessed through properties like
              cls_token_id, sep_token_id, etc.
        """
        <your code>

    @property
    def do_lower_case(self):
        """
        Get the lowercase configuration setting for the tokenizer.

        This property provides access to the tokenizer's case handling configuration,
        indicating whether text should be converted to lowercase during tokenization.

        Returns:
            bool: True if the tokenizer is configured to convert input text to lowercase
                during tokenization, False otherwise. This value is determined by the
                `do_lower_case` parameter passed during tokenizer initialization.

        Note:
            This property returns the value of the internal `lower_case` attribute,
            which stores the lowercase configuration. The lowercase conversion is
            applied during word tokenization when `do_word_tokenize` is True and
            affects how the underlying word tokenizer (BasicTokenizer, MecabTokenizer,
            SudachiTokenizer, or JumanppTokenizer) processes the input text.
        """
        <your code>

    def __getstate__(self):
        """
        Prepare the tokenizer object for serialization by creating a state dictionary.

        This method is part of Python's pickle protocol and is automatically called when the tokenizer
        object needs to be serialized (e.g., when saving to disk or sending over a network). It creates
        a copy of the object's state while handling special cases for certain word tokenizers that
        cannot be directly pickled.

        Returns:
            dict: A dictionary containing the object's state suitable for serialization. For tokenizers
                using "mecab", "sudachi", or "jumanpp" word tokenizers, the word_tokenizer attribute
                is excluded from the state since these tokenizers contain non-serializable components
                and will be reconstructed during deserialization.

        Important notes:
            - This method works in conjunction with __setstate__ to provide complete pickle support
            - The word_tokenizer is deliberately excluded for "mecab", "sudachi", and "jumanpp" 
              tokenizer types because they contain references to external libraries or system 
              resources that cannot be pickled
            - The excluded word_tokenizer will be automatically recreated when the object is 
              unpickled using the stored configuration parameters
            - For other tokenizer types ("basic"), the word_tokenizer is preserved in the state
              since it can be safely serialized
        """
        <your code>

    def __setstate__(self, state):
        """
        Restore the object's state from a pickled representation.

        This method is part of Python's pickle protocol and is automatically called during
        the unpickling process to reconstruct a BertJapaneseTokenizer object. It restores
        the object's attributes from the provided state dictionary and reinitializes any
        word tokenizers that were excluded during pickling.

        Parameters:
            state (dict): A dictionary containing the object's state that was previously
                saved by __getstate__(). This includes all instance attributes except
                for certain word tokenizers that need to be recreated.

        Returns:
            None: This method modifies the object in-place and does not return a value.

        Important notes:
            - This method is automatically called by Python's pickle module during
              deserialization and should not be called directly by users.
            - Word tokenizers of types "mecab", "sudachi", and "jumanpp" are recreated
              from their respective configuration parameters (mecab_kwargs, sudachi_kwargs,
              jumanpp_kwargs) since these objects cannot be directly pickled.
            - The method assumes that the state dictionary contains all necessary
              configuration parameters to properly reconstruct the tokenizer.
            - If the word_tokenizer_type is not one of the special cases ("mecab",
              "sudachi", "jumanpp"), the word tokenizer is expected to be preserved
              in the state and will be restored automatically.
        """
        <your code>

    def _tokenize(self, text):
        """
        Tokenizes a piece of text into tokens using the configured word and subword tokenizers.

        This is the core tokenization method that applies the tokenization pipeline configured for this
        BertJapaneseTokenizer instance. The method performs tokenization in two stages:
        1. Word-level tokenization (if enabled) using the configured word tokenizer
        2. Subword-level tokenization (if enabled) using the configured subword tokenizer

        Parameters:
            text (str): The input text to be tokenized. This should be a string containing
                Japanese text that needs to be broken down into tokens.

        Returns:
            List[str]: A list of tokens representing the tokenized input text. The tokens
                are strings that result from the applied tokenization pipeline. If both
                word and subword tokenization are enabled, the returned tokens will be
                the result of subword tokenization applied to each word token. If only
                word tokenization is enabled, returns the word tokens. If neither is
                enabled, returns the original text as a single-item list.

        Important Notes:
            - The tokenization behavior depends on the tokenizer configuration set during
              initialization (do_word_tokenize, do_subword_tokenize, word_tokenizer_type,
              subword_tokenizer_type)
            - When word tokenization is enabled, special tokens from self.all_special_tokens
              are passed to the word tokenizer's never_split parameter to prevent splitting
            - The method processes tokens sequentially through the pipeline: first word
              tokenization (if enabled), then subword tokenization (if enabled)
            - This is an internal method (indicated by the underscore prefix) and is typically
              called by the public tokenize method of the parent PreTrainedTokenizer class
        """
        <your code>

    @property
    def vocab_size(self):
        """
        Get the size of the tokenizer's vocabulary.

        This property returns the total number of tokens in the tokenizer's vocabulary,
        which varies depending on the subword tokenizer type being used.

        Returns:
            int: The size of the vocabulary. For SentencePiece tokenizers, this is the
                length of the SentencePiece model. For other tokenizer types (wordpiece,
                character), this is the length of the vocab dictionary.

        Notes:
            The vocabulary size calculation differs based on the subword_tokenizer_type:
            - For "sentencepiece": Returns len(self.subword_tokenizer.sp_model)
            - For other types ("wordpiece", "character"): Returns len(self.vocab)

            This property is commonly used for initializing model embedding layers
            and ensuring compatibility between tokenizer and model vocabularies.
        """
        <your code>

    def get_vocab(self):
        """
        Retrieve the vocabulary dictionary used by the tokenizer.

        This method returns the complete vocabulary mapping from tokens to their corresponding IDs,
        including both the base vocabulary and any additional tokens that have been added to the
        tokenizer (such as special tokens or tokens added during fine-tuning).

        Returns:
            dict: A dictionary mapping tokens (str) to their token IDs (int). The dictionary
                contains all tokens from the base vocabulary plus any additional tokens from
                the added_tokens_encoder.

        Notes:
            - For SentencePiece-based tokenizers (when subword_tokenizer_type is "sentencepiece"),
              the vocabulary is dynamically generated by converting token IDs back to tokens
              using the SentencePiece model, then merged with added tokens.
            - For other tokenizer types (wordpiece, character), the method returns the
              pre-loaded vocabulary dictionary merged with added tokens.
            - The returned dictionary includes special tokens like [CLS], [SEP], [MASK], etc.
              if they were added to the tokenizer.
            - This method is commonly used for vocabulary inspection, saving tokenizer state,
              or when you need to access the complete token-to-ID mapping.
        """
        <your code>

    def _convert_token_to_id(self, token):
        """
        Converts a token (str) in an id using the vocab.
        """
        <your code>

    def _convert_id_to_token(self, index):
        """
        Converts an index (integer) in a token (str) using the vocab.
        """
        <your code>

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (string) in a single string.
        """
        <your code>

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
        <your code>

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
        <your code>

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None
    ) -> tuple[str]:
        """
        Save the tokenizer's vocabulary files to the specified directory.

        This method saves the vocabulary used by the tokenizer to disk, handling different
        vocabulary formats based on the subword tokenizer type. For SentencePiece tokenizers,
        it saves the serialized model proto, while for other tokenizers it saves a text-based
        vocabulary file with one token per line.

        Args:
            save_directory (str): The directory path where the vocabulary file(s) should be saved.
                If this is a directory path, the vocabulary file will be saved inside it with
                the appropriate filename. If this is not a directory, it will be treated as
                a filename prefix.
            filename_prefix (Optional[str], optional): An optional prefix to add to the
                vocabulary filename. If provided, it will be prepended to the default filename
                with a hyphen separator. Defaults to None.

        Returns:
            tuple[str]: A tuple containing the path(s) to the saved vocabulary file(s).
                Currently returns a single-element tuple with the path to the saved vocabulary file.

        Notes:
            - For SentencePiece tokenizers (subword_tokenizer_type == "sentencepiece"), saves
              a binary .model file containing the serialized SentencePiece model
            - For other tokenizer types, saves a text file with one vocabulary token per line,
              sorted by token index
            - The method validates vocabulary consistency for non-SentencePiece tokenizers and
              logs a warning if vocabulary indices are not consecutive
            - The actual filename used depends on the subword_tokenizer_type: "spiece.model"
              for SentencePiece or "vocab.txt" for others
        """
        <your code>

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
        <your code>

    def tokenize(
        self,
        text,
        never_split = None,
        **kwargs
    ):
        """
        Tokenizes a piece of text.
        """
        <your code>

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
        <your code>

    def tokenize(
        self,
        text,
        never_split = None,
        **kwargs
    ):
        """
        Tokenizes a piece of text.
        """
        <your code>

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
        <your code>

    def tokenize(
        self,
        text,
        never_split = None,
        **kwargs
    ):
        """
        Tokenizes a piece of text.
        """
        <your code>

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
        <your code>

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
        <your code>

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
        """
        Initialize a WordpieceTokenizer instance for subword tokenization.

        This constructor sets up a WordPiece tokenizer that uses a greedy longest-match-first algorithm
        to break down tokens into subword pieces based on a given vocabulary. WordPiece tokenization
        is commonly used in BERT and similar transformer models to handle out-of-vocabulary words
        by splitting them into known subword units.

        Args:
            vocab (dict): A dictionary mapping vocabulary tokens to their corresponding IDs.
                The vocabulary should contain subword pieces including continuation tokens
                (typically prefixed with "##" in BERT-style tokenization).
            unk_token (str): The unknown token string used to replace tokens that cannot be
                tokenized using the vocabulary. Commonly "[UNK]" in BERT models.
            max_input_chars_per_word (int, optional): Maximum number of characters allowed
                per input word. Words exceeding this limit will be replaced with the unknown
                token. Defaults to 100. This prevents extremely long tokens from causing
                performance issues during tokenization.

        Returns:
            None: This is a constructor method.

        Notes:
            - The tokenizer expects input text to have already been processed by a basic tokenizer
              (whitespace tokenization, punctuation splitting, etc.)
            - Subword pieces that are continuations of a word are typically prefixed with "##"
            - If a word cannot be completely tokenized using available vocabulary pieces,
              the entire word is replaced with the unknown token
            - The greedy algorithm attempts to find the longest possible matches first,
              working from left to right through each word
        """
        <your code>

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
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.