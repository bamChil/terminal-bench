## Task
**Task: Implement Longformer Text Tokenization System**

**Core Functionalities:**
- Build a specialized tokenizer for the Longformer model using byte-level Byte-Pair Encoding (BPE)
- Provide both standard and fast tokenization implementations with vocabulary management
- Handle text preprocessing, token encoding/decoding, and special token integration

**Main Features & Requirements:**
- Support single and paired sequence tokenization with appropriate special token formatting
- Manage vocabulary loading/saving and BPE merge operations
- Handle space-sensitive tokenization where word encoding depends on position context
- Provide configurable special tokens (CLS, SEP, MASK, PAD, etc.) and prefix space handling
- Support batch processing and various output formats (tensors, attention masks, etc.)

**Key Challenges:**
- Correctly handle space-as-token behavior where words encode differently based on leading spaces
- Ensure compatibility between standard and fast tokenizer implementations
- Validate proper configuration for pre-tokenized inputs requiring prefix space handling
- Maintain backward compatibility with existing Longformer model expectations
- Efficiently manage large vocabularies and BPE operations for long sequence processing

**NOTE**: 
- This test is derived from the `transformers` library, but you are NOT allowed to view this codebase or call any of its interfaces. It is **VERY IMPORTANT** to note that if we detect any viewing or calling of this codebase, you will receive a ZERO for this review.
- What's more, you need to install `pytest, pytest-timeout, pytest-json-report` in your environment, otherwise our tests won't run and you'll get **ZERO POINTS**!
- **CRITICAL**: This task is derived from `transformers`, but you **MUST** implement the task description independently. It is **ABSOLUTELY FORBIDDEN** to use `pip install transformers` or some similar commands to access the original implementation—doing so will be considered cheating and will result in an immediate score of ZERO! You must keep this firmly in mind throughout your implementation.
- You are now in `/testbed/`, and originally there was a specific implementation of `transformers` under `/testbed/` that had been installed via `pip install -e .`. However, to prevent you from cheating, we've removed the code under `/testbed/`. While you can see traces of the installation via the pip show, it's an artifact, and `transformers` doesn't exist. So you can't and don't need to use `pip install transformers`, just focus on writing your `agent_code` and accomplishing our task.
- Also, don't try to `pip uninstall transformers` even if the actual `transformers` has already been deleted by us, as this will affect our evaluation of you, and uninstalling the residual `transformers` will result in you getting a ZERO because our tests won't run.

Your available resources are listed below:
- `/workspace/task/black_links.txt`: Prohibited URLs (all other web resources are allowed)


## Precautions
- You may need to install some of the libraries to support you in accomplishing our task, some of the packages are already pre-installed in your environment, you can check them out yourself via `pip list` etc. For standard installs, just run `pip install <package>`. There's no need to add `--index-url`, the domestic mirrors are already set up unless you have special requirements.
- Please note that when running `pip install <package>`, you should not include the `--force-reinstall` flag, as it may cause pre-installed packages to be reinstalled.
- **IMPORTANT**: While you can install libraries using pip, you should never access the actual implementations in the libraries you install, as the tasks we give you originate from github, and if you look at the contents of the libraries, it could result in you being awarded 0 points directly for alleged cheating. Specifically, you cannot read any files under `/usr/local/lib/python3.x` and its subfolders (here python3.x means any version of python).
- **IMPORTANT**: Your installed python library may contain a real implementation of the task, and you are prohibited from directly calling the library's interface of the same name and pretending to package it as your answer, which will also be detected and awarded 0 points.
- **CRITICAL REQUIREMENT**: After completing the task, pytest will be used to test your implementation. **YOU MUST**:
    - Build proper code hierarchy with correct import relationships shown in **Test Description** (I will give you this later)
    - Match the exact interface shown in the **Interface Description** (I will give you this later)
- I will tell you details about **CRITICAL REQUIREMENT** below.

Your final deliverable should be code in the `/testbed/agent_code` directory, and after completing the codebase, we will use testfiles in workspace/test to evaluatate your codebase, and note that you won't see workspace/test.
The final structure is like below, note that all dirs and files under agent_code/ are just examples, your codebase's structure should match import structure in **Test Description**, which I will tell you later.
```
/workspace
├── task/                 
│   ├── prompt.md          # task statement
│   ├── black_links.txt    # black links you can't access
│   ├── ...
├── test/                 # you won't see this dir
│   ├── ...
/testbed
├── agent_code/           # all your code should be put into this dir and match the specific dir structure
│   ├── __init__.py       # agent_code/ folder and ALL folders under it should contain __init__.py
│   ├── dir1/
│   │   ├── __init__.py
│   │   ├── code1.py
│   │   ├── ...
├── setup.py              # after finishing your work, you MUST generate this file
```
After you have done all your work, you need to complete three CRITICAL things: 
1. When you have generated all files or folders under `agent_code/` that match the directory structure, you need to recursively generate `__init__.py` under `agent_code/` and in all subfolders under it to ensure that we can access all functions you generate.(you can simply generate empty `__init__.py`)
2. You need to generate `/testbed/setup.py` under `/testbed/` and place the following content exactly:
```python
from setuptools import setup, find_packages
setup(
    name="agent_code",
    version="1.0.0",
    packages=find_packages(include=["agent_code", "agent_code.*"]),
    install_requires=[],
)
```
3. After you have done above two things, you need to use `cd /testbed && pip install .` command to install your code.
Remember, these things are **VERY IMPORTANT**, as they will directly affect whether you can pass our tests.

## Test and Interface Descriptions

The **Test Description** will tell you the position of the function or class which we're testing should satisfy.
This means that when you generate some files and complete the functionality we want to test in the files, you need to put these files in the specified directory, otherwise our tests won't be able to import your generated.
For example, if the **Test Description** show you this:
```python
from agent_code.transformers import LongformerTokenizer
```
This means that we will test one function/class: LongformerTokenizer.
And the defination and implementation of class LongformerTokenizer should be in `/testbed/agent_code/transformers.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
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
        """
        Initialize a LongformerTokenizer instance.

        This constructor sets up a Longformer tokenizer, which is derived from the GPT-2 tokenizer and uses 
        byte-level Byte-Pair-Encoding (BPE). The tokenizer is designed to handle long sequences and treats 
        spaces as parts of tokens, similar to sentencepiece.

        Parameters:
            vocab_file (str): Path to the vocabulary file containing the token-to-ID mappings in JSON format.
            merges_file (str): Path to the merges file containing BPE merge rules.
            errors (str, optional): Paradigm to follow when decoding bytes to UTF-8. Defaults to 'replace'.
                See Python's bytes.decode() documentation for more information.
            bos_token (str, optional): The beginning of sequence token used during pretraining. Defaults to '<s>'.
                Note that when building sequences with special tokens, cls_token is used instead.
            eos_token (str, optional): The end of sequence token. Defaults to '</s>'.
                Note that when building sequences with special tokens, sep_token is used instead.
            sep_token (str, optional): The separator token used when building sequences from multiple sequences
                or as the last token of a sequence built with special tokens. Defaults to '</s>'.
            cls_token (str, optional): The classifier token used for sequence classification tasks.
                It becomes the first token when building sequences with special tokens. Defaults to '<s>'.
            unk_token (str, optional): The unknown token used for out-of-vocabulary words. Defaults to '<unk>'.
            pad_token (str, optional): The padding token used when batching sequences of different lengths.
                Defaults to '<pad>'.
            mask_token (str, optional): The token used for masking values in masked language modeling tasks.
                Defaults to '<mask>'.
            add_prefix_space (bool, optional): Whether to add an initial space to the input text.
                This allows treating the leading word like any other word. Defaults to False.
            **kwargs: Additional keyword arguments passed to the parent PreTrainedTokenizer class.

        Returns:
            None: This is a constructor method that initializes the tokenizer instance.

        Notes:
            - The tokenizer encodes words differently depending on whether they appear at the beginning
              of a sentence (without preceding space) or not.
            - When used with is_split_into_words=True, the tokenizer adds a space before each word,
              including the first one.
            - The mask_token behaves like a normal word and includes the space before it.
            - All special tokens are converted to AddedToken objects with specific lstrip/rstrip settings.
            - The tokenizer loads vocabulary and merge files during initialization and sets up internal
              mappings for encoding/decoding operations.

        Raises:
            FileNotFoundError: If vocab_file or merges_file cannot be found.
            json.JSONDecodeError: If vocab_file contains invalid JSON.
            UnicodeDecodeError: If files cannot be decoded with UTF-8 encoding.
        """
        <your code>
...
```

In order to implement this functionality, some additional libraries etc. are often required, I don't restrict you to any libraries, you need to think about what dependencies you might need and fetch and install and call them yourself. The only thing is that you **MUST** fulfill the input/output format described by this interface, otherwise the test will not pass and you will get zero points for this feature.
And note that there may be not only one **Interface Description**, you should match all **Interface Description {n}**

### Test Description 1
Below is **Test Description 1**
```python
from agent_code.transformers import LongformerTokenizer
from agent_code.transformers import LongformerTokenizerFast
```

### Interface Description 1
Below is **Interface Description 1** for file: src-transformers-models-longformer-tokenization_longformer.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
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
        """
        Initialize a LongformerTokenizer instance.

        This constructor sets up a Longformer tokenizer, which is derived from the GPT-2 tokenizer and uses 
        byte-level Byte-Pair-Encoding (BPE). The tokenizer is designed to handle long sequences and treats 
        spaces as parts of tokens, similar to sentencepiece.

        Parameters:
            vocab_file (str): Path to the vocabulary file containing the token-to-ID mappings in JSON format.
            merges_file (str): Path to the merges file containing BPE merge rules.
            errors (str, optional): Paradigm to follow when decoding bytes to UTF-8. Defaults to 'replace'.
                See Python's bytes.decode() documentation for more information.
            bos_token (str, optional): The beginning of sequence token used during pretraining. Defaults to '<s>'.
                Note that when building sequences with special tokens, cls_token is used instead.
            eos_token (str, optional): The end of sequence token. Defaults to '</s>'.
                Note that when building sequences with special tokens, sep_token is used instead.
            sep_token (str, optional): The separator token used when building sequences from multiple sequences
                or as the last token of a sequence built with special tokens. Defaults to '</s>'.
            cls_token (str, optional): The classifier token used for sequence classification tasks.
                It becomes the first token when building sequences with special tokens. Defaults to '<s>'.
            unk_token (str, optional): The unknown token used for out-of-vocabulary words. Defaults to '<unk>'.
            pad_token (str, optional): The padding token used when batching sequences of different lengths.
                Defaults to '<pad>'.
            mask_token (str, optional): The token used for masking values in masked language modeling tasks.
                Defaults to '<mask>'.
            add_prefix_space (bool, optional): Whether to add an initial space to the input text.
                This allows treating the leading word like any other word. Defaults to False.
            **kwargs: Additional keyword arguments passed to the parent PreTrainedTokenizer class.

        Returns:
            None: This is a constructor method that initializes the tokenizer instance.

        Notes:
            - The tokenizer encodes words differently depending on whether they appear at the beginning
              of a sentence (without preceding space) or not.
            - When used with is_split_into_words=True, the tokenizer adds a space before each word,
              including the first one.
            - The mask_token behaves like a normal word and includes the space before it.
            - All special tokens are converted to AddedToken objects with specific lstrip/rstrip settings.
            - The tokenizer loads vocabulary and merge files during initialization and sets up internal
              mappings for encoding/decoding operations.

        Raises:
            FileNotFoundError: If vocab_file or merges_file cannot be found.
            json.JSONDecodeError: If vocab_file contains invalid JSON.
            UnicodeDecodeError: If files cannot be decoded with UTF-8 encoding.
        """
        <your code>

    @property
    def vocab_size(self):
        """
        Get the vocabulary size of the tokenizer.

        This property returns the total number of tokens in the base vocabulary of the 
        LongformerTokenizer, which corresponds to the number of entries in the encoder 
        dictionary loaded from the vocabulary file. This count includes all the base 
        tokens used for byte-pair encoding but excludes any additional special tokens 
        that may have been added after initialization.

        Returns:
            int: The size of the base vocabulary, representing the number of unique 
                 tokens that can be encoded by the tokenizer's base vocabulary.

        Note:
            This property only counts the base vocabulary tokens loaded from the 
            vocab.json file. It does not include added tokens or special tokens 
            that were added after the tokenizer was initialized. To get the complete 
            vocabulary including added tokens, use the get_vocab() method instead.
        """
        <your code>

    def get_vocab(self):
        """
        Retrieve the complete vocabulary dictionary for the tokenizer.

        This method returns a comprehensive vocabulary dictionary that includes both the base vocabulary
        from the encoder and any additional tokens that have been added to the tokenizer (such as
        special tokens like <pad>, <mask>, etc.).

        Returns:
            dict: A dictionary mapping vocabulary tokens (str) to their corresponding token IDs (int).
                  The dictionary contains:
                  - All tokens from the base vocabulary (self.encoder)
                  - All added tokens from the added_tokens_encoder

        Notes:
            - The returned dictionary is a copy of the encoder dictionary, so modifications to it
              will not affect the tokenizer's internal state
            - Added tokens (like special tokens) are included with their assigned IDs
            - This method is commonly used for vocabulary inspection, saving tokenizer state, or
              creating custom tokenizers with the same vocabulary
        """
        <your code>

    def bpe(self, token):
        """
        Apply Byte-Pair Encoding (BPE) algorithm to tokenize a given token into subword units.

        This method implements the core BPE algorithm that iteratively merges the most frequent
        pairs of characters or character sequences based on pre-learned merge rules. The process
        continues until no more valid merges can be performed according to the Bpe ranks.

        Parameters:
            token (str): The input token string to be processed with BPE. This should be a 
                        single token (word or word piece) that has already been preprocessed 
                        and encoded using byte-level encoding.

        Returns:
            str: A space-separated string of BPE subword tokens. Each subword represents 
                 a unit that was learned during the BPE training process. If the input 
                 token cannot be split further, the original token is returned.

        Notes:
            - The method uses caching (self.cache) to store previously computed results 
              for efficiency, avoiding recomputation of the same tokens.
            - The algorithm relies on self.bpe_ranks, which contains the learned merge 
              rules with their priority rankings from the BPE training process.
            - If no character pairs exist in the token (single character), the original 
              token is returned immediately.
            - The method uses the get_pairs() helper function to extract all adjacent 
              character pairs from the current word representation.
            - Merging continues until either all possible merges are exhausted or the 
              word is reduced to a single subword unit.
        """
        <your code>

    def _tokenize(self, text):
        """
        Tokenize a string.
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

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None
    ) -> tuple[str]:
        """
        Save the tokenizer's vocabulary files to a specified directory.

        This method saves both the vocabulary file (vocab.json) and the BPE merges file (merges.txt)
        to the given directory. These files are essential for reconstructing the tokenizer later.

        Parameters:
            save_directory (str): The directory path where the vocabulary files will be saved.
                Must be an existing directory, otherwise an error will be logged and the method
                will return early.
            filename_prefix (Optional[str], optional): An optional prefix to add to the saved
                filenames. If provided, the files will be saved as "{prefix}-vocab.json" and
                "{prefix}-merges.txt". If None, the default filenames "vocab.json" and 
                "merges.txt" will be used. Defaults to None.

        Returns:
            tuple[str]: A tuple containing the full paths to the saved vocabulary file and
                merges file, in that order: (vocab_file_path, merge_file_path). Returns None
                if the save_directory is not a valid directory.

        Notes:
            - The vocabulary file is saved as a JSON file containing the encoder dictionary
              with proper formatting (indented, sorted keys, UTF-8 encoding).
            - The merges file is saved as a text file with BPE merge rules, starting with
              a version header "#version: 0.2".
            - If BPE merge indices are not consecutive, a warning will be logged indicating
              potential tokenizer corruption.
            - Both files are saved with UTF-8 encoding to ensure proper handling of unicode
              characters.
        """
        <your code>

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
                    token_ids_0 (`list[int]`):
                        List of IDs.
                    token_ids_1 (`list[int]`, *optional*):
                        Optional second list of IDs for sequence pairs.
                    already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                        Whether or not the token list is already formatted with special tokens for the model.

                Returns:
                    `list[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.

        """
        <your code>

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
        <your code>

    def prepare_for_tokenization(
        self,
        text,
        is_split_into_words = False,
        **kwargs
    ):
        """
        Prepares text for tokenization by applying necessary preprocessing steps.

        This method handles text preprocessing before the actual tokenization process begins. It primarily manages the addition of prefix spaces based on the tokenizer's configuration and the nature of the input text.

        Parameters:
            text (str): The input text string to be prepared for tokenization.
            is_split_into_words (bool, optional): Whether the input text is already split into words. 
                Defaults to False. When True, a prefix space will be added if the text doesn't start 
                with whitespace.
            **kwargs: Additional keyword arguments that may contain tokenization options. The method 
                specifically looks for 'add_prefix_space' in kwargs, which can override the tokenizer's 
                default add_prefix_space setting.

        Returns:
            tuple: A tuple containing:
                - text (str): The preprocessed text, potentially with a prefix space added
                - kwargs (dict): The remaining keyword arguments after removing 'add_prefix_space' 
                  if it was present

        Important Notes:
            - The method adds a prefix space to the text when either is_split_into_words is True 
              or add_prefix_space is True, provided the text is non-empty and doesn't already 
              start with whitespace
            - The add_prefix_space parameter in kwargs takes precedence over the tokenizer's 
              instance variable self.add_prefix_space
            - This preprocessing is crucial for the Longformer tokenizer as it treats spaces as 
              part of tokens, similar to sentencepiece, and words are encoded differently 
              depending on whether they appear at the beginning of a sentence or not
            - The 'add_prefix_space' key is removed from kwargs before returning to avoid 
              passing it to subsequent tokenization steps
        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.

### Interface Description 2
Below is **Interface Description 2** for file: src-transformers-models-longformer-tokenization_longformer_fast.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
class LongformerTokenizerFast(PreTrainedTokenizerFast):
    """
    
        Construct a "fast" Longformer tokenizer (backed by HuggingFace's *tokenizers* library), derived from the GPT-2
        tokenizer, using byte-level Byte-Pair-Encoding.
    
        This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
        be encoded differently whether it is at the beginning of the sentence (without space) or not:
    
        ```python
        >>> from transformers import LongformerTokenizerFast
    
        >>> tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")
        >>> tokenizer("Hello world")["input_ids"]
        [0, 31414, 232, 2]
    
        >>> tokenizer(" Hello world")["input_ids"]
        [0, 20920, 232, 2]
        ```
    
        You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
        call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.
    
        <Tip>
    
        When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.
    
        </Tip>
    
        This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
        refer to this superclass for more information regarding those methods.
    
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
            trim_offsets (`bool`, *optional*, defaults to `True`):
                Whether the post processing step should trim offsets to avoid including whitespaces.
        
    """

    vocab_files_names = "VOCAB_FILES_NAMES"
    model_input_names = ['input_ids', 'attention_mask']
    slow_tokenizer_class = "LongformerTokenizer"

    def __init__(
        self,
        vocab_file = None,
        merges_file = None,
        tokenizer_file = None,
        errors = 'replace',
        bos_token = '<s>',
        eos_token = '</s>',
        sep_token = '</s>',
        cls_token = '<s>',
        unk_token = '<unk>',
        pad_token = '<pad>',
        mask_token = '<mask>',
        add_prefix_space = False,
        trim_offsets = True,
        **kwargs
    ):
        """
        Initialize a LongformerTokenizerFast instance for fast tokenization of text using the Longformer model.

        This constructor sets up a fast tokenizer based on HuggingFace's tokenizers library, derived from the GPT-2
        tokenizer using byte-level Byte-Pair-Encoding. The tokenizer is specifically designed for the Longformer
        model and handles long sequences efficiently.

        Parameters:
            vocab_file (str, optional): Path to the vocabulary file containing the token-to-id mappings.
                If None, the tokenizer will attempt to load from the default location or tokenizer_file.
            merges_file (str, optional): Path to the merges file containing BPE merge rules.
                If None, the tokenizer will attempt to load from the default location or tokenizer_file.
            tokenizer_file (str, optional): Path to a pre-built tokenizer file in JSON format.
                If provided, this takes precedence over vocab_file and merges_file.
            errors (str, optional): Error handling paradigm when decoding bytes to UTF-8.
                Defaults to 'replace'. See Python's bytes.decode() documentation for options.
            bos_token (str, optional): Beginning of sequence token used during pretraining.
                Defaults to '<s>'. Used as sequence classifier token.
            eos_token (str, optional): End of sequence token. Defaults to '</s>'.
                Note: When building sequences with special tokens, sep_token is used instead.
            sep_token (str, optional): Separator token for building sequences from multiple parts.
                Defaults to '</s>'. Also used as the last token in sequences built with special tokens.
            cls_token (str, optional): Classifier token for sequence classification tasks.
                Defaults to '<s>'. Used as the first token when building sequences with special tokens.
            unk_token (str, optional): Unknown token for out-of-vocabulary words.
                Defaults to '<unk>'. Tokens not in vocabulary are replaced with this.
            pad_token (str, optional): Padding token for batching sequences of different lengths.
                Defaults to '<pad>'.
            mask_token (str, optional): Masking token for masked language modeling tasks.
                Defaults to '<mask>'. The model predicts this token during training.
            add_prefix_space (bool, optional): Whether to add an initial space to input text.
                Defaults to False. When True, treats leading words like any other word by adding
                a preceding space, which the tokenizer uses to detect word boundaries.
            trim_offsets (bool, optional): Whether to trim token offsets to exclude whitespaces
                in post-processing. Defaults to True.
            **kwargs: Additional keyword arguments passed to the parent PreTrainedTokenizerFast class.

        Returns:
            None: This is a constructor method that initializes the instance.

        Important Notes:
            - The tokenizer treats spaces as parts of tokens, so words are encoded differently
              depending on whether they appear at the beginning of a sentence or not.
            - When using is_split_into_words=True, the tokenizer must be instantiated with
              add_prefix_space=True to work correctly with pretokenized inputs.
            - The mask_token is automatically configured as an AddedToken with lstrip=True
              to maintain backward compatibility with existing Longformer models.
            - If a tokenizer_file is provided, it takes precedence over separate vocab_file
              and merges_file parameters.
            - The constructor automatically updates the backend tokenizer's post-processor
              settings if add_prefix_space or trim_offsets differ from existing configuration.
        """
        <your code>

    @property
    def mask_token(self) -> str:
        """

                `str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while not
                having been set.

                Longformer tokenizer has a special mask token to be usable in the fill-mask pipeline. The mask token will greedily
                comprise the space before the *<mask>*.

        """
        <your code>

    @mask_token.setter
    def mask_token(self, value):
        """

                Overriding the default behavior of the mask token to have it eat the space before it.

                This is needed to preserve backward compatibility with all the previously used models based on Longformer.

        """
        <your code>

    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        """
        Internal batch encoding method for processing multiple text inputs with Longformer-specific validation.

        This method serves as an internal wrapper around the parent class's batch encoding functionality,
        adding validation specific to the Longformer tokenizer's requirements for handling pretokenized inputs.

        Args:
            *args: Variable length argument list passed to the parent class's _batch_encode_plus method.
                Typically includes batch_text_or_text_pairs as the first argument, which can be:
                - A list of strings to be tokenized
                - A list of tuples of strings for sequence pair tasks
            **kwargs: Arbitrary keyword arguments passed to the parent class's _batch_encode_plus method.
                Common parameters include:
                - is_split_into_words (bool): Whether the input is already split into words/tokens
                - add_special_tokens (bool): Whether to add special tokens like [CLS], [SEP]
                - padding (bool/str): Padding strategy
                - truncation (bool/str): Truncation strategy
                - max_length (int): Maximum sequence length
                - return_tensors (str): Format of return tensors ('pt', 'tf', 'np', etc.)

        Returns:
            BatchEncoding: A BatchEncoding object containing the tokenized and encoded batch data.
                This includes input_ids, attention_mask, and other relevant tensors based on the
                specified return parameters.

        Raises:
            AssertionError: If is_split_into_words=True is used when the tokenizer was not instantiated
                with add_prefix_space=True. This is required because the Longformer tokenizer needs
                to properly handle word boundaries when working with pretokenized inputs.

        Notes:
            - This method enforces Longformer-specific constraints on pretokenized input handling
            - The validation ensures compatibility with the model's pretraining approach to space handling
            - All other functionality is delegated to the parent class implementation
        """
        <your code>

    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        """
        Encode a single text or text pair into a BatchEncoding object with token IDs and additional information.

        This method is the internal implementation for encoding a single text input or a pair of text inputs
        into the format expected by the Longformer model. It handles tokenization, special token addition,
        and creates the necessary tensors and metadata for model input.

        Args:
            *args: Variable length argument list passed to the parent tokenizer's _encode_plus method.
                   Typically includes:
                   - text (str or List[str]): The sequence to be encoded. Can be a string or a list of strings
                     if is_split_into_words=True.
                   - text_pair (str or List[str], optional): Optional second sequence for sequence pair tasks.
            **kwargs: Arbitrary keyword arguments passed to the parent tokenizer's _encode_plus method.
                      Common arguments include:
                      - add_special_tokens (bool): Whether to add special tokens (CLS, SEP, etc.).
                      - padding (bool, str, or PaddingStrategy): Padding strategy.
                      - truncation (bool, str, or TruncationStrategy): Truncation strategy.
                      - max_length (int): Maximum length of the returned list.
                      - return_tensors (str): Type of tensors to return ('pt', 'tf', 'np', etc.).
                      - return_attention_mask (bool): Whether to return attention mask.
                      - return_token_type_ids (bool): Whether to return token type IDs.
                      - is_split_into_words (bool): Whether the input is already split into words.

        Returns:
            BatchEncoding: A BatchEncoding object containing:
                - input_ids: List of token IDs representing the encoded sequence(s).
                - attention_mask: List of integers (0s and 1s) indicating which tokens should be attended to.
                - token_type_ids: List of token type IDs (if applicable).
                - special_tokens_mask: List indicating which tokens are special tokens.
                - offset_mapping: List of tuples indicating character spans (if return_offsets_mapping=True).
                - overflowing_tokens: Information about tokens that were truncated (if applicable).

        Raises:
            AssertionError: If is_split_into_words=True is used without add_prefix_space=True during
                           tokenizer instantiation. This is required because the Longformer tokenizer
                           needs to properly handle word boundaries when working with pre-tokenized inputs.

        Notes:
            - This method validates that when using pre-tokenized inputs (is_split_into_words=True),
              the tokenizer must have been instantiated with add_prefix_space=True.
            - The method delegates the actual encoding work to the parent class implementation
              after performing the necessary validation checks.
            - This is an internal method and should typically not be called directly by users.
              Use the public encode_plus method instead.
        """
        <your code>

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None
    ) -> tuple[str]:
        """
        Save the tokenizer's vocabulary files to the specified directory.

        This method saves the vocabulary and merge files used by the tokenizer to disk,
        allowing the tokenizer to be reconstructed later or used in other applications.
        The files are saved using the tokenizer's underlying model save functionality.

        Args:
            save_directory (str): The directory path where the vocabulary files will be saved.
                The directory must exist or be creatable by the underlying save mechanism.
            filename_prefix (Optional[str], optional): An optional prefix to add to the saved
                vocabulary filenames. If None, the default filenames will be used without
                any prefix. Defaults to None.

        Returns:
            tuple[str]: A tuple containing the paths to the saved vocabulary files. Typically
                includes paths to files like vocab.json and merges.txt, depending on the
                tokenizer's configuration.

        Note:
            This method delegates to the underlying tokenizer model's save functionality.
            The exact files saved and their formats depend on the specific tokenizer
            implementation (in this case, a BPE-based tokenizer for Longformer).
        """
        <your code>

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1 = None):
        """
        Build a sequence with special tokens for the Longformer model.

        This method constructs input sequences by adding the appropriate special tokens (BOS and EOS) 
        around the provided token sequences. For single sequences, it adds BOS at the beginning and 
        EOS at the end. For sequence pairs, it creates the format: 
        [BOS] + sequence_0 + [EOS] + [EOS] + sequence_1 + [EOS].

        Args:
            token_ids_0 (list[int]): The first sequence of token IDs to be processed. This is the 
                primary sequence that will always be included in the output.
            token_ids_1 (list[int], optional): The second sequence of token IDs for sequence pair 
                tasks such as question answering or sentence similarity. If None, only the first 
                sequence will be processed. Defaults to None.

        Returns:
            list[int]: A list of token IDs with special tokens added. For single sequences, returns 
                [bos_token_id] + token_ids_0 + [eos_token_id]. For sequence pairs, returns 
                [bos_token_id] + token_ids_0 + [eos_token_id] + [eos_token_id] + token_ids_1 + [eos_token_id].

        Notes:
            - The BOS (beginning of sequence) token is added at the very beginning of the output
            - The EOS (end of sequence) token is used as a separator between sequences and at the end
            - For sequence pairs, two consecutive EOS tokens separate the first and second sequences
            - This format is specific to the Longformer model architecture and follows the same 
              pattern as RoBERTa tokenization
        """
        <your code>

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
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.