## Task
## Task: Implement a Knowledge-Enhanced Text Tokenizer

**Core Functionality:**
Develop a specialized tokenizer that processes both textual content and entity information for knowledge-enhanced language models. The tokenizer must handle standard text tokenization using BPE encoding while simultaneously managing entity spans, entity vocabularies, and task-specific entity processing.

**Main Features & Requirements:**
- **Dual Processing**: Tokenize regular text using byte-pair encoding and process entity information (spans, names, IDs) in parallel
- **Task Adaptability**: Support multiple NLP tasks (entity classification, entity pair classification, entity span classification) with automatic sequence formatting
- **Entity Management**: Handle entity vocabularies, special entity tokens, and character-to-token span conversion
- **Standard Operations**: Implement padding, truncation, batching, vocabulary management, and tensor format conversion
- **Compatibility**: Maintain interface consistency with standard tokenizer APIs while extending functionality for entities

**Key Challenges:**
- **Span Alignment**: Accurately convert character-based entity spans to token-based positions after BPE tokenization
- **Multi-Modal Processing**: Coordinate text and entity processing pipelines while maintaining sequence integrity
- **Task-Specific Logic**: Dynamically adjust entity handling based on configured task requirements and constraints
- **Memory Efficiency**: Handle variable-length sequences and entity lists efficiently during batch processing

**NOTE**: 
- This test comes from the `transformers` library, and we have given you the content of this code repository under `/testbed/`, and you need to complete based on this code repository and supplement the files we specify. Remember, all your changes must be in this codebase, and changes that are not in this codebase will not be discovered and tested by us.
- What's more, you need to install `pytest, pytest-timeout, pytest-json-report` in your environment, otherwise our tests won't run and you'll get **ZERO POINTS**!

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

Your final deliverable should be code under the `/testbed/` directory, and after completing the codebase, we will evaluate your completion and it is important that you complete our tasks with integrity and precision
The final structure is like below, note that  your codebase's structure should match import structure in **Test Description**, which I will tell you later.
```
/workspace
├── task/
│   ├── prompt.md          # task statement
│   ├── black_links.txt    # black links you can't access
│   ├── ...
├── test/                  # you won't see this dir
│   ├── ...
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
from transformers import LukeTokenizer
```
This means that we will test one function/class: LukeTokenizer.
And the defination and implementation of class LukeTokenizer should be in `/testbed/src/transformers/models/luke/tokenization_luke.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
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
        """
        Initialize a LUKE tokenizer instance.

        This constructor sets up a LUKE (Language Understanding with Knowledge-based Embeddings) tokenizer,
        which is derived from the GPT-2 tokenizer using byte-level Byte-Pair-Encoding. The tokenizer is
        designed to handle both text tokens and entity information for knowledge-enhanced language modeling.

        Parameters:
            vocab_file (str): Path to the vocabulary file containing the token-to-ID mappings in JSON format.
            merges_file (str): Path to the BPE merges file containing byte-pair merge rules.
            entity_vocab_file (str): Path to the entity vocabulary file containing entity-to-ID mappings in JSON format.
            task (str, optional): Task type for automatic entity sequence creation. Must be one of:
                - "entity_classification": Single entity classification task
                - "entity_pair_classification": Entity pair classification task  
                - "entity_span_classification": Entity span classification task
                If None, no automatic entity sequence creation is performed. Defaults to None.
            max_entity_length (int, optional): Maximum length of entity sequences. Automatically set based on task:
                - 32 for entity_span_classification or None task
                - 1 for entity_classification
                - 2 for entity_pair_classification
                Defaults to 32.
            max_mention_length (int, optional): Maximum number of tokens allowed within a single entity span. Defaults to 30.
            entity_token_1 (str, optional): Special token representing the first entity in word sequences for classification tasks. Defaults to '<ent>'.
            entity_token_2 (str, optional): Special token representing the second entity in word sequences for pair classification. Defaults to '<ent2>'.
            entity_unk_token (str, optional): Unknown token for entities not found in entity vocabulary. Defaults to '[UNK]'.
            entity_pad_token (str, optional): Padding token for entity sequences. Defaults to '[PAD]'.
            entity_mask_token (str, optional): Mask token for entities in masked language modeling. Defaults to '[MASK]'.
            entity_mask2_token (str, optional): Second mask token for entity pair tasks. Defaults to '[MASK2]'.
            errors (str, optional): Error handling strategy for UTF-8 decoding. See bytes.decode() documentation. Defaults to 'replace'.
            bos_token (str, optional): Beginning of sequence token used during pretraining. Defaults to '<s>'.
            eos_token (str, optional): End of sequence token. Defaults to '</s>'.
            sep_token (str, optional): Separator token for building sequences from multiple parts. Defaults to '</s>'.
            cls_token (str, optional): Classification token used as first token in sequence classification. Defaults to '<s>'.
            unk_token (str, optional): Unknown token for out-of-vocabulary words. Defaults to '<unk>'.
            pad_token (str, optional): Padding token for batching sequences of different lengths. Defaults to '<pad>'.
            mask_token (str, optional): Mask token for masked language modeling tasks. Defaults to '<mask>'.
            add_prefix_space (bool, optional): Whether to add initial space to input text for consistent word boundary detection. Defaults to False.
            **kwargs: Additional keyword arguments passed to the parent PreTrainedTokenizer class.

        Returns:
            None: This is a constructor method.

        Raises:
            ValueError: If the specified task is not one of the supported task types.
            ValueError: If any required entity special tokens are missing from the entity vocabulary file.
            FileNotFoundError: If any of the required vocabulary files cannot be found or opened.
            json.JSONDecodeError: If vocabulary files contain invalid JSON format.

        Notes:
            - The tokenizer treats spaces as part of tokens (similar to sentencepiece), so words are encoded differently depending on whether they appear at the beginning of a sentence.
            - When using add_prefix_space=True, a space is added before each word, including the first one.
            - All entity special tokens must exist in the provided entity vocabulary file.
            - The max_entity_length is automatically adjusted based on the specified task type.
        """
        <your code>
...
```

In order to implement this functionality, some additional libraries etc. are often required, I don't restrict you to any libraries, you need to think about what dependencies you might need and fetch and install and call them yourself. The only thing is that you **MUST** fulfill the input/output format described by this interface, otherwise the test will not pass and you will get zero points for this feature.
And note that there may be not only one **Interface Description**, you should match all **Interface Description {n}**

### Test Description 1
Below is **Test Description 1**
```python
from transformers import LukeTokenizer
```

### Interface Description 1
Below is **Interface Description 1** for file: src-transformers-models-luke-tokenization_luke.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
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
        """
        Initialize a LUKE tokenizer instance.

        This constructor sets up a LUKE (Language Understanding with Knowledge-based Embeddings) tokenizer,
        which is derived from the GPT-2 tokenizer using byte-level Byte-Pair-Encoding. The tokenizer is
        designed to handle both text tokens and entity information for knowledge-enhanced language modeling.

        Parameters:
            vocab_file (str): Path to the vocabulary file containing the token-to-ID mappings in JSON format.
            merges_file (str): Path to the BPE merges file containing byte-pair merge rules.
            entity_vocab_file (str): Path to the entity vocabulary file containing entity-to-ID mappings in JSON format.
            task (str, optional): Task type for automatic entity sequence creation. Must be one of:
                - "entity_classification": Single entity classification task
                - "entity_pair_classification": Entity pair classification task  
                - "entity_span_classification": Entity span classification task
                If None, no automatic entity sequence creation is performed. Defaults to None.
            max_entity_length (int, optional): Maximum length of entity sequences. Automatically set based on task:
                - 32 for entity_span_classification or None task
                - 1 for entity_classification
                - 2 for entity_pair_classification
                Defaults to 32.
            max_mention_length (int, optional): Maximum number of tokens allowed within a single entity span. Defaults to 30.
            entity_token_1 (str, optional): Special token representing the first entity in word sequences for classification tasks. Defaults to '<ent>'.
            entity_token_2 (str, optional): Special token representing the second entity in word sequences for pair classification. Defaults to '<ent2>'.
            entity_unk_token (str, optional): Unknown token for entities not found in entity vocabulary. Defaults to '[UNK]'.
            entity_pad_token (str, optional): Padding token for entity sequences. Defaults to '[PAD]'.
            entity_mask_token (str, optional): Mask token for entities in masked language modeling. Defaults to '[MASK]'.
            entity_mask2_token (str, optional): Second mask token for entity pair tasks. Defaults to '[MASK2]'.
            errors (str, optional): Error handling strategy for UTF-8 decoding. See bytes.decode() documentation. Defaults to 'replace'.
            bos_token (str, optional): Beginning of sequence token used during pretraining. Defaults to '<s>'.
            eos_token (str, optional): End of sequence token. Defaults to '</s>'.
            sep_token (str, optional): Separator token for building sequences from multiple parts. Defaults to '</s>'.
            cls_token (str, optional): Classification token used as first token in sequence classification. Defaults to '<s>'.
            unk_token (str, optional): Unknown token for out-of-vocabulary words. Defaults to '<unk>'.
            pad_token (str, optional): Padding token for batching sequences of different lengths. Defaults to '<pad>'.
            mask_token (str, optional): Mask token for masked language modeling tasks. Defaults to '<mask>'.
            add_prefix_space (bool, optional): Whether to add initial space to input text for consistent word boundary detection. Defaults to False.
            **kwargs: Additional keyword arguments passed to the parent PreTrainedTokenizer class.

        Returns:
            None: This is a constructor method.

        Raises:
            ValueError: If the specified task is not one of the supported task types.
            ValueError: If any required entity special tokens are missing from the entity vocabulary file.
            FileNotFoundError: If any of the required vocabulary files cannot be found or opened.
            json.JSONDecodeError: If vocabulary files contain invalid JSON format.

        Notes:
            - The tokenizer treats spaces as part of tokens (similar to sentencepiece), so words are encoded differently depending on whether they appear at the beginning of a sentence.
            - When using add_prefix_space=True, a space is added before each word, including the first one.
            - All entity special tokens must exist in the provided entity vocabulary file.
            - The max_entity_length is automatically adjusted based on the specified task type.
        """
        <your code>

    @property
    def vocab_size(self):
        """
        Get the size of the vocabulary.

        This property returns the total number of tokens in the tokenizer's vocabulary,
        which corresponds to the number of unique tokens that can be encoded by this
        tokenizer. The vocabulary size is determined by the length of the encoder
        dictionary that maps tokens to their corresponding IDs.

        Returns:
            int: The size of the vocabulary (number of unique tokens in the encoder).

        Note:
            This property only counts the base vocabulary tokens stored in the encoder
            and does not include any additional special tokens that may have been added
            after initialization. For the complete vocabulary including added tokens,
            use the `get_vocab()` method instead.
        """
        <your code>

    def get_vocab(self):
        """
        Retrieve the complete vocabulary dictionary for the tokenizer.

        This method returns a dictionary containing all tokens from the base vocabulary
        (encoder) combined with any additional special tokens that have been added to
        the tokenizer. The vocabulary maps token strings to their corresponding integer IDs.

        Returns:
            dict[str, int]: A dictionary mapping token strings to their integer IDs. This includes:
                - All tokens from the base vocabulary (self.encoder)
                - All additional special tokens (self.added_tokens_encoder)

        Notes:
            - This method creates a copy of the encoder dictionary to avoid modifying the original
            - The returned dictionary can be used for token-to-ID lookups or vocabulary inspection
            - Additional special tokens are merged with the base vocabulary, potentially overriding
              base vocabulary entries if there are conflicts
            - This is a standard method inherited from the parent PreTrainedTokenizer class
        """
        <your code>

    def bpe(self, token):
        """
        Apply Byte-Pair Encoding (BPE) to a given token.

        This method performs BPE tokenization on a single token by iteratively merging the pairs of characters/subwords with the **highest priority** 
        (as defined by the pre-learned BPE merge rules in `self.bpe_ranks`). 
        The process continues until no more merges can be applied based on the available merge vocabulary. 
    
        Parameters:
            token (str): The input token to be processed with BPE. This should be a string representing
                a single token that has already been preprocessed (e.g., byte-encoded).

        Returns:
            str: The BPE-processed token as a space-separated string of subword units. If the token
                was previously cached, returns the cached result. If no BPE merges can be applied,
                returns the original token.

        Important notes or exceptions:
            - This method uses caching to improve performance for repeated tokens
            - The BPE merge rules are determined by the `self.bpe_ranks` dictionary loaded during
              tokenizer initialization
            - If a token contains character pairs not found in the BPE vocabulary, those pairs
              will not be merged
            - The method modifies the internal cache (`self.cache`) to store results for future use
            - This is a core component of the tokenization pipeline and is typically called
              internally by other tokenization methods rather than directly by users
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
        <your code>

    def prepare_for_tokenization(
        self,
        text,
        is_split_into_words = False,
        **kwargs
    ):
        """
        Prepares text for tokenization by applying necessary preprocessing transformations.

        This method performs preprocessing steps on the input text before tokenization occurs. 
        Specifically, it handles the addition of prefix spaces based on the tokenizer configuration
        and input parameters. This is important for LUKE tokenizer since it treats spaces as part
        of tokens (similar to sentencepiece), and words are encoded differently depending on
        whether they appear at the beginning of a sentence or not.

        Parameters:
            text (str): The input text string to be prepared for tokenization.
            is_split_into_words (bool, optional): Whether the input text is already split into 
                words. When True, a space will be added before each word (even the first one).
                Defaults to False.
            **kwargs: Additional keyword arguments. The method specifically looks for 
                'add_prefix_space' parameter which can override the tokenizer's default
                add_prefix_space setting.

        Returns:
            tuple[str, dict]: A tuple containing:
                - The preprocessed text string with appropriate spacing applied
                - The updated kwargs dictionary with 'add_prefix_space' parameter removed
                  if it was present

        Important Notes:
            - If is_split_into_words is True or add_prefix_space is True, and the text is 
              non-empty and doesn't start with whitespace, a space will be prepended to the text
            - The add_prefix_space parameter in kwargs takes precedence over the tokenizer's 
              default add_prefix_space attribute
            - This preprocessing ensures consistent tokenization behavior regardless of whether
              words appear at sentence boundaries or not
            - The method modifies the input text in-place and removes the add_prefix_space
              parameter from kwargs to prevent it from being passed to subsequent methods
        """
        <your code>

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
        <your code>

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
        """
        """
        Encode a single text or text pair with entity information for the LUKE model.

        This is an internal method that handles the core encoding logic for preparing inputs
        with both textual tokens and entity information. It processes the input text(s) and
        entity spans/IDs to create model-ready encodings including special tokens, padding,
        and truncation as needed.

        Parameters:
            text (Union[TextInput]): The primary text sequence to encode. 
            text_pair (Optional[Union[TextInput]], optional): The second text sequence for 
                sequence pair tasks. Defaults to None.
            entity_spans (Optional[EntitySpanInput], optional): List of tuples containing 
                (start, end) character positions for entities in the primary text. Each tuple 
                defines the character-level boundaries of an entity mention. Defaults to None.
            entity_spans_pair (Optional[EntitySpanInput], optional): List of tuples containing 
                (start, end) character positions for entities in the second text sequence. 
                Only used when text_pair is provided. Defaults to None.
            entities (Optional[EntityInput], optional): List of entity names/identifiers 
                corresponding to the entity_spans. If not provided, entities will be filled 
                with [MASK] tokens. Must have same length as entity_spans if provided. 
                Defaults to None.
            entities_pair (Optional[EntityInput], optional): List of entity names/identifiers 
                for the second text sequence. Must have same length as entity_spans_pair if 
                provided. Defaults to None.
            add_special_tokens (bool, optional): Whether to add special tokens like [CLS] and 
                [SEP] to the sequences. Defaults to True.
            padding_strategy (PaddingStrategy, optional): Strategy for padding sequences. 
                Defaults to PaddingStrategy.DO_NOT_PAD.
            truncation_strategy (TruncationStrategy, optional): Strategy for truncating 
                sequences that exceed max_length. Defaults to TruncationStrategy.DO_NOT_TRUNCATE.
            max_length (Optional[int], optional): Maximum length for input sequences. If None, 
                no length limit is applied. Defaults to None.
            max_entity_length (Optional[int], optional): Maximum number of entities to include. 
                If None, uses the tokenizer's default max_entity_length. Defaults to None.
            stride (int, optional): Stride for handling overflowing tokens when truncation 
                creates multiple segments. Defaults to 0.
            is_split_into_words (Optional[bool], optional): Whether the input text is already 
                split into words. Currently not supported and will raise NotImplementedError. 
                Defaults to False.
            pad_to_multiple_of (Optional[int], optional): Pad sequence length to a multiple 
                of this value. Useful for tensor core optimization. Defaults to None.
            padding_side (Optional[str], optional): Side to apply padding ('left' or 'right'). 
                If None, uses tokenizer default. Defaults to None.
            return_tensors (Optional[Union[str, TensorType]], optional): Format for returned 
                tensors ('pt' for PyTorch, 'tf' for TensorFlow, 'np' for NumPy). Defaults to None.
            return_token_type_ids (Optional[bool], optional): Whether to return token type IDs. 
                If None, determined by model requirements. Defaults to None.
            return_attention_mask (Optional[bool], optional): Whether to return attention mask. 
                If None, determined by model requirements. Defaults to None.
            return_overflowing_tokens (bool, optional): Whether to return tokens that overflow 
                max_length. Defaults to False.
            return_special_tokens_mask (bool, optional): Whether to return a mask indicating 
                special tokens. Defaults to False.
            return_offsets_mapping (bool, optional): Whether to return character-to-token 
                offset mapping. Not supported in this tokenizer. Defaults to False.
            return_length (bool, optional): Whether to return the length of encoded sequences. 
                Defaults to False.
            verbose (bool, optional): Whether to print warnings and additional information. 
                Defaults to True.
            **kwargs: Additional keyword arguments passed to internal tokenization methods.

        Returns:
            BatchEncoding: A BatchEncoding object containing the encoded inputs with the following keys:
                - input_ids: Token IDs for the input text
                - attention_mask: Attention mask (if return_attention_mask=True)
                - token_type_ids: Token type IDs (if return_token_type_ids=True)
                - entity_ids: Entity IDs corresponding to the entities
                - entity_position_ids: Position IDs for entity tokens in the sequence
                - entity_attention_mask: Attention mask for entities
                - entity_token_type_ids: Token type IDs for entities (if applicable)
                - entity_start_positions: Start positions for entity spans (for span classification)
                - entity_end_positions: End positions for entity spans (for span classification)
                - special_tokens_mask: Mask for special tokens (if return_special_tokens_mask=True)
                - length: Sequence length (if return_length=True)
                - overflowing_tokens: Overflow tokens (if return_overflowing_tokens=True)

        Raises:
                NotImplementedError: If return_offsets_mapping=True or is_split_into_words=True... (thrown directly by this method)
                ValueError: If entity spans and entities have mismatched lengths, or if invalid entity span formats are provided... (thrown by internal `_create_input_sequence` method)
                TypeError: If entity_spans is not provided as a list of tuples when entities are specified... (thrown by internal `_create_input_sequence` method)

        Notes:
            - This method is typically called internally by the public encode_plus method
            - Entity spans must be character-level positions, not token positions
            - The tokenizer will automatically handle entity token alignment during encoding
            - When task-specific behavior is configured, entity processing follows the task requirements
            - Truncation may remove entities if their spans fall outside the truncated sequence
        """
        """
        <your code>

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
        """
        Encode a batch of text sequences or text pairs along with their entity information for the LUKE model.

        This is an internal method that processes multiple sequences at once, handling tokenization,
        entity processing, padding, and truncation according to the specified strategies. It supports
        both single sequences and sequence pairs, along with their corresponding entity spans and
        entity vocabularies.

        Parameters:
            batch_text_or_text_pairs (Union[list[TextInput], list[TextInputPair]]): 
                A batch of text sequences or text pairs to be encoded. Each element can be either
                a single text string or a tuple/list containing two text strings for sequence pairs.

            batch_entity_spans_or_entity_spans_pairs (Optional[Union[list[EntitySpanInput], list[tuple[EntitySpanInput, EntitySpanInput]]]], optional):
                A batch of entity spans corresponding to the text sequences. Each entity span is
                represented as a list of tuples containing (start, end) character positions.
                For sequence pairs, this should be a list of tuples containing entity spans for
                both sequences. Defaults to None.

            batch_entities_or_entities_pairs (Optional[Union[list[EntityInput], list[tuple[EntityInput, EntityInput]]]], optional):
                A batch of entity names/identifiers corresponding to the entity spans. Each element
                is a list of entity strings. For sequence pairs, this should be a list of tuples
                containing entities for both sequences. If not provided, entity spans will be
                filled with [MASK] tokens. Defaults to None.

            add_special_tokens (bool, optional): 
                Whether to add special tokens (CLS, SEP, etc.) to the sequences. Defaults to True.

            padding_strategy (PaddingStrategy, optional): 
                Strategy for padding sequences to uniform length. Defaults to PaddingStrategy.DO_NOT_PAD.

            truncation_strategy (TruncationStrategy, optional): 
                Strategy for truncating sequences that exceed maximum length. 
                Defaults to TruncationStrategy.DO_NOT_TRUNCATE.

            max_length (Optional[int], optional): 
                Maximum length for input sequences after tokenization. Defaults to None.

            max_entity_length (Optional[int], optional): 
                Maximum number of entities to include per sequence. Defaults to None.

            stride (int, optional): 
                Stride length for handling overflowing tokens when truncation occurs. Defaults to 0.

            is_split_into_words (Optional[bool], optional): 
                Whether the input text is already split into words. Currently not supported 
                and will raise NotImplementedError if True. Defaults to False.

            pad_to_multiple_of (Optional[int], optional): 
                Pad sequence lengths to be multiples of this value. Useful for tensor core optimization. 
                Defaults to None.

            padding_side (Optional[str], optional): 
                Side on which to apply padding ('left' or 'right'). If None, uses tokenizer default. 
                Defaults to None.

            return_tensors (Optional[Union[str, TensorType]], optional): 
                Format for returned tensors ('pt' for PyTorch, 'tf' for TensorFlow, 'np' for NumPy). 
                Defaults to None.

            return_token_type_ids (Optional[bool], optional): 
                Whether to return token type IDs. If None, follows model defaults. Defaults to None.

            return_attention_mask (Optional[bool], optional): 
                Whether to return attention masks. If None, follows model defaults. Defaults to None.

            return_overflowing_tokens (bool, optional): 
                Whether to return information about tokens that were truncated. Defaults to False.

            return_special_tokens_mask (bool, optional): 
                Whether to return a mask indicating which tokens are special tokens. Defaults to False.

            return_offsets_mapping (bool, optional): 
                Whether to return character-to-token offset mappings. Currently not supported 
                and will raise NotImplementedError if True. Defaults to False.

            return_length (bool, optional): 
                Whether to return the length of each encoded sequence. Defaults to False.

            verbose (bool, optional): 
                Whether to print warnings and additional information during processing. Defaults to True.

            **kwargs: Additional keyword arguments passed to internal tokenization methods.

        Returns:
            BatchEncoding: A BatchEncoding object containing the processed batch with the following fields:
                - input_ids: Token IDs for the input sequences
                - attention_mask: Attention masks (if return_attention_mask=True)
                - token_type_ids: Token type IDs (if return_token_type_ids=True)
                - entity_ids: Entity vocabulary IDs for recognized entities
                - entity_position_ids: Position mappings for entities within token sequences
                - entity_attention_mask: Attention masks for entity sequences
                - entity_token_type_ids: Token type IDs for entities (if applicable)
                - entity_start_positions: Start positions for entity spans (for span classification tasks)
                - entity_end_positions: End positions for entity spans (for span classification tasks)
                - special_tokens_mask: Mask for special tokens (if return_special_tokens_mask=True)
                - length: Sequence lengths (if return_length=True)
                - overflowing_tokens: Information about truncated tokens (if return_overflowing_tokens=True)

        Raises:
            NotImplementedError: If return_offsets_mapping=True or is_split_into_words=True, as these
                features are not supported by this tokenizer implementation.

            ValueError: If entity spans and entities have mismatched lengths, or if entity spans
                are not properly formatted as lists of (start, end) tuples.

        Notes:
            - This method processes each sequence in the batch individually before applying batch-level
              padding and formatting
            - Entity spans that become invalid due to text truncation will be automatically filtered out
              with appropriate warnings
            - The method supports different task types (entity_classification, entity_pair_classification,
              entity_span_classification) which affect how entities are processed
            - For optimal performance on modern hardware, consider using pad_to_multiple_of=8 or 16
        """
        <your code>

    def _check_entity_input_format(
        self,
        entities: Optional[EntityInput],
        entity_spans: Optional[EntitySpanInput]
    ):
        """
        Validates and checks the format of entity input parameters for the LUKE tokenizer.

        This internal method performs comprehensive validation of entity-related inputs to ensure they
        conform to the expected format and structure required by the LUKE model. It validates both
        entity spans (character-based positions) and optional entity names.

        Parameters:
            entities (Optional[EntityInput]): 
                Optional list of entity names/identifiers. Each entity should be a string representing
                either special entities (e.g., [MASK]) or entity titles from Wikipedia (e.g., "Los Angeles").
                If provided, must have the same length as entity_spans. Can be None if only entity spans
                are needed.
            entity_spans (Optional[EntitySpanInput]):
                List of entity span tuples, where each tuple contains two integers representing the
                character-based start and end positions of entities in the input text. Must be a list
                of tuples in the format [(start1, end1), (start2, end2), ...].

        Raises:
            TypeError: If entity_spans is not provided as a list.
            ValueError: If entity_spans contains elements that are not tuples, if entities is not a list
                when provided, if entities contains non-string elements, or if the lengths of entities
                and entity_spans do not match when both are provided.

        Notes:
            - This method does not return any value; it only performs validation
            - Entity spans use character-based indexing (not token-based)
            - The validation ensures consistency between entity names and their corresponding spans
            - This method is called internally during the tokenization process to prevent malformed inputs
        """
        <your code>

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
        """
        Create input sequences for LUKE tokenizer processing.

        This internal method processes text and entity inputs to generate tokenized sequences
        suitable for the LUKE model. It handles different task types (entity classification,
        entity pair classification, entity span classification) and creates appropriate
        token sequences with special entity markers when needed.

        Parameters:
            text (Union[TextInput]): The primary text sequence to be tokenized. Must be a string
                that will be processed using the tokenizer's BPE encoding.
            text_pair (Optional[Union[TextInput]], optional): The secondary text sequence for
                sequence pair tasks. Defaults to None.
            entities (Optional[EntityInput], optional): List of entity names corresponding to
                the entity spans in the primary text. If None and entity_spans is provided,
                entities will be filled with [MASK] tokens. Defaults to None.
            entities_pair (Optional[EntityInput], optional): List of entity names corresponding
                to the entity spans in the secondary text. Only used when text_pair is provided.
                Defaults to None.
            entity_spans (Optional[EntitySpanInput], optional): List of tuples containing
                character-based start and end positions of entities in the primary text.
                Each tuple should be (start_pos, end_pos). Defaults to None.
            entity_spans_pair (Optional[EntitySpanInput], optional): List of tuples containing
                character-based start and end positions of entities in the secondary text.
                Only used when text_pair is provided. Defaults to None.
            **kwargs: Additional keyword arguments passed to the tokenize method.

        Returns:
            tuple[list, list, list, list, list, list]: A 6-tuple containing:
                - first_ids (list): Token IDs for the primary text sequence
                - second_ids (list): Token IDs for the secondary text sequence (None if no text_pair)
                - first_entity_ids (list): Entity IDs for entities in the primary text (None if no entities)
                - second_entity_ids (list): Entity IDs for entities in the secondary text (None if no entities_pair)
                - first_entity_token_spans (list): Token-based spans for entities in primary text (None if no entity_spans)
                - second_entity_token_spans (list): Token-based spans for entities in secondary text (None if no entity_spans_pair)

        Important Notes:
            - The behavior varies significantly based on the tokenizer's task configuration:
                * None/entity_span_classification: Processes entities as provided
                * entity_classification: Expects exactly one entity span and adds special tokens
                * entity_pair_classification: Expects exactly two entity spans and adds different special tokens
            - Entity spans are converted from character-based to token-based positions
            - Special entity tokens (<ent>, <ent2>) are inserted around entity mentions for classification tasks
            - Invalid entity spans (extending beyond text boundaries) are handled gracefully
            - The method performs input validation for entity formats and task requirements

        Raises:
            ValueError: If entity_spans format is invalid, if entities and entity_spans have mismatched lengths,
                if task requirements are not met (wrong number of entity spans for classification tasks),
                or if an unsupported task is specified.
            TypeError: If entity_spans is not provided as a list when entities are specified.
        """
        <your code>

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
        <your code>

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
        <your code>

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
        <your code>

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
        <your code>

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None
    ) -> tuple[str]:
        """
        Save the tokenizer's vocabulary files to a specified directory.

        This method saves three vocabulary files required by the LUKE tokenizer:
        1. The main vocabulary file (vocab.json) containing token-to-ID mappings
        2. The BPE merges file (merges.txt) containing byte-pair encoding merge rules
        3. The entity vocabulary file (entity_vocab.json) containing entity-to-ID mappings

        Parameters:
            save_directory (str): The directory path where the vocabulary files will be saved. 
                Must be an existing directory.
            filename_prefix (Optional[str], optional): An optional prefix to add to the 
                vocabulary filenames. If provided, filenames will be formatted as 
                "{prefix}-{original_filename}". Defaults to None.

        Returns:
            tuple[str]: A tuple containing the full paths to the three saved vocabulary files:
                - Path to the saved vocab.json file
                - Path to the saved merges.txt file  
                - Path to the saved entity_vocab.json file

        Important Notes:
            - The save_directory must be an existing directory, otherwise an error will be logged
            - The method saves vocabulary in JSON format with proper indentation and UTF-8 encoding
            - BPE merge indices are validated for consistency during saving
            - If BPE merge indices are not consecutive, a warning will be logged
            - All files are saved with UTF-8 encoding to ensure proper character handling
        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.