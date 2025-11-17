## Task
## Task: Multi-Modal Processor Implementation for Protein-Text Models

**Core Functionality:**
Implement a dual-tokenizer processor system that handles both protein sequences (amino acids + structural data) and natural language text within a unified interface, supporting automatic model loading and registration.

**Main Features & Requirements:**
- Factory pattern for automatic processor instantiation based on model configuration
- Combined processing of protein data (amino acid sequences with foldseek structural information) and conversational text
- Bidirectional encoding/decoding capabilities for both modalities
- Standardized save/load functionality with proper directory structure management
- Registration system for custom processor-model mappings

**Key Challenges:**
- Seamless integration of heterogeneous tokenizers (protein vs. text) with different vocabularies and sequence formats
- Proper handling of variable-length sequences across modalities with appropriate padding/truncation
- Maintaining chat template formatting while preserving protein structure-aware (SA) sequence encoding
- Ensuring consistent batch processing and tensor shape management across different input types

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
from agent_code.transformers import AutoProcessor
```
This means that we will test one function/class: AutoProcessor.
And the defination and implementation of class AutoProcessor should be in `/testbed/agent_code/transformers.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
class AutoProcessor:
    """
    
        This is a generic processor class that will be instantiated as one of the processor classes of the library when
        created with the [`AutoProcessor.from_pretrained`] class method.
    
        This class cannot be instantiated directly using `__init__()` (throws an error).
        
    """

    def __init__(self):
        """
        Initialize the AutoProcessor class.

        This constructor is intentionally disabled and will raise an OSError when called directly.
        The AutoProcessor class is designed to be a factory class that automatically selects and
        instantiates the appropriate processor class based on the model configuration.

        Parameters:
            None

        Returns:
            None: This method does not return anything as it raises an exception.

        Raises:
            OSError: Always raised when this constructor is called directly. The error message
                indicates that AutoProcessor should be instantiated using the class method
                `AutoProcessor.from_pretrained(pretrained_model_name_or_path)` instead.

        Notes:
            - This class cannot be instantiated directly using `__init__()`
            - Use `AutoProcessor.from_pretrained()` class method to create processor instances
            - The actual processor class instantiated depends on the model type and configuration
            - This design pattern ensures that the correct processor class is automatically
              selected based on the pretrained model being loaded
        """
        <your code>
...
```

In order to implement this functionality, some additional libraries etc. are often required, I don't restrict you to any libraries, you need to think about what dependencies you might need and fetch and install and call them yourself. The only thing is that you **MUST** fulfill the input/output format described by this interface, otherwise the test will not pass and you will get zero points for this feature.
And note that there may be not only one **Interface Description**, you should match all **Interface Description {n}**

### Test Description 1
Below is **Test Description 1**
```python
from agent_code.transformers import AutoProcessor
from agent_code.transformers import EvollaProcessor
```

### Interface Description 1
Below is **Interface Description 1** for file: src-transformers-models-auto-processing_auto.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
class AutoProcessor:
    """
    
        This is a generic processor class that will be instantiated as one of the processor classes of the library when
        created with the [`AutoProcessor.from_pretrained`] class method.
    
        This class cannot be instantiated directly using `__init__()` (throws an error).
        
    """

    def __init__(self):
        """
        Initialize the AutoProcessor class.

        This constructor is intentionally disabled and will raise an OSError when called directly.
        The AutoProcessor class is designed to be a factory class that automatically selects and
        instantiates the appropriate processor class based on the model configuration.

        Parameters:
            None

        Returns:
            None: This method does not return anything as it raises an exception.

        Raises:
            OSError: Always raised when this constructor is called directly. The error message
                indicates that AutoProcessor should be instantiated using the class method
                `AutoProcessor.from_pretrained(pretrained_model_name_or_path)` instead.

        Notes:
            - This class cannot be instantiated directly using `__init__()`
            - Use `AutoProcessor.from_pretrained()` class method to create processor instances
            - The actual processor class instantiated depends on the model type and configuration
            - This design pattern ensures that the correct processor class is automatically
              selected based on the pretrained model being loaded
        """
        <your code>

    @classmethod
    @replace_list_option_in_docstrings(PROCESSOR_MAPPING_NAMES)
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        **kwargs
    ):
        """

                Instantiate one of the processor classes of the library from a pretrained model vocabulary.

                The processor class to instantiate is selected based on the `model_type` property of the config object (either
                passed as an argument or loaded from `pretrained_model_name_or_path` if possible):

                List options

                Params:
                    pretrained_model_name_or_path (`str` or `os.PathLike`):
                        This can be either:

                        - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on
                          huggingface.co.
                        - a path to a *directory* containing a processor files saved using the `save_pretrained()` method,
                          e.g., `./my_model_directory/`.
                    cache_dir (`str` or `os.PathLike`, *optional*):
                        Path to a directory in which a downloaded pretrained model feature extractor should be cached if the
                        standard cache should not be used.
                    force_download (`bool`, *optional*, defaults to `False`):
                        Whether or not to force to (re-)download the feature extractor files and override the cached versions
                        if they exist.
                    resume_download:
                        Deprecated and ignored. All downloads are now resumed by default when possible.
                        Will be removed in v5 of Transformers.
                    proxies (`dict[str, str]`, *optional*):
                        A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                        'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
                    token (`str` or *bool*, *optional*):
                        The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                        when running `hf auth login` (stored in `~/.huggingface`).
                    revision (`str`, *optional*, defaults to `"main"`):
                        The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                        git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                        identifier allowed by git.
                    return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                        If `False`, then this function returns just the final feature extractor object. If `True`, then this
                        functions returns a `Tuple(feature_extractor, unused_kwargs)` where *unused_kwargs* is a dictionary
                        consisting of the key/value pairs whose keys are not feature extractor attributes: i.e., the part of
                        `kwargs` which has not been used to update `feature_extractor` and is otherwise ignored.
                    trust_remote_code (`bool`, *optional*, defaults to `False`):
                        Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                        should only be set to `True` for repositories you trust and in which you have read the code, as it will
                        execute code present on the Hub on your local machine.
                    kwargs (`dict[str, Any]`, *optional*):
                        The values in kwargs of any keys which are feature extractor attributes will be used to override the
                        loaded values. Behavior concerning key/value pairs whose keys are *not* feature extractor attributes is
                        controlled by the `return_unused_kwargs` keyword parameter.

                <Tip>

                Passing `token=True` is required when you want to use a private model.

                </Tip>

                Examples:

                ```python
                >>> from transformers import AutoProcessor

                >>> # Download processor from huggingface.co and cache.
                >>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")

                >>> # If processor files are in a directory (e.g. processor was saved using *save_pretrained('./test/saved_model/')*)
                >>> # processor = AutoProcessor.from_pretrained("./test/saved_model/")
                ```
        """
        <your code>

    @staticmethod
    def register(config_class, processor_class, exist_ok = False):
        """

                Register a new processor for this class.

                Args:
                    config_class ([`PretrainedConfig`]):
                        The configuration corresponding to the model to register.
                    processor_class ([`ProcessorMixin`]): The processor to register.

        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.

### Interface Description 2
Below is **Interface Description 2** for file: src-transformers-models-evolla-processing_evolla.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
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
        """
        Initialize an EvollaProcessor instance that combines protein and text tokenizers.

        This constructor sets up a processor that wraps both a protein tokenizer (for handling amino acid sequences and structural information) and a text tokenizer (for handling natural language) into a unified interface for the EVOLLA model.

        Parameters:
            protein_tokenizer: An instance of a protein tokenizer (typically EsmTokenizer or AutoTokenizer) used to tokenize protein sequences. This parameter is required and cannot be None.
            tokenizer (optional): An instance of a text tokenizer (typically LlamaTokenizerFast or AutoTokenizer) used to tokenize natural language text. Defaults to None but is required - will raise ValueError if not provided.
            protein_max_length (int, optional): Maximum sequence length for protein tokenization. Sequences longer than this will be truncated. Defaults to 1024.
            text_max_length (int, optional): Maximum sequence length for text tokenization. Text longer than this will be truncated. Defaults to 512.
            **kwargs: Additional keyword arguments passed to the parent ProcessorMixin class.

        Returns:
            None: This is a constructor method that initializes the instance.

        Raises:
            ValueError: If protein_tokenizer is None - a protein tokenizer must be provided.
            ValueError: If tokenizer is None - a text tokenizer must be provided.

        Notes:
            - The text tokenizer's pad_token is automatically set to "<|reserved_special_token_0|>" during initialization.
            - The processor inherits from ProcessorMixin and maintains references to both tokenizers as instance attributes.
            - The protein_max_length and text_max_length parameters set default values that can be overridden in individual method calls.
        """
        <your code>

    def process_proteins(
        self,
        proteins,
        protein_max_length = 1024
    ):
        """
        Process a list of protein dictionaries into tokenized format suitable for the model.

        This method takes protein data containing amino acid sequences and foldseek structural information,
        combines them into SA (Structure-Aware) sequences, and tokenizes them using the protein tokenizer.
        Each protein's amino acid sequence and foldseek string are merged by alternating uppercase amino
        acids with lowercase foldseek characters.

        Args:
            proteins (list[dict]): A list of dictionaries where each dictionary represents a protein
                and contains the following keys:
                - "aa_seq" (str): The amino acid sequence of the protein
                - "foldseek" (str): The foldseek structural string corresponding to the amino acid sequence
            protein_max_length (int, optional): Maximum length for protein sequence tokenization.
                Sequences longer than this will be truncated. Defaults to 1024.

        Returns:
            dict: A dictionary containing tokenized protein data with the following keys:
                - "input_ids": Tensor of shape (batch_size, sequence_length) containing token IDs
                - "attention_mask": Tensor of shape (batch_size, sequence_length) containing attention masks
                - Additional keys may be present depending on the protein tokenizer configuration

        Notes:
            - The amino acid sequence and foldseek string must have the same length for proper pairing
            - The resulting SA sequence alternates between uppercase amino acids and lowercase foldseek characters
            - Sequences are automatically padded to the same length within the batch
            - Truncation is applied if sequences exceed protein_max_length
        """
        <your code>

    def process_text(
        self,
        texts,
        text_max_length: int = 512
    ):
        """
        Process text messages by applying chat templates and tokenizing them for model input.

        This method takes a list of conversation messages, applies the appropriate chat template
        to format them as prompts, and then tokenizes the resulting text for use with the model.
        The text is processed with padding, truncation, and proper formatting to ensure
        consistent input dimensions.

        Args:
            texts (Union[List[List[dict]], List[dict]]): A list of message conversations where each
                conversation is a list of dictionaries containing message data. Each message
                dictionary should have 'role' and 'content' keys. Can also be a single conversation
                (list of message dictionaries).
            text_max_length (int, optional): Maximum length for the tokenized text sequences.
                Sequences longer than this will be truncated. Defaults to 512.

        Returns:
            transformers.tokenization_utils_base.BatchEncoding: A batch encoding object containing
                the tokenized text data with the following keys:
                - 'input_ids': Token IDs for the processed text sequences
                - 'attention_mask': Attention masks indicating which tokens should be attended to
                - Additional tokenizer-specific keys as returned by the underlying tokenizer

        Notes:
            - The method applies chat templates using the tokenizer's apply_chat_template method
              with add_generation_prompt=True to format conversations appropriately
            - Text sequences are padded to the longest sequence in the batch for consistent dimensions
            - Special tokens are not automatically added (add_special_tokens=False) as they are
              handled by the chat template
            - The method assumes the tokenizer has an apply_chat_template method available
        """
        <your code>

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
        <your code>

    def batch_decode(self, *args, **kwargs):
        """
        Decodes a batch of token sequences back to text strings using the text tokenizer.

        This method is a wrapper around the underlying text tokenizer's batch_decode method,
        providing convenient access to batch decoding functionality for text sequences processed
        by the EVOLLA processor.

        Args:
            *args: Variable length argument list passed directly to the underlying tokenizer's
                batch_decode method. Typically includes:
                - token_ids (torch.Tensor or List[List[int]]): Batch of token sequences to decode
                - skip_special_tokens (bool, optional): Whether to remove special tokens from output
                - clean_up_tokenization_spaces (bool, optional): Whether to clean up tokenization spaces
            **kwargs: Arbitrary keyword arguments passed directly to the underlying tokenizer's
                batch_decode method. Common options include padding and truncation parameters.

        Returns:
            List[str]: A list of decoded text strings corresponding to the input token sequences.
                Each string represents the decoded text from one sequence in the batch.

        Note:
            This method specifically uses the text tokenizer (LlamaTokenizerFast) for decoding.
            For decoding protein sequences, use the protein_batch_decode method instead.
            All arguments are passed through directly to the underlying tokenizer, so refer
            to the LlamaTokenizerFast documentation for complete parameter details.
        """
        <your code>

    def decode(self, *args, **kwargs):
        """
        Decodes token IDs back to human-readable text using the text tokenizer.

        This method serves as a wrapper around the underlying text tokenizer's decode functionality,
        allowing for the conversion of token IDs generated by the model back into readable text strings.
        It delegates all decoding operations to the text tokenizer component of the processor.

        Args:
            *args: Variable length argument list passed directly to the underlying text tokenizer's
                decode method. Typically includes:
                - token_ids: The token IDs to decode (torch.Tensor, list, or numpy array)
                - skip_special_tokens (bool, optional): Whether to remove special tokens from 
                  the decoded text
                - clean_up_tokenization_spaces (bool, optional): Whether to clean up extra 
                  whitespace in the decoded text
            **kwargs: Arbitrary keyword arguments passed directly to the underlying text 
                tokenizer's decode method. May include additional tokenizer-specific parameters.

        Returns:
            str: The decoded text string corresponding to the input token IDs.

        Note:
            This method specifically uses the text tokenizer (LlamaTokenizerFast) for decoding.
            For protein sequence decoding, use the protein_decode method instead. The exact
            behavior and supported parameters depend on the underlying text tokenizer's 
            implementation.
        """
        <your code>

    def protein_batch_decode(self, *args, **kwargs):
        """
        Decodes a batch of protein token sequences back to their string representations.

        This method serves as a wrapper around the protein tokenizer's batch_decode functionality,
        allowing for the conversion of tokenized protein sequences (such as those containing
        amino acid and foldseek structure information) back to their original string format.

        Args:
            *args: Variable length argument list passed directly to the underlying protein
                tokenizer's batch_decode method. Typically includes:
                - token_ids: A batch of token ID sequences to decode
                - skip_special_tokens: Whether to remove special tokens from output
                - clean_up_tokenization_spaces: Whether to clean up tokenization artifacts
            **kwargs: Arbitrary keyword arguments passed directly to the underlying protein
                tokenizer's batch_decode method. May include tokenizer-specific parameters
                for controlling the decoding behavior.

        Returns:
            List[str]: A list of decoded protein sequence strings. Each string represents
                the decoded form of the corresponding input token sequence, typically
                containing amino acid sequences combined with structural information
                (foldseek tokens) in the SA (Structure-Aware) format.

        Notes:
            - This method specifically handles protein sequences that have been tokenized
              using the processor's protein_tokenizer (EsmTokenizer)
            - The decoded sequences are in SA format where amino acids are uppercase
              and foldseek structural tokens are lowercase
            - For decoding regular text sequences, use the standard batch_decode method instead
            - All arguments are forwarded to the underlying tokenizer, so refer to the
              protein tokenizer's documentation for specific parameter details
        """
        <your code>

    def protein_decode(self, *args, **kwargs):
        """
        Decodes protein token IDs back to their original protein sequence representation.

        This method is a wrapper around the protein tokenizer's decode functionality, allowing for the conversion of tokenized protein sequences (generated by the protein tokenizer) back into their string representation. It supports all the same arguments and options as the underlying protein tokenizer's decode method.

        Args:
            *args: Variable length argument list passed directly to the protein tokenizer's decode method.
                Typically includes:
                - token_ids: The token IDs to decode (usually a list or tensor of integers)
            **kwargs: Arbitrary keyword arguments passed directly to the protein tokenizer's decode method.
                Common options include:
                - skip_special_tokens (bool): Whether to remove special tokens from the decoded string
                - clean_up_tokenization_spaces (bool): Whether to clean up extra spaces in the decoded string

        Returns:
            str: The decoded protein sequence string. For EVOLLA proteins, this will be in the SA (Structure-Aware) format where amino acid sequences are represented with uppercase letters followed by corresponding foldseek structure tokens in lowercase.

        Notes:
            - This method specifically decodes protein sequences that were tokenized using the protein_tokenizer (EsmTokenizer)
            - The decoded output will be in SA format: amino acids (uppercase) + foldseek tokens (lowercase)
            - For batch decoding of multiple sequences, use protein_batch_decode instead
            - All arguments are passed through to the underlying protein tokenizer's decode method without modification
        """
        <your code>

    def save_pretrained(self, save_directory, **kwargs):
        """
        Save the processor's tokenizers to the specified directory.

        This method saves the EVOLLA processor by storing the protein tokenizer in a separate
        subdirectory and the text tokenizer in the main directory. The protein tokenizer is
        saved in a subfolder named 'protein_tokenizer' to maintain proper organization and
        enable correct loading later.

        Args:
            save_directory (str): The directory path where the processor should be saved.
                Both the main text tokenizer and the protein tokenizer (in a subdirectory)
                will be saved to this location.
            **kwargs: Additional keyword arguments passed to the parent class's save_pretrained
                method. These may include parameters like push_to_hub, commit_message, etc.

        Returns:
            The return value from the parent class's save_pretrained method, typically
            containing information about the saved files and their locations.

        Important Notes:
            - The protein tokenizer is saved in a subdirectory named 'protein_tokenizer'
              within the specified save_directory
            - The text tokenizer is saved in the main save_directory
            - The method temporarily modifies the processor's attributes list to ensure
              only the text tokenizer is saved in the main directory, then restores the
              original attributes list
            - The directory structure created by this method is required for proper loading
              using the from_pretrained class method

        Raises:
            OSError: If the save_directory cannot be created or written to
            ValueError: If save_directory is not a valid path string
        """
        <your code>

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        **kwargs
    ):
        """
        Load a pretrained EVOLLA processor from a model repository or local directory.

        This class method creates an instance of EvollaProcessor by loading both the main tokenizer and protein tokenizer from a pretrained model. The protein tokenizer is loaded from a subdirectory specified by `protein_tokenizer_dir_name`.

        Args:
            pretrained_model_name_or_path (str): 
                The model repository name on Hugging Face Hub or path to a local directory containing the pretrained processor files. The directory should contain the main tokenizer files and a subdirectory with the protein tokenizer.
            **kwargs: 
                Additional keyword arguments passed to the parent class's `from_pretrained` method. These may include parameters like `cache_dir`, `force_download`, `local_files_only`, etc.

        Returns:
            EvollaProcessor: An instance of the EvollaProcessor class with both the main tokenizer and protein tokenizer loaded and configured.

        Raises:
            OSError: If the pretrained model path does not exist or required tokenizer files are missing.
            ValueError: If the tokenizer configuration is invalid or incompatible.

        Notes:
            - The protein tokenizer is expected to be located in a subdirectory named according to `protein_tokenizer_dir_name` (default: "protein_tokenizer").
            - This method automatically handles the loading of both tokenizers and properly initializes the processor with default parameters.
            - The returned processor will have the pad token set to "<|reserved_special_token_0|>" for the main tokenizer.
        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.