## Task
## Task: Implement KTO (Kahneman-Tversky Optimization) Model Training System

**Core Functionality:**
Build a preference-based model alignment system using prospect theory principles to train language models on human feedback data containing desirable/undesirable response pairs.

**Main Features & Requirements:**
- Process datasets with prompt-completion pairs labeled as chosen/rejected
- Implement KTO loss computation with policy and reference model comparison
- Support both encoder-decoder and decoder-only model architectures
- Handle data preprocessing including tokenization, truncation, and batch creation
- Provide configurable training parameters (beta, loss types, weights)
- Enable efficient training with optional PEFT integration and reference model caching

**Key Challenges:**
- Maintain proper batch structure for KL divergence estimation through sequential sampling
- Balance computational efficiency between policy and reference model inference
- Handle mixed precision training and memory optimization for large models
- Ensure robust data collation with appropriate padding and label masking
- Implement prospect theory-based loss functions that account for asymmetric preference learning

The system should seamlessly integrate with HuggingFace transformers while providing specialized KTO-specific optimizations for preference-based model alignment.

**NOTE**: 
- This test comes from the `trl` library, and we have given you the content of this code repository under `/testbed/`, and you need to complete based on this code repository and supplement the files we specify. Remember, all your changes must be in this codebase, and changes that are not in this codebase will not be discovered and tested by us.
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
from trl import KTOConfig
```
This means that we will test one function/class: KTOConfig.
And the defination and implementation of class KTOConfig should be in `/testbed/trl/trainer/kto_config.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
class KTOConfig(TrainingArguments):
    """
    
        Configuration class for the [`KTOTrainer`].
    
        This class includes only the parameters that are specific to KTO training. For a full list of training arguments,
        please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
        differ from those in [`~transformers.TrainingArguments`].
    
        Using [`~transformers.HfArgumentParser`] we can turn this class into
        [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
        command line.
    
        Parameters:
            max_length (`int` or `None`, *optional*, defaults to `1024`):
                Maximum length of the sequences (prompt + completion) in the batch. This argument is required if you want
                to use the default data collator.
            max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
                Maximum length of the prompt. This argument is required if you want to use the default data collator.
            max_completion_length (`int`, *optional*):
                Maximum length of the completion. This argument is required if you want to use the default data collator
                and your model is an encoder-decoder.
            beta (`float`, *optional*, defaults to `0.1`):
                Parameter controlling the deviation from the reference model. Higher β means less deviation from the
                reference model.
            loss_type (`str`, *optional*, defaults to `"kto"`):
                Type of loss to use. Possible values are:
    
                    - `"kto"`: KTO loss from the [KTO](https://huggingface.co/papers/2402.01306) paper.
                    - `"apo_zero_unpaired"`: Unpaired variant of APO-zero loss from the
                      [APO](https://huggingface.co/papers/2408.06266) paper.
    
            desirable_weight (`float`, *optional*, defaults to `1.0`):
                Desirable losses are weighed by this factor to counter unequal number of desirable and undesirable paris.
            undesirable_weight (`float`, *optional*, defaults to `1.0`):
                Undesirable losses are weighed by this factor to counter unequal number of desirable and undesirable pairs.
            label_pad_token_id (`int`, *optional*, defaults to `-100`):
                Label pad token id. This argument is required if you want to use the default data collator.
            padding_value (`int`, *optional*):
                Padding value to use. If `None`, the padding value of the tokenizer is used.
            truncation_mode (`str`, *optional*, defaults to `"keep_end"`):
                Truncation mode to use when the prompt is too long. Possible values are `"keep_end"` or `"keep_start"`.
                This argument is required if you want to use the default data collator.
            generate_during_eval (`bool`, *optional*, defaults to `False`):
                If `True`, generates and logs completions from both the model and the reference model to W&B or Comet
                during evaluation.
            is_encoder_decoder (`bool`, *optional*):
                When using the `model_init` argument (callable) to instantiate the model instead of the `model` argument,
                you need to specify if the model returned by the callable is an encoder-decoder model.
            precompute_ref_log_probs (`bool`, *optional*, defaults to `False`):
                Whether to precompute reference model log probabilities for training and evaluation datasets. This is
                useful when training without the reference model to reduce the total GPU memory needed.
            model_init_kwargs (`dict[str, Any]`, *optional*):
                Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model from a
                string.
            ref_model_init_kwargs (`dict[str, Any]`, *optional*):
                Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the reference model
                from a string.
            dataset_num_proc: (`int`, *optional*):
                Number of processes to use for processing the dataset.
            disable_dropout (`bool`, *optional*, defaults to `True`):
                Whether to disable dropout in the model and reference model.
            use_liger_loss (`bool`, *optional*, defaults to `False`):
                Whether to use Liger loss. It requires liger-kernel to be installed.
            base_model_attribute_name (`str`, *optional*, defaults to `"model"`):
                Name of the attribute in the model that contains the base model. This is used to get the base model from
                the model when the model does not have a `get_decoder` method in the case when `use_liger_loss` is `True`.
        
    """

    _VALID_DICT_FIELDS = "TrainingArguments._VALID_DICT_FIELDS + ['model_init_kwargs', 'ref_model_init_kwargs']"
    learning_rate = "field(default=1e-06, metadata={'help': 'The initial learning rate for AdamW.'})"
    logging_steps = "field(default=10, metadata={'help': 'Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps.'})"
    gradient_checkpointing = "field(default=True, metadata={'help': 'If True, use gradient checkpointing to save memory at the expense of slower backward pass.'})"
    bf16 = "field(default=None, metadata={'help': 'Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if `fp16` is not set.'})"
    max_length = "field(default=1024, metadata={'help': 'Maximum length of the sequences (prompt + completion) in the batch.'})"
    max_prompt_length = "field(default=512, metadata={'help': 'Maximum length of the prompt. This argument is required if you want to use the default data collator and your model is an encoder-decoder.'})"
    max_completion_length = "field(default=None, metadata={'help': 'Maximum length of the completion. This argument is required if you want to use the default data collator and your model is an encoder-decoder.'})"
    beta = "field(default=0.1, metadata={'help': 'Parameter controlling the deviation from the reference model. Higher β means less deviation from the reference model.'})"
    loss_type = "field(default='kto', metadata={'help': 'Type of loss to use.', 'choices': ['kto', 'apo_zero_unpaired']})"
    desirable_weight = "field(default=1.0, metadata={'help': 'Desirable losses are weighed by this factor to counter unequal number of desirable and undesirable pairs.'})"
    undesirable_weight = "field(default=1.0, metadata={'help': 'Undesirable losses are weighed by this factor to counter unequal number of desirable and undesirable pairs.'})"
    label_pad_token_id = "field(default=-100, metadata={'help': 'Label pad token id. This argument is required if you want to use the default data collator.'})"
    padding_value = "field(default=None, metadata={'help': 'Padding value to use. If `None`, the padding value of the tokenizer is used.'})"
    truncation_mode = "field(default='keep_end', metadata={'help': 'Truncation mode to use when the prompt is too long.', 'choices': ['keep_end', 'keep_start']})"
    generate_during_eval = "field(default=False, metadata={'help': 'If `True`, generates and logs completions from both the model and the reference model to W&B during evaluation.'})"
    is_encoder_decoder = "field(default=None, metadata={'help': 'When using the `model_init` argument (callable) to instantiate the model instead of the `model` argument, you need to specify if the model returned by the callable is an encoder-decoder model.'})"
    disable_dropout = "field(default=True, metadata={'help': 'Whether to disable dropout in the model.'})"
    precompute_ref_log_probs = "field(default=False, metadata={'help': 'Whether to precompute reference model log probabilities for training and evaluation datasets. This is useful when training without the reference model to reduce the total GPU memory needed.'})"
    model_init_kwargs = "field(default=None, metadata={'help': 'Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model from a string.'})"
    ref_model_init_kwargs = "field(default=None, metadata={'help': 'Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the reference model from a string.'})"
    dataset_num_proc = "field(default=None, metadata={'help': 'Number of processes to use for processing the dataset.'})"
    use_liger_loss = "field(default=False, metadata={'help': 'Whether to use Liger loss. It requires liger-kernel to be installed.'})"
    base_model_attribute_name = "field(default='model', metadata={'help': 'Name of the attribute in the model that contains the base model. This is used to get the base model from the model when the model does not have a `get_decoder` method in the case when `use_liger_loss` is `True`.'})"

    def __post_init__(self):
        """
        Post-initialization method for KTOConfig that sets default values and validates configuration.

        This method is automatically called after the dataclass instance is created. It performs
        necessary setup and validation of the KTOConfig parameters before the configuration is
        used by the KTOTrainer.

        Parameters:
            self: The KTOConfig instance being initialized.

        Return value description:
            None. This method modifies the instance in-place.

        Important notes or exceptions:
            - Automatically sets bf16 to True if fp16 is False and bf16 is None, otherwise
              preserves the explicitly set bf16 value
            - Calls the parent TrainingArguments.__post_init__() method to ensure proper
              initialization of inherited configuration parameters
            - This method is part of the dataclass post-initialization protocol and should
              not be called manually
            - Any validation errors from the parent TrainingArguments.__post_init__() method
              will be propagated
        """
        <your code>
...
```

In order to implement this functionality, some additional libraries etc. are often required, I don't restrict you to any libraries, you need to think about what dependencies you might need and fetch and install and call them yourself. The only thing is that you **MUST** fulfill the input/output format described by this interface, otherwise the test will not pass and you will get zero points for this feature.
And note that there may be not only one **Interface Description**, you should match all **Interface Description {n}**

### Test Description 1
Below is **Test Description 1**
```python
from trl import KTOConfig
from trl import KTOTrainer
from trl.trainer.kto_trainer import _get_kl_dataset
from trl.trainer.kto_trainer import _process_tokens
from trl.trainer.kto_trainer import _tokenize
```

### Interface Description 1
Below is **Interface Description 1** for file: trl-trainer-kto_trainer.py

This file contains 4 top-level interface(s) that need to be implemented.

```python
def _get_kl_dataset(
    batch: dict[str, list[Any]]
) -> dict[str, list[Any]]:
    """
    
        Creates mismatched pairs of prompts and completions for the KL dataset by adding a +1 offset to the order of
        completions. For best results, the mismatched outputs y' used to estimate the KL term for a batch should be the
        same set as the matched outputs y used to estimate the rewards in that batch, just paired with different x.
        
    """
    <your code>

def _tokenize(
    batch: dict[str, list[Any]],
    tokenizer: 'PreTrainedTokenizer'
) -> dict[str, list[Any]]:
    """
    Tokenize a batch from a KTO specific dataset.
    """
    <your code>

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
    <your code>

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
        """
        Initialize a KTO (Kahneman-Tversky Optimization) Trainer for model alignment using prospect theory.

        The KTOTrainer implements the KTO algorithm for aligning language models with human preferences
        using prospect theory principles. It supports both encoder-decoder and decoder-only architectures,
        with optional PEFT (Parameter Efficient Fine-Tuning) integration and reference model comparison.

        Parameters:
            model (Union[PreTrainedModel, nn.Module, str], optional): 
                The model to train. Can be a PreTrainedModel instance, nn.Module, or string path to model.
                If string, will be loaded using AutoModelForCausalLM.from_pretrained().

            ref_model (Optional[Union[PreTrainedModel, nn.Module, str]], optional):
                Reference model for computing implicit rewards and KL divergence. If None and not using
                PEFT, a reference model will be created automatically. If using PEFT, the base model
                with adapters disabled serves as reference.

            args (KTOConfig, optional):
                Training configuration containing hyperparameters, loss settings, and training options.
                Must be KTOConfig instance, not TrainingArguments.

            train_dataset (Optional[Dataset], optional):
                Training dataset containing prompts, completions, and preference labels. Should have
                columns: 'prompt', 'completion', 'label' (boolean indicating desirable/undesirable).

            eval_dataset (Optional[Union[Dataset, dict[str, Dataset]]], optional):
                Evaluation dataset(s) with same format as train_dataset. Can be single Dataset or
                dictionary mapping names to datasets.

            processing_class (Optional[Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]], optional):
                Tokenizer or processor for handling model inputs. Required for automatic data processing
                and will be saved with the model for easier reuse.

            data_collator (Optional[DataCollator], optional):
                Custom data collator for batching. If None, uses DPODataCollatorWithPadding with
                automatic padding to maximum sequence length in batch.

            model_init (Optional[Callable[[], PreTrainedModel]], optional):
                Function that returns a fresh model instance. Used for hyperparameter search or
                when model needs to be reinitialized during training.

            callbacks (Optional[list[TrainerCallback]], optional):
                List of callback functions to customize training behavior (logging, early stopping, etc.).

            optimizers (tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR], optional):
                Tuple of (optimizer, scheduler). If (None, None), default optimizer and scheduler
                from args will be used.

            preprocess_logits_for_metrics (Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]], optional):
                Function to preprocess model logits before metric computation during evaluation.

            peft_config (Optional[dict], optional):
                PEFT configuration dictionary for parameter-efficient fine-tuning. If provided,
                model will be wrapped with PEFT. Requires peft library installation.

            compute_metrics (Optional[Callable[[EvalLoopOutput], dict]], optional):
                Function to compute custom metrics during evaluation. Takes EvalLoopOutput and
                returns dictionary of metric names to values.

            model_adapter_name (Optional[str], optional):
                Name of the target PEFT adapter to train when using LoRA with multiple adapters.
                Defaults to "default" if not specified.

            ref_adapter_name (Optional[str], optional):
                Name of the reference PEFT adapter when using LoRA with multiple adapters.
                Used for computing reference model outputs.

        Raises:
            ValueError: If args is TrainingArguments instead of KTOConfig, if model and ref_model
                are the same object, if model_init_kwargs provided with instantiated model,
                if invalid dtype in model_init_kwargs, if PEFT config provided without peft installation,
                if generate_during_eval=True without wandb/comet, if processing_class is None,
                if per_device_train_batch_size <= 1 (required for KL term calculation),
                if precompute_ref_log_probs=True with DeepSpeed ZeRO-3, if no reference model
                and model is not PEFT without precompute_ref_log_probs=True.

            ImportError: If use_liger_loss=True but liger-kernel not installed.

            AttributeError: If Trainer missing accelerator object (upgrade transformers).

        Important Notes:
            - KTO requires actual batch size > 1 for proper KL term calculation
            - The trainer automatically handles dataset preprocessing including tokenization,
              chat template application, and KL dataset creation
            - Reference log probabilities can be precomputed for efficiency when precompute_ref_log_probs=True
            - Supports both "kto" and "apo_zero_unpaired" loss types
            - Automatically balances desirable/undesirable examples and warns about suboptimal weights
            - DeepSpeed ZeRO-3 incompatible with precomputed reference log probabilities
            - Liger kernel integration available for optimized loss computation
            - Model card automatically generated and saved with checkpoints
        """
        <your code>

    @contextmanager
    def null_ref_context(self):
        """
        Context manager for handling null reference model (that is, peft adapter manipulation).
        """
        <your code>

    def get_train_dataloader(self) -> DataLoader:
        """

                Returns the training [`~torch.utils.data.DataLoader`].

                Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.

        """
        <your code>

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
        <your code>

    def compute_reference_log_probs(self, padded_batch: dict) -> dict:
        """
        Computes log probabilities of the reference model for a single padded batch of a KTO specific dataset.
        """
        <your code>

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
        <your code>

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, Union[list, torch.LongTensor]]
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Performs a forward pass through the model to compute log probabilities and logits for KTO training.

        This method processes a batch of training data through the policy model to compute completion log probabilities,
        chosen/rejected logits, and KL divergence terms needed for KTO (Kahneman-Tversky Optimization) loss calculation.
        It handles both encoder-decoder and decoder-only model architectures.

        Args:
            model (nn.Module): The policy model to perform forward pass on. Should be a PyTorch neural network module,
                typically a transformer-based language model.
            batch (dict[str, Union[list, torch.LongTensor]]): A dictionary containing the batch data with the following keys:
                - "completion_input_ids": Input token IDs for the full completion sequences
                - "completion_attention_mask": Attention mask for completion sequences  
                - "completion_labels": Labels for completion sequences (for loss computation)
                - "label": Binary labels indicating chosen (True) vs rejected (False) responses
                - "completion_decoder_input_ids": Decoder input IDs (for encoder-decoder models only)
                - Additional KL-related keys if KL calculation is enabled

        Returns:
            tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]: A tuple containing:
                - chosen_logps: Log probabilities for chosen responses, shape (num_chosen,)
                - rejected_logps: Log probabilities for rejected responses, shape (num_rejected,) 
                - chosen_logits: Raw logits for chosen responses, shape (num_chosen, seq_len, vocab_size)
                - rejected_logits: Raw logits for rejected responses, shape (num_rejected, seq_len, vocab_size)
                - KL_logps: KL divergence log probabilities if calculate_KL is True, otherwise None
                - aux_loss: Auxiliary loss if auxiliary loss is enabled, otherwise not included

        Raises:
            ValueError: If there is a mismatch between the number of examples in the batch and the number of 
                predicted output sequences, indicating an inconsistency in batch processing.

        Important Notes:
            - The method automatically separates chosen and rejected examples based on the "label" field
            - For encoder-decoder models, additional decoder input IDs and labels are used
            - KL log probabilities are computed separately if self.calculate_KL is True
            - Auxiliary loss from mixture-of-experts models is included if self.aux_loss_enabled is True
            - The method handles gradient computation appropriately for training
        """
        <your code>

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
        <your code>

    def _compute_kl_logps(self, model, batch):
        """
        Compute KL log probabilities for a given batch.
        """
        <your code>

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
        <your code>

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, Union[list, torch.LongTensor]]
    ):
        """
        Compute the KTO loss and other metrics for the given batch of inputs for train or test.
        """
        <your code>

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs = False,
        num_items_in_batch = None
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        """
        Compute the KTO (Kahneman-Tversky Optimization) loss for the given model and inputs.

        This method serves as the main loss computation function for KTO training. It processes the input batch through the model, computes policy and reference log probabilities, and calculates the KTO loss based on the specified loss type. The method handles both chosen (desirable) and rejected (undesirable) examples, applying appropriate weights and computing auxiliary losses when enabled.

        Parameters:
            model (Union[PreTrainedModel, nn.Module]): The policy model to compute loss for. Can be either a HuggingFace PreTrainedModel or a PyTorch nn.Module.
            inputs (dict[str, Union[torch.Tensor, Any]]): Dictionary containing the input batch data including:
                - completion_input_ids: Token IDs for the full completion sequences
                - completion_attention_mask: Attention masks for completion sequences  
                - completion_labels: Labels for computing loss
                - label: Binary labels indicating chosen (True) vs rejected (False) examples
                - prompt_input_ids: Token IDs for prompt sequences (encoder-decoder models)
                - KL_* keys: Corresponding inputs for KL divergence computation (if enabled)
                - reference_logps: Precomputed reference log probabilities (optional)
            return_outputs (bool, optional): Whether to return additional outputs along with the loss. Defaults to False.
            num_items_in_batch (int, optional): Number of items in the batch. Currently unused but kept for compatibility. Defaults to None.

        Returns:
            Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]: 
                If return_outputs is False, returns only the computed loss tensor.
                If return_outputs is True, returns a tuple containing:
                    - loss (torch.Tensor): The computed KTO loss
                    - metrics (dict[str, torch.Tensor]): Dictionary of computed metrics including:
                        - kl: KL divergence between policy and reference models
                        - rewards/chosen_sum: Sum of rewards for chosen examples
                        - rewards/rejected_sum: Sum of rewards for rejected examples  
                        - logps/*_sum: Sum of log probabilities for chosen/rejected examples
                        - logits/*_sum: Sum of logits for chosen/rejected examples
                        - count/*: Counts of chosen/rejected examples

        Important notes:
            - The method automatically handles mixed precision training when PEFT models are cast to bf16
            - Metrics are stored internally and logged on the main process only
            - The loss is moved to the appropriate device specified in training arguments
            - Auxiliary loss from mixture-of-experts models is added when enabled
            - The method supports both standard KTO loss and APO (Alignment via Prospect Optimization) variants
            - KL divergence computation can be disabled for certain loss types like 'apo_zero_unpaired'
        """
        <your code>

    def store_metrics(
        self,
        metrics: dict[str, float],
        train_eval: Literal['train', 'eval'] = 'train'
    ) -> None:
        """
        Store training or evaluation metrics in the internal metrics storage.

        This method accumulates metrics from training or evaluation batches into the trainer's
        internal storage system for later aggregation and logging. The metrics are organized
        by training phase (train/eval) and metric type.

        Parameters:
            metrics (dict[str, float]): A dictionary containing metric names as keys and their
                corresponding float values. Common metrics include loss values, rewards,
                log probabilities, logits, and counts for chosen/rejected samples.
            train_eval (Literal['train', 'eval'], optional): Specifies whether the metrics
                are from training or evaluation phase. Defaults to 'train'.

        Returns:
            None: This method modifies the internal state by storing metrics in
                self._stored_metrics but does not return any value.

        Notes:
            - Metrics are stored in a nested defaultdict structure organized by phase
              (train/eval) and then by metric name
            - The stored metrics are later processed and aggregated in the log() method
            - This method is typically called automatically during training and evaluation
              loops to accumulate batch-level metrics
            - The accumulated metrics are used to compute averages and other aggregate
              statistics across batches
        """
        <your code>

    def _get_train_sampler(
        self,
        dataset: Optional[Dataset] = None
    ) -> Optional[torch.utils.data.Sampler]:
        """
        Get the training data sampler for the KTO trainer.

        This method creates and returns a data sampler specifically for the training dataset.
        Unlike the default Trainer behavior which may use random sampling, this implementation
        always returns a SequentialSampler to ensure deterministic ordering of training data,
        which is important for KTO's batch-based KL divergence calculations.

        Parameters:
            dataset (Optional[Dataset], default=None): The dataset to create a sampler for.
                If None, uses self.train_dataset. The dataset should be a valid Dataset
                object with a defined length.

        Returns:
            Optional[torch.utils.data.Sampler]: A SequentialSampler instance if the dataset
                is valid and has length, None otherwise. The SequentialSampler ensures that
                data is accessed in sequential order rather than randomly shuffled.

        Important notes:
            - This method overrides the parent Trainer's _get_train_sampler to enforce
              sequential sampling, which is crucial for KTO's algorithm that relies on
              batch structure for KL term estimation
            - Returns None if the dataset is None or doesn't have a defined length,
              following the same pattern as the parent class
            - The sequential ordering is essential because KTO creates mismatched pairs
              within each batch for KL divergence estimation by applying offsets to
              completion ordering
        """
        <your code>

    def generate_from_model_and_ref(
        self,
        model,
        batch: dict[str, torch.LongTensor]
    ) -> tuple[str, str]:
        """
        Generate samples from the model and reference model for the given batch of inputs.
        """
        <your code>

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None
    ):
        """
        Perform a single prediction step on a batch of inputs during evaluation or inference.

        This method processes a batch of inputs through the model to compute loss and metrics,
        and optionally returns predictions. It handles the KTO-specific loss computation and
        metric collection for both training and evaluation phases.

        Parameters:
            model (Union[PreTrainedModel, nn.Module]): The model to use for prediction. Can be
                either a HuggingFace PreTrainedModel or a PyTorch nn.Module.
            inputs (dict[str, Union[torch.Tensor, Any]]): Dictionary containing the input data
                for the batch. Expected to contain keys like 'completion_input_ids', 
                'completion_attention_mask', 'completion_labels', and 'label' among others.
            prediction_loss_only (bool): If True, only returns the loss without computing
                predictions or additional outputs. If False, also returns logits and labels.
            ignore_keys (Optional[list[str]], optional): List of keys to ignore when extracting
                predictions from the model outputs. If None, uses the model's default ignored
                keys from config.keys_to_ignore_at_inference. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - loss (torch.Tensor): The computed loss for the batch (detached from computation graph)
                - logits (torch.Tensor or None): Model logits if prediction_loss_only is False, None otherwise
                - labels (torch.Tensor or None): Labels tensor if prediction_loss_only is False, None otherwise

                When prediction_loss_only is False, logits contains chosen and rejected logits sums,
                and labels is a zero tensor with matching dimensions.

        Important notes:
            - The method automatically handles mixed precision training contexts when PEFT models
              are cast to bf16
            - Metrics are automatically stored for later logging when running on the main process
            - The method uses torch.no_grad() context to prevent gradient computation during inference
            - Loss computation follows the KTO (Kahneman-Tversky Optimization) algorithm for
              preference-based model alignment
        """
        <your code>

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
        <your code>

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
        <your code>

    def _save_checkpoint(self, model, trial):
        """
        Save a model checkpoint during training.

        This method extends the parent class's checkpoint saving functionality by automatically
        creating a model card before saving the checkpoint. The model card contains metadata
        about the training process, model configuration, and relevant citations.

        Parameters:
            model: The model instance to be saved in the checkpoint. This should be the trained
                model that will be serialized along with its current state.
            trial: Trial information used by hyperparameter optimization frameworks like Optuna.
                Contains trial-specific metadata that may be included in the checkpoint.

        Important notes:
            - The model card is created using either the hub_model_id (if specified) or the
              output directory name as the model name
            - The model card generation only occurs on the main process to avoid conflicts
              in distributed training scenarios
            - This method calls the parent class's _save_checkpoint method after creating
              the model card, ensuring all standard checkpoint saving functionality is preserved
            - The model card is saved as README.md in the output directory and includes
              information about the KTO training process, citations, and model metadata
        """
        <your code>

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
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.

### Interface Description 2
Below is **Interface Description 2** for file: trl-trainer-kto_config.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
@dataclass
class KTOConfig(TrainingArguments):
    """
    
        Configuration class for the [`KTOTrainer`].
    
        This class includes only the parameters that are specific to KTO training. For a full list of training arguments,
        please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
        differ from those in [`~transformers.TrainingArguments`].
    
        Using [`~transformers.HfArgumentParser`] we can turn this class into
        [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
        command line.
    
        Parameters:
            max_length (`int` or `None`, *optional*, defaults to `1024`):
                Maximum length of the sequences (prompt + completion) in the batch. This argument is required if you want
                to use the default data collator.
            max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
                Maximum length of the prompt. This argument is required if you want to use the default data collator.
            max_completion_length (`int`, *optional*):
                Maximum length of the completion. This argument is required if you want to use the default data collator
                and your model is an encoder-decoder.
            beta (`float`, *optional*, defaults to `0.1`):
                Parameter controlling the deviation from the reference model. Higher β means less deviation from the
                reference model.
            loss_type (`str`, *optional*, defaults to `"kto"`):
                Type of loss to use. Possible values are:
    
                    - `"kto"`: KTO loss from the [KTO](https://huggingface.co/papers/2402.01306) paper.
                    - `"apo_zero_unpaired"`: Unpaired variant of APO-zero loss from the
                      [APO](https://huggingface.co/papers/2408.06266) paper.
    
            desirable_weight (`float`, *optional*, defaults to `1.0`):
                Desirable losses are weighed by this factor to counter unequal number of desirable and undesirable paris.
            undesirable_weight (`float`, *optional*, defaults to `1.0`):
                Undesirable losses are weighed by this factor to counter unequal number of desirable and undesirable pairs.
            label_pad_token_id (`int`, *optional*, defaults to `-100`):
                Label pad token id. This argument is required if you want to use the default data collator.
            padding_value (`int`, *optional*):
                Padding value to use. If `None`, the padding value of the tokenizer is used.
            truncation_mode (`str`, *optional*, defaults to `"keep_end"`):
                Truncation mode to use when the prompt is too long. Possible values are `"keep_end"` or `"keep_start"`.
                This argument is required if you want to use the default data collator.
            generate_during_eval (`bool`, *optional*, defaults to `False`):
                If `True`, generates and logs completions from both the model and the reference model to W&B or Comet
                during evaluation.
            is_encoder_decoder (`bool`, *optional*):
                When using the `model_init` argument (callable) to instantiate the model instead of the `model` argument,
                you need to specify if the model returned by the callable is an encoder-decoder model.
            precompute_ref_log_probs (`bool`, *optional*, defaults to `False`):
                Whether to precompute reference model log probabilities for training and evaluation datasets. This is
                useful when training without the reference model to reduce the total GPU memory needed.
            model_init_kwargs (`dict[str, Any]`, *optional*):
                Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model from a
                string.
            ref_model_init_kwargs (`dict[str, Any]`, *optional*):
                Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the reference model
                from a string.
            dataset_num_proc: (`int`, *optional*):
                Number of processes to use for processing the dataset.
            disable_dropout (`bool`, *optional*, defaults to `True`):
                Whether to disable dropout in the model and reference model.
            use_liger_loss (`bool`, *optional*, defaults to `False`):
                Whether to use Liger loss. It requires liger-kernel to be installed.
            base_model_attribute_name (`str`, *optional*, defaults to `"model"`):
                Name of the attribute in the model that contains the base model. This is used to get the base model from
                the model when the model does not have a `get_decoder` method in the case when `use_liger_loss` is `True`.
        
    """

    _VALID_DICT_FIELDS = "TrainingArguments._VALID_DICT_FIELDS + ['model_init_kwargs', 'ref_model_init_kwargs']"
    learning_rate = "field(default=1e-06, metadata={'help': 'The initial learning rate for AdamW.'})"
    logging_steps = "field(default=10, metadata={'help': 'Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps.'})"
    gradient_checkpointing = "field(default=True, metadata={'help': 'If True, use gradient checkpointing to save memory at the expense of slower backward pass.'})"
    bf16 = "field(default=None, metadata={'help': 'Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if `fp16` is not set.'})"
    max_length = "field(default=1024, metadata={'help': 'Maximum length of the sequences (prompt + completion) in the batch.'})"
    max_prompt_length = "field(default=512, metadata={'help': 'Maximum length of the prompt. This argument is required if you want to use the default data collator and your model is an encoder-decoder.'})"
    max_completion_length = "field(default=None, metadata={'help': 'Maximum length of the completion. This argument is required if you want to use the default data collator and your model is an encoder-decoder.'})"
    beta = "field(default=0.1, metadata={'help': 'Parameter controlling the deviation from the reference model. Higher β means less deviation from the reference model.'})"
    loss_type = "field(default='kto', metadata={'help': 'Type of loss to use.', 'choices': ['kto', 'apo_zero_unpaired']})"
    desirable_weight = "field(default=1.0, metadata={'help': 'Desirable losses are weighed by this factor to counter unequal number of desirable and undesirable pairs.'})"
    undesirable_weight = "field(default=1.0, metadata={'help': 'Undesirable losses are weighed by this factor to counter unequal number of desirable and undesirable pairs.'})"
    label_pad_token_id = "field(default=-100, metadata={'help': 'Label pad token id. This argument is required if you want to use the default data collator.'})"
    padding_value = "field(default=None, metadata={'help': 'Padding value to use. If `None`, the padding value of the tokenizer is used.'})"
    truncation_mode = "field(default='keep_end', metadata={'help': 'Truncation mode to use when the prompt is too long.', 'choices': ['keep_end', 'keep_start']})"
    generate_during_eval = "field(default=False, metadata={'help': 'If `True`, generates and logs completions from both the model and the reference model to W&B during evaluation.'})"
    is_encoder_decoder = "field(default=None, metadata={'help': 'When using the `model_init` argument (callable) to instantiate the model instead of the `model` argument, you need to specify if the model returned by the callable is an encoder-decoder model.'})"
    disable_dropout = "field(default=True, metadata={'help': 'Whether to disable dropout in the model.'})"
    precompute_ref_log_probs = "field(default=False, metadata={'help': 'Whether to precompute reference model log probabilities for training and evaluation datasets. This is useful when training without the reference model to reduce the total GPU memory needed.'})"
    model_init_kwargs = "field(default=None, metadata={'help': 'Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model from a string.'})"
    ref_model_init_kwargs = "field(default=None, metadata={'help': 'Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the reference model from a string.'})"
    dataset_num_proc = "field(default=None, metadata={'help': 'Number of processes to use for processing the dataset.'})"
    use_liger_loss = "field(default=False, metadata={'help': 'Whether to use Liger loss. It requires liger-kernel to be installed.'})"
    base_model_attribute_name = "field(default='model', metadata={'help': 'Name of the attribute in the model that contains the base model. This is used to get the base model from the model when the model does not have a `get_decoder` method in the case when `use_liger_loss` is `True`.'})"

    def __post_init__(self):
        """
        Post-initialization method for KTOConfig that sets default values and validates configuration.

        This method is automatically called after the dataclass instance is created. It performs
        necessary setup and validation of the KTOConfig parameters before the configuration is
        used by the KTOTrainer.

        Parameters:
            self: The KTOConfig instance being initialized.

        Return value description:
            None. This method modifies the instance in-place.

        Important notes or exceptions:
            - Automatically sets bf16 to True if fp16 is False and bf16 is None, otherwise
              preserves the explicitly set bf16 value
            - Calls the parent TrainingArguments.__post_init__() method to ensure proper
              initialization of inherited configuration parameters
            - This method is part of the dataclass post-initialization protocol and should
              not be called manually
            - Any validation errors from the parent TrainingArguments.__post_init__() method
              will be propagated
        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.