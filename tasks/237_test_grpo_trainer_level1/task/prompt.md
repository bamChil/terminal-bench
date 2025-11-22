## Task
## Task: Implement Group Relative Policy Optimization (GRPO) Trainer for Language Model Fine-tuning

**Core Functionality:**
Develop a specialized trainer that fine-tunes language models using reward-based optimization through the GRPO algorithm, which generates multiple completions per prompt and optimizes the policy using relative advantages within groups.

**Main Features & Requirements:**
- Support multiple reward functions (model-based or custom) for evaluating generated text quality
- Handle both text-only and multimodal (vision-language) inputs with appropriate tokenization
- Implement efficient batch generation with completion reuse across multiple training steps
- Provide distributed training support (FSDP, DeepSpeed) with proper parameter synchronization
- Integrate with vLLM backend for accelerated inference during training
- Support parameter-efficient fine-tuning (PEFT) and importance sampling corrections
- Include comprehensive logging, evaluation, and checkpoint management

**Key Challenges:**
- **Memory efficiency**: Managing large generation batches while avoiding OOM during distributed training
- **Reward computation**: Coordinating multi-process reward calculation and proper advantage normalization across devices  
- **Generation-training synchronization**: Efficiently reusing completions across gradient steps while maintaining model weight consistency
- **Backend integration**: Seamlessly switching between standard transformers and vLLM generation with parameter sync
- **Multimodal handling**: Processing diverse input types (text, images) through different model architectures and reward functions

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
from trl import GRPOConfig
```
This means that we will test one function/class: GRPOConfig.
And the defination and implementation of class GRPOConfig should be in `/testbed/trl/trainer/grpo_config.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
class GRPOTrainer(Trainer):
    """
    
        Trainer for the Group Relative Policy Optimization 
...
```

In order to implement this functionality, some additional libraries etc. are often required, I don't restrict you to any libraries, you need to think about what dependencies you might need and fetch and install and call them yourself. The only thing is that you **MUST** fulfill the input/output format described by this interface, otherwise the test will not pass and you will get zero points for this feature.
And note that there may be not only one **Interface Description**, you should match all **Interface Description {n}**

### Test Description 1
Below is **Test Description 1**
```python
from trl import GRPOConfig
from trl import GRPOTrainer
```

### Interface Description 1
Below is **Interface Description 1** for file: trl-trainer-grpo_trainer.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
class GRPOTrainer(Trainer):
    """
    
        Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
        paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
        Models](https://huggingface.co/papers/2402.03300).
    
        Example:
    
        ```python
        from datasets import load_dataset
        from trl import GRPOTrainer
    
        dataset = load_dataset("trl-lib/tldr", split="train")
    
    
        def reward_func(completions, **kwargs):
            # Dummy reward function that rewards completions with more unique letters.
            return [float(len(set(completion))) for completion in completions]
    
    
        trainer = GRPOTrainer(
            model="Qwen/Qwen2-0.5B-Instruct",
            reward_funcs=reward_func,
            train_dataset=dataset,
        )
    
        trainer.train()
        ```
    
        Args:
            model (`Union[str, PreTrainedModel]`):
                Model to be trained. Can be either:
    
                - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
                  path to a *directory* containing model weights saved using
                  [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                  using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
                  `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
            reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
                Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
                functions with the prompts and completions and sum the rewards. Can be either:
    
                - A single reward function, such as:
                    - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                    path to a *directory* containing model weights saved using
                    [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                    using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                    keyword arguments in `args.model_init_kwargs`.
                    - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                    - A custom reward function: The function is provided with the prompts and the generated completions,
                      plus any additional columns in the dataset. It should return a list of rewards. Custom reward
                      functions can also return `None` when the reward is not applicable to those samples. This is useful
                      for multi-task training where different reward functions apply to different types of samples. When a
                      reward function returns `None` for a sample, that reward function is excluded from the reward
                      calculation for that sample. For more details, see [Using a custom reward
                      function](#using-a-custom-reward-function).
    
                      The trainer's state is also passed to the reward function. The trainer's state is an instance of
                      [`~transformers.TrainerState`] and can be accessed by accessing the `trainer_state` argument to the
                      reward function's signature.
                - A list of reward functions, where each item can independently be any of the above types. Mixing different
                types within the list (e.g., a string model ID and a custom reward function) is allowed.
            args ([`GRPOConfig`], *optional*):
                Configuration for this trainer. If `None`, a default configuration is used.
            train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
                Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
                ignored. The format of the samples can be either:
    
                - [Standard](dataset_formats#standard): Each sample contains plain text.
                - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
                  and content).
            eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
                Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
            processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
                Processing class used to process the data. The padding side must be set to "left". If `None`, the
                processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
                padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
                `tokenizer.eos_token` will be used as the default.
            reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*):
                Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:
    
                - A single processing class: Used when `reward_funcs` contains only one reward function.
                - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
                If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
                `None`, the tokenizer for the model is automatically loaded using
                [`~transformers.AutoTokenizer.from_pretrained`]. For elements in `reward_funcs` that are custom reward
                functions (not [`~transformers.PreTrainedModel`]), the corresponding entries in `reward_processing_classes`
                are ignored.
            callbacks (list of [`~transformers.TrainerCallback`], *optional*):
                List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
                in [here](https://huggingface.co/docs/transformers/main_classes/callback).
    
                If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
                method.
            optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
                A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
                model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
            peft_config ([`~peft.PeftConfig`], *optional*):
                PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
        
    """

    _tag_names = ['trl', 'grpo']

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional['PeftConfig'] = None
    ):
        """
        Initialize a GRPOTrainer instance for Group Relative Policy Optimization training.

        This trainer implements the GRPO algorithm for training language models using reward-based optimization.
        It supports multiple reward functions, importance sampling, and various generation backends including
        transformers and vLLM.

        Parameters:
            model (Union[str, PreTrainedModel]): 
                Model to be trained. Can be either a string model ID/path for loading with 
                AutoModelForCausalLM.from_pretrained, or an already instantiated PreTrainedModel.
                Only causal language models are supported.

            reward_funcs (Union[RewardFunc, list[RewardFunc]]): 
                Reward functions for computing rewards. Can be a single function or list of functions.
                Each function can be: a string model ID for sequence classification, a PreTrainedModel
                for sequence classification, or a custom callable that takes prompts and completions
                and returns rewards. Custom functions can return None for inapplicable samples.

            args (Optional[GRPOConfig], default=None): 
                Training configuration. If None, uses default GRPOConfig with model name.

            train_dataset (Optional[Union[Dataset, IterableDataset]], default=None): 
                Training dataset. Must include a "prompt" column. Supports both standard text
                and conversational message formats.

            eval_dataset (Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]], default=None): 
                Evaluation dataset(s). Must meet same requirements as train_dataset.

            processing_class (Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]], default=None): 
                Tokenizer or processor for text processing. Must have padding_side="left".
                If None, loads from model using AutoProcessor.from_pretrained.

            reward_processing_classes (Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]], default=None): 
                Processing classes for reward model inputs. Should match the order and length
                of reward_funcs. If None, automatically loads tokenizers for PreTrainedModel
                reward functions.

            callbacks (Optional[list[TrainerCallback]], default=None): 
                Additional training callbacks to customize the training loop.

            optimizers (tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]], default=(None, None)): 
                Custom optimizer and scheduler. If None, uses AdamW with linear warmup schedule.

            peft_config (Optional[PeftConfig], default=None): 
                PEFT configuration for parameter-efficient fine-tuning. If None, trains full model.

        Raises:
            ValueError: If reward_weights length doesn't match reward_funcs length, or if
                reward_processing_classes length doesn't match reward_funcs length.
            ImportError: If required dependencies (liger-kernel, vllm) are missing when
                corresponding features are enabled.
            NotImplementedError: If using IterableDataset (not yet supported), or incompatible
                combinations of liger_loss with certain sampling configurations.

        Notes:
            - Automatically handles model preparation for distributed training (DeepSpeed, FSDP)
            - Sets up reference model for KL divergence computation when beta > 0
            - Configures vLLM backend if use_vllm=True in args
            - Initializes reward models and processing classes
            - Sets device-specific random seeds for generation diversity
            - Supports both PEFT and full model training
            - Handles multimodal inputs when images are present in dataset
        """
        <your code>

    def _set_signature_columns_if_needed(self):
        """
        Set signature columns for the trainer if they haven't been set already.

        This method overrides the parent class implementation to customize column handling for GRPO training.
        In the standard Trainer, signature columns are set based on the model's expected inputs (like "input_ids").
        However, in GRPOTrainer, data is preprocessed differently and the model signature columns don't apply.
        Instead, this method sets the signature columns to match what the training_step method expects.

        The signature columns determine which columns are kept when `args.remove_unused_columns` is True.
        By setting them to ["prompt", "image"], the trainer will preserve these columns during data preprocessing
        and remove any other columns that aren't needed for training.

        This method is called automatically by the Trainer framework during initialization and doesn't need
        to be called manually.

        Note:
            This method modifies the instance variable `self._signature_columns` in place. It only sets the
            columns if they haven't been set already (i.e., if `self._signature_columns` is None).
        """
        <your code>

    def get_train_dataloader(self):
        """
        Retrieves the training dataloader for the GRPO trainer.

        This method overrides the parent class implementation to support GRPO's custom batching strategy.
        Instead of returning a standard per-step batch (i.e., `per_device_batch_size`), the dataloader loads
        a generation batch (i.e., `per_device_batch_size × steps_per_generation`). This allows the trainer
        to generate completions once every `steps_per_generation` steps rather than once per accumulation
        step, which is significantly more efficient.

        The method multiplies the standard batch size by `steps_per_generation`, so `_prepare_inputs` is
        called with this larger generation batch and handles the splitting internally across multiple
        training steps.

        Parameters:
            self: The GRPOTrainer instance.

        Returns:
            torch.utils.data.DataLoader: A DataLoader configured for GRPO training with batch size equal to
                `_train_batch_size * steps_per_generation`. The dataloader uses the custom RepeatSampler
                to ensure proper prompt distribution across processes and generation reuse across multiple
                updates.

        Raises:
            ValueError: If `train_dataset` is None, as training requires a dataset.

        Notes:
            - This is a copy-paste of the parent `Trainer.get_train_dataloader` method with only the batch
              size calculation modified to include `steps_per_generation` multiplier
            - Some parts of the method may not be directly relevant to GRPO but are kept for compatibility
              and easier maintenance
            - The dataloader is prepared by the accelerator for distributed training support
            - Uses a custom sampler (`RepeatSampler`) that handles prompt repetition and batch reuse
              according to GRPO's multi-iteration training strategy
        """
        <your code>

    def _get_train_sampler(
        self,
        dataset: Optional[Dataset] = None
    ) -> Sampler:
        """
        Get the training sampler for the dataset.

        This method returns a custom sampler that handles the specific requirements of GRPO training:
        1. Ensures each prompt is repeated across multiple processes to guarantee identical prompts
           are distributed to different GPUs for proper reward computation and normalization
        2. Repeats batches multiple times to allow reusing generations across multiple updates
        3. Uses consistent seeding across processes to prevent discrepancies in group formation

        The sampler creates a pattern where prompts are duplicated according to `num_generations`
        and repeated for `num_iterations * steps_per_generation` to support the multi-step training
        approach where completions are generated once and reused across multiple gradient updates.

        Parameters:
            dataset (Optional[Dataset], optional): The dataset to sample from. If None, uses
                `self.train_dataset`. Defaults to None.

        Returns:
            Sampler: A RepeatSampler configured for GRPO training that handles prompt repetition
                and batch reuse according to the training configuration.

        Notes:
            - The sampler ensures deterministic prompt assignment across processes using a shared seed
            - Batch size is calculated as `generation_batch_size // num_generations` 
            - Total repeat count is `num_iterations * steps_per_generation` to support completion reuse
            - Shuffling is controlled by the `shuffle_dataset` configuration parameter
        """
        <your code>

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        """
        Get the evaluation sampler for the given evaluation dataset.

        This method creates a RepeatSampler specifically configured for evaluation purposes.
        Unlike the training sampler, the evaluation sampler does not include multiple
        iterations or complex batching strategies since evaluation is performed once
        per dataset without gradient accumulation or multi-step generation reuse.

        Args:
            eval_dataset: The evaluation dataset to create a sampler for. This should be
                a Dataset object that will be used during the evaluation phase.

        Returns:
            Sampler: A RepeatSampler instance configured for evaluation with the following
                characteristics:
                - Uses the provided eval_dataset as the data source
                - Repeats each sample num_generations times to match the generation strategy
                - Uses the same seed as specified in training arguments for reproducibility
                - Does not include the complex repeat_count logic used in training

        Note:
            The sampler ensures that each prompt in the evaluation dataset is repeated
            num_generations times, which is necessary for the GRPO algorithm to generate
            multiple completions per prompt for reward computation and comparison. The
            seed parameter ensures reproducible evaluation results across different runs.
        """
        <your code>

    @profiling_decorator
    def _get_last_hidden_state(
        self,
        unwrapped_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values = None,
        image_grid_thw = None,
        pixel_attention_mask = None,
        image_sizes = None
    ):
        """
        Extract the last hidden state from a model for completion tokens.

        This method computes the last hidden state of the model for the completion portion of the input sequence. It handles various multimodal inputs (images, pixel values, attention masks) and supports models with or without the `logits_to_keep` parameter optimization.

        Args:
            unwrapped_model: The unwrapped model instance to extract hidden states from. If the model is a PEFT model, it will be further unwrapped to access the base model.
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, sequence_length) containing both prompt and completion tokens.
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, sequence_length) indicating which tokens should be attended to.
            logits_to_keep (int): Number of completion tokens to keep from the end of the sequence. Used to extract only the relevant hidden states.
            pixel_values (torch.Tensor, optional): Pixel values for image inputs, used by vision-language models. Shape depends on the specific model architecture.
            image_grid_thw (torch.Tensor, optional): Image grid dimensions (time, height, width) for models like Qwen that require this information.
            pixel_attention_mask (torch.Tensor, optional): Attention mask for pixel values, used by models like SmolVLM2.
            image_sizes (torch.Tensor, optional): Original image sizes before preprocessing, used by models like LLaVa-Next.

        Returns:
            torch.Tensor: Last hidden state tensor of shape (batch_size, logits_to_keep, hidden_size) containing the hidden representations for the completion tokens only.

        Notes:
            - The method automatically detects whether the model supports the `logits_to_keep` parameter and adjusts the forward pass accordingly
            - For models that don't support `logits_to_keep`, the method computes the full forward pass and then slices the output
            - The last token's hidden state is excluded as it corresponds to the next token prediction
            - Caching is disabled during the forward pass to avoid memory issues during training
            - The method is decorated with `@profiling_decorator` for performance monitoring
        """
        <your code>

    def get_high_entropy_mask(
        self,
        entropies: torch.Tensor,
        mask: torch.Tensor,
        threshold: float
    ) -> torch.Tensor:
        """

                Returns a binary mask identifying tokens whose entropy exceeds a given quantile threshold.

                Args:
                    entropies (`torch.Tensor`):
                        Tensor of shape (batch_size, seq_len) with per-token entropy values.
                    mask (`torch.Tensor`):
                        Binary mask of the same shape as `entropies`, where `1` indicates valid tokens and `0` padding.
                    threshold (`float`):
                        Quantile threshold between `0.0` and `1.0` to select high-entropy tokens.

                Returns:
                    `torch.Tensor`:
                        Boolean mask of shape (batch_size, seq_len), where `True` indicates tokens with entropy >= threshold
                        and `False` otherwise.

        """
        <your code>

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size = None,
        compute_entropy = False,
        pixel_values = None,
        image_grid_thw = None,
        pixel_attention_mask = None,
        image_sizes = None
    ) -> dict[str, Optional[torch.Tensor]]:
        """
        Compute log-probs and (optionally) entropies for each token.
        """
        <your code>

    def _fix_param_name_to_vllm(
        self,
        name,
        extra_prefixes: Optional[list[str]] = None
    ):
        """
        Convert parameter names from checkpoint format to vLLM-compatible format by removing specific prefixes.

        This method handles parameter name transformations needed when synchronizing model weights
        with vLLM instances. It removes checkpoint-specific prefixes that are not recognized by vLLM's
        parameter loading system.

        Args:
            name (str): The original parameter name from the model checkpoint that needs to be
                converted to vLLM format.
            extra_prefixes (Optional[list[str]], optional): Additional prefixes to remove beyond
                the default ones. If None, only default prefixes are removed. Defaults to None.

        Returns:
            str: The transformed parameter name with all specified prefixes removed, making it
                compatible with vLLM's parameter naming convention.

        Notes:
            - Default prefixes that are always removed include "_checkpoint_wrapped_module."
            - This method is essential for proper weight synchronization between the training
              model and vLLM inference engine
            - The order of prefix removal follows the order in the prefixes list
            - If a parameter name doesn't contain any of the specified prefixes, it is returned unchanged
        """
        <your code>

    def _sync_fsdp1_params_to_vllm(
        self,
        module: nn.Module,
        prefix: str = '',
        visited = None
    ):
        """
        Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM.
        """
        <your code>

    def _sync_fsdp2_params_to_vllm(self, module: nn.Module):
        """
        Synchronizes FSDP2 (Fully Sharded Data Parallel version 2) model parameters to vLLM engine.

        This method extracts parameters from an FSDP2-wrapped PyTorch module and updates the corresponding
        parameters in the vLLM engine. Unlike FSDP1, FSDP2's state_dict() method already provides access
        to all parameters without requiring recursive traversal of child modules.

        Args:
            module (nn.Module): The FSDP2-wrapped PyTorch module whose parameters need to be synchronized
                to vLLM. This should be a model that has been prepared with FSDP2 sharding.

        Returns:
            None: This method performs in-place parameter synchronization and does not return any value.

        Notes:
            - This method handles both "server" and "colocate" vLLM modes differently:
                * In "server" mode: Parameters are sent to the vLLM server via the client interface
                  (only executed on the main process)
                * In "colocate" mode: Parameters are directly loaded into the colocated vLLM engine
            - Parameters are automatically moved to CUDA if they are on CPU
            - The method calls full_tensor() on parameters to ensure they are fully materialized
              before synchronization
            - Parameter names are processed through _fix_param_name_to_vllm() to ensure compatibility
              with vLLM's expected naming conventions
            - This is a memory-efficient approach compared to FSDP1 as it doesn't require manual
              recursion through the module hierarchy

        Important:
            - This method should only be called when using FSDP2 for model sharding
            - Ensure the vLLM engine is properly initialized before calling this method
            - The method assumes parameters can fit in GPU memory after being moved to CUDA
        """
        <your code>

    @profiling_decorator
    def _move_model_to_vllm(self):
        """
        Move the current model weights to the vLLM inference engine for generation.

        This method synchronizes the training model's parameters with the vLLM engine to ensure
        that completions are generated using the most up-to-date model weights. The synchronization
        process varies depending on the distributed training setup (DeepSpeed ZeRO-3, FSDP, or
        standard training) and whether PEFT adapters are being used.

        The method handles different scenarios:
        - For PEFT models: Temporarily merges adapters, syncs weights, then unmerges
        - For FSDP: Uses memory-efficient post-order traversal to gather and sync parameters
        - For DeepSpeed ZeRO-3: Gathers parameters using DeepSpeed's context manager
        - For standard models: Directly syncs parameters without special handling

        Important notes:
        - This method is only called when using vLLM for generation (use_vllm=True)
        - Parameter gathering is memory-intensive and may cause OOM for very large models
        - The method automatically handles parameter name transformations required by vLLM
        - After syncing weights, the vLLM prefix cache is reset to ensure consistency
        - For PEFT models with FSDP/DeepSpeed, all parameters must be gathered before merging adapters

        Exceptions:
        - May raise CUDA out-of-memory errors if the model is too large to gather on a single device
        - Network timeouts may occur when syncing with vLLM server mode
        - FSDP parameter gathering may fail if the model structure is incompatible
        """
        <your code>

    @profiling_decorator
    def _prepare_inputs(
        self,
        generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Prepares inputs for model training/evaluation by managing completion generation and batch handling.

        This method serves as the main input preparation pipeline for the GRPO trainer, handling different
        behaviors for training and evaluation modes. It manages completion generation, batch splitting,
        and input buffering to optimize the training process.

        Args:
            generation_batch (dict[str, Union[torch.Tensor, Any]]): A dictionary containing the input batch
                data. During training, this is a local generation batch with size 
                (per_device_train_batch_size × steps_per_generation). During evaluation, this is a standard
                local batch. The dictionary must contain a "prompt" key and may contain additional keys
                like "image" for multimodal inputs.

        Returns:
            dict[str, Union[torch.Tensor, Any]]: A single local batch prepared for model training or
                evaluation. The returned dictionary contains processed inputs including:
                - prompt_ids: Tokenized prompt sequences
                - completion_ids: Generated completion sequences  
                - attention_mask: Attention masks for the sequences
                - advantages: Computed advantage values for training
                - Various other keys depending on model type (e.g., pixel_values for vision models)

        Important Notes:
            - During training, the method implements an optimization where completions are generated once
              every (steps_per_generation × num_iterations) steps and then reused across multiple
              accumulation steps by buffering and splitting the generation batch
            - During evaluation, completions are generated fresh for each batch without buffering
            - The method handles both text-only and multimodal inputs (with images)
            - Completion generation can use either transformers generation, vLLM, or paged attention
              depending on the trainer configuration
            - The method manages reward calculation, advantage computation, and various metrics logging
            - Memory optimization is performed through batch splitting and selective computation
        """
        <your code>

    @profiling_decorator
    def _calculate_rewards(
        self,
        inputs,
        prompts,
        completions,
        completion_ids_list
    ):
        """
        Calculate rewards for generated completions using the configured reward functions.

        This method computes rewards for each completion by calling all configured reward functions
        and aggregating their outputs. It handles both model-based reward functions (PreTrainedModel)
        and custom callable reward functions, applying proper tokenization and processing for each type.

        Args:
            inputs (list[dict]): List of input examples containing prompt and completion data.
                Each dictionary should contain keys like "prompt", "completion", "completion_ids"
                and any additional columns from the dataset.
            prompts (list): List of prompts corresponding to the completions. Format depends on
                whether conversational data is used (list of message dicts vs plain strings).
            completions (list): List of generated completions. Format matches prompts - either
                list of message dicts for conversational data or plain strings.
            completion_ids_list (list[list[int]]): List of tokenized completion sequences as
                lists of token IDs, used by reward functions that operate on tokens directly.

        Returns:
            torch.Tensor: Tensor of shape (num_completions, num_reward_functions) containing
                rewards from each reward function for each completion. Values are gathered
                across all processes for proper normalization. NaN values indicate cases
                where a reward function returned None for that completion.

        Notes:
            - Rewards are computed by calling each reward function with prompts, completions,
              completion_ids, and any additional dataset columns as keyword arguments
            - For PreTrainedModel reward functions, inputs are tokenized using the corresponding
              reward_processing_class and passed through the model to get logits
            - Custom reward functions receive the trainer state for dynamic reward shaping
            - All rewards are gathered across processes since normalization happens per group
              and completions may be distributed across different devices
            - The method handles conversational vs standard text formats automatically
            - Reward functions can return None to indicate the reward is not applicable,
              which gets converted to NaN in the output tensor
        """
        <your code>

    def _generate_and_score_completions(
        self,
        inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Generate completions for prompts and compute their rewards for training or evaluation.

        This method handles the core workflow of the GRPO algorithm by:
        1. Processing input prompts (text and optional images)
        2. Generating completions using either vLLM or standard transformers generation
        3. Computing rewards using the configured reward functions
        4. Calculating advantages for policy optimization
        5. Preparing all necessary tensors for loss computation

        Args:
            inputs (list[dict[str, Union[torch.Tensor, Any]]]): 
                List of input examples, where each example is a dictionary containing:
                - "prompt": The input prompt (text or conversational format)
                - "image" (optional): Associated image data for multimodal inputs
                - Additional columns that may be passed to reward functions

        Returns:
            dict[str, Union[torch.Tensor, Any]]: 
                Dictionary containing processed data for training/evaluation:
                - "prompt_ids": Tokenized prompt input IDs (torch.Tensor)
                - "prompt_mask": Attention mask for prompts (torch.Tensor) 
                - "completion_ids": Generated completion token IDs (torch.Tensor)
                - "completion_mask": Mask for valid completion tokens (torch.Tensor)
                - "advantages": Computed advantages for policy optimization (torch.Tensor)
                - "num_items_in_batch": Total number of completion tokens across processes (int)
                - "old_per_token_logps" (optional): Log probabilities from previous model state for importance sampling
                - "importance_sampling_ratio" (optional): Ratios for vLLM importance sampling correction
                - "ref_per_token_logps" (optional): Reference model log probabilities for KL regularization
                - "pixel_values" (optional): Processed image pixel values for multimodal models
                - "image_grid_thw" (optional): Image grid dimensions for certain VLM architectures
                - "pixel_attention_mask" (optional): Attention mask for image pixels
                - "image_sizes" (optional): Original image sizes for certain VLM architectures

        Important Notes:
            - This method automatically handles both training and evaluation modes with different batching strategies
            - During training, completions are generated once per steps_per_generation and reused across multiple gradient accumulation steps for efficiency
            - Rewards are computed across all processes and normalized per prompt group to ensure proper advantage calculation
            - The method supports both text-only and multimodal (vision-language) inputs
            - When using vLLM, importance sampling corrections may be applied to account for distribution mismatch
            - Completion sequences are automatically masked after the first EOS token
            - All metrics and logs are automatically updated during execution
        """
        <your code>

    def compute_liger_loss(self, unwrapped_model, inputs):
        """
        Computes the loss using the Liger kernel implementation for GRPO (Group Relative Policy Optimization).

        This method leverages the LigerFusedLinearGRPOLoss kernel to efficiently compute the GRPO loss
        by fusing the linear layer computation with the loss calculation. This approach provides memory
        and computational efficiency compared to the standard implementation.

        Args:
            unwrapped_model: The unwrapped model instance (without any distributed wrappers) that
                contains the language model head and other components needed for loss computation.
            inputs (dict): A dictionary containing the input data with the following keys:
                - "prompt_ids" (torch.Tensor): Token IDs for the prompt portion
                - "prompt_mask" (torch.Tensor): Attention mask for the prompt
                - "completion_ids" (torch.Tensor): Token IDs for the completion portion  
                - "completion_mask" (torch.Tensor): Attention mask for the completion
                - "advantages" (torch.Tensor): Computed advantages for each sample
                - "old_per_token_logps" (torch.Tensor, optional): Log probabilities from previous model state
                - "ref_per_token_logps" (torch.Tensor, optional): Reference model log probabilities
                - "pixel_values" (torch.Tensor, optional): Image pixel values for multimodal inputs
                - "image_grid_thw" (torch.Tensor, optional): Image grid dimensions
                - "pixel_attention_mask" (torch.Tensor, optional): Attention mask for image pixels
                - "image_sizes" (torch.Tensor, optional): Original image sizes

        Returns:
            torch.Tensor: The computed GRPO loss value, scaled by the current gradient accumulation steps.
                The loss is ready for backpropagation and has been normalized appropriately.

        Notes:
            - This method requires the Liger kernel to be available and use_liger_loss to be True
            - The method automatically handles the computation of last hidden states and applies
              the fused linear GRPO loss computation
            - Metrics such as KL divergence and clip ratios are automatically computed and logged
            - The loss is scaled by gradient accumulation steps to ensure proper gradient scaling
            - Only supports token-level importance sampling when using Liger kernels
            - Does not support entropy-based token masking (top_entropy_quantile must be 1.0)
        """
        <your code>

    @profiling_decorator
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs = False,
        num_items_in_batch = None
    ):
        """
        Computes the loss for the GRPO (Group Relative Policy Optimization) training step.

        This method handles the core loss computation for GRPO training, supporting both standard
        PyTorch implementation and optimized Liger kernel implementation. The loss is computed
        based on policy gradients with clipped importance sampling and optional KL divergence
        regularization against a reference model.

        Parameters:
            model: The model being trained (wrapped by accelerator).
            inputs (dict): Dictionary containing the processed training inputs with keys:
                - "prompt_ids": Token IDs for the prompts
                - "prompt_mask": Attention mask for prompts  
                - "completion_ids": Token IDs for generated completions
                - "completion_mask": Attention mask for completions
                - "advantages": Computed advantages for each sample
                - "num_items_in_batch": Total number of items across all processes
                - "old_per_token_logps" (optional): Log probabilities from previous model state
                - "ref_per_token_logps" (optional): Reference model log probabilities
                - "importance_sampling_ratio" (optional): Importance sampling corrections for vLLM
                - Visual inputs (optional): "pixel_values", "image_grid_thw", etc. for multimodal models
            return_outputs (bool, optional): Whether to return model outputs. Defaults to False.
                Currently not supported and will raise ValueError if True.
            num_items_in_batch (int, optional): Number of items in batch. Currently unused as this
                information is extracted from inputs.

        Returns:
            torch.Tensor: The computed loss value, scaled by gradient accumulation steps for proper
                gradient scaling during distributed training.

        Important Notes:
            - The method automatically detects whether to use Liger kernel optimization based on
              the `use_liger_loss` configuration
            - Loss is scaled by `current_gradient_accumulation_steps` to ensure proper gradient
              magnitudes during accumulation
            - Supports multiple loss types: "grpo", "bnpo", "dr_grpo", and "dapo"
            - Handles importance sampling corrections when using vLLM for generation
            - Computes and logs various training metrics including KL divergence, entropy, and
              clipping ratios
            - For multimodal models, processes visual inputs alongside text

        Exceptions:
            ValueError: Raised if `return_outputs` is set to True, as output return is not
                currently supported in GRPOTrainer.
        """
        <your code>

    def _compute_loss(self, model, inputs):
        """
        Compute the loss for the GRPO (Group Relative Policy Optimization) training step.

        This method calculates the GRPO loss by computing per-token log probabilities, KL divergence
        with a reference model (if applicable), and applying importance sampling corrections. The loss
        incorporates advantages from reward normalization and uses clipped probability ratios to
        stabilize training.

        Args:
            model: The model being trained. Should be a PyTorch module that can compute forward passes
                and return logits for the given inputs.
            inputs (dict): Dictionary containing the processed training inputs with the following keys:
                - "prompt_ids" (torch.Tensor): Token IDs for the prompts, shape (batch_size, prompt_length)
                - "prompt_mask" (torch.Tensor): Attention mask for prompts, shape (batch_size, prompt_length)  
                - "completion_ids" (torch.Tensor): Token IDs for completions, shape (batch_size, completion_length)
                - "completion_mask" (torch.Tensor): Attention mask for completions, shape (batch_size, completion_length)
                - "advantages" (torch.Tensor): Computed advantages for each sequence, shape (batch_size,)
                - "num_items_in_batch" (int): Total number of items across all processes (used for DAPO loss)
                - "old_per_token_logps" (torch.Tensor, optional): Log probabilities from previous model state
                - "ref_per_token_logps" (torch.Tensor, optional): Reference model log probabilities for KL computation
                - "importance_sampling_ratio" (torch.Tensor, optional): Importance sampling corrections for vLLM
                - "pixel_values" (torch.Tensor, optional): Image pixel values for vision-language models
                - "image_grid_thw" (torch.Tensor, optional): Image grid dimensions for vision models
                - "pixel_attention_mask" (torch.Tensor, optional): Attention mask for image pixels
                - "image_sizes" (torch.Tensor, optional): Original image sizes before preprocessing

        Returns:
            torch.Tensor: Computed loss value, scaled by gradient accumulation steps. The loss combines
                the clipped policy gradient loss with optional KL divergence penalty and importance
                sampling corrections.

        Notes:
            - The method supports different loss types: "grpo", "bnpo", "dr_grpo", and "dapo"
            - When beta > 0, KL divergence with reference model is added as regularization
            - Importance sampling corrections are applied when using vLLM generation
            - High-entropy token masking can be applied based on top_entropy_quantile setting
            - Various metrics are logged including entropy, KL divergence, and clipping ratios
            - The loss is automatically scaled by gradient accumulation steps except for DAPO loss type
        """
        <your code>

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys: Optional[list[str]] = None
    ):
        """
        Perform a single prediction step on a batch of inputs during model evaluation.

        This method handles the evaluation pipeline by preparing inputs, computing the loss,
        and returning the results in the format expected by the Trainer's evaluation loop.
        It is called automatically during model evaluation (e.g., when `trainer.evaluate()` 
        is invoked).

        Args:
            model: The model to evaluate. This is typically the wrapped model from the trainer.
            inputs: A batch of input data containing prompts and associated metadata. The exact
                structure depends on the dataset format but typically includes prompt text and
                may include images for multimodal models.
            prediction_loss_only (bool): If True, only compute and return the loss without
                generating predictions. For GRPO evaluation, this parameter is respected but
                the method always returns None for predictions and labels.
            ignore_keys (Optional[list[str]], optional): List of keys to ignore when processing
                inputs. Defaults to None. Currently not used in the GRPO implementation.

        Returns:
            tuple: A 3-tuple containing:
                - loss (torch.Tensor): The computed evaluation loss as a scalar tensor
                - predictions (None): Always None for GRPO evaluation
                - labels (None): Always None for GRPO evaluation

        Important Notes:
            - This method automatically handles the generation of completions and reward computation
              for the evaluation batch through the `_prepare_inputs` method
            - The method runs in no-gradient mode (`torch.no_grad()`) for efficiency during evaluation
            - Loss computation includes reward calculation, advantage estimation, and policy optimization
              objectives as defined by the GRPO algorithm
            - The returned loss is detached from the computation graph and represents the mean loss
              across the batch
        """
        <your code>

    def log(
        self,
        logs: dict[str, float],
        start_time: Optional[float] = None
    ) -> None:
        """
        Log metrics and training statistics to the configured logging backends.

        This method aggregates and logs various training and evaluation metrics collected during
        the GRPO training process, including rewards, completion statistics, KL divergence,
        clipping ratios, and other relevant metrics. It also handles logging of completion
        samples to Weights & Biases if configured.

        Args:
            logs (dict[str, float]): Dictionary containing metrics to be logged. Keys are
                metric names and values are the corresponding metric values.
            start_time (Optional[float], optional): Start time for timing calculations.
                If provided, used to compute elapsed time metrics. Defaults to None.

        Returns:
            None: This method does not return any value.

        Important notes:
            - Automatically determines training vs evaluation mode based on model.training state
            - Aggregates metrics by computing averages across accumulated values
            - Adds "eval_" prefix to metric names when in evaluation mode
            - Clears accumulated metrics after logging to prevent memory buildup
            - Only logs completion samples on the main process when log_completions is enabled
            - Creates Weights & Biases tables with prompts, completions, rewards, and advantages
            - Supports logging of images for multimodal models
            - Can optionally deduplicate prompts in W&B logs based on wandb_log_unique_prompts setting
        """
        <your code>

    def _save_checkpoint(self, model, trial):
        """
        Save a model checkpoint during training.

        This method extends the base Trainer's checkpoint saving functionality by automatically
        generating and saving a model card alongside the checkpoint. The model card contains
        metadata about the training process, model configuration, and relevant citations.

        Args:
            model: The model being trained. This is typically the wrapped model from the training
                process that contains the current state of the parameters.
            trial: Trial information from hyperparameter optimization frameworks like Optuna.
                Can be None if not using hyperparameter optimization. Contains trial-specific
                metadata that may be included in the checkpoint.

        Returns:
            None: This method performs checkpoint saving as a side effect and does not return
                any value.

        Notes:
            - The model card is created using either the hub_model_id (if specified) or the
              output directory name as the model name
            - The model card generation only occurs on the main process to avoid duplicate
              file creation in distributed training scenarios
            - This method calls the parent class's _save_checkpoint method to handle the
              actual model state saving
            - The model card is saved as "README.md" in the output directory and includes
              information about the GRPO training method, base model, dataset, and citations
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