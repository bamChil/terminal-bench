## Task
## Task: Implement Hybrid Mamba-Attention Model with Dynamic Caching

**Core Functionality:**
Develop a hybrid neural network architecture that combines Mamba (state space model) layers with traditional attention layers for causal language modeling, featuring a specialized dynamic caching system.

**Main Features & Requirements:**
- **Dual Architecture Support**: Handle both attention layers (with sequence-dependent key-value caching) and Mamba layers (with fixed-size convolution/SSM state caching)
- **Dynamic Cache Management**: Implement cache that adapts tensor shapes based on layer type - growing cache for attention layers, fixed cache for Mamba layers
- **Generation Optimization**: Support efficient autoregressive text generation with proper cache reordering for beam search
- **Configuration Flexibility**: Allow configurable layer arrangements (which layers use attention vs. Mamba) and model dimensions

**Key Challenges:**
- **Heterogeneous Caching**: Managing two fundamentally different cache types (sequence-growing vs. fixed-size) within a unified interface
- **Memory Efficiency**: Balancing cache growth for attention layers while maintaining constant memory for Mamba layers
- **Generation Compatibility**: Ensuring seamless integration with standard text generation pipelines and beam search mechanisms
- **Performance Optimization**: Supporting multiple attention implementations (eager, SDPA, Flash Attention) while maintaining cache consistency

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
from transformers import BambaConfig
```
This means that we will test one function/class: BambaConfig.
And the defination and implementation of class BambaConfig should be in `/testbed/src/transformers/models/bamba/configuration_bamba.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
class BambaConfig(PretrainedConfig):
    """
    
        This is the configuration class to store the configuration of a [`BambaModel`]. It is used to instantiate a
        BambaModel model according to the specified arguments, defining the model architecture. Instantiating a configuration
        with defaults taken from [ibm-fms/Bamba-9.8b-2.2T-hf](https://huggingface.co/ibm-fms/Bamba-9.8b-2.2T-hf).
    
        The BambaModel is a hybrid [mamba2](https://github.com/state-spaces/mamba) architecture with SwiGLU.
        The checkpoints are  jointly trained by IBM, Princeton, and UIUC.
    
        Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
        documentation from [`PretrainedConfig`] for more information.
    
        Args:
            vocab_size (`int`, *optional*, defaults to 128000):
                Vocabulary size of the Bamba model. Defines the number of different tokens that can be represented by the
                `inputs_ids` passed when calling [`BambaModel`]
            tie_word_embeddings (`bool`, *optional*, defaults to `False`):
                Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the
                model has an output word embedding layer.
            hidden_size (`int`, *optional*, defaults to 4096):
                Dimension of the hidden representations.
            intermediate_size (`int`, *optional*, defaults to 14336):
                Dimension of the MLP representations.
            num_hidden_layers (`int`, *optional*, defaults to 32):
                Number of hidden layers in the Transformer encoder.
            num_attention_heads (`int`, *optional*, defaults to 32):
                Number of attention heads for each attention layer in the Transformer encoder.
            num_key_value_heads (`int`, *optional*, defaults to 8):
                This is the number of key_value heads that should be used to implement Grouped Query Attention. If
                `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
                `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
                converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
                by meanpooling all the original heads within that group. For more details, check out [this
                paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `8`.
            hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
                The non-linear activation function (function or string) in the decoder.
            initializer_range (`float`, *optional*, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            rms_norm_eps (`float`, *optional*, defaults to 1e-05):
                The epsilon used by the rms normalization layers.
            use_cache (`bool`, *optional*, defaults to `True`):
                Whether or not the model should return the last key/values attentions (not used by all models). Only
                relevant if `config.is_decoder=True`.
            num_logits_to_keep (`int` or `None`, *optional*, defaults to 1):
                Number of prompt logits to calculate during generation. If `None`, all logits will be calculated. If an
                integer value, only last `num_logits_to_keep` logits will be calculated. Default is 1 because only the
                logits of the last prompt token are needed for generation. For long sequences, the logits for the entire
                sequence may use a lot of memory so, setting `num_logits_to_keep=1` will reduce memory footprint
                significantly.
            pad_token_id (`int`, *optional*, defaults to 0):
                The id of the padding token.
            bos_token_id (`int`, *optional*, defaults to 1):
                The id of the "beginning-of-sequence" token.
            eos_token_id (`int`, *optional*, defaults to 2):
                The id of the "end-of-sequence" token.
            max_position_embeddings (`int`, *optional*, defaults to 262144):
                Max cached sequence length for the model
            attention_dropout (`float`, *optional*, defaults to 0.0):
                The dropout ratio for the attention probabilities.
            attn_layer_indices (`list`, *optional*):
                Specifies the layer indices that will have full attention. Must contain values at most num_hidden_layers.
            mamba_n_heads (`int`, *optional*, defaults to 128):
                The number of mamba heads used in the v2 implementation.
            mamba_d_head (`int`, *optional*, defaults to `"auto"`):
                Head embedding dimension size
            mamba_n_groups (`int`, *optional*, defaults to 1):
                The number of the mamba groups used in the v2 implementation.
            mamba_d_state (`int`, *optional*, defaults to 256):
                The dimension the mamba state space latents
            mamba_d_conv (`int`, *optional*, defaults to 4):
                The size of the mamba convolution kernel
            mamba_expand (`int`, *optional*, defaults to 2):
                Expanding factor (relative to hidden_size) used to determine the mamba intermediate size
            mamba_chunk_size (`int`, *optional*, defaults to 256):
                The chunks in which to break the sequence when doing prefill/training
            mamba_conv_bias (`bool`, *optional*, defaults to `True`):
                Flag indicating whether or not to use bias in the convolution layer of the mamba mixer block.
            mamba_proj_bias (`bool`, *optional*, defaults to `False`):
                Flag indicating whether or not to use bias in the input and output projections (["in_proj", "out_proj"]) of the mamba mixer block
            z_loss_coefficient (`float`, *optional*, defaults to 0.0):
                Coefficient for auxiliary z-loss used to control logit growth during training
    
        
    """

    model_type = "bamba"
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        vocab_size = 128000,
        tie_word_embeddings = False,
        hidden_size = 4096,
        intermediate_size = 14336,
        num_hidden_layers = 32,
        num_attention_heads = 32,
        num_key_value_heads = 8,
        hidden_act = 'silu',
        initializer_range = 0.02,
        rms_norm_eps = 1e-05,
        use_cache = True,
        num_logits_to_keep = 1,
        pad_token_id = 0,
        bos_token_id = 1,
        eos_token_id = 2,
        max_position_embeddings = 262144,
        attention_dropout = 0.0,
        attn_layer_indices = None,
        mamba_n_heads = 128,
        mamba_d_head = 'auto',
        mamba_n_groups = 1,
        mamba_d_state = 256,
        mamba_d_conv = 4,
        mamba_expand = 2,
        mamba_chunk_size = 256,
        mamba_conv_bias = True,
        mamba_proj_bias = False,
        z_loss_coefficient = 0.0,
        **kwargs
    ):
        """
        Initialize a BambaConfig instance for configuring a Bamba model.

        This constructor sets up the configuration parameters for a Bamba model, which is a hybrid
        mamba2 architecture with SwiGLU. The configuration defines the model architecture including
        vocabulary size, hidden dimensions, attention mechanisms, and mamba-specific parameters.

        Parameters:
            vocab_size (int, optional): Vocabulary size of the Bamba model. Defines the number of 
                different tokens that can be represented by the inputs_ids. Defaults to 128000.
            tie_word_embeddings (bool, optional): Whether the model's input and output word 
                embeddings should be tied. Only relevant if the model has an output word embedding 
                layer. Defaults to False.
            hidden_size (int, optional): Dimension of the hidden representations. Defaults to 4096.
            intermediate_size (int, optional): Dimension of the MLP representations. Defaults to 14336.
            num_hidden_layers (int, optional): Number of hidden layers in the Transformer encoder. 
                Defaults to 32.
            num_attention_heads (int, optional): Number of attention heads for each attention layer 
                in the Transformer encoder. Defaults to 32.
            num_key_value_heads (int, optional): Number of key_value heads for Grouped Query Attention.
                If equal to num_attention_heads, uses Multi Head Attention (MHA). If 1, uses Multi 
                Query Attention (MQA). Otherwise uses GQA. Defaults to 8.
            hidden_act (str or function, optional): The non-linear activation function in the decoder. 
                Defaults to 'silu'.
            initializer_range (float, optional): Standard deviation of the truncated_normal_initializer 
                for initializing all weight matrices. Defaults to 0.02.
            rms_norm_eps (float, optional): Epsilon used by the RMS normalization layers. 
                Defaults to 1e-05.
            use_cache (bool, optional): Whether the model should return the last key/values attentions.
                Only relevant if config.is_decoder=True. Defaults to True.
            num_logits_to_keep (int or None, optional): Number of prompt logits to calculate during 
                generation. If None, all logits calculated. If int, only last num_logits_to_keep 
                logits calculated. Defaults to 1.
            pad_token_id (int, optional): ID of the padding token. Defaults to 0.
            bos_token_id (int, optional): ID of the "beginning-of-sequence" token. Defaults to 1.
            eos_token_id (int, optional): ID of the "end-of-sequence" token. Defaults to 2.
            max_position_embeddings (int, optional): Maximum cached sequence length for the model. 
                Defaults to 262144.
            attention_dropout (float, optional): Dropout ratio for attention probabilities. 
                Defaults to 0.0.
            attn_layer_indices (list, optional): Layer indices that will have full attention. 
                Values must be at most num_hidden_layers. Defaults to None.
            mamba_n_heads (int, optional): Number of mamba heads used in the v2 implementation. 
                Defaults to 128.
            mamba_d_head (int or str, optional): Head embedding dimension size. If 'auto', 
                calculated as mamba_intermediate // mamba_n_heads. Defaults to 'auto'.
            mamba_n_groups (int, optional): Number of mamba groups used in the v2 implementation. 
                Defaults to 1.
            mamba_d_state (int, optional): Dimension of the mamba state space latents. Defaults to 256.
            mamba_d_conv (int, optional): Size of the mamba convolution kernel. Defaults to 4.
            mamba_expand (int, optional): Expanding factor (relative to hidden_size) used to 
                determine the mamba intermediate size. Defaults to 2.
            mamba_chunk_size (int, optional): Chunks to break the sequence when doing prefill/training. 
                Defaults to 256.
            mamba_conv_bias (bool, optional): Whether to use bias in the convolution layer of the 
                mamba mixer block. Defaults to True.
            mamba_proj_bias (bool, optional): Whether to use bias in the input and output projections 
                of the mamba mixer block. Defaults to False.
            z_loss_coefficient (float, optional): Coefficient for auxiliary z-loss used to control 
                logit growth during training. Defaults to 0.0.
            **kwargs: Additional keyword arguments passed to the parent PretrainedConfig class.

        Raises:
            ValueError: If mamba_n_heads does not divide (mamba_expand * hidden_size).
            ValueError: If mamba_d_head * mamba_n_heads does not equal mamba_intermediate.

        Notes:
            - The configuration defaults are taken from ibm-fms/Bamba-9.8b-2.2T-hf model.
            - For backward compatibility, if num_key_value_heads is None, it defaults to num_attention_heads.
            - When mamba_d_head is 'auto', it is automatically calculated as mamba_intermediate // mamba_n_heads.
            - The model uses RoPE (Rotary Position Embedding) with theta=10000.0 and partial_rotary_factor=0.5.
        """
        <your code>
...
```

In order to implement this functionality, some additional libraries etc. are often required, I don't restrict you to any libraries, you need to think about what dependencies you might need and fetch and install and call them yourself. The only thing is that you **MUST** fulfill the input/output format described by this interface, otherwise the test will not pass and you will get zero points for this feature.
And note that there may be not only one **Interface Description**, you should match all **Interface Description {n}**

### Test Description 1
Below is **Test Description 1**
```python
from transformers import BambaConfig
from transformers import BambaForCausalLM
from transformers import BambaModel
from transformers.models.bamba.modeling_bamba import HybridMambaAttentionDynamicCache
```


### Interface Description 1
Below is **Interface Description 1** for file: src-transformers-models-bamba-configuration_bamba.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
class BambaConfig(PretrainedConfig):
    """
    
        This is the configuration class to store the configuration of a [`BambaModel`]. It is used to instantiate a
        BambaModel model according to the specified arguments, defining the model architecture. Instantiating a configuration
        with defaults taken from [ibm-fms/Bamba-9.8b-2.2T-hf](https://huggingface.co/ibm-fms/Bamba-9.8b-2.2T-hf).
    
        The BambaModel is a hybrid [mamba2](https://github.com/state-spaces/mamba) architecture with SwiGLU.
        The checkpoints are  jointly trained by IBM, Princeton, and UIUC.
    
        Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
        documentation from [`PretrainedConfig`] for more information.
    
        Args:
            vocab_size (`int`, *optional*, defaults to 128000):
                Vocabulary size of the Bamba model. Defines the number of different tokens that can be represented by the
                `inputs_ids` passed when calling [`BambaModel`]
            tie_word_embeddings (`bool`, *optional*, defaults to `False`):
                Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the
                model has an output word embedding layer.
            hidden_size (`int`, *optional*, defaults to 4096):
                Dimension of the hidden representations.
            intermediate_size (`int`, *optional*, defaults to 14336):
                Dimension of the MLP representations.
            num_hidden_layers (`int`, *optional*, defaults to 32):
                Number of hidden layers in the Transformer encoder.
            num_attention_heads (`int`, *optional*, defaults to 32):
                Number of attention heads for each attention layer in the Transformer encoder.
            num_key_value_heads (`int`, *optional*, defaults to 8):
                This is the number of key_value heads that should be used to implement Grouped Query Attention. If
                `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
                `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
                converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
                by meanpooling all the original heads within that group. For more details, check out [this
                paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `8`.
            hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
                The non-linear activation function (function or string) in the decoder.
            initializer_range (`float`, *optional*, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            rms_norm_eps (`float`, *optional*, defaults to 1e-05):
                The epsilon used by the rms normalization layers.
            use_cache (`bool`, *optional*, defaults to `True`):
                Whether or not the model should return the last key/values attentions (not used by all models). Only
                relevant if `config.is_decoder=True`.
            num_logits_to_keep (`int` or `None`, *optional*, defaults to 1):
                Number of prompt logits to calculate during generation. If `None`, all logits will be calculated. If an
                integer value, only last `num_logits_to_keep` logits will be calculated. Default is 1 because only the
                logits of the last prompt token are needed for generation. For long sequences, the logits for the entire
                sequence may use a lot of memory so, setting `num_logits_to_keep=1` will reduce memory footprint
                significantly.
            pad_token_id (`int`, *optional*, defaults to 0):
                The id of the padding token.
            bos_token_id (`int`, *optional*, defaults to 1):
                The id of the "beginning-of-sequence" token.
            eos_token_id (`int`, *optional*, defaults to 2):
                The id of the "end-of-sequence" token.
            max_position_embeddings (`int`, *optional*, defaults to 262144):
                Max cached sequence length for the model
            attention_dropout (`float`, *optional*, defaults to 0.0):
                The dropout ratio for the attention probabilities.
            attn_layer_indices (`list`, *optional*):
                Specifies the layer indices that will have full attention. Must contain values at most num_hidden_layers.
            mamba_n_heads (`int`, *optional*, defaults to 128):
                The number of mamba heads used in the v2 implementation.
            mamba_d_head (`int`, *optional*, defaults to `"auto"`):
                Head embedding dimension size
            mamba_n_groups (`int`, *optional*, defaults to 1):
                The number of the mamba groups used in the v2 implementation.
            mamba_d_state (`int`, *optional*, defaults to 256):
                The dimension the mamba state space latents
            mamba_d_conv (`int`, *optional*, defaults to 4):
                The size of the mamba convolution kernel
            mamba_expand (`int`, *optional*, defaults to 2):
                Expanding factor (relative to hidden_size) used to determine the mamba intermediate size
            mamba_chunk_size (`int`, *optional*, defaults to 256):
                The chunks in which to break the sequence when doing prefill/training
            mamba_conv_bias (`bool`, *optional*, defaults to `True`):
                Flag indicating whether or not to use bias in the convolution layer of the mamba mixer block.
            mamba_proj_bias (`bool`, *optional*, defaults to `False`):
                Flag indicating whether or not to use bias in the input and output projections (["in_proj", "out_proj"]) of the mamba mixer block
            z_loss_coefficient (`float`, *optional*, defaults to 0.0):
                Coefficient for auxiliary z-loss used to control logit growth during training
    
        
    """

    model_type = "bamba"
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        vocab_size = 128000,
        tie_word_embeddings = False,
        hidden_size = 4096,
        intermediate_size = 14336,
        num_hidden_layers = 32,
        num_attention_heads = 32,
        num_key_value_heads = 8,
        hidden_act = 'silu',
        initializer_range = 0.02,
        rms_norm_eps = 1e-05,
        use_cache = True,
        num_logits_to_keep = 1,
        pad_token_id = 0,
        bos_token_id = 1,
        eos_token_id = 2,
        max_position_embeddings = 262144,
        attention_dropout = 0.0,
        attn_layer_indices = None,
        mamba_n_heads = 128,
        mamba_d_head = 'auto',
        mamba_n_groups = 1,
        mamba_d_state = 256,
        mamba_d_conv = 4,
        mamba_expand = 2,
        mamba_chunk_size = 256,
        mamba_conv_bias = True,
        mamba_proj_bias = False,
        z_loss_coefficient = 0.0,
        **kwargs
    ):
        """
        Initialize a BambaConfig instance for configuring a Bamba model.

        This constructor sets up the configuration parameters for a Bamba model, which is a hybrid
        mamba2 architecture with SwiGLU. The configuration defines the model architecture including
        vocabulary size, hidden dimensions, attention mechanisms, and mamba-specific parameters.

        Parameters:
            vocab_size (int, optional): Vocabulary size of the Bamba model. Defines the number of 
                different tokens that can be represented by the inputs_ids. Defaults to 128000.
            tie_word_embeddings (bool, optional): Whether the model's input and output word 
                embeddings should be tied. Only relevant if the model has an output word embedding 
                layer. Defaults to False.
            hidden_size (int, optional): Dimension of the hidden representations. Defaults to 4096.
            intermediate_size (int, optional): Dimension of the MLP representations. Defaults to 14336.
            num_hidden_layers (int, optional): Number of hidden layers in the Transformer encoder. 
                Defaults to 32.
            num_attention_heads (int, optional): Number of attention heads for each attention layer 
                in the Transformer encoder. Defaults to 32.
            num_key_value_heads (int, optional): Number of key_value heads for Grouped Query Attention.
                If equal to num_attention_heads, uses Multi Head Attention (MHA). If 1, uses Multi 
                Query Attention (MQA). Otherwise uses GQA. Defaults to 8.
            hidden_act (str or function, optional): The non-linear activation function in the decoder. 
                Defaults to 'silu'.
            initializer_range (float, optional): Standard deviation of the truncated_normal_initializer 
                for initializing all weight matrices. Defaults to 0.02.
            rms_norm_eps (float, optional): Epsilon used by the RMS normalization layers. 
                Defaults to 1e-05.
            use_cache (bool, optional): Whether the model should return the last key/values attentions.
                Only relevant if config.is_decoder=True. Defaults to True.
            num_logits_to_keep (int or None, optional): Number of prompt logits to calculate during 
                generation. If None, all logits calculated. If int, only last num_logits_to_keep 
                logits calculated. Defaults to 1.
            pad_token_id (int, optional): ID of the padding token. Defaults to 0.
            bos_token_id (int, optional): ID of the "beginning-of-sequence" token. Defaults to 1.
            eos_token_id (int, optional): ID of the "end-of-sequence" token. Defaults to 2.
            max_position_embeddings (int, optional): Maximum cached sequence length for the model. 
                Defaults to 262144.
            attention_dropout (float, optional): Dropout ratio for attention probabilities. 
                Defaults to 0.0.
            attn_layer_indices (list, optional): Layer indices that will have full attention. 
                Values must be at most num_hidden_layers. Defaults to None.
            mamba_n_heads (int, optional): Number of mamba heads used in the v2 implementation. 
                Defaults to 128.
            mamba_d_head (int or str, optional): Head embedding dimension size. If 'auto', 
                calculated as mamba_intermediate // mamba_n_heads. Defaults to 'auto'.
            mamba_n_groups (int, optional): Number of mamba groups used in the v2 implementation. 
                Defaults to 1.
            mamba_d_state (int, optional): Dimension of the mamba state space latents. Defaults to 256.
            mamba_d_conv (int, optional): Size of the mamba convolution kernel. Defaults to 4.
            mamba_expand (int, optional): Expanding factor (relative to hidden_size) used to 
                determine the mamba intermediate size. Defaults to 2.
            mamba_chunk_size (int, optional): Chunks to break the sequence when doing prefill/training. 
                Defaults to 256.
            mamba_conv_bias (bool, optional): Whether to use bias in the convolution layer of the 
                mamba mixer block. Defaults to True.
            mamba_proj_bias (bool, optional): Whether to use bias in the input and output projections 
                of the mamba mixer block. Defaults to False.
            z_loss_coefficient (float, optional): Coefficient for auxiliary z-loss used to control 
                logit growth during training. Defaults to 0.0.
            **kwargs: Additional keyword arguments passed to the parent PretrainedConfig class.

        Raises:
            ValueError: If mamba_n_heads does not divide (mamba_expand * hidden_size).
            ValueError: If mamba_d_head * mamba_n_heads does not equal mamba_intermediate.

        Notes:
            - The configuration defaults are taken from ibm-fms/Bamba-9.8b-2.2T-hf model.
            - For backward compatibility, if num_key_value_heads is None, it defaults to num_attention_heads.
            - When mamba_d_head is 'auto', it is automatically calculated as mamba_intermediate // mamba_n_heads.
            - The model uses RoPE (Rotary Position Embedding) with theta=10000.0 and partial_rotary_factor=0.5.
        """
        <your code>

    @property
    def layers_block_type(self):
        """
        Get the block type for each layer in the Bamba model architecture.

        This property determines whether each layer in the model uses attention or mamba blocks
        based on the configuration. The Bamba model is a hybrid architecture that can contain
        both attention layers and mamba layers at different positions.

        Returns:
            list[str]: A list of strings with length equal to `num_hidden_layers`, where each
                element is either "attention" or "mamba". The value at index i indicates the
                block type for layer i. If `attn_layer_indices` is specified and contains
                index i, then layer i will be "attention", otherwise it will be "mamba".

        Notes:
            - This property is computed dynamically based on the `attn_layer_indices` and
              `num_hidden_layers` configuration parameters
            - If `attn_layer_indices` is None or empty, all layers will be "mamba" type
            - The `attn_layer_indices` should contain values that are valid layer indices
              (i.e., between 0 and `num_hidden_layers - 1`)
        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.

### Interface Description 2
Below is **Interface Description 2** for file: src-transformers-models-bamba-modeling_bamba.py

This file contains 3 top-level interface(s) that need to be implemented.

```python
class HybridMambaAttentionDynamicCache:
    """
    
        A dynamic cache that can handle both the attention cache (which has a seq_len dimension) and the mamba cache
        (which has a constant shape regardless of seq_len).
    
        This cache has two sets of lists of tensors: `key_cache` and `value_cache` for attention cache and `conv_states`
        and `ssm_states` for mamba cache. Each of these lists has `num_layers` tensors. The expected shape for each tensor
        For attention layers, `key_cache` and `value_cache` have a shape of `(batch_size, num_heads, seq_len, head_dim)`,
        while `conv_states` and `ssm_states` have a shape of `(batch_size, 0)` (empty tensors).
        For mamba layers, `key_cache` and `value_cache` have a shape of `(batch_size, 0)` (empty tensors),
        while `conv_states` represents the convolution state and has a shape of `(batch_size, d_inner, d_conv)`,
        and `ssm_states` represents the ssm state and has a shape of `(batch_size, d_inner, d_state)`.
        
    """

    is_compileable = False

    def __init__(
        self,
        config: BambaConfig,
        batch_size,
        dtype = torch.float16,
        device = None
    ):
        """
        Initialize a HybridMambaAttentionDynamicCache instance for handling both attention and mamba cache states.

        This cache manages two types of cache states:
        - Attention cache: key_cache and value_cache tensors with shape (batch_size, num_heads, seq_len, head_dim)
        - Mamba cache: conv_states and ssm_states tensors with fixed shapes regardless of sequence length

        For attention layers, the mamba cache tensors are empty, and for mamba layers, the attention cache tensors are empty.

        Args:
            config (BambaConfig): The model configuration containing layer specifications and dimensions.
            batch_size (int): The batch size for which to initialize the cache tensors.
            dtype (torch.dtype, optional): The data type for cache tensors. Defaults to torch.float16.
            device (torch.device, optional): The device on which to create cache tensors. If None, uses default device.

        Returns:
            None: This is an __init__ method that initializes the cache instance.

        Notes:
            - The cache structure depends on config.layers_block_type which specifies whether each layer is "mamba" or "attention"
            - For mamba layers: conv_states shape is (batch_size, expanded_hidden_size + 2*n_groups*state_size, conv_kernel_size)
                               ssm_states shape is (batch_size, n_heads, d_head, state_size)
            - For attention layers: conv_states and ssm_states are empty tensors
            - The has_previous_state flag is used internally by mamba layers to track cache state
            - All cache tensors are initialized with zeros for mamba layers and empty tensors for attention layers
        """
        <your code>

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the key and value cache for attention layers in the hybrid Mamba-Attention dynamic cache.

        This method manages the caching mechanism for attention layers by either initializing the cache
        with new key-value states or concatenating new states to existing cached states along the
        sequence dimension.

        Parameters:
            key_states (torch.Tensor): The key states tensor to be cached. Expected shape is
                (batch_size, num_heads, seq_len, head_dim) for attention layers.
            value_states (torch.Tensor): The value states tensor to be cached. Expected shape is
                (batch_size, num_heads, seq_len, head_dim) for attention layers.
            layer_idx (int): The index of the layer for which the cache is being updated. Used to
                access the correct cache entry in the layer-specific cache lists.
            cache_kwargs (Optional[dict[str, Any]], optional): Additional cache-related keyword
                arguments. Currently unused but maintained for interface compatibility. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the updated key and value cache
                tensors for the specified layer. Both tensors have shape 
                (batch_size, num_heads, total_seq_len, head_dim) where total_seq_len includes
                both previously cached and newly added sequence tokens.

        Important Notes:
            - This method is specifically designed for attention layers in the hybrid cache system
            - For Mamba layers, the key_cache and value_cache contain empty tensors
            - The method performs in-place updates to the cache lists
            - New key-value states are concatenated along dimension 2 (sequence length dimension)
            - If the cache is empty (shape[-1] == 0), it initializes with the provided states
            - The cache grows dynamically as new tokens are processed during generation
        """
        <your code>

    def reorder_cache(
        self,
        beam_idx: torch.LongTensor
    ):
        """
        Reorders the cache for beam search, given the selected beam indices.
        """
        <your code>

    def get_seq_length(
        self,
        layer_idx: Optional[int] = 0
    ) -> int:
        """
        Returns the sequence length of the cached states. A layer index can be optionally passed.
        """
        <your code>

@auto_docstring
class BambaModel(BambaPreTrainedModel):
    """
    """
    A hybrid transformer-mamba model that combines attention and mamba (state space model) layers.
    
    BambaModel is the core model class that implements a hybrid architecture alternating between
    attention layers and mamba layers based on the configuration. It processes input sequences
    through embedding, multiple decoder layers, and final normalization to produce contextualized
    hidden representations.
    
    Main Attributes:
        embed_tokens (nn.Embedding): Token embedding layer that converts input IDs to dense vectors
        layers (nn.ModuleList): List of decoder layers, each being either attention or mamba type
        final_layernorm (BambaRMSNorm): RMS normalization applied to final hidden states
        rotary_emb (BambaRotaryEmbedding): Rotary position embeddings shared across attention layers
        padding_idx (int): Index used for padding tokens
        vocab_size (int): Size of the vocabulary
    
    Main Methods:
        forward: Main forward pass that processes input through all layers and returns model outputs.
            Handles both training and inference modes, supports caching for efficient generation,
            and can output attention weights and hidden states from intermediate layers.
        
        _update_causal_mask: Creates and updates 4D causal attention masks for attention layers,
            handling different attention implementations (eager, SDPA, flash attention).
        
        _update_mamba_mask: Updates 2D attention masks specifically for mamba layers, with
            optimizations for cached forward passes and full attention scenarios.
        
        _prepare_4d_causal_attention_mask_with_cache_position: Static method that converts 2D
            attention masks to 4D causal masks with proper cache position handling.
    
    Usage Examples:
        ```python
        from transformers import BambaModel, BambaConfig
        
        # Initialize model with hybrid attention-mamba architecture
        config = BambaConfig(
            hidden_size=768,
            num_hidden_layers=12,
            layers_block_type=["attention", "mamba"] * 6  # Alternating layers
        )
        model = BambaModel(config)
        
        # Forward pass
        input_ids = torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10
        outputs = model(input_ids)
        hidden_states = outputs.last_hidden_state
        
        # With caching for generation
        from transformers.models.bamba.modeling_bamba import HybridMambaAttentionDynamicCache
        cache = HybridMambaAttentionDynamicCache(config, batch_size=2, device=model.device)
        outputs = model(input_ids, past_key_values=cache, use_cache=True)
        ```
    
    The model supports advanced features like gradient checkpointing, different attention implementations
    (eager, SDPA, flash attention), and specialized caching mechanisms for efficient autoregressive generation.
    """
    """

    def __init__(self, config: BambaConfig):
        """
        Initialize a Bamba model with the given configuration.

        This constructor sets up all the necessary components for a Bamba model, including:
        - Model configuration and basic parameters (padding index, vocabulary size)
        - Token embedding layer for converting input token IDs to embeddings
        - Decoder layers with hybrid Mamba/Attention architecture based on the layer configuration
        - Final layer normalization and rotary position embeddings
        - Gradient checkpointing support

        Args:
            config (BambaConfig): Configuration object containing all model hyperparameters
                and architectural settings. This includes:
                - Model dimensions (hidden_size, vocab_size, num_hidden_layers)
                - Layer type configuration (layers_block_type) specifying which layers
                  use Mamba vs attention mechanisms
                - Normalization parameters (rms_norm_eps)
                - Attention implementation settings (_attn_implementation)
                - Token padding configuration (pad_token_id)

        Returns:
            None: This is a constructor method that initializes the model instance.

        Important Notes:
            - The model uses a hybrid architecture where each layer can be either a Mamba
              layer or an attention layer, as specified in config.layers_block_type
            - Rotary position embeddings are initialized for attention layers
            - The model supports gradient checkpointing for memory-efficient training
            - Post-initialization processing (weight initialization) is automatically
              called after construction
            - The attention implementation can be configured via config._attn_implementation
              to use different backends (eager, flash_attention_2, sdpa)
        """
        <your code>

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridMambaAttentionDynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[BambaFlashAttentionKwargs]
    ) -> BaseModelOutputWithPast:
        """
        Performs a forward pass through the Bamba model, processing input sequences through hybrid Mamba-Attention layers.

        This method handles the core forward computation of the Bamba model, which combines Mamba (selective state space) layers and traditional attention layers in a hybrid architecture. It processes input tokens through embedding, applies positional encodings, and passes through decoder layers to produce contextualized representations.

        Args:
            input_ids (torch.LongTensor, optional): 
                Indices of input sequence tokens in the vocabulary of shape `(batch_size, sequence_length)`. 
                Cannot be provided simultaneously with `inputs_embeds`.
            attention_mask (torch.Tensor, optional): 
                Mask to avoid performing attention on padding token indices of shape `(batch_size, sequence_length)`. 
                Mask values are in `[0, 1]`: 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
            position_ids (torch.LongTensor, optional): 
                Indices of positions of each input sequence token in the position embeddings of shape 
                `(batch_size, sequence_length)`. If None, will be automatically generated.
            past_key_values (HybridMambaAttentionDynamicCache, optional): 
                Pre-computed hidden states (keys, values, conv states, and SSM states) that can be used to speed up 
                sequential decoding. Contains both attention cache and Mamba cache components.
            inputs_embeds (torch.FloatTensor, optional): 
                Embedded representation of input tokens of shape `(batch_size, sequence_length, hidden_size)`. 
                Cannot be provided simultaneously with `input_ids`.
            use_cache (bool, optional): 
                If set to True, past_key_values are returned and can be used to speed up decoding. 
                Defaults to model configuration value.
            output_attentions (bool, optional): 
                Whether to return attention weights from attention layers. Note that Mamba layers do not produce 
                attention weights. Defaults to model configuration value.
            output_hidden_states (bool, optional): 
                Whether to return hidden states from all layers. Defaults to model configuration value.
            cache_position (torch.LongTensor, optional): 
                Indices depicting the position of input sequence tokens in the sequence of shape `(sequence_length,)`. 
                Used for caching and positional embeddings.
            **kwargs: 
                Additional keyword arguments of type `BambaFlashAttentionKwargs` for advanced Flash Attention usage, 
                including `cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, `max_length_k`, and `seq_idx` for 
                padding-free training and optimized kernel usage.

        Returns:
            BaseModelOutputWithPast: A dataclass containing:
                - last_hidden_state (torch.FloatTensor): Hidden states from the last layer of shape 
                  `(batch_size, sequence_length, hidden_size)`
                - past_key_values (HybridMambaAttentionDynamicCache): Updated cache containing attention and Mamba states
                - hidden_states (tuple of torch.FloatTensor, optional): Hidden states from all layers if 
                  `output_hidden_states=True`
                - attentions (tuple of torch.FloatTensor, optional): Attention weights from attention layers if 
                  `output_attentions=True`

        Raises:
            ValueError: If both `input_ids` and `inputs_embeds` are provided or if neither is provided.

        Notes:
            - The model uses a hybrid architecture combining Mamba (selective state space) and attention layers
            - Mamba layers use a specialized cache structure different from traditional attention caches
            - When using gradient checkpointing during training, `use_cache` will be automatically set to False
            - The model automatically handles different attention implementations (eager, flash_attention_2, sdpa)
            - For Mamba layers, a 2D attention mask is used, while attention layers use a 4D causal mask
        """
        <your code>

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: HybridMambaAttentionDynamicCache,
        output_attentions: bool
    ):
        """
        Updates the causal attention mask for attention layers in the hybrid Bamba model.

        This method generates and processes the appropriate attention mask based on the attention implementation
        being used (eager, SDPA, or Flash Attention 2). It handles different scenarios including cached
        generation, training vs inference, and various mask formats.

        Parameters:
            attention_mask (torch.Tensor): The input attention mask of shape (batch_size, sequence_length)
                where padding elements are indicated by 0. Can be None if no masking is needed.
            input_tensor (torch.Tensor): The input tensor to the model, used to determine batch size,
                sequence length, and dtype for mask generation.
            cache_position (torch.Tensor): Tensor containing indices depicting the position of input
                sequence tokens in the sequence, used for cache-aware mask generation.
            past_key_values (HybridMambaAttentionDynamicCache): The dynamic cache containing both
                attention cache (key/value states) and mamba cache (conv/ssm states). Used to determine
                the number of past tokens seen.
            output_attentions (bool): Whether attention weights will be returned. Affects SDPA
                implementation choice and mask processing.

        Returns:
            torch.Tensor or None: The processed causal attention mask of shape 
                (batch_size, 1, query_length, key_value_length) for eager/SDPA implementations,
                or the original 2D mask for Flash Attention 2, or None if no mask is needed.

        Important notes:
            - For Flash Attention 2: Returns None unless the attention_mask contains padding (0.0 values)
            - For SDPA: May return None when causal masking can be handled implicitly via is_causal argument
            - For eager implementation: Always returns a 4D causal mask
            - The method handles both 2D input masks (converted to 4D causal) and pre-existing 4D masks
            - Applies special handling for SDPA on CUDA/XPU/NPU devices to ensure memory-efficient attention compatibility
            - The mask values use the minimum representable value for the tensor dtype to indicate masked positions
        """
        <your code>

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs
    ):
        """

                Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
                `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

                Args:
                    attention_mask (`torch.Tensor`):
                        A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                        `(batch_size, 1, query_length, key_value_length)`.
                    sequence_length (`int`):
                        The sequence length being processed.
                    target_length (`int`):
                        The target length: when generating with static cache, the mask should be as long as the static cache,
                        to account for the 0 padding, the part of the cache that is not filled yet.
                    dtype (`torch.dtype`):
                        The dtype to use for the 4D attention mask.
                    cache_position (`torch.Tensor`):
                        Indices depicting the position of the input sequence tokens in the sequence.
                    batch_size (`torch.Tensor`):
                        Batch size.

        """
        <your code>

    def _update_mamba_mask(self, attention_mask, cache_position):
        """

                No need for zeroing states when
                    1. Cached forward
                    2. Attending to all inputs

        """
        <your code>

@auto_docstring
class BambaForCausalLM(BambaPreTrainedModel, GenerationMixin):
    """
    A hybrid transformer model that combines Mamba (state space model) and attention mechanisms for causal language modeling tasks. This model implements the Bamba architecture which alternates between Mamba layers for efficient sequence processing and attention layers for capturing long-range dependencies.
    
    BambaForCausalLM is designed for autoregressive text generation and can be used for tasks such as language modeling, text completion, and conversational AI. The model leverages a specialized hybrid cache system (HybridMambaAttentionDynamicCache) to efficiently handle the different caching requirements of Mamba and attention layers.
    
    Args:
        config (BambaConfig): Model configuration containing hyperparameters and architecture settings.
    
    Attributes:
        model (BambaModel): The core Bamba model containing embedding layers, decoder layers, and normalization.
        vocab_size (int): Size of the vocabulary.
        lm_head (nn.Linear): Linear layer that projects hidden states to vocabulary logits for next token prediction.
        z_loss_coefficient (float): Coefficient for auxiliary z-loss regularization to improve training stability.
    
    Methods:
        forward: Performs forward pass through the model, computing logits and optionally loss for training.
            Handles both training (with labels) and inference modes. Supports various output options
            including hidden states and attention weights.
        
        prepare_inputs_for_generation: Prepares model inputs for text generation, handling cache management
            and input slicing for efficient autoregressive generation. Creates and manages the hybrid
            cache system required for the mixed Mamba-attention architecture.
    
    Example:
        ```python
        from transformers import AutoTokenizer, BambaForCausalLM
        
        # Load model and tokenizer
        model = BambaForCausalLM.from_pretrained("bamba-model-name")
        tokenizer = AutoTokenizer.from_pretrained("bamba-model-name")
        
        # Text generation
        prompt = "The future of AI is"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=50,
                temperature=0.7,
                do_sample=True
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(generated_text)
        
        # Training mode with labels
        labels = inputs.input_ids.clone()
        outputs = model(input_ids=inputs.input_ids, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        ```
    
    Note:
        This model requires specific dependencies (mamba_ssm, causal_conv1d) for optimal performance.
        When these are available and running on GPU, the model uses optimized CUDA kernels.
        Otherwise, it falls back to a PyTorch implementation with reduced performance.
    """

    _tied_weights_keys = ['lm_head.weight']
    _tp_plan = {'lm_head': 'colwise_rep'}
    _pp_plan = {'lm_head': (['hidden_states'], ['logits'])}

    def __init__(self, config):
        """
          Initialize a Bamba-based language model with a causal language modeling head.

            This constructor sets up the core components of a Bamba model for text generation tasks,
            including the main model body (hybrid Mamba/attention layers) and a language model head
            for token prediction. It inherits from the base model class and initializes weights
            according to the provided configuration.

            The Bamba model is a hybrid architecture combining:
            - Mamba layers (selective state space models) for efficient sequence processing
            - Attention layers for capturing long-range dependencies (where configured)

            Parameters:
                config (BambaConfig): Configuration object containing all hyperparameters that define
                    the model architecture. Key parameters include:
                    - Model dimensions: hidden_size, vocab_size, num_hidden_layers
                    - Attention settings: num_attention_heads, num_key_value_heads, attn_layer_indices
                    - Mamba-specific parameters: mamba_d_state, mamba_d_conv, mamba_expand
                    - Training configurations: dropout rates, bias settings, z_loss_coefficient
                    - Normalization and activation specifications

            Important notes:
                - The main model body is initialized as `self.model = BambaModel(config)`, which contains
                the hybrid Mamba/attention layers as defined by `config.attn_layer_indices`
                - `self.lm_head` maps hidden states to token logits (no bias, as specified in the config)
                - Model weights are initialized via `self.post_init()` following Hugging Face conventions
                - Requires Mamba dependencies (mamba_ssm, causal_conv1d) for optimal performance
                - This file may be auto-generated from modular_bamba.py; edits should target the source template
        """
        <your code>

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridMambaAttentionDynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """

                labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                    Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                    config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                    (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

                Example:

                ```python
                >>> from transformers import AutoTokenizer, BambaForCausalLM

                >>> model = BambaForCausalLM.from_pretrained("...")
                >>> tokenizer = AutoTokenizer.from_pretrained("...")

                >>> prompt = "Hey, are you conscious? Can you talk to me?"
                >>> inputs = tokenizer(prompt, return_tensors="pt")

                >>> # Generate
                >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
                >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
                ```
        """
        <your code>

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values = None,
        attention_mask = None,
        inputs_embeds = None,
        cache_position = None,
        position_ids = None,
        use_cache = True,
        **kwargs
    ):
        """
        Prepares inputs for text generation by processing and formatting various input parameters.

        This method handles the preparation of input tensors for generation tasks, including managing
        cached states, attention masks, position IDs, and input embeddings. It performs special handling
        for the HybridMambaAttentionDynamicCache used by the Bamba model architecture.

        Parameters:
            input_ids (torch.LongTensor, optional): 
                Input token IDs of shape (batch_size, sequence_length). Contains the tokenized input text.
            past_key_values (HybridMambaAttentionDynamicCache, optional): 
                Cached key-value states from previous generation steps. If None, a new cache will be 
                initialized for the current batch.
            attention_mask (torch.Tensor, optional): 
                Attention mask of shape (batch_size, sequence_length) where 1 indicates tokens to attend 
                to and 0 indicates padding tokens to ignore.
            inputs_embeds (torch.FloatTensor, optional): 
                Pre-computed input embeddings of shape (batch_size, sequence_length, hidden_size). 
                If provided, will be used instead of input_ids for the first generation step.
            cache_position (torch.LongTensor, optional): 
                Position indices indicating which tokens in the sequence are being processed in the 
                current forward pass.
            position_ids (torch.LongTensor, optional): 
                Position indices for each token in the input sequence. If None and attention_mask is 
                provided, position_ids will be automatically generated.
            use_cache (bool, optional): 
                Whether to use caching for faster generation. Defaults to True.
            **kwargs: 
                Additional keyword arguments passed to the method.

        Returns:
            dict: A dictionary containing the prepared model inputs with the following keys:
                - 'input_ids' or 'inputs_embeds': The processed input tokens or embeddings
                - 'position_ids': Position indices for the input tokens
                - 'past_key_values': The cache object (newly created if was None)
                - 'use_cache': Boolean flag for cache usage
                - 'attention_mask': The attention mask tensor
                - 'logits_to_keep': Number of logits to keep from config
                - 'cache_position': Position indices for caching

        Important Notes:
            - This method performs intelligent slicing of input_ids when cache is present to only 
              process unprocessed tokens
            - Creates a new HybridMambaAttentionDynamicCache if none is provided
            - Handles special cases for input_embeds usage and cache position boundaries
            - Automatically generates position_ids from attention_mask when not provided
            - The returned input_ids tensor is made contiguous for compilation compatibility
        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.