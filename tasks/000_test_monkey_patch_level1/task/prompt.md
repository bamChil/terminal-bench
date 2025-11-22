## Task
## Task: Implement High-Performance Neural Network Optimization Library

**Core Functionalities:**
- Develop optimized kernel implementations for common transformer operations (RMSNorm, LayerNorm, SwiGLU, GEGLU)
- Create drop-in replacements for standard PyTorch modules with performance improvements
- Provide automatic model patching system for popular language models (LLaMA, Gemma, Qwen, etc.)

**Main Features & Requirements:**
- **Performance Optimization**: Replace standard operations with memory-efficient, GPU-accelerated kernels
- **Model Compatibility**: Support multiple transformer architectures with configurable activation functions
- **Seamless Integration**: Maintain identical APIs to original PyTorch modules for easy adoption
- **Flexible Configuration**: Allow selective enabling/disabling of optimizations per model component

**Key Challenges:**
- **Numerical Stability**: Ensure optimized kernels maintain mathematical correctness across different precisions
- **Memory Management**: Balance in-place operations for efficiency while preserving gradient computation
- **Architecture Diversity**: Handle varying model configurations (hidden sizes, activation functions, normalization types)
- **Backward Compatibility**: Maintain compatibility with existing model checkpoints and training pipelines

The task focuses on creating a high-performance optimization layer that accelerates transformer model training and inference while preserving functional equivalence to standard implementations.

**NOTE**: 
- This test comes from the `liger-kernel` library, and we have given you the content of this code repository under `/testbed/`, and you need to complete based on this code repository and supplement the files we specify. Remember, all your changes must be in this codebase, and changes that are not in this codebase will not be discovered and tested by us.
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
from liger_kernel.transformers import LigerBlockSparseTop2MLP
```
This means that we will test one function/class: LigerBlockSparseTop2MLP.
And the defination and implementation of class LigerBlockSparseTop2MLP should be in `/testbed/src/liger_kernel/transformers/swiglu.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
class LigerBlockSparseTop2MLP(nn.Module):
    """
    A PyTorch neural network module implementing a block-sparse top-2 Multi-Layer Perceptron (MLP) with SwiGLU activation.
    
    This class provides an optimized implementation of a feed-forward network layer commonly used in transformer architectures. It uses the Liger kernel's optimized SiLU multiplication function for improved performance and implements a gated linear unit structure with SwiGLU activation.
    
    Attributes:
        ffn_dim (int): The dimension of the feed-forward network intermediate layer, derived from config.intermediate_size
        hidden_dim (int): The dimension of the hidden layer, derived from config.hidden_size
        w1 (nn.Linear): First linear transformation layer (hidden_dim -> ffn_dim) without bias
        w2 (nn.Linear): Output linear transformation layer (ffn_dim -> hidden_dim) without bias  
        w3 (nn.Linear): Gate linear transformation layer (hidden_dim -> ffn_dim) without bias
    
    Methods:
        __init__(config): Initializes the MLP layers with dimensions from the provided configuration.
            Validates that the activation function is either 'silu' or 'swish'.
        
        forward(x): Performs the forward pass through the MLP using SwiGLU activation.
            Computes w2(SiLU(w1(x)) * w3(x)) where * denotes element-wise multiplication.
    
    Usage Example:
        ```python
        import torch
        from types import SimpleNamespace
        
        # Create configuration
        config = SimpleNamespace()
        config.hidden_size = 768
        config.intermediate_size = 3072
        config.hidden_act = "silu"
        
        # Initialize the MLP
        mlp = LigerBlockSparseTop2MLP(config)
        
        # Forward pass
        batch_size, seq_len = 32, 128
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        output = mlp(x)  # Shape: (32, 128, 768)
        ```
    
    Note:
        This implementation requires the activation function to be either 'silu' or 'swish', 
        and uses the optimized LigerSiLUMulFunction for efficient computation of the gated activation.
    """

    def __init__(self, config):
        """
        Initialize a LigerSwiGLUMLP module with SwiGLU (Swish-Gated Linear Unit) architecture.

        This constructor sets up a three-layer MLP with gating mechanism that uses SiLU/Swish activation
        function. The architecture consists of gate projection, up projection, and down projection layers,
        implementing the SwiGLU variant commonly used in transformer models for improved performance.

        Parameters:
            config: Configuration object containing model hyperparameters. Must have the following attributes:
                - hidden_size (int): Dimension of the input hidden states
                - intermediate_size (int): Dimension of the intermediate/feed-forward layer
                - hidden_act (str): Activation function name, must be either "silu" or "swish"

        Raises:
            ValueError: If config.hidden_act is not "silu" or "swish"

        Notes:
            - All linear layers are created without bias terms (bias=False)
            - The gate_proj and up_proj layers both transform from hidden_size to intermediate_size
            - The down_proj layer transforms from intermediate_size back to hidden_size
            - This implementation is optimized to work with LigerSiLUMulFunction for efficient computation
        """
        <your code>
...
```

In order to implement this functionality, some additional libraries etc. are often required, I don't restrict you to any libraries, you need to think about what dependencies you might need and fetch and install and call them yourself. The only thing is that you **MUST** fulfill the input/output format described by this interface, otherwise the test will not pass and you will get zero points for this feature.
And note that there may be not only one **Interface Description**, you should match all **Interface Description {n}**

### Test Description 1
Below is **Test Description 1**
```python
from liger_kernel.transformers import LigerBlockSparseTop2MLP
from liger_kernel.transformers import LigerGEGLUMLP
from liger_kernel.transformers import LigerPhi3SwiGLUMLP
from liger_kernel.transformers import LigerQwen3MoeSwiGLUMLP
from liger_kernel.transformers import LigerRMSNorm
from liger_kernel.transformers import LigerSwiGLUMLP
from liger_kernel.transformers.layer_norm import LigerLayerNorm
from liger_kernel.transformers.monkey_patch import _apply_liger_kernel
from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from liger_kernel.transformers import apply_liger_kernel_to_gemma
from liger_kernel.transformers import apply_liger_kernel_to_gemma2
from liger_kernel.transformers import apply_liger_kernel_to_gemma3
from liger_kernel.transformers import apply_liger_kernel_to_gemma3_text
from liger_kernel.transformers import apply_liger_kernel_to_glm4
from liger_kernel.transformers import apply_liger_kernel_to_glm4v
from liger_kernel.transformers import apply_liger_kernel_to_glm4v_moe
from liger_kernel.transformers import apply_liger_kernel_to_llama
from liger_kernel.transformers import apply_liger_kernel_to_mistral
from liger_kernel.transformers import apply_liger_kernel_to_mixtral
from liger_kernel.transformers import apply_liger_kernel_to_mllama
from liger_kernel.transformers import apply_liger_kernel_to_phi3
from liger_kernel.transformers import apply_liger_kernel_to_qwen2
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
from liger_kernel.transformers import apply_liger_kernel_to_qwen3
from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe
from liger_kernel.transformers import apply_liger_kernel_to_smollm3
from liger_kernel.transformers.rms_norm import LigerRMSNormForGlm4
```

### Interface Description 1
Below is **Interface Description 1** for file: src-liger_kernel-transformers-geglu.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
class LigerGEGLUMLP(nn.Module):
    """
    A PyTorch neural network module implementing a Gated Expert GLU (GEGLU) Multi-Layer Perceptron with optimized GELU activation.
    
    This class provides an efficient implementation of the GEGLU MLP architecture commonly used in transformer models, particularly in Gemma models. It combines gating mechanisms with GLU (Gated Linear Units) and uses the Liger kernel's optimized GELU multiplication function for improved performance.
    
    Attributes:
        config: Configuration object containing model hyperparameters
        hidden_size (int): Dimension of the input and output hidden states
        intermediate_size (int): Dimension of the intermediate projection layer
        gate_proj (nn.Linear): Linear projection for the gating mechanism (no bias)
        up_proj (nn.Linear): Linear projection for the up-scaling transformation (no bias)
        down_proj (nn.Linear): Linear projection for down-scaling back to hidden size (no bias)
    
    Methods:
        __init__(config): Initializes the GEGLU MLP with the specified configuration, setting up
            the three linear projection layers based on hidden_size and intermediate_size.
        
        forward(x): Performs the forward pass through the GEGLU MLP. Applies gate and up projections
            to the input, processes them through the optimized GELU multiplication function, and
            then applies the down projection to return to the original hidden dimension.
    
    Usage Example:
        ```python
        import torch
        from types import SimpleNamespace
        
        # Create configuration
        config = SimpleNamespace()
        config.hidden_size = 768
        config.intermediate_size = 3072
        
        # Initialize the GEGLU MLP
        mlp = LigerGEGLUMLP(config)
        
        # Forward pass
        batch_size, seq_len = 32, 128
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        output = mlp(x)  # Shape: (32, 128, 768)
        ```
    
    Note:
        This implementation uses the tanh approximation form of GELU as used in Gemma models.
        The LigerGELUMulFunction provides an optimized kernel implementation for better performance
        compared to standard PyTorch operations.
    """

    def __init__(self, config):
        """
        Initialize a GEGLU (Gated Exponential Linear Unit) Multi-Layer Perceptron module.

        This constructor sets up a three-layer MLP architecture with GEGLU activation, commonly used
        in transformer models like Gemma. The module consists of gate projection, up projection, and
        down projection layers, where the gate and up projections work together to create a gated
        activation mechanism.

        Parameters:
            config: Configuration object containing model hyperparameters. Must have the following
                   attributes:
                   - hidden_size (int): The dimensionality of the input and output hidden states
                   - intermediate_size (int): The dimensionality of the intermediate layer, typically
                     larger than hidden_size (often 4x or 8x)

        Returns:
            None: This is a constructor method.

        Notes:
            - All linear layers are initialized without bias terms (bias=False)
            - The implementation uses tanh approximation for GELU activation, which is compatible
              with Gemma model variants (1, 1.1, and 2)
            - The gate_proj and up_proj layers both transform from hidden_size to intermediate_size
            - The down_proj layer transforms back from intermediate_size to hidden_size
            - Exact GELU support is planned for future implementation

        Raises:
            AttributeError: If the config object is missing required attributes (hidden_size or
                           intermediate_size)
        """
        <your code>

    def forward(self, x):
        """
        Forward pass of the GEGLU MLP layer.

        This method implements the forward propagation through a Gated Linear Unit (GLU) variant
        that uses GELU activation function. The computation follows the pattern:
        GEGLU(x) = down_proj(GELU(gate_proj(x)) * up_proj(x))

        The gate projection applies GELU activation and is element-wise multiplied with the
        up projection before being passed through the down projection to produce the final output.

        Parameters:
            x (torch.Tensor): Input tensor of shape (..., hidden_size) where the last dimension
                             must match the model's hidden_size configuration parameter.

        Returns:
            torch.Tensor: Output tensor of shape (..., hidden_size) after applying the GEGLU
                         transformation. The output maintains the same shape as the input tensor.

        Notes:
            - Uses GELU with tanh approximation as implemented in LigerGELUMulFunction
            - The intermediate computation expands to intermediate_size before projecting back
              to hidden_size
            - All linear projections are performed without bias terms
            - This implementation is optimized for Gemma model architectures (versions 1, 1.1, and 2)
              which use the tanh approximation form of GELU
        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.

### Interface Description 2
Below is **Interface Description 2** for file: src-liger_kernel-transformers-auto_model.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
class AutoLigerKernelForCausalLM(AutoModelForCausalLM):
    """
    
        This class is a drop-in replacement for AutoModelForCausalLM that applies the Liger Kernel to the model
        if applicable.
        
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        **kwargs
    ):
        """
        Load a pretrained causal language model with Liger Kernel optimizations automatically applied.

        This method is a drop-in replacement for AutoModelForCausalLM.from_pretrained() that 
        automatically detects the model type and applies appropriate Liger Kernel optimizations
        if supported. The Liger Kernel provides optimized implementations of common transformer
        operations to improve training and inference performance.

        Args:
            pretrained_model_name_or_path (str or os.PathLike): 
                Can be either:
                - A string, the model id of a pretrained model hosted inside a model repo 
                  on huggingface.co
                - A path to a directory containing model weights saved using 
                  save_pretrained()
            *model_args: 
                Additional positional arguments passed along to the underlying model's 
                __init__ method.
            **kwargs: 
                Additional keyword arguments passed to the underlying model's __init__ method
                and/or used to configure Liger Kernel optimizations. Note that kwargs specific
                to Liger Kernel functions will be filtered out before model initialization to
                prevent errors.

        Returns:
            PreTrainedModel: A causal language model instance with Liger Kernel optimizations
            applied if the model type is supported. The exact return type depends on the 
            model architecture (e.g., LlamaForCausalLM, MistralForCausalLM, etc.).

        Notes:
            - Liger Kernel optimizations are only applied if the model type is supported.
              Unsupported model types will load normally without optimizations.
            - The method automatically filters out Liger-specific kwargs before passing
              them to the model constructor to prevent initialization errors.
            - All standard AutoModelForCausalLM.from_pretrained() functionality is preserved,
              including device mapping, quantization, and other transformers features.
        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.

### Interface Description 3
Below is **Interface Description 3** for file: src-liger_kernel-transformers-monkey_patch.py

This file contains 20 top-level interface(s) that need to be implemented.

```python
def apply_liger_kernel_to_llama(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None
) -> None:
    """
    
        Apply Liger kernels to replace original implementation in HuggingFace Llama models (2 and 3)
    
        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.
        
    """
    <your code>

def apply_liger_kernel_to_smollm3(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None
) -> None:
    """
    
        Apply Liger kernels to replace original implementation in HuggingFace SmolLM3 model
    
        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.
        
    """
    <your code>

def apply_liger_kernel_to_mllama(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    layer_norm: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None
) -> None:
    """
    
        Apply Liger kernels to replace original implementation in HuggingFace MLlama models.
        NOTE: MLlama is not available in transformers<4.45.0
    
        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.
        
    """
    <your code>

def apply_liger_kernel_to_mistral(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None
) -> None:
    """
    
        Apply Liger kernels to replace original implementation in HuggingFace Mistral models
    
        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is False.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is True.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.
        
    """
    <your code>

def apply_liger_kernel_to_mixtral(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None
) -> None:
    """
    
        Apply Liger kernels to replace original implementation in HuggingFace Mixtral models
    
        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.
        
    """
    <your code>

def apply_liger_kernel_to_gemma(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    geglu: bool = True,
    model: PreTrainedModel = None
) -> None:
    """
    
        Apply Liger kernels to replace original implementation in HuggingFace Gemma
        (Gemma 1 and 1.1 supported, for Gemma2 please use `apply_liger_kernel_to_gemma2` ) to make GPU go burrr.
    
        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            geglu (bool): Whether to apply Liger's GeGLU MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.
        
    """
    <your code>

def apply_liger_kernel_to_gemma2(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    geglu: bool = True,
    model: PreTrainedModel = None
) -> None:
    """
    
        Apply Liger kernels to replace original implementation in HuggingFace Gemma2
        (for Gemma1 please use `apply_liger_kernel_to_gemma`) to make GPU go burrr.
    
        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            geglu (bool): Whether to apply Liger's GeGLU MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.
        
    """
    <your code>

def apply_liger_kernel_to_gemma3_text(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    geglu: bool = True,
    model: PreTrainedModel = None
) -> None:
    """
    
        Apply Liger kernels to replace original implementation in HuggingFace Gemma3
    
        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            geglu (bool): Whether to apply Liger's GeGLU MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.
        
    """
    <your code>

def apply_liger_kernel_to_gemma3(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    layer_norm: bool = True,
    rms_norm: bool = True,
    geglu: bool = True,
    model: PreTrainedModel = None
) -> None:
    """
    
        Apply Liger kernels to replace original implementation in HuggingFace Gemma3
    
        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            layer_norm (bool): Whether to apply Liger's LayerNorm. Default is True.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            geglu (bool): Whether to apply Liger's GeGLU MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.
        
    """
    <your code>

def apply_liger_kernel_to_qwen2(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None
) -> None:
    """
    
        Apply Liger kernels to replace original implementation in HuggingFace Qwen2 models
    
        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.
        
    """
    <your code>

def apply_liger_kernel_to_qwen3(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None
) -> None:
    """
    
        Apply Liger kernels to replace original implementation in HuggingFace Qwen3 models.
        
    """
    <your code>

def apply_liger_kernel_to_qwen3_moe(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None
) -> None:
    """
    
        Apply Liger kernels to replace original implementation in HuggingFace Qwen3 models.
        
    """
    <your code>

def apply_liger_kernel_to_qwen2_vl(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    layer_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None
) -> None:
    """
    
        Apply Liger kernels to replace original implementation in HuggingFace Qwen2-VL models.
        NOTE: Qwen2-VL is not supported in transformers<4.52.4
    
        Args:
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            layer_norm (bool): Whether to apply Liger's LayerNorm. Default is True.
            swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.
        
    """
    <your code>

def apply_liger_kernel_to_qwen2_5_vl(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None
) -> None:
    """
    
        Apply Liger kernels to replace original implementation in HuggingFace Qwen2.5-VL models.
        NOTE: Qwen2.5-VL is not available in transformers<4.48.2
    
        Args:
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.
        
    """
    <your code>

def apply_liger_kernel_to_phi3(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None
) -> None:
    """
    
        Apply Liger kernels to replace original implementation in HuggingFace Phi3 models.
    
        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            swiglu (bool): Whether to apply Liger's SwiGLU Phi3MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.
        
    """
    <your code>

def apply_liger_kernel_to_glm4(
    rope: bool = False,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None
) -> None:
    """
    
        Apply Liger kernels to replace original implementation in HuggingFace GLM-4 models.
    
        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is False.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            swiglu (bool): Whether to apply Liger's SwiGLU Glm4MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.
        
    """
    <your code>

def apply_liger_kernel_to_glm4v(
    rope: bool = False,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None
) -> None:
    """
    
        Apply Liger kernels to replace original implementation in HuggingFace GLM-4v models.
    
        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is False.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            swiglu (bool): Whether to apply Liger's SwiGLU Glm4MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.
        
    """
    <your code>

def apply_liger_kernel_to_glm4v_moe(
    rope: bool = False,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None
) -> None:
    """
    
        Apply Liger kernels to replace original implementation in HuggingFace GLM4v_moe models.
    
        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is False.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            swiglu (bool): Whether to apply Liger's SwiGLUMLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.
        
    """
    <your code>

def _apply_liger_kernel(model_type: str, **kwargs) -> None:
    """
    
        Applies Liger kernels based on the specified model type. The custom
        kernels for the specified model type will be applied with the provided
        keyword arguments, otherwise the default configuration will be used.
    
        ** Note: Calling _apply_liger_kernel() after model initialization
        will not be able to fully patch models. This must be called before model initialization.
        If the model has already been instantiated
    
        Args:
            - model_type: the model types as defined in transformers/models/auto/modeling_auto.py
              and specified in the model's config.json
            - kwargs: keyword arguments that are passed to the corresponding apply_liger_kernel_to_* function.
        
    """
    <your code>

def _apply_liger_kernel_to_instance(
    model: PreTrainedModel,
    **kwargs
) -> None:
    """
    
        Applies Liger kernels to the provided model instance.
    
        Args:
            - model: the model instance to apply Liger kernels to
            - kwargs: keyword arguments that are passed to the corresponding apply_liger_kernel_to_* function.
        
    """
    <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.

### Interface Description 4
Below is **Interface Description 4** for file: src-liger_kernel-transformers-rms_norm.py

This file contains 2 top-level interface(s) that need to be implemented.

```python
class LigerRMSNorm(nn.Module):
    """
    A high-performance implementation of Root Mean Square Layer Normalization (RMSNorm) optimized for large language models.
    
    LigerRMSNorm provides an efficient alternative to standard layer normalization by computing normalization
    using only the root mean square of the input activations, without centering around the mean. This approach
    reduces computational overhead while maintaining training stability and model performance.
    
    Attributes:
        weight (nn.Parameter): Learnable scaling parameter of shape (hidden_size,). Initialized to ones or zeros
            based on the init_fn parameter.
        variance_epsilon (float): Small constant added to variance for numerical stability. Default is 1e-6.
        offset (float): Additive offset applied during normalization. Default is 0.0.
        casting_mode (str): Determines the computation precision and casting behavior. Default is "llama".
        in_place (bool): Whether to perform in-place operations for memory efficiency. Default is True.
        row_mode (str or None): Optional row-wise processing mode for specific optimization patterns.
    
    Methods:
        __init__(hidden_size, eps=1e-6, offset=0.0, casting_mode="llama", init_fn="ones", in_place=True, row_mode=None):
            Initializes the RMSNorm layer with specified configuration parameters.
            
        forward(hidden_states):
            Applies RMS normalization to the input tensor using the optimized LigerRMSNormFunction.
            Returns normalized tensor with the same shape as input.
            
        extra_repr():
            Returns a string representation of the layer's key parameters for debugging and logging.
    
    Usage Examples:
        Basic usage with default parameters:
        >>> rms_norm = LigerRMSNorm(hidden_size=768)
        >>> input_tensor = torch.randn(32, 128, 768)  # (batch, seq_len, hidden_size)
        >>> normalized = rms_norm(input_tensor)
        
        Custom configuration for specific model requirements:
        >>> rms_norm = LigerRMSNorm(
        ...     hidden_size=1024,
        ...     eps=1e-5,
        ...     offset=1.0,
        ...     casting_mode="gemma",
        ...     init_fn="zeros",
        ...     in_place=False
        ... )
        >>> output = rms_norm(hidden_states)
    
    Note:
        This implementation is optimized for GPU acceleration and memory efficiency. The casting_mode
        parameter should be chosen based on the target model architecture (e.g., "llama" for LLaMA-style
        models, "gemma" for Gemma-style models).
    """

    def __init__(
        self,
        hidden_size,
        eps = 1e-06,
        offset = 0.0,
        casting_mode = 'llama',
        init_fn = 'ones',
        in_place = True,
        row_mode = None
    ):
        """
        Initialize a Liger RMS Normalization layer.

        RMS (Root Mean Square) Normalization is a normalization technique that normalizes
        the input by dividing by the root mean square of the input elements. This implementation
        provides an optimized version with configurable parameters for different model architectures.

        Parameters:
            hidden_size (int): The size of the hidden dimension to be normalized. This determines
                the size of the learnable weight parameter.
            eps (float, optional): A small value added to the denominator for numerical stability
                to avoid division by zero. Defaults to 1e-6.
            offset (float, optional): An offset value added during normalization computation.
                Different models use different offset values (e.g., 0.0 for LLaMA, 1.0 for Gemma).
                Defaults to 0.0.
            casting_mode (str, optional): Specifies the casting behavior for different model
                architectures. Supported modes include 'llama' and 'gemma'. Defaults to 'llama'.
            init_fn (str, optional): Initialization function for the weight parameter. Must be
                either 'ones' or 'zeros'. When 'ones', weights are initialized to 1.0; when
                'zeros', weights are initialized to 0.0. Defaults to 'ones'.
            in_place (bool, optional): Whether to perform the normalization operation in-place
                to save memory. When True, the input tensor may be modified directly.
                Defaults to True.
            row_mode (optional): Specifies the row processing mode for the normalization operation.
                The exact behavior depends on the underlying kernel implementation. Defaults to None.

        Raises:
            AssertionError: If init_fn is not 'ones' or 'zeros'.

        Notes:
            - The weight parameter is created as a learnable parameter with shape (hidden_size,)
            - Different model architectures may require different parameter configurations
            - The actual normalization computation is performed by LigerRMSNormFunction in the forward pass
            - In-place operations can reduce memory usage but may affect gradient computation in some cases
        """
        <your code>

    def forward(self, hidden_states):
        """
        Performs forward pass of the Liger RMS normalization layer.

        This method applies Root Mean Square (RMS) normalization to the input hidden states
        using an optimized kernel implementation. RMS normalization normalizes the input by
        dividing by the root mean square of the elements, which helps stabilize training
        and improve model performance.

        Parameters:
            hidden_states (torch.Tensor): Input tensor to be normalized. Expected to be
                a multi-dimensional tensor where normalization is applied along the last
                dimension. The last dimension should match the hidden_size specified
                during initialization.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as the input hidden_states.
                The output is computed as: hidden_states / sqrt(mean(hidden_states^2) + eps) * weight + offset,
                where the exact computation depends on the casting_mode configuration.

        Notes:
            - The normalization is performed using the LigerRMSNormFunction which provides
              optimized CUDA kernels for better performance compared to standard PyTorch operations
            - The behavior varies based on initialization parameters:
                * variance_epsilon: Small value added for numerical stability
                * offset: Additive offset applied after normalization
                * casting_mode: Determines the specific normalization variant (e.g., "llama", "gemma")
                * in_place: Whether to perform the operation in-place to save memory
                * row_mode: Optional row-wise processing mode for specific use cases
            - When in_place=True, the input tensor may be modified directly to save memory
            - The weight parameter is learned during training and scales the normalized output
        """
        <your code>

    def extra_repr(self):
        """
        Return a string representation of the extra parameters for this LigerRMSNorm module.

        This method provides a human-readable string containing the key configuration 
        parameters of the RMS normalization layer, which is useful for debugging, 
        logging, and model inspection purposes.

        Returns:
            str: A formatted string containing the weight tensor shape as a tuple, 
                 epsilon value for numerical stability, offset value, in-place operation 
                 flag, and row processing mode. The format is:
                 "(weight_shape), eps=variance_epsilon, offset=offset, in_place=in_place, row_mode=row_mode"

        Notes:
            - This method is typically called automatically by PyTorch when printing 
              the module or converting it to string representation
            - The weight shape is displayed as a tuple for better readability
            - All configuration parameters that affect the normalization behavior are included
            - This method inherits from nn.Module.extra_repr() and follows PyTorch conventions
        """
        <your code>

class LigerRMSNormForGlm4(LigerRMSNorm):
    """
    A specialized RMS normalization layer optimized for GLM-4 (General Language Model 4) architecture.
    
    This class extends LigerRMSNorm with GLM-4 specific default parameters, providing an efficient
    implementation of Root Mean Square Layer Normalization tailored for GLM-4 model requirements.
    It uses optimized kernel operations for better performance compared to standard PyTorch implementations.
    
    Attributes:
        weight (nn.Parameter): Learnable scaling parameter of shape (hidden_size,), initialized to ones
        variance_epsilon (float): Small constant added to variance for numerical stability (default: 1e-6)
        offset (float): Offset value added during normalization computation (default: 0.0)
        casting_mode (str): Computation mode, set to "llama" for GLM-4 compatibility
        in_place (bool): Whether to perform in-place operations, set to False for GLM-4
        row_mode (str or None): Row processing mode for the normalization operation
    
    Methods:
        __init__(hidden_size, eps=1e-6, offset=0.0, casting_mode="llama", init_fn="ones", 
                 in_place=False, row_mode=None):
            Initializes the GLM-4 RMS normalization layer with model-specific defaults.
    
    Usage Examples:
        # Basic usage in GLM-4 model
        rms_norm = LigerRMSNormForGlm4(hidden_size=4096)
        normalized_output = rms_norm(hidden_states)
        
        # Custom epsilon for different numerical precision
        rms_norm = LigerRMSNormForGlm4(hidden_size=2048, eps=1e-8)
        
        # Integration in GLM-4 transformer block
        class GLM4Block(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.norm = LigerRMSNormForGlm4(hidden_size)
                # ... other components
                
            def forward(self, x):
                return self.norm(x)
    
    Note:
        This implementation is specifically configured for GLM-4 architecture with in_place=False
        to ensure compatibility with GLM-4's computational graph requirements. The "llama" casting
        mode provides the appropriate numerical behavior for GLM-4 model training and inference.
    """

    def __init__(
        self,
        hidden_size,
        eps = 1e-06,
        offset = 0.0,
        casting_mode = 'llama',
        init_fn = 'ones',
        in_place = False,
        row_mode = None
    ):
        """
        Initialize a Liger RMS Normalization layer.

        RMS (Root Mean Square) Normalization is a normalization technique that normalizes
        the input by dividing by the root mean square of the input elements. This implementation
        provides an optimized version with configurable parameters for different model architectures.

        Parameters:
            hidden_size (int): The size of the hidden dimension to be normalized. This determines
                the size of the learnable weight parameter.
            eps (float, optional): A small value added to the denominator for numerical stability
                to avoid division by zero. Defaults to 1e-6.
            offset (float, optional): An offset value added during normalization computation.
                Different models use different offset values (e.g., 0.0 for LLaMA, 1.0 for Gemma).
                Defaults to 0.0.
            casting_mode (str, optional): Specifies the casting behavior for different model
                architectures. Supported modes include 'llama' and 'gemma'. Defaults to 'llama'.
            init_fn (str, optional): Initialization function for the weight parameter. Must be
                either 'ones' or 'zeros'. When 'ones', weights are initialized to 1.0; when
                'zeros', weights are initialized to 0.0. Defaults to 'ones'.
            in_place (bool, optional): Whether to perform the normalization operation in-place
                to save memory. When True, the input tensor may be modified directly.
                Defaults to False.
            row_mode (optional): Specifies the row processing mode for the normalization operation.
                The exact behavior depends on the underlying kernel implementation. Defaults to None.

        Raises:
            AssertionError: If init_fn is not 'ones' or 'zeros'.

        Notes:
            - The weight parameter is created as a learnable parameter with shape (hidden_size,)
            - Different model architectures may require different parameter configurations
            - The actual normalization computation is performed by LigerRMSNormFunction in the forward pass
            - This is the base class for architecture-specific RMS normalization variants
        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.

### Interface Description 5
Below is **Interface Description 5** for file: src-liger_kernel-transformers-layer_norm.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
class LigerLayerNorm(nn.Module):
    """
    A custom implementation of Layer Normalization using Liger kernel operations for improved performance.
    
    This class provides a PyTorch nn.Module implementation of layer normalization that leverages
    the LigerLayerNormFunction for potentially optimized computation. Layer normalization normalizes
    the inputs across the features dimension, helping to stabilize training and improve convergence
    in neural networks.
    
    Attributes:
        hidden_size (int): The size of the hidden dimension to be normalized.
        eps (float): A small epsilon value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scale parameter initialized based on init_fn.
        bias (nn.Parameter): Learnable shift parameter, initialized as zeros or random values.
        variance_epsilon (float): Alias for eps used in the underlying computation.
    
    Methods:
        __init__(hidden_size, eps=1e-6, bias=False, init_fn="ones"):
            Initializes the layer normalization module with specified parameters.
            
            Args:
                hidden_size (int): Dimension of the input features to normalize
                eps (float): Small value for numerical stability (default: 1e-6)
                bias (bool): Whether to include learnable bias parameter (default: False)
                init_fn (str): Weight initialization method, either "ones" or "zeros" (default: "ones")
        
        forward(hidden_states):
            Applies layer normalization to the input tensor using the Liger kernel function.
            
            Args:
                hidden_states (torch.Tensor): Input tensor to normalize
                
            Returns:
                torch.Tensor: Layer normalized output tensor
        
        extra_repr():
            Returns a string representation of the module's key parameters for debugging.
            
            Returns:
                str: String containing hidden_size and eps values
    
    Usage Examples:
        Basic usage:
        >>> layer_norm = LigerLayerNorm(hidden_size=768)
        >>> input_tensor = torch.randn(32, 128, 768)  # (batch, seq_len, hidden_size)
        >>> output = layer_norm(input_tensor)
        
        With bias enabled:
        >>> layer_norm_with_bias = LigerLayerNorm(hidden_size=512, bias=True)
        >>> input_tensor = torch.randn(16, 512)
        >>> output = layer_norm_with_bias(input_tensor)
        
        Custom epsilon and zero initialization:
        >>> layer_norm_custom = LigerLayerNorm(hidden_size=256, eps=1e-5, init_fn="zeros")
        >>> input_tensor = torch.randn(8, 64, 256)
        >>> output = layer_norm_custom(input_tensor)
    """

    def __init__(
        self,
        hidden_size,
        eps = 1e-06,
        bias = False,
        init_fn = 'ones'
    ):
        """
        Initialize a Liger Layer Normalization module.

        This constructor creates a custom layer normalization module that uses the LigerLayerNormFunction
        for efficient computation. Layer normalization normalizes inputs across the feature dimension,
        helping to stabilize training and improve convergence in neural networks.

        Parameters:
            hidden_size (int): The number of features in the input tensor. This determines the size
                of the learnable weight and bias parameters.
            eps (float, optional): A small value added to the denominator for numerical stability
                when computing the variance. Defaults to 1e-6.
            bias (bool, optional): If True, initializes the bias parameter with random values from
                a normal distribution. If False, initializes bias as zeros. Defaults to False.
            init_fn (str, optional): Initialization function for the weight parameter. Must be either
                'ones' or 'zeros'. If 'ones', weights are initialized to 1.0. If 'zeros', weights
                are initialized to 0.0. Defaults to 'ones'.

        Raises:
            AssertionError: If init_fn is not 'ones' or 'zeros'.

        Notes:
            - The weight parameter is always created as a learnable parameter regardless of init_fn
            - The bias parameter is always created, but only initialized with random values if bias=True
            - The variance_epsilon attribute stores the eps value for use in the forward pass
            - This implementation is designed to work with the LigerLayerNormFunction for optimized performance
        """
        <your code>

    def forward(self, hidden_states):
        """
        Performs forward pass of the Liger Layer Normalization.

        This method applies layer normalization to the input hidden states using the Liger kernel
        implementation for optimized performance. Layer normalization normalizes the input across
        the feature dimension, applying learned weight and bias parameters.

        Args:
            hidden_states (torch.Tensor): Input tensor to be normalized. Expected to have shape
                (..., hidden_size) where the last dimension matches the hidden_size specified
                during initialization. The tensor can have any number of leading dimensions
                (batch size, sequence length, etc.).

        Returns:
            torch.Tensor: Layer normalized tensor with the same shape as the input hidden_states.
                The output is computed as: weight * (hidden_states - mean) / sqrt(variance + eps) + bias,
                where normalization statistics are computed across the last dimension.

        Note:
            This method uses the LigerLayerNormFunction which provides an optimized implementation
            of layer normalization. The normalization is applied across the last dimension of the
            input tensor using the variance_epsilon value specified during initialization.
        """
        <your code>

    def extra_repr(self):
        """
        Return a string containing extra representation information for this layer.

        This method provides a concise string representation of the key parameters
        that define this LigerLayerNorm layer instance. It is automatically called
        by PyTorch when displaying the module (e.g., when printing the model or
        using repr()).

        Parameters:
            None

        Returns:
            str: A formatted string containing the hidden_size and eps parameters
                 in the format "{hidden_size}, eps={eps}". For example, if 
                 hidden_size=768 and eps=1e-6, returns "768, eps=1e-06".

        Notes:
            This method is part of PyTorch's nn.Module interface and is used to
            provide additional information when the module is printed or represented
            as a string. The returned string appears within the parentheses of the
            module's string representation alongside the module's class name.
        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.

### Interface Description 6
Below is **Interface Description 6** for file: src-liger_kernel-transformers-swiglu.py

This file contains 4 top-level interface(s) that need to be implemented.

```python
class LigerSwiGLUMLP(nn.Module):
    """
    A PyTorch neural network module implementing the SwiGLU (Swish-Gated Linear Unit) MLP architecture optimized with Liger kernel operations.
    
    This class provides an efficient implementation of the SwiGLU activation function commonly used in transformer architectures. SwiGLU combines a gating mechanism with the SiLU (Swish) activation function to create a more expressive feedforward layer. The implementation leverages the LigerSiLUMulFunction for optimized computation.
    
    Attributes:
        config: Configuration object containing model hyperparameters
        hidden_size (int): Dimension of the input and output hidden states
        intermediate_size (int): Dimension of the intermediate feedforward layer
        gate_proj (nn.Linear): Linear projection for the gating mechanism (no bias)
        up_proj (nn.Linear): Linear projection for the up-scaling transformation (no bias)
        down_proj (nn.Linear): Linear projection for the down-scaling transformation (no bias)
    
    Methods:
        __init__(config): Initializes the MLP layers and validates the activation function.
            Requires config.hidden_act to be either "silu" or "swish".
        
        forward(x): Performs the forward pass through the SwiGLU MLP.
            Applies gate and up projections, then uses LigerSiLUMulFunction for
            element-wise multiplication with SiLU activation, followed by down projection.
    
    Usage Example:
        ```python
        import torch
        from types import SimpleNamespace
        
        # Create configuration
        config = SimpleNamespace(
            hidden_size=768,
            intermediate_size=3072,
            hidden_act="silu"
        )
        
        # Initialize the MLP
        mlp = LigerSwiGLUMLP(config)
        
        # Forward pass
        batch_size, seq_len = 32, 128
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        output = mlp(x)  # Shape: (32, 128, 768)
        ```
    
    Note:
        This implementation only supports "silu" and "swish" activation functions.
        Other activation functions will raise a ValueError during initialization.
    """

    def __init__(self, config):
        """
        Initialize a LigerSwiGLUMLP module with SwiGLU (Swish-Gated Linear Unit) architecture.

        This constructor sets up a three-layer MLP with gating mechanism that uses SiLU/Swish activation
        function. The architecture consists of gate projection, up projection, and down projection layers,
        implementing the SwiGLU variant commonly used in transformer models for improved performance.

        Parameters:
            config: Configuration object containing model hyperparameters. Must have the following attributes:
                - hidden_size (int): Dimension of the input hidden states
                - intermediate_size (int): Dimension of the intermediate/feed-forward layer
                - hidden_act (str): Activation function name, must be either "silu" or "swish"

        Raises:
            ValueError: If config.hidden_act is not "silu" or "swish"

        Notes:
            - All linear layers are created without bias terms (bias=False)
            - The gate_proj and up_proj layers both transform from hidden_size to intermediate_size
            - The down_proj layer transforms from intermediate_size back to hidden_size
            - This implementation is optimized to work with LigerSiLUMulFunction for efficient computation
        """
        <your code>

    def forward(self, x):
        """
        
            Performs the forward pass of the LigerSwiGLUMLP layer with SwiGLU activation.

            The computation process is as follows:
            1. Applies separate linear projections to the input tensor via `gate_proj` and `up_proj`.
            2. Combines the projected results using the SiLU activation function (via `LigerSiLUMulFunction`),
            which implements the SwiGLU logic (SiLU(gate) * up).
            3. Applies a final linear projection via `down_proj` to map back to the original hidden size.

            Parameters:
                x (torch.Tensor): Input tensor of shape (..., hidden_size), where the last
                    dimension must match `self.hidden_size` (from the model configuration).

            Returns:
                torch.Tensor: Output tensor of shape (..., hidden_size), maintaining the same
                    shape as the input except for the MLP transformation on the last dimension.

            Notes:
                - Requires the model configuration to specify "silu" or "swish" as `hidden_act`;
                a ValueError is raised during initialization for other activation functions.
                - `LigerSiLUMulFunction.apply()` is used for efficient computation of SiLU activation
                combined with element-wise multiplication.
                - The `intermediate_size` (from configuration) determines the expansion factor of the
                hidden layer between the projections.
    
        """
        <your code>

class LigerBlockSparseTop2MLP(nn.Module):
    """
    A PyTorch neural network module implementing a block-sparse top-2 Multi-Layer Perceptron (MLP) with SwiGLU activation.
    
    This class provides an optimized implementation of a feed-forward network layer commonly used in transformer architectures. It uses the Liger kernel's optimized SiLU multiplication function for improved performance and implements a gated linear unit structure with SwiGLU activation.
    
    Attributes:
        ffn_dim (int): The dimension of the feed-forward network intermediate layer, derived from config.intermediate_size
        hidden_dim (int): The dimension of the hidden layer, derived from config.hidden_size
        w1 (nn.Linear): First linear transformation layer (hidden_dim -> ffn_dim) without bias
        w2 (nn.Linear): Output linear transformation layer (ffn_dim -> hidden_dim) without bias  
        w3 (nn.Linear): Gate linear transformation layer (hidden_dim -> ffn_dim) without bias
    
    Methods:
        __init__(config): Initializes the MLP layers with dimensions from the provided configuration.
            Validates that the activation function is either 'silu' or 'swish'.
        
        forward(x): Performs the forward pass through the MLP using SwiGLU activation.
            Computes w2(SiLU(w1(x)) * w3(x)) where * denotes element-wise multiplication.
    
    Usage Example:
        ```python
        import torch
        from types import SimpleNamespace
        
        # Create configuration
        config = SimpleNamespace()
        config.hidden_size = 768
        config.intermediate_size = 3072
        config.hidden_act = "silu"
        
        # Initialize the MLP
        mlp = LigerBlockSparseTop2MLP(config)
        
        # Forward pass
        batch_size, seq_len = 32, 128
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        output = mlp(x)  # Shape: (32, 128, 768)
        ```
    
    Note:
        This implementation requires the activation function to be either 'silu' or 'swish', 
        and uses the optimized LigerSiLUMulFunction for efficient computation of the gated activation.
    """

    def __init__(self, config):
        """
        Initialize a LigerSwiGLUMLP module with SwiGLU (Swish-Gated Linear Unit) architecture.

        This constructor sets up a three-layer MLP with gating mechanism that uses SiLU/Swish activation
        function. The architecture consists of gate projection, up projection, and down projection layers,
        implementing the SwiGLU variant commonly used in transformer models for improved performance.

        Parameters:
            config: Configuration object containing model hyperparameters. Must have the following attributes:
                - hidden_size (int): Dimension of the input hidden states
                - intermediate_size (int): Dimension of the intermediate/feed-forward layer
                - hidden_act (str): Activation function name, must be either "silu" or "swish"

        Raises:
            ValueError: If config.hidden_act is not "silu" or "swish"

        Notes:
            - All linear layers are created without bias terms (bias=False)
            - The gate_proj and up_proj layers both transform from hidden_size to intermediate_size
            - The down_proj layer transforms from intermediate_size back to hidden_size
            - This implementation is optimized to work with LigerSiLUMulFunction for efficient computation
        """
        <your code>

    def forward(self, x):
        """
        Performs the forward pass of the SwiGLU MLP layer.

        This method implements the SwiGLU (Swish-Gated Linear Unit) activation function combined with 
        a multi-layer perceptron. The forward pass applies linear transformations through gate and up 
        projections, applies the SwiGLU activation (SiLU activation on gate multiplied by up projection), 
        and then applies a down projection to return to the original hidden dimension.

        The computation flow is:
            1. Apply w1 (gate) projection to input x
            2. Apply w3 (up) projection to input x  
            3. Apply LigerSiLUMulFunction (SiLU(w1) * w3) to combine the projections
            4. Apply w2 (down) projection to reduce back to hidden_size

        Parameters:
            x (torch.Tensor): Input tensor of shape (..., hidden_size) where ... represents 
                             any number of leading dimensions (typically batch_size, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (..., hidden_size), same shape as input but 
                         transformed through the SwiGLU MLP layers.

        Notes:
            - This implementation uses LigerSiLUMulFunction for efficient computation of SiLU activation
              combined with element-wise multiplication
            - The activation function must be "silu" or "swish" as validated during initialization
            - All linear layers are configured without bias terms
            - The intermediate_size is typically larger than hidden_size to provide the MLP with 
              additional representational capacity
        """
        <your code>

class LigerPhi3SwiGLUMLP(nn.Module):
    """
    
        Patch Phi3MLP to use LigerSiLUMulFunction
        https://github.com/huggingface/transformers/blob/v4.41.0/src/transformers/models/phi3/modeling_phi3.py#L241
        
    """

    def __init__(self, config):
        """
        Initialize a LigerSwiGLUMLP module with SwiGLU (Swish-Gated Linear Unit) architecture.

        This constructor sets up a three-layer MLP with gating mechanism that uses SiLU/Swish activation
        function. The architecture consists of gate projection, up projection, and down projection layers,
        implementing the SwiGLU variant commonly used in transformer models for improved performance.

        Parameters:
            config: Configuration object containing model hyperparameters. Must have the following attributes:
                - hidden_size (int): Dimension of the input hidden states
                - intermediate_size (int): Dimension of the intermediate/feed-forward layer
                - hidden_act (str): Activation function name, must be either "silu" or "swish"

        Raises:
            ValueError: If config.hidden_act is not "silu" or "swish"

        Notes:
            - All linear layers are created without bias terms (bias=False)
            - The gate_proj and up_proj layers both transform from hidden_size to intermediate_size
            - The down_proj layer transforms from intermediate_size back to hidden_size
            - This implementation is optimized to work with LigerSiLUMulFunction for efficient computation
        """
        <your code>

    def forward(self, x):
        """
            Performs the forward pass of the SwiGLU MLP layer with combined gate-up projection.

            This method implements the SwiGLU (Swish-Gated Linear Unit) activation function combined with
            a multi-layer perceptron. The forward pass first applies a combined linear transformation 
            (gate_up_proj) that produces both gate and up states in a single projection, which are then
            split into separate tensors. The SwiGLU activation (SiLU on gate multiplied by up states)
            is applied, followed by a down projection to return to the original hidden dimension.

            The computation flow is:
            1. Apply combined gate_up_proj to input x to get merged states
            2. Split merged states into gate and up tensors along the last dimension
            3. Apply LigerSiLUMulFunction (SiLU(gate) * up) to combine the projections
            4. Apply down_proj to reduce back to hidden_size

            Parameters:
                x (torch.Tensor): Input tensor of shape (..., hidden_size) where ... represents
                                any number of leading dimensions (typically batch_size, sequence_length).

            Returns:
                torch.Tensor: Output tensor of shape (..., hidden_size), same shape as input but
                            transformed through the SwiGLU MLP layers.

            Notes:
                - This implementation uses a combined gate_up_proj followed by chunking, which can be
                more efficient than separate projections in some cases
                - LigerSiLUMulFunction is used for efficient computation of SiLU activation combined
                with element-wise multiplication
                - The activation function must be "silu" or "swish" as validated during initialization
                - All linear layers are configured without bias terms
                - The output dimension of gate_up_proj is typically 2 * intermediate_size to accommodate
                both gate and up states after splitting
        """
        <your code>

class LigerQwen3MoeSwiGLUMLP(nn.Module):
    """
    
        Patch Qwen3MoeMLP to use LigerSiLUMulFunction.
        https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3_moe/modular_qwen3_moe.py#L57
        
    """

    def __init__(
        self,
        config,
        intermediate_size = None
    ):
        """
        Initialize a Qwen3 Mixture of Experts (MoE) SwiGLU MLP layer with Liger kernel optimization.

        This constructor sets up a three-layer MLP architecture (gate projection, up projection, and down projection)
        that uses the LigerSiLUMulFunction for efficient SwiGLU activation computation. This is a patched version
        of the Qwen3MoeMLP that leverages optimized kernel operations for better performance.

        Parameters:
            config: Configuration object containing model hyperparameters. Must have the following attributes:
                - hidden_size (int): The size of the hidden layer
                - intermediate_size (int): The size of the intermediate/feed-forward layer
                - hidden_act (str): The activation function name, must be "silu" or "swish"
            intermediate_size (int, optional): Override for the intermediate layer size. If provided,
                this value will be used instead of config.intermediate_size. Defaults to None.

        Raises:
            ValueError: If config.hidden_act is not "silu" or "swish", as these are the only
                supported activation functions for the SwiGLU implementation.

        Notes:
            - All linear layers are created without bias terms (bias=False)
            - The gate_proj and up_proj layers both transform from hidden_size to intermediate_size
            - The down_proj layer transforms from intermediate_size back to hidden_size
            - This implementation is specifically designed to work with the LigerSiLUMulFunction
              for optimized SwiGLU computation during the forward pass
        """
        <your code>

    def forward(self, x):
        """
        Performs the forward pass of the SwiGLU MLP layer.

        This method implements the SwiGLU (Swish-Gated Linear Unit) activation pattern, which applies
        linear transformations followed by element-wise multiplication with SiLU activation. The specific
        implementation varies depending on the MLP class:

        - LigerSwiGLUMLP: Applies gate_proj and up_proj transformations, then SiLU multiplication, 
          followed by down_proj
        - LigerBlockSparseTop2MLP: Similar pattern using w1, w3 for gating/up projection and w2 for down projection
        - LigerPhi3SwiGLUMLP: Uses a combined gate_up_proj that is chunked into gate and up components
        - LigerQwen3MoeSwiGLUMLP: Same pattern as LigerSwiGLUMLP with configurable intermediate size

        Parameters:
            x (torch.Tensor): Input tensor of shape (..., hidden_size) where hidden_size matches
                             the configured hidden dimension of the model.

        Returns:
            torch.Tensor: Output tensor of shape (..., hidden_size) after applying the SwiGLU
                         transformation. The output maintains the same shape as input except for
                         the last dimension which is projected back to hidden_size.

        Notes:
            - Uses LigerSiLUMulFunction.apply() for efficient SiLU activation and element-wise multiplication
            - All linear layers are configured without bias terms
            - The activation function must be either "silu" or "swish" as validated during initialization
            - This implementation is optimized for memory efficiency compared to standard PyTorch operations
        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.