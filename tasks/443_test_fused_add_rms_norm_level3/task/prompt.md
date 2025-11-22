## Task
**Task: Implement Fused Addition and RMS Normalization Operations**

**Core Functionality:**
Develop optimized neural network operations that combine residual connection addition with RMS (Root Mean Square) normalization in a single fused kernel for transformer architectures.

**Main Features & Requirements:**
- Fuse two sequential operations (tensor addition + RMS normalization) into one optimized computation
- Support both functional interface and PyTorch module wrapper
- Handle different precision casting modes for various model architectures (Llama, Gemma)
- Implement custom autograd functions with proper forward and backward passes
- Provide configurable parameters (epsilon, offset, in-place operations)
- Support tensor broadcasting and flexible input shapes

**Key Challenges:**
- Memory optimization through kernel fusion and in-place operations
- Numerical stability across different floating-point precisions
- Compatibility with multiple model architectures' casting behaviors
- Efficient gradient computation for backpropagation
- Maintaining accuracy while optimizing performance

The task focuses on creating high-performance building blocks for transformer models by combining common sequential operations into optimized fused kernels.

**NOTE**: 
- This test is derived from the `liger-kernel` library, but you are NOT allowed to view this codebase or call any of its interfaces. It is **VERY IMPORTANT** to note that if we detect any viewing or calling of this codebase, you will receive a ZERO for this review.
- What's more, you need to install `pytest, pytest-timeout, pytest-json-report` in your environment, otherwise our tests won't run and you'll get **ZERO POINTS**!
- **CRITICAL**: This task is derived from `liger-kernel`, but you **MUST** implement the task description independently. It is **ABSOLUTELY FORBIDDEN** to use `pip install liger-kernel` or some similar commands to access the original implementation—doing so will be considered cheating and will result in an immediate score of ZERO! You must keep this firmly in mind throughout your implementation.
- You are now in `/testbed/`, and originally there was a specific implementation of `liger-kernel` under `/testbed/` that had been installed via `pip install -e .`. However, to prevent you from cheating, we've removed the code under `/testbed/`. While you can see traces of the installation via the pip show, it's an artifact, and `liger-kernel` doesn't exist. So you can't and don't need to use `pip install liger-kernel`, just focus on writing your `agent_code` and accomplishing our task.
- Also, don't try to `pip uninstall liger-kernel` even if the actual `liger-kernel` has already been deleted by us, as this will affect our evaluation of you, and uninstalling the residual `liger-kernel` will result in you getting a ZERO because our tests won't run.

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

Your final deliverable should be code in the `/testbed/agent_code` directory.
The final structure is like below, note that all dirs and files under agent_code/ are just examples, your codebase's structure should match import structure in **Test Description**, which I will tell you later.
```
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
from agent_code.liger_kernel.ops.fused_add_rms_norm import LigerFusedAddRMSNormFunction
```
This means that we will test one function/class: LigerFusedAddRMSNormFunction.
And the defination and implementation of class LigerFusedAddRMSNormFunction should be in `/testbed/agent_code/liger_kernel/ops/fused_add_rms_norm.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
class LigerFusedAddRMSNormFunction(torch.autograd.Function):
    """
    
        Performs a fused operation that first adds a residual tensor to the hidden_states tensor (`X`), then applies RMSNorm (Root Mean Square Normalization) to the result using the weight tensor `W`, with optional offset and casting mode.
    
        This class implements the following sequence, commonly used in transformer decoder layers:
            1. hidden_states = residual + hidden_states
            2. residual = hidden_states (after addition)
            3. hidden_states = rmsnorm(hidden_states)
    
        Both the normalized hidden_states and the updated residual are returned as outputs.
    
        Some models use an 'offset' to shift the weight tensor `W` by a constant value. For example, Gemma
        uses an offset of 1.0, so the computation becomes `(X / RMS(X)) * (W + 1.0)` instead of the usual
        `(X / RMS(X)) * W`. You can pass the offset value as an argument to the forward function.
    
        In addition, different models cast their inputs at different places during RMSNorm computation. For
        example, Gemma casts everything to fp32 before starting the computation, while Llama casts only the
        inverse RMS to fp32. You can specify the casting mode using the `casting_mode` argument. We currently
        support the following casting modes (they match HuggingFace Transformers' implementations):
        - 'llama': matches the Llama implementation, where only the inverse RMS is computed on fp32.
        - 'gemma': matches the Gemma implementation, where everything is cast to fp32, then computed, then cast back to the original dtype.
        - 'none': no casting is done. The computation is done in the original dtype. This saves memory and is slightly faster, but has more error w.r.t. the original implementation.
    
        The `in_place` option determines whether to modify dY in-place to store dX. This defaults to `True` to save memory.
        
    """

    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        X,
        R,
        W,
        eps,
        offset = 0.0,
        casting_mode = 'llama',
        in_place = False
    ):
        """

                X: (B, T, H) or (BxT, H)
                W: (H,)

        """
        <your code>
...
```

In order to implement this functionality, some additional libraries etc. are often required, I don't restrict you to any libraries, you need to think about what dependencies you might need and fetch and install and call them yourself. The only thing is that you **MUST** fulfill the input/output format described by this interface, otherwise the test will not pass and you will get zero points for this feature.
And note that there may be not only one **Interface Description**, you should match all **Interface Description {n}**

### Test Description 1
Below is **Test Description 1**
```python
from agent_code.liger_kernel.ops.fused_add_rms_norm import LigerFusedAddRMSNormFunction
from agent_code.liger_kernel.transformers.functional import liger_fused_add_rms_norm
from agent_code.liger_kernel.transformers.fused_add_rms_norm import LigerFusedAddRMSNorm
```

### Interface Description 1
Below is **Interface Description 1** for file: src-liger_kernel-ops-fused_add_rms_norm.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
class LigerFusedAddRMSNormFunction(torch.autograd.Function):
    """
    
        Performs a fused operation that first adds a residual tensor to the hidden_states tensor (`X`), then applies RMSNorm (Root Mean Square Normalization) to the result using the weight tensor `W`, with optional offset and casting mode.
    
        This class implements the following sequence, commonly used in transformer decoder layers:
            1. hidden_states = residual + hidden_states
            2. residual = hidden_states (after addition)
            3. hidden_states = rmsnorm(hidden_states)
    
        Both the normalized hidden_states and the updated residual are returned as outputs.
    
        Some models use an 'offset' to shift the weight tensor `W` by a constant value. For example, Gemma
        uses an offset of 1.0, so the computation becomes `(X / RMS(X)) * (W + 1.0)` instead of the usual
        `(X / RMS(X)) * W`. You can pass the offset value as an argument to the forward function.
    
        In addition, different models cast their inputs at different places during RMSNorm computation. For
        example, Gemma casts everything to fp32 before starting the computation, while Llama casts only the
        inverse RMS to fp32. You can specify the casting mode using the `casting_mode` argument. We currently
        support the following casting modes (they match HuggingFace Transformers' implementations):
        - 'llama': matches the Llama implementation, where only the inverse RMS is computed on fp32.
        - 'gemma': matches the Gemma implementation, where everything is cast to fp32, then computed, then cast back to the original dtype.
        - 'none': no casting is done. The computation is done in the original dtype. This saves memory and is slightly faster, but has more error w.r.t. the original implementation.
    
        The `in_place` option determines whether to modify dY in-place to store dX. This defaults to `True` to save memory.
        
    """

    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        X,
        R,
        W,
        eps,
        offset = 0.0,
        casting_mode = 'llama',
        in_place = False
    ):
        """

                X: (B, T, H) or (BxT, H)
                W: (H,)

        """
        <your code>

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY, dS_out):
        """

                Y: (B, T, H) or (BxT, H)

        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.

### Interface Description 2
Below is **Interface Description 2** for file: src-liger_kernel-transformers-functional.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
def liger_fused_add_rms_norm(
    X,
    R,
    W,
    eps,
    offset: float = 0.0,
    casting_mode: str = 'llama',
    in_place: bool = True
):
    """
    Applies fused addition and RMS normalization in a single optimized operation.
    
    This function performs element-wise addition of input tensor X and residual tensor R, 
    followed by RMS (Root Mean Square) normalization. The fusion of these operations 
    provides better memory efficiency and computational performance compared to separate 
    addition and normalization steps.
    
    Args:
        X: Input tensor to be normalized after addition with residual.
        R: Residual tensor to be added to the input tensor X. Must be broadcastable with X.
        W: Weight tensor for scaling the normalized output. Should have the same shape as 
           the last dimension of X or be broadcastable.
        eps: Small epsilon value added to the denominator for numerical stability during 
             RMS normalization. Prevents division by zero.
        offset (float, optional): Offset value applied during normalization computation. 
                                 Defaults to 0.0.
        casting_mode (str, optional): Specifies the precision casting behavior during 
                                     computation. Supported modes include 'llama' and others 
                                     depending on implementation. Defaults to 'llama'.
        in_place (bool, optional): If True, performs the operation in-place to save memory. 
                                  If False, creates a new tensor for the result. 
                                  Defaults to True.
    
    Returns:
        Tensor: The result of fused addition and RMS normalization. If in_place is True, 
                the input tensor X is modified and returned. Otherwise, a new tensor with 
                the same shape as X is returned.
    
    Notes:
        - The RMS normalization is computed as: (X + R) / sqrt(mean((X + R)^2) + eps) * W
        - This fused operation is particularly beneficial in transformer architectures 
          where residual connections are followed by normalization layers
        - The casting_mode parameter affects numerical precision and should be chosen 
          based on the specific model requirements
        - When in_place=True, the original input tensor X will be modified
    """
    <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.

### Interface Description 3
Below is **Interface Description 3** for file: src-liger_kernel-transformers-fused_add_rms_norm.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
class LigerFusedAddRMSNorm(nn.Module):
    """
    A PyTorch neural network module that performs fused addition and RMS (Root Mean Square) normalization operations in a single, optimized kernel.
    
    This class combines residual connection addition with RMS normalization, which is commonly used in transformer architectures like LLaMA. The fused implementation provides better memory efficiency and computational performance compared to separate operations.
    
    Attributes:
        weight (nn.Parameter): Learnable scaling parameter of shape (hidden_size,). Used to scale the normalized output.
        variance_epsilon (float): Small constant added to variance for numerical stability during normalization.
        offset (float): Offset value applied during the normalization computation.
        casting_mode (str): Mode for type casting operations, typically set to 'llama' for LLaMA-style models.
        in_place (bool): Whether to perform operations in-place to save memory.
    
    Methods:
        __init__(hidden_size, eps=1e-6, offset=0.0, casting_mode='llama', init_fn='ones', in_place=False):
            Initializes the fused add-RMSNorm layer with specified parameters.
            
            Args:
                hidden_size (int): Dimension of the hidden states
                eps (float): Epsilon value for numerical stability
                offset (float): Offset for normalization computation
                casting_mode (str): Type casting mode ('llama' for LLaMA models)
                init_fn (str): Weight initialization function ('ones' or 'zeros')
                in_place (bool): Whether to use in-place operations
        
        forward(hidden_states, residual):
            Performs the fused addition and RMS normalization operation.
            
            Args:
                hidden_states (torch.Tensor): Input tensor to be normalized
                residual (torch.Tensor): Residual tensor to be added before normalization
                
            Returns:
                torch.Tensor: Normalized output after adding residual and applying RMS normalization
        
        extra_repr():
            Returns a string representation of the module's key parameters for debugging.
    
    Usage Example:
        ```python
        # Initialize the layer
        hidden_size = 768
        norm_layer = LigerFusedAddRMSNorm(hidden_size, eps=1e-5)
        
        # Forward pass
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        residual = torch.randn(batch_size, seq_len, hidden_size)
        output = norm_layer(hidden_states, residual)
        ```
    
    Note:
        This implementation is optimized for transformer models and provides significant performance benefits over separate addition and normalization operations, especially for large models and long sequences.
    """

    def __init__(
        self,
        hidden_size,
        eps = 1e-06,
        offset = 0.0,
        casting_mode = 'llama',
        init_fn = 'ones',
        in_place = False
    ):
        """
        Initialize a Liger Fused Add RMS Normalization layer.

        This layer performs a fused addition and RMS (Root Mean Square) normalization operation,
        which combines residual connection addition with RMS normalization for improved efficiency.
        RMS normalization is a variant of layer normalization that normalizes using the root mean
        square of the input values.

        Parameters:
            hidden_size (int): The size of the hidden dimension/feature dimension of the input tensors.
                              This determines the size of the learnable weight parameter.
            eps (float, optional): A small epsilon value added to the denominator for numerical 
                                  stability during normalization. Defaults to 1e-6.
            offset (float, optional): An offset value applied during the normalization computation.
                                     Defaults to 0.0.
            casting_mode (str, optional): The casting mode for type conversions during computation.
                                         Currently supports 'llama' mode. Defaults to 'llama'.
            init_fn (str, optional): Initialization function for the weight parameter. Must be either
                                    'ones' (initialize weights to 1.0) or 'zeros' (initialize weights 
                                    to 0.0). Defaults to 'ones'.
            in_place (bool, optional): Whether to perform the operation in-place to save memory.
                                      When True, input tensors may be modified directly. 
                                      Defaults to False.

        Raises:
            AssertionError: If init_fn is not 'ones' or 'zeros'.

        Notes:
            - The weight parameter is a learnable parameter of shape (hidden_size,) that scales
              the normalized output.
            - This fused operation is more memory and computationally efficient than performing
              addition and RMS normalization separately.
            - The layer expects two input tensors during forward pass: hidden_states and residual.
        """
        <your code>

    def forward(self, hidden_states, residual):
        """
        Performs fused addition and RMS normalization on the input tensors.

        This forward method applies a fused operation that combines element-wise addition of hidden_states 
        and residual tensors, followed by Root Mean Square (RMS) normalization. The operation is optimized 
        for performance by fusing these two common neural network operations into a single kernel call.

        Parameters:
            hidden_states (torch.Tensor): The primary input tensor to be normalized. This tensor contains
                the hidden states from the current layer that need to be combined with the residual
                connection and then normalized.
            residual (torch.Tensor): The residual tensor to be added to hidden_states. This typically
                represents the skip connection from a previous layer. Must have the same shape as
                hidden_states or be broadcastable to it.

        Returns:
            torch.Tensor: The output tensor after performing fused addition and RMS normalization.
                The tensor has the same shape as the input hidden_states and contains the normalized
                result of (hidden_states + residual).

        Notes:
            - The normalization uses the weight parameter initialized during module construction
            - The variance epsilon, offset, casting mode, and in-place operation settings are
              determined by the module's configuration set during initialization
            - This fused operation is more memory and computationally efficient than performing
              addition and RMS normalization separately
            - The actual computation is delegated to LigerFusedAddRMSNormFunction.apply() which
              implements the optimized kernel
        """
        <your code>

    def extra_repr(self):
        """
        Return a string representation of the extra parameters of this module.

        This method provides a human-readable string containing the key configuration
        parameters of the LigerFusedAddRMSNorm module, which is useful for debugging
        and logging purposes.

        Returns:
            str: A formatted string containing the weight tensor shape, epsilon value,
                 offset value, and in-place operation flag. The format is:
                 "(shape_tuple), eps=epsilon_value, offset=offset_value, in_place=in_place_flag"

        Notes:
            This method is automatically called by PyTorch when printing the module
            or converting it to a string representation. It extends the default
            module representation with module-specific parameters to provide more
            detailed information about the layer's configuration.
        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.