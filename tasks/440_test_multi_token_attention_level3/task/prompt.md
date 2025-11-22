## Task
**Task: Implement Multi-Token Attention Module**

**Core Functionality:**
Develop a neural network module that applies multi-token attention mechanism using masked convolution operations on attention scores.

**Main Features & Requirements:**
- Apply sequential operations: negative infinity masking → softmax normalization → 2D convolution → zero masking
- Support configurable convolution parameters (channels, kernel size, stride, padding, dilation, groups)
- Provide both class-based and functional interfaces
- Handle optional bias and sparse computation modes
- Implement proper parameter initialization (Kaiming uniform for weights, zeros for bias)

**Key Challenges:**
- Efficiently implement the masking operations before and after convolution
- Ensure gradient flow through custom autograd functions
- Handle 4D tensor operations with proper shape transformations
- Maintain numerical stability during softmax computation on masked scores
- Optimize performance for both dense and sparse tensor operations

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
from agent_code.liger_kernel.transformers.functional import liger_multi_token_attention
```
This means that we will test one function/class: liger_multi_token_attention.
And the defination and implementation of class liger_multi_token_attention should be in `/testbed/agent_code/liger_kernel/transformers/functional.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
def liger_multi_token_attention(
    scores,
    weight,
    bias = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    sparse: bool = False
):
    """
    
        Functional interface for multi-token attention.
    
        Args:
            scores: Input tensor of shape (B, C_in, L, L)
            weight: Convolution weight tensor of shape (C_out, C_in // groups, K, K)
            bias: Optional bias tensor of shape (C_out,)
            stride: Stride for the convolution (default: 1)
            padding: Padding for the convolution (default: 0)
            dilation: Dilation factor for the convolution (default: 1)
            groups: Number of groups for the convolution (default: 1)
            sparse: Specifies if input tensors are expected to be sparse (default: False)
        Returns:
            Output tensor after applying multi-token attention.
        
    """
    <your code>
...
```

In order to implement this functionality, some additional libraries etc. are often required, I don't restrict you to any libraries, you need to think about what dependencies you might need and fetch and install and call them yourself. The only thing is that you **MUST** fulfill the input/output format described by this interface, otherwise the test will not pass and you will get zero points for this feature.
And note that there may be not only one **Interface Description**, you should match all **Interface Description {n}**

### Test Description 1
Below is **Test Description 1**
```python
from agent_code.liger_kernel.transformers.functional import liger_multi_token_attention
from agent_code.liger_kernel.transformers.multi_token_attention import LigerMultiTokenAttention
```

### Interface Description 1
Below is **Interface Description 1** for file: src-liger_kernel-transformers-multi_token_attention.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
class LigerMultiTokenAttention(nn.Module):
    """
    
        Multi-Token Attention:
            out = mask_{0}(conv2d(softmax(mask_{-\inf}(scores))))
    
        Reference: https://arxiv.org/pdf/2504.00927
        
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        sparse: bool = False
    ):
        """
        Initialize a Multi-Token Attention layer.

        This layer implements the Multi-Token Attention mechanism as described in the reference paper,
        which applies a convolution operation on softmax-normalized attention scores with masking:
        out = mask_{0}(conv2d(softmax(mask_{-∞}(scores))))

        Parameters:
            in_channels (int): Number of channels in the input tensor. Must be divisible by groups.
            out_channels (int): Number of channels produced by the convolution operation.
            kernel_size (int): Size of the convolving kernel. Will be converted to a 2D kernel size pair.
            stride (int, optional): Stride of the convolution operation. Defaults to 1.
                Will be converted to a 2D stride pair.
            padding (int, optional): Padding added to all four sides of the input. Defaults to 0.
                Will be converted to a 2D padding pair.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
                Will be converted to a 2D dilation pair.
            groups (int, optional): Number of blocked connections from input channels to output channels.
                Defaults to 1. in_channels and out_channels must be divisible by groups.
            bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
            sparse (bool, optional): If True, enables sparse computation mode. Defaults to False.

        Raises:
            ValueError: If in_channels is not divisible by groups.

        Notes:
            - The weight parameter is initialized using Kaiming uniform initialization.
            - The bias parameter (if enabled) is initialized to zeros.
            - All spatial parameters (kernel_size, stride, padding, dilation) are converted to 2D pairs
              using torch.nn.modules.utils._pair for compatibility with 2D convolution operations.
            - This implementation is based on the Multi-Token Attention paper: https://arxiv.org/pdf/2504.00927
        """
        <your code>

    def reset_parameters(self):
        """
        Reset the parameters of the Multi-Token Attention layer to their initial values.

        This method initializes the weight and bias parameters using standard initialization
        schemes commonly used in neural networks. The weight parameter is initialized using
        Kaiming uniform initialization, while the bias parameter (if present) is initialized
        to zeros.

        Parameters:
            None

        Returns:
            None

        Notes:
            - The weight parameter is initialized using `nn.init.kaiming_uniform_` with 
              `a=math.sqrt(5)`, which is suitable for layers with ReLU-like activations
            - The bias parameter is initialized to zeros using `nn.init.zeros_` if bias
              is enabled during layer construction
            - This method is automatically called during layer initialization but can
              be called manually to reinitialize the parameters
            - The initialization follows PyTorch's standard practices for convolutional layers
        """
        <your code>

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Multi-Token Attention module.

        This method computes the multi-token attention using a custom autograd function
        that implements the attention mechanism. The underlying computation typically
        includes attention score processing, normalization, and convolution operations
        optimized for multi-token attention tasks.

        Parameters:
            scores (torch.Tensor): Input attention scores tensor. The exact shape depends
                                on the specific configuration but generally follows 
                                attention tensor conventions.

        Returns:
            torch.Tensor: Output tensor after applying multi-token attention. The output shape
                        is determined by the input shape and layer configuration.

        Notes:
            - This method uses LigerMultiTokenAttentionFunction.apply() to perform the
            core computation with optimized kernel operations
            - Convolution parameters (stride, padding, dilation, groups) and sparse setting
            are passed to the underlying function to configure the attention behavior
            - Gradients are preserved for backpropagation through the custom autograd function
        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.

### Interface Description 2
Below is **Interface Description 2** for file: src-liger_kernel-transformers-functional.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
def liger_multi_token_attention(
    scores,
    weight,
    bias = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    sparse: bool = False
):
    """
    
        Functional interface for multi-token attention.
    
        Args:
            scores: Input tensor of shape (B, C_in, L, L)
            weight: Convolution weight tensor of shape (C_out, C_in // groups, K, K)
            bias: Optional bias tensor of shape (C_out,)
            stride: Stride for the convolution (default: 1)
            padding: Padding for the convolution (default: 0)
            dilation: Dilation factor for the convolution (default: 1)
            groups: Number of groups for the convolution (default: 1)
            sparse: Specifies if input tensors are expected to be sparse (default: False)
        Returns:
            Output tensor after applying multi-token attention.
        
    """
    <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.