## Task
**Task: Implement Memory-Efficient Preference-Based Language Model Training**

**Core Functionality:**
Develop a fused linear layer with SimPO (Simple Preference Optimization) loss for training language models on preference data without requiring reference models.

**Main Features & Requirements:**
- Combine linear transformation and preference loss computation in a single memory-efficient operation
- Implement SimPO algorithm using length-normalized log probabilities with configurable margin and smoothing
- Support both forward and backward passes with proper gradient computation
- Handle chunked processing for large sequences
- Provide configurable hyperparameters (beta, gamma, label smoothing)

**Key Challenges:**
- Optimize memory usage by fusing operations rather than computing them separately
- Correctly implement the SimPO loss formula with sigmoid-based preference comparison
- Ensure proper gradient flow through custom autograd functions
- Balance computational efficiency with numerical stability for large-scale training

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
from agent_code.liger_kernel.chunked_loss import LigerFusedLinearSimPOLoss
```
This means that we will test one function/class: LigerFusedLinearSimPOLoss.
And the defination and implementation of class LigerFusedLinearSimPOLoss should be in `/testbed/agent_code/liger_kernel/chunked_loss.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
class LigerFusedLinearSimPOLoss(torch.nn.Module):
    """
    
        Fused linear layer with SimPO loss.
        
    """

    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        alpha: float = 1.0,
        label_smoothing: float = 0.0,
        compute_nll_loss: bool = True,
        compiled: bool = True,
        gamma: float = 0.5,
        chunk_size: int = 1
    ):
        """

                Args:
                    ignore_index (int): Index to ignore in the loss.
                    beta (float): Weight for the odds ratio loss.
                    alpha (float): Weight for the alpha parameter.
                    label_smoothing (float): Label smoothing factor.
                    compute_nll_loss (bool): Whether to compute the NLL loss.
                    compiled (bool): Whether to use the torch compiled kernel.
                    gamma (float): Weight for the gamma parameter.
                    chunk_size (int): Size of chunks for processing.

        """
        <your code>
...
```

In order to implement this functionality, some additional libraries etc. are often required, I don't restrict you to any libraries, you need to think about what dependencies you might need and fetch and install and call them yourself. The only thing is that you **MUST** fulfill the input/output format described by this interface, otherwise the test will not pass and you will get zero points for this feature.
And note that there may be not only one **Interface Description**, you should match all **Interface Description {n}**

### Test Description 1
Below is **Test Description 1**
```python
from agent_code.liger_kernel.chunked_loss import LigerFusedLinearSimPOLoss
from agent_code.liger_kernel.chunked_loss.simpo_loss import LigerFusedLinearSimPOFunction
```

### Interface Description 1
Below is **Interface Description 1** for file: src-liger_kernel-chunked_loss-simpo_loss.py

This file contains 2 top-level interface(s) that need to be implemented.

```python
class LigerFusedLinearSimPOFunction(LigerFusedLinearPreferenceBase):
    """
    A PyTorch autograd function that implements the SimPO (Simple Preference Optimization) loss combined with a fused linear layer for efficient preference-based model training.
    
    This class extends LigerFusedLinearPreferenceBase to provide a memory-efficient implementation of SimPO loss, which is used for training language models on preference data without requiring a reference model. The SimPO loss compares chosen and rejected sequences using their average log probabilities and applies a sigmoid-based loss function with configurable margin and smoothing parameters.
    
    **Key Features:**
    - Fuses linear layer computation with SimPO loss calculation for improved memory efficiency
    - Implements the SimPO algorithm from "SimPO: Simple Preference Optimization with a Reference-Free Reward" (https://arxiv.org/pdf/2405.14734)
    - Supports chunked processing for handling large sequences
    - Provides both forward and backward pass implementations with gradient computation
    
    **Main Methods:**
    
    - `preference_loss_fn(chosen_logps, rejected_logps, full_target, beta, gamma, label_smoothing)`: 
      Computes the core SimPO loss using the formula: L_SimPO(π_θ) = -E [log σ(β/|y_w| log π_θ(y_w|x) - β/|y_l| log π_θ(y_l|x) - γ)].
      Returns the loss value along with chosen and rejected rewards.
    
    - `forward(cls, ctx, _input, weight, target, bias, ignore_index, beta, alpha, label_smoothing, compute_nll_loss, compiled, gamma, chunk_size)`:
      Performs the forward pass by calling the parent class implementation with SimPO-specific parameters.
      Handles the fused linear transformation and loss computation in a single operation.
    
    - `backward(ctx, *grad_output)`:
      Computes gradients for the input, weight, target, and bias tensors during backpropagation.
      Returns gradients while setting None for non-differentiable parameters.
    
    **Usage Example:**
    ```python
    # Typically used through LigerFusedLinearSimPOLoss module
    simpo_loss = LigerFusedLinearSimPOLoss(beta=0.1, gamma=0.5, label_smoothing=0.0)
    loss = simpo_loss(linear_weight, input_tensor, target_tensor, bias)
    
    # Or directly as an autograd function
    loss = LigerFusedLinearSimPOFunction.apply(
        input_tensor, weight, target, bias, -100, 0.1, 1.0, 0.0, False, True, 0.5, 1
    )
    ```
    
    **Parameters:**
    - `beta`: Controls the strength of the preference signal (default: 0.1)
    - `gamma`: Margin term that affects the decision boundary (default: 0.5)  
    - `label_smoothing`: Smoothing factor for the loss function (default: 0.0)
    - `chunk_size`: Size of chunks for memory-efficient processing (default: 1)
    """

    @staticmethod
    def preference_loss_fn(
        chosen_logps,
        rejected_logps,
        full_target,
        beta = 0.1,
        gamma = 0.5,
        label_smoothing = 0.0
    ):
        """

                Paper: https://arxiv.org/pdf/2405.14734

                Formula:
                L_SimPO(π_θ) = -E [log σ(β/|y_w| log π_θ(y_w|x) - β/|y_l| log π_θ(y_l|x) - γ)]

                Where:
                - π_θ(y|x): Policy (model) probability
                - y_w: Chosen sequence
                - y_l: Rejected sequence
                - |y_w|, |y_l|: Sequence lengths
                - σ: Sigmoid function
                - β: beta weight
                - γ: gemma margin term

                Args:
                    chosen_logps (torch.Tensor): Avg log probabilities of chosen tokens. Shape: (batch_size,).
                    rejected_logps (torch.Tensor): Avg log probabilities of rejected tokens. Shape: (batch_size,).
                    full_target: Non chunked full target tensor
                    beta (float): beta weight
                    gamma (float): gemma margin term
                    label_smoothing (float): Label smoothing factor, will reduce to Equation above when label_smoothing -> 0.

        """
        <your code>

    @classmethod
    def forward(
        cls,
        ctx,
        _input,
        weight,
        target,
        bias = None,
        ignore_index = -100,
        beta = 0.1,
        alpha = 1.0,
        label_smoothing = 0.0,
        compute_nll_loss = False,
        compiled = True,
        gamma = 0.5,
        chunk_size = 1
    ):
        """

                Fused linear layer with SimPO loss.
                Args:
                    _input (torch.Tensor): Input tensor. Shape: (batch_size * seq_len, hidden_size)
                    weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size)
                    target (torch.LongTensor): Target tensor. Shape: (batch_size * seq_len,)
                    bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,)
                    ignore_index (int): Index to ignore in loss computation
                    beta (float): Weight for the odds ratio loss
                    alpha (float): Weight for the alpha parameter
                    label_smoothing (float): Label smoothing factor
                    compute_nll_loss (bool): Whether to compute the NLL loss
                    compiled (bool): Whether to use torch compile
                    gamma (float): Weight for the gamma parameter
                    chunk_size (int): Size of chunks for processing
                Returns:
                    torch.Tensor: Computed loss

        """
        <your code>

    @staticmethod
    def backward(ctx, *grad_output):
        """
        Performs the backward pass for the LigerFusedLinearSimPOFunction.

        This static method computes gradients with respect to the input parameters during backpropagation
        for the fused linear layer with SimPO (Simple Preference Optimization) loss. It delegates the
        gradient computation to the parent class and then filters and pads the results appropriately.

        Args:
            ctx: The context object that was saved during the forward pass, containing
                 intermediate values and tensors needed for gradient computation.
            *grad_output: Variable length argument list containing the gradients of the loss
                          with respect to the outputs of the forward pass. Typically contains
                          the gradient tensor from the subsequent layer in the computation graph.

        Returns:
            tuple: A tuple containing gradients with respect to all input parameters of the forward
                   method, in the same order as the forward method signature:
                   - Gradient w.r.t. _input (torch.Tensor or None)
                   - Gradient w.r.t. weight (torch.Tensor or None) 
                   - Gradient w.r.t. target (torch.Tensor or None)
                   - Gradient w.r.t. bias (torch.Tensor or None)
                   - None for ignore_index (no gradient needed)
                   - None for beta (no gradient needed)
                   - None for alpha (no gradient needed)
                   - None for label_smoothing (no gradient needed)
                   - None for compute_nll_loss (no gradient needed)
                   - None for compiled (no gradient needed)
                   - None for gamma (no gradient needed)
                   - None for chunk_size (no gradient needed)

        Notes:
            - This method is automatically called by PyTorch's autograd system during backpropagation
            - The method only computes gradients for the first 4 parameters (input tensors) and returns
              None for all hyperparameters and boolean flags as they don't require gradients
            - The actual gradient computation is handled by the parent class LigerFusedLinearPreferenceBase
        """
        <your code>

class LigerFusedLinearSimPOLoss(torch.nn.Module):
    """
    
        Fused linear layer with SimPO loss.
        
    """

    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        alpha: float = 1.0,
        label_smoothing: float = 0.0,
        compute_nll_loss: bool = True,
        compiled: bool = True,
        gamma: float = 0.5,
        chunk_size: int = 1
    ):
        """

                Args:
                    ignore_index (int): Index to ignore in the loss.
                    beta (float): Weight for the odds ratio loss.
                    alpha (float): Weight for the alpha parameter.
                    label_smoothing (float): Label smoothing factor.
                    compute_nll_loss (bool): Whether to compute the NLL loss.
                    compiled (bool): Whether to use the torch compiled kernel.
                    gamma (float): Weight for the gamma parameter.
                    chunk_size (int): Size of chunks for processing.

        """
        <your code>

    def forward(
        self,
        lin_weight,
        _input,
        target,
        bias = None
    ):
        """
        Computes the fused linear transformation with SimPO (Simple Preference Optimization) loss.

        This method performs a forward pass through a linear layer followed by SimPO loss computation,
        which is designed for preference-based training without requiring a reference model. The SimPO
        loss uses length-normalized log probabilities and includes a margin term for improved training stability.

        Parameters:
            lin_weight (torch.Tensor): The weight matrix for the linear transformation.
                Shape: (vocab_size, hidden_size)
            _input (torch.Tensor): Input tensor containing token embeddings.
                Shape: (batch_size * seq_len, hidden_size)
            target (torch.LongTensor): Target token indices for loss computation.
                Shape: (batch_size * seq_len,)
            bias (torch.Tensor, optional): Bias vector for the linear transformation.
                Shape: (vocab_size,). Defaults to None.

        Returns:
            torch.Tensor: The computed SimPO loss value as a scalar tensor.

        Notes:
            - This implementation follows the SimPO paper (https://arxiv.org/pdf/2405.14734)
            - The loss function uses length-normalized log probabilities: L_SimPO(π_θ) = -E [log σ(β/|y_w| log π_θ(y_w|x) - β/|y_l| log π_θ(y_l|x) - γ)]
            - The method leverages the class instance parameters (beta, gamma, alpha, etc.) set during initialization
            - Input tensors are expected to contain both chosen and rejected sequences in the batch
            - The computation is performed using a fused kernel for improved efficiency
            - Gradients are automatically computed for backpropagation through the custom autograd function
        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.