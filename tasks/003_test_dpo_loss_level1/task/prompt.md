## Task
**Task: Implement Fused Linear Layer with Direct Preference Optimization (DPO) Loss**

**Core Functionality:**
Create a memory-efficient PyTorch autograd function that combines linear layer computation with DPO loss calculation for preference-based language model training.

**Main Features & Requirements:**
- Fuse linear transformation and DPO loss computation in a single operation
- Support multiple loss variants (sigmoid, APO, SPPO, NCA) 
- Handle chosen/rejected sequence pairs for preference learning
- Integrate optional reference model comparisons
- Implement chunked processing for memory efficiency
- Provide proper gradient computation for backpropagation

**Key Challenges:**
- Memory optimization through operation fusion
- Correct gradient flow in custom autograd functions
- Efficient handling of large sequence batches
- Supporting multiple DPO loss formulations
- Managing optional reference model integration
- Proper masking of ignored tokens during loss computation

The task focuses on creating an efficient implementation of DPO training that reduces memory overhead while maintaining flexibility across different preference optimization approaches.

**NOTE**: 
- This test comes from the `liger-kernel` library, and we have given you the content of this code repository under `/testbed/`, and you need to complete based on this code repository and supplement the files we specify. Remember, all your changes must be in this codebase, and changes that are not in this codebase will not be discovered and tested by us.
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
from liger_kernel.chunked_loss import LigerFusedLinearDPOLoss
```
This means that we will test one function/class: LigerFusedLinearDPOLoss.
And the defination and implementation of class LigerFusedLinearDPOLoss should be in `/testbed/src/liger_kernel/chunked_loss/dpo_loss.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
class LigerFusedLinearDPOLoss(torch.nn.Module):
    """
    
        Fused linear layer with DPO loss.
        
    """

    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        compute_nll_loss: bool = False,
        compiled: bool = True,
        use_ref_model: bool = True,
        average_log_prob: bool = False,
        chunk_size: int = 1,
        loss_type: str = 'sigmoid'
    ):
        """

                Args:
                    ignore_index (int): Index to ignore in the loss.
                    beta (float): Weight for the odds ratio loss.
                    compute_nll_loss (bool): Whether to compute the NLL loss.
                    compiled (bool): Whether to use the torch compiled kernel.
                    use_ref_model (bool): Whether to use a reference model for the DPO loss.
                    average_log_prob (bool): Whether to average the log probability per non-masked token.
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
from liger_kernel.chunked_loss import LigerFusedLinearDPOLoss
from liger_kernel.chunked_loss.dpo_loss import LigerFusedLinearDPOFunction
```

### Interface Description 1
Below is **Interface Description 1** for file: src-liger_kernel-chunked_loss-dpo_loss.py

This file contains 2 top-level interface(s) that need to be implemented.

```python
class LigerFusedLinearDPOFunction(LigerFusedLinearPreferenceBase):
    """
    A PyTorch autograd function that implements fused linear layer computation with Direct Preference Optimization (DPO) loss for efficient preference-based model training.
    
    This class extends LigerFusedLinearPreferenceBase to provide a memory-efficient implementation of DPO loss computation by fusing the linear layer forward pass with the preference loss calculation. DPO is a method for training language models to align with human preferences without requiring reinforcement learning.
    
    Main Methods:
        preference_loss_fn: Computes the DPO loss using various loss formulations including sigmoid (standard DPO), 
                           APO variants (apo_zero, apo_down), SPPO hard variant (sppo_hard), and NCA pair loss (nca_pair).
                           Implements the core DPO formula: -E[log_sigmoid(β * (log_ratio_chosen - log_ratio_rejected))].
        
        forward: Performs the fused forward pass combining linear layer computation with DPO loss calculation.
                Handles both policy and reference model computations, supports chunked processing for memory efficiency,
                and provides options for NLL loss computation and log probability averaging.
        
        backward: Computes gradients for the fused operation, returning gradients only for the first 4 input tensors
                 (input, weight, target, bias) while ignoring gradients for configuration parameters.
    
    Key Features:
        - Memory-efficient fused computation of linear layer + DPO loss
        - Support for multiple DPO loss variants (sigmoid, APO, SPPO, NCA)
        - Optional reference model integration for computing log probability ratios
        - Chunked processing capability for handling large sequences
        - Configurable ignore index for masked token handling
        - Optional NLL loss computation alongside preference loss
    
    Usage Example:
        ```python
        # Create input tensors
        input_tensor = torch.randn(batch_size * seq_len, hidden_size)
        weight = torch.randn(vocab_size, hidden_size, requires_grad=True)
        target = torch.randint(0, vocab_size, (batch_size * seq_len,))
        
        # Apply fused DPO function
        loss = LigerFusedLinearDPOFunction.apply(
            input_tensor, weight, target, 
            bias=None, beta=0.1, loss_type='sigmoid'
        )
        ```
    
    The class is designed for training scenarios where chosen and rejected sequences are provided in pairs,
    with the target tensor containing both chosen and rejected token sequences concatenated together.
    """

    @staticmethod
    def preference_loss_fn(
        chosen_logps,
        rejected_logps,
        full_target,
        ref_chosen_logps = None,
        ref_rejected_logps = None,
        beta = 0.1,
        loss_type = 'sigmoid'
    ):
        """

                Paper: https://arxiv.org/pdf/2305.18290

                Formula:
                L_DPO = -E[ log_sigmoid( β * (log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x))) ) ]

                Where:
                - π(y|x): Policy (model) probability
                - π_ref(y|x): Reference model probability
                - y_w: Chosen sequence
                - y_l: Rejected sequence
                - β: Weight for the direct preference loss
                - E: Expected value over the dataset

                Args:
                    chosen_logps: Log probabilities of chosen tokens (batch_size,)
                    rejected_logps: Log probabilities of rejected tokens (batch_size,)
                    full_target: Non chunked full target tensor
                    ref_chosen_logps: Reference log probs of chosen tokens (batch_size,)
                    ref_rejected_logps: Reference log probs of rejected tokens (batch_size,)
                    beta: Weight for the direct preference loss

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
        ref_input = None,
        ref_weight = None,
        ref_bias = None,
        ignore_index = -100,
        beta = 0.1,
        compute_nll_loss = False,
        compiled = True,
        use_ref_model = True,
        average_log_prob = False,
        chunk_size = 1,
        loss_type = 'sigmoid'
    ):
        """

                Fused linear layer with DPO loss.
                Args:
                    _input (torch.Tensor): Input tensor. Shape: (batch_size * seq_len, hidden_size)
                    weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size)
                    target (torch.LongTensor): Target tensor. Shape: (batch_size * seq_len,)
                    bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,)
                    ref_input (torch.Tensor, optional): Reference model input tensor. Shape: (batch_size * seq_len, hidden_size)
                    ref_weight (torch.Tensor, optional): Reference model weight tensor. Shape: (vocab_size, hidden_size)
                    ref_bias (torch.Tensor, optional): Reference model bias tensor. Shape: (vocab_size,)
                    ignore_index (int): Index to ignore in loss computation
                    beta (float): Weight for the odds ratio loss
                    compute_nll_loss (bool): Whether to compute the NLL loss
                    compiled (bool): Whether to use torch compile
                    use_ref_model (bool): Whether to use a reference model
                    average_log_prob (bool): Whether to average the log probability per non-masked token
                    chunk_size (int): Size of chunks for processing.
                Returns:
                    torch.Tensor: Computed loss

        """
        <your code>

    @staticmethod
    def backward(ctx, *grad_output):
        """
        Computes gradients for the fused linear DPO (Direct Preference Optimization) function during backpropagation.

        This static method implements the backward pass for automatic differentiation in PyTorch's autograd system. It delegates the gradient computation to the parent class's backward method and then filters the results to match the expected number of parameters for the DPO function.

        Parameters:
            ctx: PyTorch autograd context object containing saved tensors and other information from the forward pass
            *grad_output: Variable-length argument list of gradient tensors flowing back from the loss function. Each tensor represents the gradient with respect to the corresponding output of the forward pass

        Returns:
            tuple: A tuple containing gradients with respect to the input parameters of the forward function:
                - Gradient w.r.t. input tensor (_input)
                - Gradient w.r.t. weight tensor (weight) 
                - Gradient w.r.t. target tensor (target)
                - Gradient w.r.t. bias tensor (bias)
                - None values for parameters that don't require gradients (ref_input, ref_weight, ref_bias, ignore_index, beta, compute_nll_loss, compiled, use_ref_model, average_log_prob, chunk_size, loss_type)

        Important Notes:
            - This method is part of PyTorch's Function interface for custom autograd operations
            - The method truncates the parent class gradients to the first 4 elements using [:4] slicing
            - Additional None values are returned to match the signature of the forward method parameters
            - The actual gradient computation logic is inherited from LigerFusedLinearPreferenceBase.backward()
            - This ensures proper gradient flow for DPO loss optimization while maintaining compatibility with PyTorch's autograd system
        """
        <your code>

class LigerFusedLinearDPOLoss(torch.nn.Module):
    """
    
        Fused linear layer with DPO loss.
        
    """

    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        compute_nll_loss: bool = False,
        compiled: bool = True,
        use_ref_model: bool = True,
        average_log_prob: bool = False,
        chunk_size: int = 1,
        loss_type: str = 'sigmoid'
    ):
        """

                Args:
                    ignore_index (int): Index to ignore in the loss.
                    beta (float): Weight for the odds ratio loss.
                    compute_nll_loss (bool): Whether to compute the NLL loss.
                    compiled (bool): Whether to use the torch compiled kernel.
                    use_ref_model (bool): Whether to use a reference model for the DPO loss.
                    average_log_prob (bool): Whether to average the log probability per non-masked token.
                    chunk_size (int): Size of chunks for processing.

        """
        <your code>

    def forward(
        self,
        lin_weight,
        _input,
        target,
        bias = None,
        ref_input = None,
        ref_weight = None,
        ref_bias = None
    ):
        """
        Computes the fused linear layer forward pass with Direct Preference Optimization (DPO) loss.

        This method performs a forward pass through a linear layer while simultaneously computing
        the DPO loss, which is used for training language models to align with human preferences
        by comparing chosen and rejected response pairs.

        Parameters:
            lin_weight (torch.Tensor): Linear layer weight matrix with shape (vocab_size, hidden_size).
                Contains the learnable parameters for the linear transformation.
            _input (torch.Tensor): Input tensor with shape (batch_size * seq_len, hidden_size).
                Represents the hidden states from the model that will be transformed to logits.
            target (torch.LongTensor): Target token indices with shape (batch_size * seq_len,).
                Contains the ground truth token IDs for both chosen and rejected sequences.
            bias (torch.Tensor, optional): Bias vector with shape (vocab_size,). Defaults to None.
                Optional bias term to be added to the linear transformation.
            ref_input (torch.Tensor, optional): Reference model input tensor with shape 
                (batch_size * seq_len, hidden_size). Defaults to None. Used when comparing
                against a reference model for DPO loss computation.
            ref_weight (torch.Tensor, optional): Reference model weight matrix with shape
                (vocab_size, hidden_size). Defaults to None. Weight parameters of the reference model.
            ref_bias (torch.Tensor, optional): Reference model bias vector with shape (vocab_size,).
                Defaults to None. Bias parameters of the reference model.

        Returns:
                torch.Tensor: Normalized DPO loss scalar tensor. The loss is averaged over the number of 
                    "chosen-rejected" sequence pairs, and used for model parameter optimization via backpropagation.

        Notes:
            - This method applies the configured DPO loss function using the instance parameters
              (ignore_index, beta, compute_nll_loss, compiled, use_ref_model, average_log_prob,
              chunk_size, loss_type) that were set during initialization.
            - The input tensors are expected to contain both chosen and rejected sequences,
              typically concatenated along the batch dimension.
            - The method leverages the underlying LigerFusedLinearDPOFunction for efficient
              computation of the fused linear transformation and loss calculation.
            - Reference model parameters (ref_input, ref_weight, ref_bias) are only used
              when use_ref_model is True during initialization.
        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.