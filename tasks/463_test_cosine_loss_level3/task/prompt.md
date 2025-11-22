## Task
**Task: Implement Fused Linear Layer with Cosine Similarity Knowledge Distillation**

**Core Functionality:**
Create an efficient PyTorch module that combines linear transformations with cosine similarity-based knowledge distillation, enabling a student model to learn from a teacher model by matching both ground truth labels and directional patterns in the output space.

**Main Features & Requirements:**
- Fuse linear layer computation with distillation loss calculation for performance optimization
- Implement cosine similarity loss between normalized student and teacher logits
- Support configurable weighting between hard loss (cross-entropy) and soft loss (cosine similarity)
- Handle memory-efficient processing through chunking for large vocabularies
- Provide proper gradient computation for backpropagation training

**Key Challenges:**
- Balance computational efficiency with memory usage in fused operations
- Implement custom autograd functions with correct gradient flow
- Handle numerical stability in cosine similarity calculations with normalization
- Manage hyperparameter tuning for loss weighting and temperature scaling
- Ensure compatibility with chunked processing for scalability

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
from agent_code.liger_kernel.chunked_loss.cosine_similarity_loss import LigerFusedLinearCosineSimilarityFunction
```
This means that we will test one function/class: LigerFusedLinearCosineSimilarityFunction.
And the defination and implementation of class LigerFusedLinearCosineSimilarityFunction should be in `/testbed/agent_code/liger_kernel/chunked_loss/cosine_similarity_loss.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
class LigerFusedLinearCosineSimilarityFunction(LigerFusedLinearDistillationBase):
    """
    A PyTorch autograd function that implements fused linear layer computation with cosine similarity-based knowledge distillation loss.
    
    This class extends LigerFusedLinearDistillationBase to provide an efficient implementation of knowledge distillation using cosine similarity loss between student and teacher model outputs. It combines linear transformations with distillation loss computation in a single fused operation for improved performance.
    
    Main Methods:
        distillation_loss_fn(student_logits, teacher_logits, beta=1.0):
            Computes the cosine similarity loss between normalized student and teacher logits.
            The loss is calculated as beta * (1 - cosine_similarity), where cosine similarity
            measures the alignment between the normalized logit vectors.
    
        forward(cls, ctx, student_input, student_weight, teacher_input, teacher_weight, 
                true_labels, student_bias, teacher_bias, **kwargs):
            Performs the forward pass by computing linear transformations for both student
            and teacher networks, then applies the combined hard loss (cross-entropy with
            true labels) and soft loss (cosine similarity distillation loss). Supports
            chunked processing for memory efficiency.
    
        backward(ctx, grad_output):
            Computes gradients with respect to the input tensors during backpropagation.
            Only propagates gradients for trainable parameters (student inputs/weights),
            while setting gradients to None for hyperparameters and teacher bias.
    
    Key Parameters:
        - beta: Controls the strength of the cosine similarity loss component
        - weight_hard_loss/weight_soft_loss: Balance between supervised and distillation losses
        - temperature: Temperature scaling for knowledge distillation
        - chunk_size: Memory optimization parameter for processing large tensors
        - ignore_index: Label index to ignore in loss computation
    
    Usage Example:
        ```python
        # Used internally by LigerFusedLinearCosineSimilarityLoss
        loss_fn = LigerFusedLinearCosineSimilarityLoss(
            weight_hard_loss=0.3,
            weight_soft_loss=0.7,
            beta=1.0,
            temperature=3.0
        )
        
        loss = loss_fn(
            student_input=student_hidden_states,
            student_weight=student_lm_head.weight,
            teacher_input=teacher_hidden_states,
            teacher_weight=teacher_lm_head.weight,
            true_labels=target_labels
        )
        ```
    """

    @staticmethod
    def distillation_loss_fn(student_logits, teacher_logits, beta = 1.0):
        """

                Compute Cosine loss (Cosine Similarity Loss).
                Args:
                    student_logits (torch.Tensor): Logits of student tokens. Shape: (batch_size * seq_len,).
                    teacher_logits (torch.Tensor): Logits of teacher tokens. Shape: (batch_size * seq_len,).
                    beta: Coefficient beta of generalized Cosine Similarity in the interval [0, 1]. Default: `1.0` (float): .
                Returns:
                    torch.Tensor: cosine similarity loss

        """
        <your code>
...
```

In order to implement this functionality, some additional libraries etc. are often required, I don't restrict you to any libraries, you need to think about what dependencies you might need and fetch and install and call them yourself. The only thing is that you **MUST** fulfill the input/output format described by this interface, otherwise the test will not pass and you will get zero points for this feature.
And note that there may be not only one **Interface Description**, you should match all **Interface Description {n}**

### Test Description 1
Below is **Test Description 1**
```python
from agent_code.liger_kernel.chunked_loss.cosine_similarity_loss import LigerFusedLinearCosineSimilarityFunction
from agent_code.liger_kernel.chunked_loss.cosine_similarity_loss import LigerFusedLinearCosineSimilarityLoss
```

### Interface Description 1
Below is **Interface Description 1** for file: src-liger_kernel-chunked_loss-cosine_similarity_loss.py

This file contains 2 top-level interface(s) that need to be implemented.

```python
class LigerFusedLinearCosineSimilarityFunction(LigerFusedLinearDistillationBase):
    """
    A PyTorch autograd function that implements fused linear layer computation with cosine similarity-based knowledge distillation loss.
    
    This class extends LigerFusedLinearDistillationBase to provide an efficient implementation of knowledge distillation using cosine similarity loss between student and teacher model outputs. It combines linear transformations with distillation loss computation in a single fused operation for improved performance.
    
    Main Methods:
        distillation_loss_fn(student_logits, teacher_logits, beta=1.0):
            Computes the cosine similarity loss between normalized student and teacher logits.
            The loss is calculated as beta * (1 - cosine_similarity), where cosine similarity
            measures the alignment between the normalized logit vectors.
    
        forward(cls, ctx, student_input, student_weight, teacher_input, teacher_weight, 
                true_labels, student_bias, teacher_bias, **kwargs):
            Performs the forward pass by computing linear transformations for both student
            and teacher networks, then applies the combined hard loss (cross-entropy with
            true labels) and soft loss (cosine similarity distillation loss). Supports
            chunked processing for memory efficiency.
    
        backward(ctx, grad_output):
            Computes gradients with respect to the input tensors during backpropagation.
            Only propagates gradients for trainable parameters (student inputs/weights),
            while setting gradients to None for hyperparameters and teacher bias.
    
    Key Parameters:
        - beta: Controls the strength of the cosine similarity loss component
        - weight_hard_loss/weight_soft_loss: Balance between supervised and distillation losses
        - temperature: Temperature scaling for knowledge distillation
        - chunk_size: Memory optimization parameter for processing large tensors
        - ignore_index: Label index to ignore in loss computation
    
    Usage Example:
        ```python
        # Used internally by LigerFusedLinearCosineSimilarityLoss
        loss_fn = LigerFusedLinearCosineSimilarityLoss(
            weight_hard_loss=0.3,
            weight_soft_loss=0.7,
            beta=1.0,
            temperature=3.0
        )
        
        loss = loss_fn(
            student_input=student_hidden_states,
            student_weight=student_lm_head.weight,
            teacher_input=teacher_hidden_states,
            teacher_weight=teacher_lm_head.weight,
            true_labels=target_labels
        )
        ```
    """

    @staticmethod
    def distillation_loss_fn(student_logits, teacher_logits, beta = 1.0):
        """

                Compute Cosine loss (Cosine Similarity Loss).
                Args:
                    student_logits (torch.Tensor): Logits of student tokens. Shape: (batch_size * seq_len,).
                    teacher_logits (torch.Tensor): Logits of teacher tokens. Shape: (batch_size * seq_len,).
                    beta: Coefficient beta of generalized Cosine Similarity in the interval [0, 1]. Default: `1.0` (float): .
                Returns:
                    torch.Tensor: cosine similarity loss

        """
        <your code>

    @classmethod
    def forward(
        cls,
        ctx,
        student_input: torch.Tensor,
        student_weight: torch.Tensor,
        teacher_input: torch.Tensor,
        teacher_weight: torch.Tensor,
        true_labels: torch.LongTensor,
        student_bias: torch.Tensor,
        teacher_bias: torch.Tensor,
        weight_hard_loss: float = 0.5,
        weight_soft_loss: float = 0.5,
        beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
        compiled: bool = True,
        chunk_size: int = 1024
    ):
        """
        Forward pass for the Liger Fused Linear Cosine Similarity distillation function.

        This classmethod implements the forward pass of a custom autograd function that combines
        linear transformations with cosine similarity-based knowledge distillation. It computes
        a combined loss consisting of both hard loss (cross-entropy with true labels) and soft
        loss (cosine similarity between student and teacher logits).

        Parameters:
            ctx: PyTorch autograd context for storing information needed in backward pass
            student_input (torch.Tensor): Input tensor for the student model, typically hidden states
                from a transformer layer with shape (batch_size, seq_len, hidden_dim)
            student_weight (torch.Tensor): Weight matrix for the student's linear layer with shape
                (vocab_size, hidden_dim)
            teacher_input (torch.Tensor): Input tensor for the teacher model with same shape as
                student_input
            teacher_weight (torch.Tensor): Weight matrix for the teacher's linear layer with shape
                (vocab_size, hidden_dim)
            true_labels (torch.LongTensor): Ground truth labels for computing hard loss with shape
                (batch_size, seq_len)
            student_bias (torch.Tensor): Optional bias tensor for student's linear layer with shape
                (vocab_size,). Can be None if no bias is used
            teacher_bias (torch.Tensor): Optional bias tensor for teacher's linear layer with shape
                (vocab_size,). Can be None if no bias is used
            weight_hard_loss (float, optional): Weight coefficient for the hard loss component.
                Default: 0.5
            weight_soft_loss (float, optional): Weight coefficient for the soft loss component.
                Default: 0.5
            beta (float, optional): Coefficient for the generalized cosine similarity loss in the
                interval [0, 1]. Default: 0.5
            ignore_index (int, optional): Index to ignore when computing the loss, typically used
                for padding tokens. Default: -100
            temperature (float, optional): Temperature parameter for softmax scaling in distillation.
                Default: 1.0
            compiled (bool, optional): Whether to use compiled operations for better performance.
                Default: True
            chunk_size (int, optional): Size of chunks for memory-efficient processing.
                Default: 1024

        Returns:
            torch.Tensor: Combined distillation loss scalar value computed as a weighted sum of
                hard loss (cross-entropy) and soft loss (cosine similarity)

        Important Notes:
            - This function delegates the actual computation to the parent class's forward method
            - The cosine similarity loss encourages the student model's output distribution to
              align with the teacher's output distribution in terms of directional similarity
            - Memory usage is optimized through chunked processing to handle large vocabulary sizes
            - The function is part of a custom autograd function and should be called via the
              apply() method rather than directly
        """
        <your code>

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute gradients for the fused linear cosine similarity distillation loss function.

        This static method implements the backward pass for the LigerFusedLinearCosineSimilarityFunction,
        computing gradients with respect to the input parameters during backpropagation. It delegates
        the gradient computation to the parent class's backward method and then formats the return
        values to match the expected signature by setting non-differentiable parameters to None.

        Args:
            ctx: The context object that stores information from the forward pass, including
                 saved tensors and other data needed for gradient computation.
            grad_output (torch.Tensor): The gradient of the loss with respect to the output
                                       of the forward pass. This gradient flows backward from
                                       the subsequent layers in the computation graph.

        Returns:
            tuple: A tuple containing gradients for all input parameters of the forward method:
                - Gradient w.r.t. student_input (torch.Tensor or None)
                - Gradient w.r.t. student_weight (torch.Tensor or None)  
                - Gradient w.r.t. teacher_input (torch.Tensor or None)
                - Gradient w.r.t. teacher_weight (torch.Tensor or None)
                - Gradient w.r.t. true_labels (None, as labels are not differentiable)
                - Gradient w.r.t. student_bias (torch.Tensor or None)
                - None for teacher_bias (non-differentiable parameter)
                - None for weight_hard_loss (non-differentiable parameter)
                - None for weight_soft_loss (non-differentiable parameter)
                - None for beta (non-differentiable parameter)
                - None for ignore_index (non-differentiable parameter)
                - None for temperature (non-differentiable parameter)
                - None for compiled (non-differentiable parameter)
                - None for chunk_size (non-differentiable parameter)

        Notes:
            - This method is part of PyTorch's autograd system and should not be called directly
            - Only the first 6 gradients from the parent class backward method are used
            - All hyperparameters and configuration options return None as they are not differentiable
            - The method assumes the parent class LigerFusedLinearDistillationBase properly implements
              the gradient computation for the core distillation loss components
        """
        <your code>

class LigerFusedLinearCosineSimilarityLoss(torch.nn.Module):
    """
    A PyTorch neural network module that implements a fused linear layer with cosine similarity-based distillation loss for knowledge distillation tasks.
    
    This loss function combines hard loss (cross-entropy with true labels) and soft loss (cosine similarity between student and teacher logits) to enable efficient knowledge transfer from a teacher model to a student model. The cosine similarity loss measures the angular similarity between normalized student and teacher representations.
    
    Attributes:
        weight_hard_loss (float): Weight coefficient for the hard loss component (cross-entropy with true labels). Default: 0.5.
        weight_soft_loss (float): Weight coefficient for the soft loss component (cosine similarity loss). Default: 0.5.
        beta (float): Coefficient for generalized cosine similarity loss in the interval [0, 1]. Controls the strength of the cosine similarity penalty. Default: 0.5.
        ignore_index (int): Index to ignore in loss computation, typically used for padding tokens. Default: -100.
        temperature (float): Temperature parameter for softmax scaling in distillation. Must be non-zero. Default: 1.0.
        compiled (bool): Whether to use compiled operations for performance optimization. Default: True.
        chunk_size (int): Size of chunks for memory-efficient processing of large tensors. Default: 1024.
    
    Methods:
        __init__: Initializes the loss module with specified hyperparameters and validates temperature is non-zero.
        forward: Computes the combined distillation loss by applying the fused linear cosine similarity function to student and teacher inputs, weights, and true labels.
    
    Usage Example:
        ```python
        import torch
        
        # Initialize the loss function
        loss_fn = LigerFusedLinearCosineSimilarityLoss(
            weight_hard_loss=0.7,
            weight_soft_loss=0.3,
            beta=0.8,
            temperature=3.0
        )
        
        # Prepare inputs
        batch_size, seq_len, hidden_dim = 2, 10, 768
        vocab_size = 1000
        
        student_input = torch.randn(batch_size * seq_len, hidden_dim)
        teacher_input = torch.randn(batch_size * seq_len, hidden_dim)
        student_weight = torch.randn(vocab_size, hidden_dim)
        teacher_weight = torch.randn(vocab_size, hidden_dim)
        true_labels = torch.randint(0, vocab_size, (batch_size * seq_len,))
        
        # Compute loss
        loss = loss_fn(
            student_input=student_input,
            student_weight=student_weight,
            teacher_input=teacher_input,
            teacher_weight=teacher_weight,
            true_labels=true_labels
        )
        
        # Backpropagate
        loss.backward()
        ```
    """

    def __init__(
        self,
        weight_hard_loss: float = 0.5,
        weight_soft_loss: float = 0.5,
        beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
        compiled: bool = True,
        chunk_size: int = 1024
    ):
        """
        Initialize the LigerFusedLinearCosineSimilarityLoss module.

        This constructor sets up a fused linear layer with cosine similarity-based distillation loss,
        which combines both hard loss (cross-entropy with true labels) and soft loss (cosine similarity
        between student and teacher logits) for knowledge distillation in neural networks.

        Parameters:
            weight_hard_loss (float, optional): Weight coefficient for the hard loss component 
                (cross-entropy loss with true labels). Must be in range [0, 1]. Default: 0.5.
            weight_soft_loss (float, optional): Weight coefficient for the soft loss component 
                (cosine similarity loss between student and teacher). Must be in range [0, 1]. Default: 0.5.
            beta (float, optional): Coefficient for generalized cosine similarity computation 
                in the distillation loss. Controls the strength of the cosine similarity penalty. Default: 0.5.
            ignore_index (int, optional): Index to ignore in loss computation, typically used 
                for padding tokens. Default: -100.
            temperature (float, optional): Temperature parameter for softmax scaling in knowledge 
                distillation. Higher values create softer probability distributions. Must be non-zero. Default: 1.0.
            compiled (bool, optional): Whether to use compiled operations for potential performance 
                optimization. Default: True.
            chunk_size (int, optional): Size of chunks for processing large tensors to manage 
                memory usage during computation. Default: 1024.

        Raises:
            AssertionError: If temperature is set to 0, which would cause division by zero errors.

        Notes:
            - The total loss is computed as: weight_hard_loss * hard_loss + weight_soft_loss * soft_loss
            - The cosine similarity loss encourages the student model to produce logits with similar 
              directional patterns to the teacher model
            - Chunked processing helps manage memory for large vocabulary sizes or long sequences
            - The ignore_index parameter is useful for masked language modeling where certain positions 
              should not contribute to the loss
        """
        <your code>

    def forward(
        self,
        student_input: torch.Tensor,
        student_weight: torch.Tensor,
        teacher_input: torch.Tensor,
        teacher_weight: torch.Tensor,
        true_labels: torch.LongTensor,
        student_bias: torch.Tensor = None,
        teacher_bias: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute the fused linear cosine similarity distillation loss between student and teacher models.

        This method performs a forward pass through the fused linear cosine similarity loss function,
        which combines both hard loss (cross-entropy with true labels) and soft loss (cosine similarity
        between student and teacher logits) for knowledge distillation. The computation is optimized
        using chunked processing to handle large vocabularies efficiently.

        Parameters:
            student_input (torch.Tensor): Input tensor for the student model, typically hidden states
                from the student network. Shape: (batch_size * seq_len, hidden_dim).
            student_weight (torch.Tensor): Weight matrix for the student model's linear layer.
                Shape: (vocab_size, hidden_dim).
            teacher_input (torch.Tensor): Input tensor for the teacher model, typically hidden states
                from the teacher network. Shape: (batch_size * seq_len, hidden_dim).
            teacher_weight (torch.Tensor): Weight matrix for the teacher model's linear layer.
                Shape: (vocab_size, hidden_dim).
            true_labels (torch.LongTensor): Ground truth labels for computing the hard loss.
                Shape: (batch_size * seq_len,). Values should be token indices.
            student_bias (torch.Tensor, optional): Bias vector for the student model's linear layer.
                Shape: (vocab_size,). Defaults to None.
            teacher_bias (torch.Tensor, optional): Bias vector for the teacher model's linear layer.
                Shape: (vocab_size,). Defaults to None.

        Returns:
            torch.Tensor: A scalar tensor representing the combined distillation loss. The loss is
                computed as a weighted combination of hard loss (cross-entropy) and soft loss
                (cosine similarity), where the weights are determined by the instance's
                weight_hard_loss and weight_soft_loss parameters.

        Notes:
            - The function uses the instance's configuration parameters (weight_hard_loss,
              weight_soft_loss, beta, ignore_index, temperature, compiled, chunk_size) that
              were set during initialization.
            - The cosine similarity loss is computed using L2-normalized logits from both
              student and teacher models.
            - Chunked processing is employed to manage memory usage when dealing with large
              vocabulary sizes.
            - The temperature parameter is used for softmax temperature scaling in the base
              distillation computation.
            - Tokens with labels equal to ignore_index are excluded from loss computation.
        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.