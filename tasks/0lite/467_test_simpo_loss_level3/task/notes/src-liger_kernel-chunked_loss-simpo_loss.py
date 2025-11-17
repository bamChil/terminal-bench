import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.fused_linear_preference import LigerFusedLinearPreferenceBase


class LigerFusedLinearSimPOFunction(LigerFusedLinearPreferenceBase):

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
        raise NotImplementedError('This function has been masked for testing')

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
        raise NotImplementedError('This function has been masked for testing')

    @staticmethod
    def backward(ctx, *grad_output):
        raise NotImplementedError('This function has been masked for testing')


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
        raise NotImplementedError('This function has been masked for testing')

    def forward(
            self,
            lin_weight,
            _input,
            target,
            bias = None
        ):
        raise NotImplementedError('This function has been masked for testing')