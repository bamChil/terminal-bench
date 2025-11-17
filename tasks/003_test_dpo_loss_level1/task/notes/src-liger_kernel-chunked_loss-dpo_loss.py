import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.fused_linear_preference import LigerFusedLinearPreferenceBase


class LigerFusedLinearDPOFunction(LigerFusedLinearPreferenceBase):

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
        raise NotImplementedError('This function has been masked for testing')

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
        raise NotImplementedError('This function has been masked for testing')

    @staticmethod
    def backward(ctx, *grad_output):
        raise NotImplementedError('This function has been masked for testing')


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
        raise NotImplementedError('This function has been masked for testing')

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
        raise NotImplementedError('This function has been masked for testing')