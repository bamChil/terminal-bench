import torch
import torch.nn.functional as F

from liger_kernel.chunked_loss.fused_linear_distillation import LigerFusedLinearDistillationBase


class LigerFusedLinearCosineSimilarityFunction(LigerFusedLinearDistillationBase):

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
        raise NotImplementedError('This function has been masked for testing')

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
        raise NotImplementedError('This function has been masked for testing')

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError('This function has been masked for testing')


class LigerFusedLinearCosineSimilarityLoss(torch.nn.Module):

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
        raise NotImplementedError('This function has been masked for testing')

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
        raise NotImplementedError('This function has been masked for testing')