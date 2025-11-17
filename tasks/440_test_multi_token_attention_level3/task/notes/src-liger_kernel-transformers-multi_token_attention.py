import math

import torch
import torch.nn as nn

from torch.nn.modules.utils import _pair

from liger_kernel.ops.multi_token_attention import LigerMultiTokenAttentionFunction


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
        raise NotImplementedError('This function has been masked for testing')

    def reset_parameters(self):
        raise NotImplementedError('This function has been masked for testing')

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('This function has been masked for testing')