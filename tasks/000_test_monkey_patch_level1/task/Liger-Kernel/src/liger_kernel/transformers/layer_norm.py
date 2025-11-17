import torch
import torch.nn as nn

from liger_kernel.ops.layer_norm import LigerLayerNormFunction


class LigerLayerNorm(nn.Module):

    def __init__(
            self,
            hidden_size,
            eps = 1e-06,
            bias = False,
            init_fn = 'ones'
        ):
        raise NotImplementedError('This function has been masked for testing')

    def forward(self, hidden_states):
        raise NotImplementedError('This function has been masked for testing')

    def extra_repr(self):
        raise NotImplementedError('This function has been masked for testing')