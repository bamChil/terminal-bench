import torch
import torch.nn as nn

from liger_kernel.ops.fused_add_rms_norm import LigerFusedAddRMSNormFunction


class LigerFusedAddRMSNorm(nn.Module):

    def __init__(
            self,
            hidden_size,
            eps = 1e-06,
            offset = 0.0,
            casting_mode = 'llama',
            init_fn = 'ones',
            in_place = False
        ):
        raise NotImplementedError('This function has been masked for testing')

    def forward(self, hidden_states, residual):
        raise NotImplementedError('This function has been masked for testing')

    def extra_repr(self):
        raise NotImplementedError('This function has been masked for testing')