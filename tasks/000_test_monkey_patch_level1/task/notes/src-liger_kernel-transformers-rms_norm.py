import torch
import torch.nn as nn

from liger_kernel.ops.rms_norm import LigerRMSNormFunction


class LigerRMSNorm(nn.Module):

    def __init__(
            self,
            hidden_size,
            eps = 1e-06,
            offset = 0.0,
            casting_mode = 'llama',
            init_fn = 'ones',
            in_place = True,
            row_mode = None
        ):
        raise NotImplementedError('This function has been masked for testing')

    def forward(self, hidden_states):
        raise NotImplementedError('This function has been masked for testing')

    def extra_repr(self):
        raise NotImplementedError('This function has been masked for testing')


class LigerRMSNormForGemma(LigerRMSNorm):
    def __init__(
        self, hidden_size, eps=1e-6, offset=1.0, casting_mode="gemma", init_fn="zeros", in_place=True, row_mode=None
    ):
        super().__init__(hidden_size, eps, offset, casting_mode, init_fn, in_place, row_mode)


class LigerRMSNormForGemma2(LigerRMSNorm):
    def __init__(
        self, hidden_size, eps=1e-6, offset=1.0, casting_mode="gemma", init_fn="zeros", in_place=False, row_mode=None
    ):
        super().__init__(hidden_size, eps, offset, casting_mode, init_fn, in_place, row_mode)


class LigerRMSNormForGemma3(LigerRMSNorm):
    """Gemma3RMSNorm has a dim argument not hidden_size used in q_norm and k_norm."""

    def __init__(self, dim, eps=0.000001, offset=1.0, casting_mode="gemma", init_fn="zeros", in_place=False):
        super().__init__(dim, eps, offset, casting_mode, init_fn, in_place)


class LigerRMSNormForOlmo2(LigerRMSNorm):
    def __init__(
        self, hidden_size, eps=1e-6, offset=0.0, casting_mode="llama", init_fn="ones", in_place=False, row_mode=None
    ):
        super().__init__(hidden_size, eps, offset, casting_mode, init_fn, in_place, row_mode)


class LigerRMSNormForGlm4(LigerRMSNorm):

    def __init__(
            self,
            hidden_size,
            eps = 1e-06,
            offset = 0.0,
            casting_mode = 'llama',
            init_fn = 'ones',
            in_place = False,
            row_mode = None
        ):
        raise NotImplementedError('This function has been masked for testing')