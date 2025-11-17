import torch.nn as nn

from liger_kernel.ops.swiglu import LigerSiLUMulFunction


class LigerSwiGLUMLP(nn.Module):

    def __init__(self, config):
        raise NotImplementedError('This function has been masked for testing')

    def forward(self, x):
        raise NotImplementedError('This function has been masked for testing')


class LigerBlockSparseTop2MLP(nn.Module):

    def __init__(self, config):
        raise NotImplementedError('This function has been masked for testing')

    def forward(self, x):
        raise NotImplementedError('This function has been masked for testing')


class LigerPhi3SwiGLUMLP(nn.Module):
    """

        Patch Phi3MLP to use LigerSiLUMulFunction
        https://github.com/huggingface/transformers/blob/v4.41.0/src/transformers/models/phi3/modeling_phi3.py#L241

    """

    def __init__(self, config):
        raise NotImplementedError('This function has been masked for testing')

    def forward(self, x):
        raise NotImplementedError('This function has been masked for testing')


class LigerQwen3MoeSwiGLUMLP(nn.Module):
    """

        Patch Qwen3MoeMLP to use LigerSiLUMulFunction.
        https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3_moe/modular_qwen3_moe.py#L57

    """

    def __init__(
            self,
            config,
            intermediate_size = None
        ):
        raise NotImplementedError('This function has been masked for testing')

    def forward(self, x):
        raise NotImplementedError('This function has been masked for testing')