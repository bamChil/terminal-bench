import math
import operator

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import compare_version
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import torch_to_triton_dtype

if compare_version("triton", operator.ge, "3.0.0"):
    try:
        # typical import path with dispatch available
        from triton.language.extra.libdevice import rsqrt
    except ModuleNotFoundError:
        # for working with NGC containers
        from triton.language.extra.cuda.libdevice import rsqrt
else:
    from triton.language.math import rsqrt


_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)






_str_to_casting_mode = {
    "llama": _CASTING_MODE_LLAMA.value,
    "gemma": _CASTING_MODE_GEMMA.value,
    "none": _CASTING_MODE_NONE.value,
}






class LigerFusedAddRMSNormFunction(torch.autograd.Function):
    """

        Performs a fused operation that first adds a residual tensor to the hidden_states tensor (`X`), then applies RMSNorm (Root Mean Square Normalization) to the result using the weight tensor `W`, with optional offset and casting mode.

        This class implements the following sequence, commonly used in transformer decoder layers:
            1. hidden_states = residual + hidden_states
            2. residual = hidden_states (after addition)
            3. hidden_states = rmsnorm(hidden_states)

        Both the normalized hidden_states and the updated residual are returned as outputs.

        Some models use an 'offset' to shift the weight tensor `W` by a constant value. For example, Gemma
        uses an offset of 1.0, so the computation becomes `(X / RMS(X)) * (W + 1.0)` instead of the usual
        `(X / RMS(X)) * W`. You can pass the offset value as an argument to the forward function.

        In addition, different models cast their inputs at different places during RMSNorm computation. For
        example, Gemma casts everything to fp32 before starting the computation, while Llama casts only the
        inverse RMS to fp32. You can specify the casting mode using the `casting_mode` argument. We currently
        support the following casting modes (they match HuggingFace Transformers' implementations):
        - 'llama': matches the Llama implementation, where only the inverse RMS is computed on fp32.
        - 'gemma': matches the Gemma implementation, where everything is cast to fp32, then computed, then cast back to the original dtype.
        - 'none': no casting is done. The computation is done in the original dtype. This saves memory and is slightly faster, but has more error w.r.t. the original implementation.

        The `in_place` option determines whether to modify dY in-place to store dX. This defaults to `True` to save memory.

    """

    @staticmethod
    @ensure_contiguous
    def forward(
            ctx,
            X,
            R,
            W,
            eps,
            offset = 0.0,
            casting_mode = 'llama',
            in_place = False
        ):
        """

                X: (B, T, H) or (BxT, H)
                W: (H,)

        """
        raise NotImplementedError('This function has been masked for testing')

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY, dS_out):
        """

                Y: (B, T, H) or (BxT, H)

        """
        raise NotImplementedError('This function has been masked for testing')