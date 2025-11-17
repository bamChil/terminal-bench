from typing import Tuple

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import ensure_contiguous




@triton.jit
def _sparsemax_backward_kernel(
    o_ptr, go_ptr, gi_ptr, stride, n_cols, BLOCK_SIZE: tl.constexpr, num_warps: tl.constexpr
):
    row = tl.program_id(0)
    o_row = o_ptr + row * stride
    go_row = go_ptr + row * stride
    gi_row = gi_ptr + row * stride

    offs = tl.arange(0, BLOCK_SIZE)

    supp_cnt = tl.zeros((), tl.float32)
    go_sum = tl.zeros((), tl.float32)

    for i in tl.range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        offs_iter = i * BLOCK_SIZE + offs
        mask_iter = offs_iter < n_cols
        o_val = tl.load(o_row + offs_iter, mask=mask_iter, other=0.0, cache_modifier=".ca").to(tl.float32)
        go_val = tl.load(go_row + offs_iter, mask=mask_iter, other=0.0).to(tl.float32)
        supp = o_val > 0.0
        go_sum += tl.sum(tl.where(supp, go_val, 0.0))
        supp_cnt += tl.sum(supp.to(tl.float32))

    for i in tl.range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        offs_iter = i * BLOCK_SIZE + offs
        mask_iter = offs_iter < n_cols
        o_val = tl.load(o_row + offs_iter, mask=mask_iter, other=0.0, cache_modifier=".ca").to(tl.float32)
        go_val = tl.load(go_row + offs_iter, mask=mask_iter, other=0.0).to(tl.float32)
        supp = o_val > 0.0
        gi_val = tl.where(
            supp,
            go_val - tl.cast(go_sum / tl.maximum(supp_cnt, 1e-6), gi_row.dtype.element_ty).to(tl.float32),
            0.0,
        )
        tl.store(gi_row + offs_iter, gi_val.to(gi_row.dtype.element_ty), mask=mask_iter, cache_modifier=".wb")




def _sparsemax_backward(
    grad_out: torch.Tensor,
    out_flat: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    grad_sw = grad_out.transpose(dim, -1).contiguous()
    n_cols = grad_sw.size(-1)
    n_rows = grad_sw.numel() // n_cols
    go_flat = grad_sw.view(n_rows, n_cols)

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    dx_flat = torch.empty_like(go_flat)
    grid = (n_rows,)
    _sparsemax_backward_kernel[grid](
        out_flat,
        go_flat,
        dx_flat,
        out_flat.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    dx = dx_flat.view_as(grad_sw).transpose(dim, -1)
    return dx


class LigerSparsemaxFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x: torch.Tensor, dim: int):
        y, out_flat = _sparsemax_forward(x, dim)
        ctx.save_for_backward(out_flat)
        ctx.dim = dim
        return y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_out: torch.Tensor):
        (out_flat,) = ctx.saved_tensors
        dx = _sparsemax_backward(grad_out, out_flat, ctx.dim)
        return dx, None