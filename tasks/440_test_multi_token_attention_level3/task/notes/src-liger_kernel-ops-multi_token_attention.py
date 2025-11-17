import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from torch.nn.modules.utils import _pair

from liger_kernel.ops.softmax import _softmax_forward
from liger_kernel.ops.sparsemax import _sparsemax_backward
from liger_kernel.ops.sparsemax import _sparsemax_forward
from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import ensure_contiguous













