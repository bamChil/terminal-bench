"""
Liger Kernel implementation of Llama4 Rotary Position Embedding (RoPE).
Supports both text and vision RoPE variants with fused operations for optimal performance.
"""

import torch

from liger_kernel.ops.llama4_rope import LigerLlama4RopeFunction






# Note: We only patch the functions, not the classes
# The original Llama4TextRotaryEmbedding and Llama4VisionRotaryEmbedding classes remain unchanged


# Convenience functions for monkey patching