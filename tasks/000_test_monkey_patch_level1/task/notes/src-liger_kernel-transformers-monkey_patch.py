import inspect
import logging

from functools import partial
from types import MethodType
from typing import Callable

import transformers

from packaging import version
from transformers import PreTrainedModel

from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from liger_kernel.transformers.functional import liger_cross_entropy
from liger_kernel.transformers.geglu import LigerGEGLUMLP
from liger_kernel.transformers.layer_norm import LigerLayerNorm
from liger_kernel.transformers.model.gemma import lce_forward as gemma_lce_forward
from liger_kernel.transformers.model.gemma import lce_forward_deprecated as gemma_lce_forward_deprecated
from liger_kernel.transformers.model.gemma2 import lce_forward as gemma2_lce_forward
from liger_kernel.transformers.model.gemma2 import lce_forward_deprecated as gemma2_lce_forward_deprected
from liger_kernel.transformers.model.llama import lce_forward as llama_lce_forward
from liger_kernel.transformers.model.llama import lce_forward_deprecated as llama_lce_forward_deprecated
from liger_kernel.transformers.model.llava import lce_forward as llava_lce_forward
from liger_kernel.transformers.model.llava import lce_forward_deprecated as llava_lce_forward_deprecated
from liger_kernel.transformers.model.mistral import lce_forward as mistral_lce_forward
from liger_kernel.transformers.model.mixtral import lce_forward as mixtral_lce_forward
from liger_kernel.transformers.model.mixtral import lce_forward_deprecated as mixtral_lce_forward_deprecated
from liger_kernel.transformers.model.phi3 import lce_forward as phi3_lce_forward
from liger_kernel.transformers.model.qwen2 import lce_forward as qwen2_lce_forward
from liger_kernel.transformers.model.qwen2 import lce_forward_deprecated as qwen2_lce_forward_deprecated
from liger_kernel.transformers.model.smollm3 import lce_forward as smollm3_lce_forward
from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from liger_kernel.transformers.swiglu import LigerBlockSparseTop2MLP
from liger_kernel.transformers.swiglu import LigerPhi3SwiGLUMLP
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

try:
    import peft

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

transformer_version = version.parse(transformers.__version__)

logger = logging.getLogger(__name__)
SUPPORTED_TRANSFORMER_VERSION = "4.46.1"
TRANSFORMER_DEPRECATION_WARNING = "Support for transformers versions < 4.46.1 will soon be discontinued due to issues with incorrect gradient accumulation. \n Please consider upgrading to avoid potential issues. See details: https://github.com/huggingface/transformers/pull/34191"












def apply_liger_kernel_to_granite(
    rope: bool = True,
    cross_entropy: bool = True,
    fused_linear_cross_entropy: bool = False,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Granite 3 models

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is True.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is False.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.



    Debugging notes:
        If LigerSwiGLUMLP is OK for Llama, it should be fine for Granite, but it's not.
    """

    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.granite import modeling_granite
    from transformers.models.granite.modeling_granite import GraniteModel

    if swiglu:
        modeling_granite.GraniteMLP = LigerSwiGLUMLP

    if rms_norm:
        modeling_granite.GraniteRMSNorm = LigerRMSNorm

    if rope:
        modeling_granite.apply_rotary_pos_emb = liger_rotary_pos_emb

    if cross_entropy:
        if transformer_version >= version.parse(SUPPORTED_TRANSFORMER_VERSION):
            from transformers.loss.loss_utils import nn

            nn.functional.cross_entropy = liger_cross_entropy
        else:
            logger.warning(TRANSFORMER_DEPRECATION_WARNING)
            modeling_granite.CrossEntropyLoss = LigerCrossEntropyLoss

    if fused_linear_cross_entropy:
        raise NotImplementedError("LigerFusedLinearCrossEntropy is not available for Granite models.")
        # NOTE: Granite model `GraniteForCausalLM.forward` scales logits each
        # call, so we can't sidestep logit materialization. A bit more work
        # would be needed to add a scaling term to the `LigerFusedLinearCrossEntropyFunction`
        # for the logit output.

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules (e.g. GraniteRMSNorm or GraniteMLP)

        # get the base model from the model instance
        base_model: GraniteModel = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


def apply_liger_kernel_to_llama(
        rope: bool = True,
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        rms_norm: bool = True,
        swiglu: bool = True,
        model: PreTrainedModel = None
    ) -> None:
    """

        Apply Liger kernels to replace original implementation in HuggingFace Llama models (2 and 3)

        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.

    """
    raise NotImplementedError('This function has been masked for testing')


def apply_liger_kernel_to_smollm3(
        rope: bool = True,
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        rms_norm: bool = True,
        swiglu: bool = True,
        model: PreTrainedModel = None
    ) -> None:
    """

        Apply Liger kernels to replace original implementation in HuggingFace SmolLM3 model

        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.

    """
    raise NotImplementedError('This function has been masked for testing')


def apply_liger_kernel_to_llava(
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    model: PreTrainedModel = None,
    **kwargs,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Llava models.
    Due to the characteristics of LlaVa, the model must be passed to apply Liger-Kernel's patch to other models connected to LLaVa.
    However, if an LM not supported by Liger-Kernel is connected to LLaVa, unexpected side effects may occur.
    NOTE: Llava is not available in transformers<4.36.0

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from transformers.models.llava import modeling_llava

    if cross_entropy:
        logger.warning(TRANSFORMER_DEPRECATION_WARNING)
        modeling_llava.nn.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        if transformer_version >= version.parse("4.52.0"):
            if model is not None:
                model.forward = MethodType(llava_lce_forward, model)
            else:
                modeling_llava.LlavaForConditionalGeneration.forward = llava_lce_forward
        elif transformer_version >= version.parse("4.49.0") and transformer_version < version.parse("4.52.0"):
            if model is not None:
                model.forward = MethodType(llava_lce_forward_deprecated, model)
            else:
                modeling_llava.LlavaForConditionalGeneration.forward = llava_lce_forward_deprecated
        else:  # if version < 4.49.0
            logger.warning(
                "The latest version of Liger does not support transformers < 4.49.0 for llava. Please downgrade your liger version or upgrade your transformer version."
            )

    if model is not None:
        text_model_name, vision_model_name = model.config.text_config.model_type, model.config.vision_config.model_type
        text_liger_fn = MODEL_TYPE_TO_APPLY_LIGER_FN.get(text_model_name, None)
        vision_liger_fn = MODEL_TYPE_TO_APPLY_LIGER_FN.get(vision_model_name, None)

        kwargs = {"cross_entropy": False, "fused_linear_cross_entropy": False, **kwargs}
        if text_liger_fn:
            accept_params = inspect.signature(text_liger_fn).parameters
            remain_params = set(kwargs) - (set(accept_params) & set(kwargs))
            text_kwargs = {k: v for k, v in kwargs.items() if k not in remain_params}

            if remain_params:
                logger.warning(
                    f"These parameters are not supported by {text_model_name}. Enter the remaining {list(text_kwargs.keys())} except for {list(remain_params)}\n"
                    f"Parameters accepted by {text_model_name}: {list(accept_params.keys())}"
                )
            text_kwargs["model"] = model.language_model
            text_liger_fn(**text_kwargs)
        elif text_model_name not in MODEL_TYPE_TO_APPLY_LIGER_FN:
            logger.warning(f"{text_model_name} is not supported by Liger kernel.")

        if vision_liger_fn:
            accept_params = inspect.signature(vision_liger_fn).parameters
            remain_params = set(kwargs) - (set(accept_params) & set(kwargs))
            vision_kwargs = {k: v for k, v in kwargs.items() if k not in remain_params}

            if remain_params:
                logger.warning(
                    f"These parameters are not supported by {vision_model_name}. Enter the remaining {list(vision_kwargs.keys())} except for {list(remain_params)}\n"
                    f"Parameters accepted by {vision_model_name}: {list(accept_params.keys())}"
                )
            vision_kwargs["model"] = model.vision_tower
            vision_liger_fn(**vision_kwargs)
        elif vision_model_name not in MODEL_TYPE_TO_APPLY_LIGER_FN:
            logger.warning(f"{vision_model_name} is not supported by Liger kernel.")




def apply_liger_kernel_to_mllama(
        rope: bool = True,
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        layer_norm: bool = True,
        rms_norm: bool = True,
        swiglu: bool = True,
        model: PreTrainedModel = None
    ) -> None:
    """

        Apply Liger kernels to replace original implementation in HuggingFace MLlama models.
        NOTE: MLlama is not available in transformers<4.45.0

        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.

    """
    raise NotImplementedError('This function has been masked for testing')


def apply_liger_kernel_to_mistral(
        rope: bool = True,
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        rms_norm: bool = True,
        swiglu: bool = True,
        model: PreTrainedModel = None
    ) -> None:
    """

        Apply Liger kernels to replace original implementation in HuggingFace Mistral models

        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is False.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is True.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.

    """
    raise NotImplementedError('This function has been masked for testing')


def apply_liger_kernel_to_mixtral(
        rope: bool = True,
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        rms_norm: bool = True,
        swiglu: bool = True,
        model: PreTrainedModel = None
    ) -> None:
    """

        Apply Liger kernels to replace original implementation in HuggingFace Mixtral models

        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.

    """
    raise NotImplementedError('This function has been masked for testing')


def apply_liger_kernel_to_gemma(
        rope: bool = True,
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        rms_norm: bool = True,
        geglu: bool = True,
        model: PreTrainedModel = None
    ) -> None:
    """

        Apply Liger kernels to replace original implementation in HuggingFace Gemma
        (Gemma 1 and 1.1 supported, for Gemma2 please use `apply_liger_kernel_to_gemma2` ) to make GPU go burrr.

        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            geglu (bool): Whether to apply Liger's GeGLU MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.

    """
    raise NotImplementedError('This function has been masked for testing')


def apply_liger_kernel_to_gemma2(
        rope: bool = True,
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        rms_norm: bool = True,
        geglu: bool = True,
        model: PreTrainedModel = None
    ) -> None:
    """

        Apply Liger kernels to replace original implementation in HuggingFace Gemma2
        (for Gemma1 please use `apply_liger_kernel_to_gemma`) to make GPU go burrr.

        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            geglu (bool): Whether to apply Liger's GeGLU MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.

    """
    raise NotImplementedError('This function has been masked for testing')


def apply_liger_kernel_to_gemma3_text(
        rope: bool = True,
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        rms_norm: bool = True,
        geglu: bool = True,
        model: PreTrainedModel = None
    ) -> None:
    """

        Apply Liger kernels to replace original implementation in HuggingFace Gemma3

        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            geglu (bool): Whether to apply Liger's GeGLU MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.

    """
    raise NotImplementedError('This function has been masked for testing')


def apply_liger_kernel_to_gemma3(
        rope: bool = True,
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        layer_norm: bool = True,
        rms_norm: bool = True,
        geglu: bool = True,
        model: PreTrainedModel = None
    ) -> None:
    """

        Apply Liger kernels to replace original implementation in HuggingFace Gemma3

        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            layer_norm (bool): Whether to apply Liger's LayerNorm. Default is True.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            geglu (bool): Whether to apply Liger's GeGLU MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.

    """
    raise NotImplementedError('This function has been masked for testing')




def apply_liger_kernel_to_qwen2(
        rope: bool = True,
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        rms_norm: bool = True,
        swiglu: bool = True,
        model: PreTrainedModel = None
    ) -> None:
    """

        Apply Liger kernels to replace original implementation in HuggingFace Qwen2 models

        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.

    """
    raise NotImplementedError('This function has been masked for testing')


def apply_liger_kernel_to_qwen3(
        rope: bool = True,
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        rms_norm: bool = True,
        swiglu: bool = True,
        model: PreTrainedModel = None
    ) -> None:
    """

        Apply Liger kernels to replace original implementation in HuggingFace Qwen3 models.

    """
    raise NotImplementedError('This function has been masked for testing')


def apply_liger_kernel_to_qwen3_moe(
        rope: bool = True,
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        rms_norm: bool = True,
        swiglu: bool = True,
        model: PreTrainedModel = None
    ) -> None:
    """

        Apply Liger kernels to replace original implementation in HuggingFace Qwen3 models.

    """
    raise NotImplementedError('This function has been masked for testing')


def apply_liger_kernel_to_qwen2_vl(
        rope: bool = True,
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        rms_norm: bool = True,
        layer_norm: bool = True,
        swiglu: bool = True,
        model: PreTrainedModel = None
    ) -> None:
    """

        Apply Liger kernels to replace original implementation in HuggingFace Qwen2-VL models.
        NOTE: Qwen2-VL is not supported in transformers<4.52.4

        Args:
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            layer_norm (bool): Whether to apply Liger's LayerNorm. Default is True.
            swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.

    """
    raise NotImplementedError('This function has been masked for testing')


def apply_liger_kernel_to_qwen2_5_vl(
        rope: bool = True,
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        rms_norm: bool = True,
        swiglu: bool = True,
        model: PreTrainedModel = None
    ) -> None:
    """

        Apply Liger kernels to replace original implementation in HuggingFace Qwen2.5-VL models.
        NOTE: Qwen2.5-VL is not available in transformers<4.48.2

        Args:
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.

    """
    raise NotImplementedError('This function has been masked for testing')


def apply_liger_kernel_to_phi3(
        rope: bool = True,
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        rms_norm: bool = True,
        swiglu: bool = True,
        model: PreTrainedModel = None
    ) -> None:
    """

        Apply Liger kernels to replace original implementation in HuggingFace Phi3 models.

        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            swiglu (bool): Whether to apply Liger's SwiGLU Phi3MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.

    """
    raise NotImplementedError('This function has been masked for testing')




def apply_liger_kernel_to_glm4(
        rope: bool = False,
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        rms_norm: bool = True,
        swiglu: bool = True,
        model: PreTrainedModel = None
    ) -> None:
    """

        Apply Liger kernels to replace original implementation in HuggingFace GLM-4 models.

        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is False.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            swiglu (bool): Whether to apply Liger's SwiGLU Glm4MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.

    """
    raise NotImplementedError('This function has been masked for testing')


def apply_liger_kernel_to_glm4v(
        rope: bool = False,
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        rms_norm: bool = True,
        swiglu: bool = True,
        model: PreTrainedModel = None
    ) -> None:
    """

        Apply Liger kernels to replace original implementation in HuggingFace GLM-4v models.

        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is False.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            swiglu (bool): Whether to apply Liger's SwiGLU Glm4MLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.

    """
    raise NotImplementedError('This function has been masked for testing')


def apply_liger_kernel_to_glm4v_moe(
        rope: bool = False,
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        rms_norm: bool = True,
        swiglu: bool = True,
        model: PreTrainedModel = None
    ) -> None:
    """

        Apply Liger kernels to replace original implementation in HuggingFace GLM4v_moe models.

        Args:
            rope (bool): Whether to apply Liger's rotary position embedding. Default is False.
            cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
            fused_linear_cross_entropy (bool):
                Whether to apply Liger's fused linear cross entropy loss. Default is True.
                `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
            rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
            swiglu (bool): Whether to apply Liger's SwiGLUMLP. Default is True.
            model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
            loaded. Default is None.

    """
    raise NotImplementedError('This function has been masked for testing')


# Model type corresponds to the keys defined in transformers/models/auto/modeling_auto.py
MODEL_TYPE_TO_APPLY_LIGER_FN = {
    "gemma": apply_liger_kernel_to_gemma,
    "gemma2": apply_liger_kernel_to_gemma2,
    "gemma3_text": apply_liger_kernel_to_gemma3_text,
    "gemma3": apply_liger_kernel_to_gemma3,
    "glm4": apply_liger_kernel_to_glm4,
    "glm4v": apply_liger_kernel_to_glm4v,
    "glm4v_moe": apply_liger_kernel_to_glm4v_moe,
    "llama": apply_liger_kernel_to_llama,
    "llama4_text": apply_liger_kernel_to_llama4,
    "llama4": apply_liger_kernel_to_llama4,
    "llava": apply_liger_kernel_to_llava,
    "granite": apply_liger_kernel_to_granite,
    "mllama": apply_liger_kernel_to_mllama,
    "mllama_text_model": apply_liger_kernel_to_mllama,
    "mistral": apply_liger_kernel_to_mistral,
    "mixtral": apply_liger_kernel_to_mixtral,
    "olmo2": apply_liger_kernel_to_olmo2,
    "qwen2": apply_liger_kernel_to_qwen2,
    "qwen3": apply_liger_kernel_to_qwen3,
    "qwen3_moe": apply_liger_kernel_to_qwen3_moe,
    "qwen2_vl": apply_liger_kernel_to_qwen2_vl,
    "qwen2_vl_text": apply_liger_kernel_to_qwen2_vl,
    "qwen2_5_vl": apply_liger_kernel_to_qwen2_5_vl,
    "qwen2_5_vl_text": apply_liger_kernel_to_qwen2_5_vl,
    "smollm3": apply_liger_kernel_to_smollm3,
    "phi3": apply_liger_kernel_to_phi3,
    "paligemma": apply_liger_kernel_to_paligemma,
}


def _apply_liger_kernel(model_type: str, **kwargs) -> None:
    """

        Applies Liger kernels based on the specified model type. The custom
        kernels for the specified model type will be applied with the provided
        keyword arguments, otherwise the default configuration will be used.

        ** Note: Calling _apply_liger_kernel() after model initialization
        will not be able to fully patch models. This must be called before model initialization.
        If the model has already been instantiated

        Args:
            - model_type: the model types as defined in transformers/models/auto/modeling_auto.py
              and specified in the model's config.json
            - kwargs: keyword arguments that are passed to the corresponding apply_liger_kernel_to_* function.

    """
    raise NotImplementedError('This function has been masked for testing')


def _apply_liger_kernel_to_instance(
        model: PreTrainedModel,
        **kwargs
    ) -> None:
    """

        Applies Liger kernels to the provided model instance.

        Args:
            - model: the model instance to apply Liger kernels to
            - kwargs: keyword arguments that are passed to the corresponding apply_liger_kernel_to_* function.

    """
    raise NotImplementedError('This function has been masked for testing')