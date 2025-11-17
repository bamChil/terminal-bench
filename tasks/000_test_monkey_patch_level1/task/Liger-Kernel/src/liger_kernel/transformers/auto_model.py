import inspect

from transformers import AutoConfig
from transformers import AutoModelForCausalLM

from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN
from liger_kernel.transformers.monkey_patch import _apply_liger_kernel




class AutoLigerKernelForCausalLM(AutoModelForCausalLM):
    """

        This class is a drop-in replacement for AutoModelForCausalLM that applies the Liger Kernel to the model
        if applicable.

    """

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path,
            *model_args,
            **kwargs
        ):
        raise NotImplementedError('This function has been masked for testing')