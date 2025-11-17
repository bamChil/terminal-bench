# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AutoProcessor class."""

import importlib
import inspect
import json
import warnings
from collections import OrderedDict

# Build the list of all feature extractors
from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...feature_extraction_utils import FeatureExtractionMixin
from ...image_processing_utils import ImageProcessingMixin
from ...processing_utils import ProcessorMixin
from ...tokenization_utils import TOKENIZER_CONFIG_FILE
from ...utils import FEATURE_EXTRACTOR_NAME, PROCESSOR_NAME, VIDEO_PROCESSOR_NAME, cached_file, logging
from ...video_processing_utils import BaseVideoProcessor
from .auto_factory import _LazyAutoMapping
from .configuration_auto import (
    CONFIG_MAPPING_NAMES,
    AutoConfig,
    model_type_to_module_name,
    replace_list_option_in_docstrings,
)
from .feature_extraction_auto import AutoFeatureExtractor
from .image_processing_auto import AutoImageProcessor
from .tokenization_auto import AutoTokenizer


logger = logging.get_logger(__name__)

PROCESSOR_MAPPING_NAMES = OrderedDict(
    [
        ("aimv2", "CLIPProcessor"),
        ("align", "AlignProcessor"),
        ("altclip", "AltCLIPProcessor"),
        ("aria", "AriaProcessor"),
        ("aya_vision", "AyaVisionProcessor"),
        ("bark", "BarkProcessor"),
        ("blip", "BlipProcessor"),
        ("blip-2", "Blip2Processor"),
        ("bridgetower", "BridgeTowerProcessor"),
        ("chameleon", "ChameleonProcessor"),
        ("chinese_clip", "ChineseCLIPProcessor"),
        ("clap", "ClapProcessor"),
        ("clip", "CLIPProcessor"),
        ("clipseg", "CLIPSegProcessor"),
        ("clvp", "ClvpProcessor"),
        ("cohere2_vision", "Cohere2VisionProcessor"),
        ("colpali", "ColPaliProcessor"),
        ("colqwen2", "ColQwen2Processor"),
        ("deepseek_vl", "DeepseekVLProcessor"),
        ("deepseek_vl_hybrid", "DeepseekVLHybridProcessor"),
        ("dia", "DiaProcessor"),
        ("emu3", "Emu3Processor"),
        ("evolla", "EvollaProcessor"),
        ("flava", "FlavaProcessor"),
        ("florence2", "Florence2Processor"),
        ("fuyu", "FuyuProcessor"),
        ("gemma3", "Gemma3Processor"),
        ("gemma3n", "Gemma3nProcessor"),
        ("git", "GitProcessor"),
        ("glm4v", "Glm4vProcessor"),
        ("glm4v_moe", "Glm4vProcessor"),
        ("got_ocr2", "GotOcr2Processor"),
        ("granite_speech", "GraniteSpeechProcessor"),
        ("grounding-dino", "GroundingDinoProcessor"),
        ("groupvit", "CLIPProcessor"),
        ("hubert", "Wav2Vec2Processor"),
        ("idefics", "IdeficsProcessor"),
        ("idefics2", "Idefics2Processor"),
        ("idefics3", "Idefics3Processor"),
        ("instructblip", "InstructBlipProcessor"),
        ("instructblipvideo", "InstructBlipVideoProcessor"),
        ("internvl", "InternVLProcessor"),
        ("janus", "JanusProcessor"),
        ("kosmos-2", "Kosmos2Processor"),
        ("kosmos-2.5", "Kosmos2_5Processor"),
        ("kyutai_speech_to_text", "KyutaiSpeechToTextProcessor"),
        ("layoutlmv2", "LayoutLMv2Processor"),
        ("layoutlmv3", "LayoutLMv3Processor"),
        ("llama4", "Llama4Processor"),
        ("llava", "LlavaProcessor"),
        ("llava_next", "LlavaNextProcessor"),
        ("llava_next_video", "LlavaNextVideoProcessor"),
        ("llava_onevision", "LlavaOnevisionProcessor"),
        ("markuplm", "MarkupLMProcessor"),
        ("mctct", "MCTCTProcessor"),
        ("metaclip_2", "CLIPProcessor"),
        ("mgp-str", "MgpstrProcessor"),
        ("mistral3", "PixtralProcessor"),
        ("mllama", "MllamaProcessor"),
        ("mm-grounding-dino", "GroundingDinoProcessor"),
        ("moonshine", "Wav2Vec2Processor"),
        ("oneformer", "OneFormerProcessor"),
        ("ovis2", "Ovis2Processor"),
        ("owlv2", "Owlv2Processor"),
        ("owlvit", "OwlViTProcessor"),
        ("paligemma", "PaliGemmaProcessor"),
        ("perception_lm", "PerceptionLMProcessor"),
        ("phi4_multimodal", "Phi4MultimodalProcessor"),
        ("pix2struct", "Pix2StructProcessor"),
        ("pixtral", "PixtralProcessor"),
        ("pop2piano", "Pop2PianoProcessor"),
        ("qwen2_5_omni", "Qwen2_5OmniProcessor"),
        ("qwen2_5_vl", "Qwen2_5_VLProcessor"),
        ("qwen2_audio", "Qwen2AudioProcessor"),
        ("qwen2_vl", "Qwen2VLProcessor"),
        ("sam", "SamProcessor"),
        ("sam2", "Sam2Processor"),
        ("sam_hq", "SamHQProcessor"),
        ("seamless_m4t", "SeamlessM4TProcessor"),
        ("sew", "Wav2Vec2Processor"),
        ("sew-d", "Wav2Vec2Processor"),
        ("shieldgemma2", "ShieldGemma2Processor"),
        ("siglip", "SiglipProcessor"),
        ("siglip2", "Siglip2Processor"),
        ("smolvlm", "SmolVLMProcessor"),
        ("speech_to_text", "Speech2TextProcessor"),
        ("speech_to_text_2", "Speech2Text2Processor"),
        ("speecht5", "SpeechT5Processor"),
        ("trocr", "TrOCRProcessor"),
        ("tvlt", "TvltProcessor"),
        ("tvp", "TvpProcessor"),
        ("udop", "UdopProcessor"),
        ("unispeech", "Wav2Vec2Processor"),
        ("unispeech-sat", "Wav2Vec2Processor"),
        ("video_llava", "VideoLlavaProcessor"),
        ("vilt", "ViltProcessor"),
        ("vipllava", "LlavaProcessor"),
        ("vision-text-dual-encoder", "VisionTextDualEncoderProcessor"),
        ("voxtral", "VoxtralProcessor"),
        ("wav2vec2", "Wav2Vec2Processor"),
        ("wav2vec2-bert", "Wav2Vec2Processor"),
        ("wav2vec2-conformer", "Wav2Vec2Processor"),
        ("wavlm", "Wav2Vec2Processor"),
        ("whisper", "WhisperProcessor"),
        ("xclip", "XCLIPProcessor"),
    ]
)

PROCESSOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, PROCESSOR_MAPPING_NAMES)




class AutoProcessor:
    """

        This is a generic processor class that will be instantiated as one of the processor classes of the library when
        created with the [`AutoProcessor.from_pretrained`] class method.

        This class cannot be instantiated directly using `__init__()` (throws an error).

    """

    def __init__(self):
        raise NotImplementedError('This function has been masked for testing')

    @classmethod
    @replace_list_option_in_docstrings(PROCESSOR_MAPPING_NAMES)
    def from_pretrained(
            cls,
            pretrained_model_name_or_path,
            **kwargs
        ):
        """

                Instantiate one of the processor classes of the library from a pretrained model vocabulary.

                The processor class to instantiate is selected based on the `model_type` property of the config object (either
                passed as an argument or loaded from `pretrained_model_name_or_path` if possible):

                List options

                Params:
                    pretrained_model_name_or_path (`str` or `os.PathLike`):
                        This can be either:

                        - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on
                          huggingface.co.
                        - a path to a *directory* containing a processor files saved using the `save_pretrained()` method,
                          e.g., `./my_model_directory/`.
                    cache_dir (`str` or `os.PathLike`, *optional*):
                        Path to a directory in which a downloaded pretrained model feature extractor should be cached if the
                        standard cache should not be used.
                    force_download (`bool`, *optional*, defaults to `False`):
                        Whether or not to force to (re-)download the feature extractor files and override the cached versions
                        if they exist.
                    resume_download:
                        Deprecated and ignored. All downloads are now resumed by default when possible.
                        Will be removed in v5 of Transformers.
                    proxies (`dict[str, str]`, *optional*):
                        A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                        'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
                    token (`str` or *bool*, *optional*):
                        The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                        when running `hf auth login` (stored in `~/.huggingface`).
                    revision (`str`, *optional*, defaults to `"main"`):
                        The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                        git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                        identifier allowed by git.
                    return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                        If `False`, then this function returns just the final feature extractor object. If `True`, then this
                        functions returns a `Tuple(feature_extractor, unused_kwargs)` where *unused_kwargs* is a dictionary
                        consisting of the key/value pairs whose keys are not feature extractor attributes: i.e., the part of
                        `kwargs` which has not been used to update `feature_extractor` and is otherwise ignored.
                    trust_remote_code (`bool`, *optional*, defaults to `False`):
                        Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                        should only be set to `True` for repositories you trust and in which you have read the code, as it will
                        execute code present on the Hub on your local machine.
                    kwargs (`dict[str, Any]`, *optional*):
                        The values in kwargs of any keys which are feature extractor attributes will be used to override the
                        loaded values. Behavior concerning key/value pairs whose keys are *not* feature extractor attributes is
                        controlled by the `return_unused_kwargs` keyword parameter.

                <Tip>

                Passing `token=True` is required when you want to use a private model.

                </Tip>

                Examples:

                ```python
                >>> from transformers import AutoProcessor

                >>> # Download processor from huggingface.co and cache.
                >>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")

                >>> # If processor files are in a directory (e.g. processor was saved using *save_pretrained('./test/saved_model/')*)
                >>> # processor = AutoProcessor.from_pretrained("./test/saved_model/")
                ```
        """
        raise NotImplementedError('This function has been masked for testing')

    @staticmethod
    def register(config_class, processor_class, exist_ok = False):
        """

                Register a new processor for this class.

                Args:
                    config_class ([`PretrainedConfig`]):
                        The configuration corresponding to the model to register.
                    processor_class ([`ProcessorMixin`]): The processor to register.

        """
        raise NotImplementedError('This function has been masked for testing')


__all__ = ["PROCESSOR_MAPPING", "AutoProcessor"]