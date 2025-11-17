# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""
Processor class for Pix2Struct.
"""

from typing import Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput
from ...utils import logging






logger = logging.get_logger(__name__)


class Pix2StructProcessor(ProcessorMixin):
    """

        Constructs a PIX2STRUCT processor which wraps a BERT tokenizer and PIX2STRUCT image processor into a single
        processor.

        [`Pix2StructProcessor`] offers all the functionalities of [`Pix2StructImageProcessor`] and [`T5TokenizerFast`]. See
        the docstring of [`~Pix2StructProcessor.__call__`] and [`~Pix2StructProcessor.decode`] for more information.

        Args:
            image_processor (`Pix2StructImageProcessor`):
                An instance of [`Pix2StructImageProcessor`]. The image processor is a required input.
            tokenizer (Union[`T5TokenizerFast`, `T5Tokenizer`]):
                An instance of ['T5TokenizerFast`] or ['T5Tokenizer`]. The tokenizer is a required input.

    """

    attributes = ['image_processor', 'tokenizer']
    image_processor_class = "Pix2StructImageProcessor"
    tokenizer_class = ('T5Tokenizer', 'T5TokenizerFast')

    def __init__(self, image_processor, tokenizer):
        raise NotImplementedError('This function has been masked for testing')

    def __call__(
            self,
            images = None,
            text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
            audio = None,
            videos = None,
            **kwargs: Unpack[Pix2StructProcessorKwargs]
        ) -> Union[BatchEncoding, BatchFeature]:
        """

                This method uses [`Pix2StructImageProcessor.preprocess`] method to prepare image(s) for the model, and
                [`T5TokenizerFast.__call__`] to prepare text for the model.

                Please refer to the docstring of the above two methods for more information.

        """
        raise NotImplementedError('This function has been masked for testing')

    @property
    def model_input_names(self):
        raise NotImplementedError('This function has been masked for testing')


__all__ = ["Pix2StructProcessor"]