# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
""" Customized (Adam H.) SpecTUS model configuration"""

from __future__ import annotations
from typing import Optional
from transformers import BartConfig

class SpectusConfig(BartConfig):
    """
        separate_encoder_decoder_embeds (`bool`, defaults to `False` - Adam's customization):
            If True, separate embeddings are used for encoder and decoder.
            If False, encoder and decoder share embedding matrix - partly overlapping.
            If None, encoder and decoder share embedding matrix - for older models compatibility.
        max_log_id (`int`, defaults to `None` - Adam's customizaiton):
            If not None, positional embeddings up to this value can be trained (if provided) and summed with standard embeddings.
        max_mz (`int`, defaults to None - Adam's customization):
            If not None, embeddings up to this value can be trained (if provided) and summed with positional embeddings.
    """
    def __init__(
        self,
        separate_encoder_decoder_embeds: bool | None = None,
        max_log_id: int | None = None,
        max_mz: int | None = None,
        decoder_max_position_embeddings: int = 200,
        encoder_max_position_embeddings: int | None = None,
        **kwargs):

        self.max_log_id=max_log_id
        self.max_mz=max_mz
        self.separate_encoder_decoder_embeds=separate_encoder_decoder_embeds
        self.decoder_max_position_embeddings=decoder_max_position_embeddings
        self.encoder_max_position_embeddings=encoder_max_position_embeddings
        super().__init__(
            **kwargs,
        )