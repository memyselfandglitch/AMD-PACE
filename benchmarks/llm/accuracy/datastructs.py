# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    model_name: str
    tokenizer_name: str
    dtype: str
    llm_operators: Optional[dict] = None
    spec_config: Optional[dict] = None


@dataclass
class GenerationArgs:
    batch_size: int
    kv_cache_type: str
    think_end_token: Optional[str] = None
    apply_chat_template: bool = False
    fewshot_as_multiturn: bool = True
    system_instruction: Optional[str] = None
