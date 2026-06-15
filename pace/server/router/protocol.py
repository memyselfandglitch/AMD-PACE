# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""OpenAI-compatible protocol models for the PACE v1/completions API.

Defines request and response Pydantic models that match the OpenAI
completions specification (https://platform.openai.com/docs/api-reference/completions).
The router accepts these native OpenAI shapes and translates them into the
engine's internal ``generation_config`` dict via ``to_engine_config()``.
"""

import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from pace.utils.logging import PACE_WARNING


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class CompletionRequest(BaseModel):
    """OpenAI v1/completions request body."""

    # --- Standard OpenAI fields ---
    model: str
    prompt: Union[str, List[str], List[int], List[List[int]]]
    best_of: Optional[int] = None
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[int] = Field(default=None, ge=0, le=5)
    max_tokens: Optional[int] = Field(default=16, ge=1)
    n: Optional[int] = Field(default=1, ge=1)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    suffix: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    user: Optional[str] = None

    # --- PACE extensions (not in OpenAI spec) ---
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    min_tokens: Optional[int] = None
    repetition_penalty: Optional[float] = None
    ignore_eos: Optional[bool] = None
    do_sample: Optional[bool] = None
    mlperf_mode: bool = False

    @model_validator(mode="before")
    @classmethod
    def validate_and_normalize(cls, data):
        if not isinstance(data, dict):
            return data

        if data.get("best_of") is not None and data["best_of"] != 1:
            raise ValueError(
                "best_of is deprecated by OpenAI and not supported by PACE. "
                "Remove 'best_of' from your request."
            )
        n = data.get("n")
        if n is not None and n != 1:
            raise ValueError(
                "n > 1 is not currently supported. The PACE engine generates "
                "one completion per request. Submit separate requests with "
                "different seeds to obtain multiple completions."
            )

        return data

    def to_engine_config(self) -> Optional[Dict[str, Any]]:
        """Translate OpenAI request fields into the engine's generation_config dict.

        The returned dict uses key names matching ``SamplingConfig.__init__``
        in ``pace/llm/configs.py``.  The engine server is never modified.

        Sampling mode inference (critical for performance):
          PACE uses ``do_sample`` as the primary greedy/random switch, while
          the OpenAI API uses ``temperature``.  When ``do_sample`` is not
          explicitly provided, we infer it from the OpenAI fields:
            - temperature == 0 or temperature is None  ->  do_sample=False (greedy)
            - temperature > 0                          ->  do_sample=True  (random)
          This ensures that ``openai.completions.create(temperature=0)``
          produces the same greedy behavior as vLLM bench defaults.
        """
        cfg: Dict[str, Any] = {}

        if self.max_tokens is not None:
            cfg["max_new_tokens"] = self.max_tokens
        if self.min_tokens is not None:
            cfg["min_new_tokens"] = self.min_tokens
        if self.temperature is not None:
            cfg["temperature"] = self.temperature
        if self.top_p is not None:
            cfg["top_p"] = self.top_p
        if self.top_k is not None:
            cfg["top_k"] = self.top_k
        if self.min_p is not None:
            cfg["min_p"] = self.min_p
        if self.seed is not None:
            cfg["seed"] = self.seed
        # stop_strings are handled by the router, not the engine
        if self.frequency_penalty is not None and self.frequency_penalty != 0.0:
            cfg["frequency_penalty"] = self.frequency_penalty
        if self.repetition_penalty is not None:
            cfg["repetition_penalty"] = self.repetition_penalty
        if self.ignore_eos is not None:
            cfg["ignore_eos"] = self.ignore_eos

        # Infer do_sample from OpenAI semantics when not explicitly set.
        # PACE defaults do_sample=False (greedy), but OpenAI/vLLM treat
        # temperature > 0 as random sampling.  We bridge the gap here.
        if self.do_sample is not None:
            cfg["do_sample"] = self.do_sample
        else:
            temp = self.temperature
            if temp is not None and temp > 0:
                cfg["do_sample"] = True
            else:
                # temperature=0, temperature=None, or omitted -> greedy
                cfg["do_sample"] = False

        if self.presence_penalty is not None and self.presence_penalty != 0.0:
            PACE_WARNING(
                "presence_penalty is accepted but has no effect on the PACE engine."
            )
        if self.logit_bias is not None:
            PACE_WARNING("logit_bias is accepted but has no effect on the PACE engine.")

        return cfg if cfg else None


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    """Non-streaming response matching the OpenAI text_completion object."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4()}")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


class CompletionStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class CompletionStreamResponse(BaseModel):
    """Single SSE chunk matching the OpenAI streaming text_completion object."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4()}")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionStreamChoice]


# ---------------------------------------------------------------------------
# Internal status response (for /v1/status/{request_id} -- PACE-specific)
# ---------------------------------------------------------------------------


class RequestStatusResponse(BaseModel):
    request_id: str
    status: str
    message: Optional[str] = None
    created_at: Optional[str] = None
