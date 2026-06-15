# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-Generated code.
# ******************************************************************************

"""Internal data structures, metrics histograms, and HTTP configuration.

Protocol-level models (OpenAI request/response shapes) live in ``protocol.py``.
This module owns ``Request`` (the internal envelope that wraps a protocol
request for scheduling), ``RequestStats``, ``RequestStatus``, and the
Prometheus histogram definitions.
"""

import os
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from prometheus_client import Histogram

from pace.server.router.protocol import CompletionRequest  # noqa: F401 - re-export


TTFT_BUCKETS = [
    0.1,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    7.5,
    10.0,
    15.0,
    20.0,
    30.0,
    60.0,
]
TPOT_BUCKETS = [
    0.05,
    0.075,
    0.1,
    0.125,
    0.15,
    0.175,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.5,
    0.6,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
]

ttft_histogram = Histogram(
    "pace_ttft_seconds",
    "Time To First Token (seconds)",
    buckets=TTFT_BUCKETS,
)

tpot_histogram = Histogram(
    "pace_tpot_seconds",
    "Time Per Output Token (seconds)",
    buckets=TPOT_BUCKETS,
)


class RequestStatus(str, Enum):
    """Enum for request status values."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class RequestStats:
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    generated_tokens_count: int = 0
    input_length: int = 0
    prefill_finished_at: Optional[float] = None
    TTFT: Optional[float] = None
    TPOT: Optional[float] = None
    end_wait_time: Optional[float] = None

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Mimic dict.get(): safely get attribute or default"""
        return getattr(self, key, default)


class Request:
    """Internal request envelope used by scheduler and engine_client.

    Wraps the OpenAI ``CompletionRequest`` from ``protocol.py`` and holds
    scheduling state, token queue, pre-translated engine config, and
    per-request statistics.
    """

    def __init__(
        self,
        req: CompletionRequest,
        token_queue: asyncio.Queue,
        engine_gen_config: Optional[Dict[str, Any]] = None,
        prompt_index: int = 0,
        group_id: Optional[str] = None,
    ):
        self.req: CompletionRequest = req
        self.request_id: str = str(uuid.uuid4())
        self.token_queue: asyncio.Queue = token_queue
        self.status: RequestStatus = RequestStatus.QUEUED
        self.assigned_engine_index: int = 0
        self.assigned_engine_url: Optional[str] = None
        self.response_queue: asyncio.Queue = asyncio.Queue()
        self.batch_submit_time: Optional[float] = None
        self.priority: str = "normal"
        self.req_stats = RequestStats()

        self.prompt_index: int = prompt_index
        self.group_id: Optional[str] = group_id

        if engine_gen_config is not None:
            self.engine_gen_config = engine_gen_config
        else:
            self.engine_gen_config = req.to_engine_config()

        self.mlperf_mode: bool = getattr(req, "mlperf_mode", False)
        self.echo: bool = getattr(req, "echo", False) or False
        self.suffix: Optional[str] = getattr(req, "suffix", None)
        self.finish_reason: str = "stop"

        stop = getattr(req, "stop", None)
        if isinstance(stop, str):
            self.stop_strings: List[str] = [stop]
        elif isinstance(stop, list):
            self.stop_strings: List[str] = list(stop)
        else:
            self.stop_strings: List[str] = []


class HTTPConfig:
    """HTTP timeout configuration for engine requests.

    Defaults optimized for LLM inference: total=300s, connect/sock_connect/sock_read=30s.
    Override via HTTP_TIMEOUT_* environment variables.
    """

    def __init__(self):
        self.total = float(os.environ.get("HTTP_TIMEOUT_TOTAL", "300"))
        self.connect = float(os.environ.get("HTTP_TIMEOUT_CONNECT", "30"))
        self.sock_connect = float(os.environ.get("HTTP_TIMEOUT_SOCK_CONNECT", "30"))
        self.sock_read = float(os.environ.get("HTTP_TIMEOUT_SOCK_READ", "30"))

        for name, value in vars(self).items():
            if value <= 0:
                raise ValueError(f"{name} timeout must be positive, got {value}")


http_config = HTTPConfig()
