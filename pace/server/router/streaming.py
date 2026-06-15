# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""SSE and plain-text streaming helpers for token delivery to clients.

Owns the per-request token queues (active_streams) and the async generator
that drains them.  Streaming chunks are formatted as OpenAI-compatible
``CompletionStreamResponse`` objects.

The engine now returns only token IDs.  This module detokenizes them
before sending text to the client (unless ``mlperf_mode`` is active).

For multi-prompt requests, ``merge_stream_generators`` multiplexes N
single-prompt generators into one interleaved SSE stream with correct
``choices[].index`` values.
"""

import asyncio
import json
import time
from typing import AsyncGenerator, Dict, List, Optional

from pace.server.router.tokenizer_utils import decode_token_ids
from pace.utils.logging import PACE_DEBUG, PACE_ERROR, PACE_INFO

active_streams: Dict[str, asyncio.Queue] = {}

_scheduler_ref = None


def set_scheduler(scheduler):
    """Called once at startup to wire the scheduler into the streaming layer."""
    global _scheduler_ref
    _scheduler_ref = scheduler


def _make_stream_chunk(
    completion_id: str,
    model: str,
    text: str,
    index: int = 0,
    finish_reason: Optional[str] = None,
    created: Optional[int] = None,
) -> str:
    """Build a single OpenAI-format CompletionStreamResponse JSON string."""
    chunk = {
        "id": completion_id,
        "object": "text_completion",
        "created": created or int(time.time()),
        "model": model,
        "choices": [
            {
                "index": index,
                "text": text,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }
    return json.dumps(chunk)


async def stream_generator(
    request_id: str,
    model: str,
    completion_id: str,
    index: int = 0,
    mlperf_mode: bool = False,
    req: Optional[object] = None,
) -> AsyncGenerator[str, None]:
    """Drain the token queue for *request_id* and yield SSE / plain-text chunks.

    ``index`` is the choice index for this prompt within a multi-prompt batch.
    The engine sends only token IDs (int).  In normal mode the router
    detokenizes them; in MLPerf mode raw IDs are streamed as plain text.
    """
    queue: Optional[asyncio.Queue] = active_streams.get(request_id)
    created = int(time.time())

    if not queue:
        if mlperf_mode:
            yield "ERROR\n"
        else:
            yield f"data: {json.dumps({'error': 'Stream not found'})}\n\n"
        return
    try:
        while True:
            token_data = await queue.get()
            PACE_DEBUG(f"Received token data for {request_id}: {token_data}")

            if token_data is None:
                if mlperf_mode:
                    yield "[DONE]\n"
                else:
                    finish = getattr(req, "finish_reason", "stop") if req else "stop"
                    chunk = _make_stream_chunk(
                        completion_id,
                        model,
                        "",
                        index=index,
                        finish_reason=finish,
                        created=created,
                    )
                    yield f"data: {chunk}\n\n"
                break

            if token_data == "ERROR":
                if mlperf_mode:
                    yield "ERROR\n"
                else:
                    yield f"data: {json.dumps({'error': 'An error occurred during processing'})}\n\n"
                break

            if mlperf_mode:
                yield " ".join(str(tid) for tid in token_data) + "\n"
            else:
                text = decode_token_ids(token_data)
                chunk = _make_stream_chunk(
                    completion_id,
                    model,
                    text,
                    index=index,
                    created=created,
                )
                yield f"data: {chunk}\n\n"
    except asyncio.CancelledError:
        PACE_INFO(f"Stream for request {request_id} cancelled by client")
        if _scheduler_ref is not None:
            _scheduler_ref.cancelled_requests.append(request_id)
            if request_id in _scheduler_ref.active_requests:
                del _scheduler_ref.active_requests[request_id]
    except Exception as e:
        PACE_ERROR(f"Error in stream generator for {request_id}: {str(e)}")
        if mlperf_mode:
            yield "ERROR\n"
        else:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        active_streams.pop(request_id, None)


async def merge_stream_generators(
    generators: List[AsyncGenerator[str, None]],
) -> AsyncGenerator[str, None]:
    """Multiplex N SSE stream generators into one interleaved stream.

    Yields chunks from whichever generator produces output first (like
    vLLM's ``merge_async_iterators``).  Sends ``data: [DONE]`` once all
    generators are exhausted.
    """
    if len(generators) == 1:
        async for chunk in generators[0]:
            yield chunk
        yield "data: [DONE]\n\n"
        return

    loop = asyncio.get_running_loop()
    pending = {loop.create_task(_anext_or_none(gen)): gen for gen in generators}

    try:
        while pending:
            done, _ = await asyncio.wait(
                pending.keys(), return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                gen = pending.pop(task)
                result = task.result()
                if result is None:
                    continue
                yield result
                pending[loop.create_task(_anext_or_none(gen))] = gen
    except asyncio.CancelledError:
        for task in pending:
            task.cancel()
        raise

    yield "data: [DONE]\n\n"


async def _anext_or_none(gen: AsyncGenerator) -> Optional[str]:
    """Advance *gen* by one step; return ``None`` when exhausted."""
    try:
        return await gen.__anext__()
    except StopAsyncIteration:
        return None
