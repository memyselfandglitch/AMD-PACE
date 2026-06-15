# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""OpenAI-compatible API routes for the PACE router.

Each public function is a FastAPI route handler.  They are registered on the
app by frontend.py at import time via ``APIRouter``.  The scheduler, args,
and tokenizer references are injected at startup through ``set_dependencies``.

Multi-prompt requests (``prompt`` is a list) are fanned out into separate
internal ``Request`` objects, each submitted independently to the scheduler.
Responses are merged back into a single OpenAI ``CompletionResponse`` (or
interleaved SSE stream) with correct ``choices[].index`` values.
"""

import asyncio
import uuid
from typing import List, Tuple

import aiohttp
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from pace.server.router.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    RequestStatusResponse,
    UsageInfo,
)
from pace.server.router.streaming import (
    active_streams,
    merge_stream_generators,
    stream_generator,
)
from pace.server.router.tokenizer_utils import (
    decode_token_ids,
    init_tokenizer,
    normalize_prompts,
    truncate_at_stop_string,
)
from pace.server.router.utils import Request, http_config
from pace.utils.logging import PACE_INFO, PACE_WARNING

router = APIRouter()

_scheduler = None
_args = None
_tokenizer = None


def set_dependencies(scheduler, args, tokenizer=None):
    """Inject runtime dependencies after startup. Called once from frontend.py."""
    global _scheduler, _args, _tokenizer
    _scheduler = scheduler
    _args = args
    _tokenizer = tokenizer
    if tokenizer is not None:
        init_tokenizer(tokenizer)


@router.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Handle OpenAI-compatible completion requests with multi-prompt support."""
    if _args.model.lower() != request.model.lower():
        PACE_WARNING(
            f"Requested model {request.model} differs from configured model {_args.model}"
        )
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"The model `{request.model}` does not exist or is not available.",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )

    prompt_token_lists = normalize_prompts(request.prompt, _tokenizer)

    if request.mlperf_mode and len(prompt_token_lists) > 1:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "mlperf_mode only supports single-prompt requests.",
                    "type": "invalid_request_error",
                    "param": "prompt",
                    "code": "invalid_value",
                }
            },
        )

    engine_gen_config = request.to_engine_config()
    group_id = str(uuid.uuid4())

    sub_requests: List[Tuple[int, Request]] = []
    for idx, token_ids in enumerate(prompt_token_lists):
        token_queue: asyncio.Queue = asyncio.Queue()
        sub_req = Request(
            req=request,
            token_queue=token_queue,
            engine_gen_config=engine_gen_config,
            prompt_index=idx,
            group_id=group_id,
        )
        sub_req.req = request.model_copy(update={"prompt": token_ids})
        sub_req.req_stats.input_length = len(token_ids)

        sub_requests.append((idx, sub_req))

    PACE_INFO(
        f"Received request group {group_id} with {len(prompt_token_lists)} prompt(s), "
        f"stream={request.stream}, mlperf_mode={request.mlperf_mode}"
    )

    if request.stream:
        return await _handle_streaming(sub_requests, request, group_id)

    return await _handle_non_streaming(sub_requests, request, group_id)


async def _handle_streaming(
    sub_requests: List[Tuple[int, Request]],
    openai_req: CompletionRequest,
    group_id: str,
):
    """Set up interleaved SSE streams for all sub-requests."""
    completion_id = f"cmpl-{group_id}"
    mlperf_mode = openai_req.mlperf_mode

    if mlperf_mode:
        idx, req = sub_requests[0]
        PACE_INFO(f"Processing MLPerf streaming request {req.request_id}")
        active_streams[req.request_id] = req.token_queue
        await _scheduler.submit_request(req)
        return StreamingResponse(
            stream_generator(
                request_id=req.request_id,
                model=openai_req.model,
                completion_id=completion_id,
                index=0,
                mlperf_mode=True,
                req=req,
            ),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": req.request_id,
            },
        )

    generators = []
    for idx, req in sub_requests:
        PACE_INFO(f"Processing streaming sub-request {req.request_id} (index={idx})")
        active_streams[req.request_id] = req.token_queue
        await _scheduler.submit_request(req)
        generators.append(
            stream_generator(
                request_id=req.request_id,
                model=openai_req.model,
                completion_id=completion_id,
                index=idx,
                mlperf_mode=False,
                req=req,
            )
        )

    return StreamingResponse(
        merge_stream_generators(generators),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-ID": group_id,
        },
    )


async def _handle_non_streaming(
    sub_requests: List[Tuple[int, Request]],
    openai_req: CompletionRequest,
    group_id: str,
):
    """Submit all sub-requests, collect token IDs, detokenize, build CompletionResponse."""
    for _, req in sub_requests:
        await _scheduler.submit_request(req)

    async def _collect_token_ids(req: Request) -> List[int]:
        """Drain the token queue; items are token-ID lists or single ints."""
        token_ids: List[int] = []
        while True:
            item = await req.token_queue.get()
            if item is None:
                break
            if item == "ERROR":
                raise RuntimeError(f"Engine error processing request {req.request_id}")
            if isinstance(item, list):
                token_ids.extend(item)
            elif isinstance(item, int):
                token_ids.append(item)
        return token_ids

    tasks = [asyncio.create_task(_collect_token_ids(req)) for _, req in sub_requests]

    try:
        results = await asyncio.gather(*tasks)
    except RuntimeError as e:
        PACE_WARNING(f"Request group {group_id} failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": "server_error",
                }
            },
        )

    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for (idx, req), token_ids in zip(sub_requests, results):
        if req.mlperf_mode:
            completion_text = " ".join(str(tid) for tid in token_ids)
        else:
            completion_text = decode_token_ids(token_ids, tokenizer=_tokenizer)
            completion_text = truncate_at_stop_string(completion_text, req.stop_strings)

        if req.echo:
            prompt_ids = req.req.prompt
            if isinstance(prompt_ids, list):
                prompt_text = decode_token_ids(prompt_ids, tokenizer=_tokenizer)
            else:
                prompt_text = str(prompt_ids)
            completion_text = prompt_text + completion_text

        if req.suffix:
            completion_text = completion_text + req.suffix

        choices.append(
            CompletionResponseChoice(
                index=idx,
                text=completion_text,
                logprobs=None,
                finish_reason=req.finish_reason,
            )
        )
        total_prompt_tokens += req.req_stats.input_length
        total_completion_tokens += len(token_ids)

    response = CompletionResponse(
        id=f"cmpl-{group_id}",
        model=openai_req.model,
        choices=choices,
        usage=UsageInfo(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        ),
    )
    return JSONResponse(response.model_dump())


@router.get("/v1/health")
async def health_check():
    """Health check endpoint for the frontend service."""
    return JSONResponse(
        {
            "status": "healthy",
            "service": "frontend",
            "scheduler_running": _scheduler.is_running,
            "queue_size": _scheduler.get_queue_size(),
            "active_requests": _scheduler.get_active_requests_count(),
            "server_metrics_enabled": _scheduler.scheduler_metrics_enabled,
        }
    )


@router.get("/v1/status/{request_id}")
async def get_request_status(request_id: str):
    """Get the status of a specific request by its request_id."""
    status = await _scheduler.get_request_status(request_id)
    if status:
        return RequestStatusResponse(
            request_id=request_id,
            status=status["status"],
            message=status.get("message"),
            created_at=status.get("created_at"),
        )

    gid = request_id.removeprefix("cmpl-")
    matches = [
        req
        for req in _scheduler.active_requests.values()
        if getattr(req, "group_id", None) == gid
    ]
    if matches:
        return JSONResponse(
            {
                "id": f"cmpl-{gid}",
                "statuses": [
                    {"index": req.prompt_index, "status": str(req.status.name)}
                    for req in sorted(matches, key=lambda r: r.prompt_index)
                ],
            }
        )
    raise HTTPException(status_code=404, detail="Request not found")


@router.get("/v1/queue/status")
async def get_queue_status():
    """Get the current queue status from the scheduler."""
    return {
        "queue_size": _scheduler.get_queue_size(),
        "active_requests": _scheduler.get_active_requests_count(),
    }


@router.get("/v1/server_metrics")
async def get_server_wide_metrics():
    """Get aggregate server metrics from the scheduler."""
    return _scheduler.server_metrics()


@router.get("/v1/models")
async def get_models():
    """Proxy endpoint to retrieve available models from the backend."""
    url = f"http://{_args.server_host}:{_args.server_port}/get_models"
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=http_config.total,
                connect=http_config.connect,
                sock_connect=http_config.sock_connect,
                sock_read=http_config.sock_read,
            )
        ) as client:
            async with client.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Backend error: {error_text}",
                    )
    except aiohttp.ClientError as e:
        raise HTTPException(status_code=502, detail="Failed to fetch models") from e
