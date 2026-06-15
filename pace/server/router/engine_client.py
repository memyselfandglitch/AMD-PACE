# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""HTTP client for communicating with PACE engine backends.

Encapsulates all HTTP calls to the engine /step and /remove_sequence endpoints.
Schedulers call into these helpers rather than managing aiohttp sessions
directly, keeping network I/O out of the scheduling logic.

The engine payload uses its native ``generation_config`` format (matching
``SamplingConfig`` kwargs).  Translation from the OpenAI request shape is
done once at the router boundary (``CompletionRequest.to_engine_config()``)
and stored on ``Request.engine_gen_config``.
"""

import time
from typing import AsyncGenerator, Dict, List, Union

import aiohttp

from pace.server.router.utils import Request, RequestStatus, http_config
from pace.utils.logging import PACE_DEBUG, PACE_ERROR, PACE_INFO


def _make_timeout() -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(
        total=http_config.total,
        connect=http_config.connect,
        sock_connect=http_config.sock_connect,
        sock_read=http_config.sock_read,
    )


async def clear_concluded_request(
    req_id: str,
    session: aiohttp.ClientSession,
    engine_url: str,
):
    """Tell the engine to drop a finished / cancelled / errored sequence."""
    try:
        PACE_INFO(f"Removing concluded sequence {req_id} from engine {engine_url}")
        payload = {"sequence_ids": [req_id]}
        async with session.post(
            f"{engine_url}/remove_sequence", json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                PACE_INFO(
                    f"Successfully removed sequence {req_id} from engine: {result}"
                )
            else:
                PACE_ERROR(
                    f"Failed to remove sequence {req_id} from engine: {response.status}"
                )
    except Exception as e:
        PACE_ERROR(f"Error removing sequence {req_id} from engine: {str(e)}")


async def iterative_generate_tokens(
    req: Request,
    active_requests: Dict[str, Request],
    engine_url: str,
) -> AsyncGenerator[Union[List[int], str, None], None]:
    """Run prefill-then-decode loop for a single request (iterative scheduler).

    Yields ``list[int]`` token-ID batches (one per engine step), the
    sentinel ``"ERROR"`` on engine failure, or ``None`` to signal stream
    end.  The router handles all detokenization.
    """
    gen_config = req.engine_gen_config
    prompt = req.req.prompt
    engine_req_id = req.request_id
    assigned_engine_url = req.assigned_engine_url or engine_url

    PACE_INFO(f"Request {engine_req_id} will use engine: {assigned_engine_url}")
    PACE_DEBUG(
        f"Prompt token count (req={engine_req_id}): {len(prompt) if isinstance(prompt, list) else 'N/A'}"
    )

    async with aiohttp.ClientSession(timeout=_make_timeout()) as session:
        is_prefill = True

        while req.status not in (RequestStatus.COMPLETED, RequestStatus.ERROR):
            if engine_req_id not in active_requests:
                PACE_INFO(f"Request {engine_req_id} cancelled, stopping generation")
                break

            try:
                worker_id = getattr(req, "assigned_engine_index", 0)
                if is_prefill:
                    PACE_INFO(
                        f"[WORKER-{worker_id}] Executing PREFILL step for request {engine_req_id}"
                    )
                    payload = {
                        "prefill_batch": [
                            {
                                "is_prefill": True,
                                "prompt": prompt,
                                "req_id": engine_req_id,
                                "generation_config": gen_config,
                            }
                        ]
                    }
                else:
                    PACE_INFO(
                        f"[WORKER-{worker_id}] Executing DECODE step {req.req_stats['generated_tokens_count']} for request {engine_req_id}"
                    )
                    payload = {"is_decode": True}

                async with session.post(
                    f"{assigned_engine_url}/step", json=payload
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    PACE_INFO(f"Received new gen token at {time.time()}")
                    if result.get("status") != "success":
                        PACE_ERROR(f"Engine error (req={engine_req_id}): {result}")
                        req.status = RequestStatus.ERROR
                        yield "ERROR"
                        break

                    if is_prefill:
                        req.req_stats["prefill_finished_at"] = time.time()
                        results = result.get("results", [])
                        result_data = results[0].get("result", {}) if results else {}
                        if result_data == {}:
                            result_data = result.get("results")[0]
                        is_prefill = False
                    else:
                        result_data = result.get("result", {})
                    PACE_DEBUG(f"Result data (req={engine_req_id}): {result_data}")

                    if "error" in result_data:
                        PACE_ERROR(f"Engine error (req={engine_req_id}): {result}")
                        req.status = RequestStatus.ERROR
                        yield "ERROR"
                        break
                    req_result = result_data.get(engine_req_id, {})
                    if "error" in req_result:
                        PACE_ERROR(f"Engine error (req={engine_req_id}): {result}")
                        req.status = RequestStatus.ERROR
                        yield "ERROR"
                        break
                    status = req_result.get("status")
                    token_ids = req_result.get("token_ids", [])

                    req.req_stats["generated_tokens_count"] += len(token_ids)
                    if token_ids:
                        yield token_ids

                    if status == "COMPLETED":
                        req.finish_reason = req_result.get("stop_reason", "stop")
                        break

            except Exception as e:
                PACE_ERROR(
                    f"Error generating token for request {engine_req_id}: {str(e)}"
                )
                req.status = RequestStatus.ERROR
                yield "ERROR"
                break

        await clear_concluded_request(engine_req_id, session, assigned_engine_url)
    yield None


async def perform_prefill(
    req: Request,
    worker_id: int,
    fallback_engine_url: str,
) -> str:
    """Execute the prefill step for a single request. Returns status string.

    Return values: "OK", "COMPLETED", or "ERROR".
    Engine now returns token_id only (no decoded text).
    """
    req.status = RequestStatus.PROCESSING
    PACE_INFO(f"[WORKER-{worker_id}] Performing PREFILL for request {req.request_id}")

    try:
        gen_config = req.engine_gen_config
        prompt = req.req.prompt
        engine_req_id = req.request_id
        assigned_engine_url = req.assigned_engine_url or fallback_engine_url

        PACE_INFO(f"Worker {worker_id} using engine: {assigned_engine_url} for prefill")
        PACE_DEBUG(
            f"Prompt token count for prefill (req={engine_req_id}): {len(prompt) if isinstance(prompt, list) else 'N/A'}"
        )

        async with aiohttp.ClientSession(timeout=_make_timeout()) as session:
            payload = {
                "prefill_batch": [
                    {
                        "is_prefill": True,
                        "prompt": prompt,
                        "req_id": engine_req_id,
                        "generation_config": gen_config,
                    }
                ]
            }

            async with session.post(
                f"{assigned_engine_url}/step", json=payload
            ) as response:
                response.raise_for_status()
                result = await response.json()
                req.req_stats["prefill_finished_at"] = time.time()

                if result.get("status") != "success":
                    PACE_ERROR(f"Server error during prefill: {result}")
                    await clear_concluded_request(
                        req.request_id, session, assigned_engine_url
                    )
                    return "ERROR"

                results = result.get("results", [])
                if not results:
                    PACE_ERROR("Empty results in prefill response")
                    await clear_concluded_request(
                        req.request_id, session, assigned_engine_url
                    )
                    return "ERROR"

                result_data = results[0].get("result", {})
                req_result = result_data.get(engine_req_id, {})

                prefill_status = req_result.get("status", "")
                token_ids = req_result.get("token_ids", [])

                if prefill_status == "COMPLETED":
                    PACE_INFO(f"Request {engine_req_id} fully completed during prefill")
                    req.status = RequestStatus.COMPLETED
                    req.finish_reason = req_result.get("stop_reason", "stop")
                    req.req_stats["finished_at"] = time.time()
                    req.req_stats["generated_tokens_count"] += len(token_ids)
                    if req.token_queue:
                        if token_ids:
                            await req.token_queue.put(token_ids)
                        await req.token_queue.put(None)
                    return "COMPLETED"

                if prefill_status != "PREFILL_COMPLETED":
                    PACE_ERROR(f"Prefill failed, status: {prefill_status}")
                    await clear_concluded_request(
                        req.request_id, session, assigned_engine_url
                    )
                    return "ERROR"

                PACE_INFO(
                    f"[WORKER-{worker_id}] PREFILL successful for request {engine_req_id}"
                )

                if req.token_queue and token_ids:
                    req.req_stats["generated_tokens_count"] += len(token_ids)
                    await req.token_queue.put(token_ids)

                return "OK"

    except Exception as e:
        PACE_INFO(f"Error during prefill for request {req.request_id}: {str(e)}")
        return "ERROR"


async def decode_step(
    session: aiohttp.ClientSession,
    worker_id: int,
    assigned_engine_url: str,
    decode_queue: Dict[str, Request],
    finalize_callback,
) -> bool:
    """Execute a single decode step across all requests in *decode_queue*.

    Returns True on success, False if the entire batch errored.
    *finalize_callback* is called for each completed/errored request.
    """
    num_requests = len(decode_queue)
    PACE_INFO(
        f"[WORKER-{worker_id}] Executing DECODE step for {num_requests} request(s)"
    )

    payload = {"is_decode": True}
    async with session.post(f"{assigned_engine_url}/step", json=payload) as response:
        response.raise_for_status()
        result = await response.json()

        if result.get("status") != "success":
            PACE_DEBUG(f"[WORKER-{worker_id}] Server error during DECODE: {result}")
            for req_id in list(decode_queue):
                req = decode_queue[req_id]
                await finalize_callback(req, req_id, "ERROR", None, worker_id, session)
            return False

        result_data = result.get("result", {})
        PACE_DEBUG(f"[WORKER-{worker_id}] DECODE result data: {result_data}")

        for req_id, decode_result in result_data.items():
            if not isinstance(decode_result, dict):
                continue
            PACE_INFO(
                f"[WORKER-{worker_id}] DECODE step for request {req_id}: {decode_result}"
            )
            status = decode_result.get("status", "DECODING_IN_PROGRESS")
            token_ids = decode_result.get("token_ids", [])

            if req_id in decode_queue:
                req = decode_queue[req_id]
                req.req_stats["generated_tokens_count"] += len(token_ids)
                if token_ids:
                    await req.token_queue.put(token_ids)

                if status in ("COMPLETED", "ERROR"):
                    if status == "COMPLETED":
                        req.finish_reason = decode_result.get("stop_reason", "stop")
                    final_token = None if status == "COMPLETED" else "ERROR"
                    await finalize_callback(
                        req, req_id, status, final_token, worker_id, session
                    )
    return True
