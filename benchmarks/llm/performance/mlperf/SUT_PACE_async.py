# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content.
# ******************************************************************************
"""MLPerf SUT for Server scenario (online accuracy/performance).

Both PACE and vLLM expose the same OpenAI v1/completions API, so the
payload and SSE streaming format are identical.  The only difference is
that PACE accepts an extra ``mlperf_mode`` flag which switches its
streaming output to raw token-ID lines (no SSE envelope, no JSON),
avoiding redundant tokenize/detokenize round-trips during benchmarking.
"""

import array
import asyncio
import json
import logging
import threading
from typing import Optional

import aiohttp
import numpy as np
import mlperf_loadgen as lg
from transformers import AutoTokenizer

from dataset import Dataset

import uvloop

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# SUT (base)
# -----------------------------------------------------------------------------


class SUT:
    def __init__(
        self,
        model_path=None,
        dtype="bfloat16",
        total_sample_count=13368,
        dataset_path=None,
        endpoint_url="http://localhost:8080/v1/completions",
        timeout_sec=60.0,
        max_tokens=128,
        server_type="pace",
    ):
        self.model_path = model_path or "meta-llama/Llama-3.1-8B-Instruct"
        self.endpoint_url = endpoint_url
        self.timeout_sec = timeout_sec
        self.max_tokens = max_tokens
        self.server_type = server_type.lower()

        if self.server_type not in ("pace", "vllm"):
            raise ValueError(
                f"Invalid server_type: {self.server_type}. Must be 'pace' or 'vllm'"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side="left",
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path,
            dataset_path=self.dataset_path,
            total_sample_count=total_sample_count,
            dtype=dtype,
        )

        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam,
        )

        self.mlperf_mode = self.server_type == "pace"

        self.session: Optional[aiohttp.ClientSession] = None
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        self.loop_thread: Optional[threading.Thread] = None
        self.pending_tasks = []

    def _run_event_loop(self):
        asyncio.set_event_loop(self.event_loop)
        self.event_loop.run_forever()

    def start(self):
        self.event_loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()

        async def _create_session():
            connector = aiohttp.TCPConnector(
                limit=0,
                limit_per_host=0,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=60,
                enable_cleanup_closed=True,
                force_close=False,
            )

            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.timeout_sec),
            )

        future = asyncio.run_coroutine_threadsafe(_create_session(), self.event_loop)
        future.result()

    def stop(self):
        if self.event_loop and self.session:

            async def _cleanup():
                if self.session:
                    await self.session.close()

            future = asyncio.run_coroutine_threadsafe(_cleanup(), self.event_loop)
            future.result()

            self.event_loop.call_soon_threadsafe(self.event_loop.stop)
            if self.loop_thread:
                self.loop_thread.join(timeout=5.0)

    def _format_request_payload(self, prompt_text, stream=False, query_id=None):
        """Build the /v1/completions request body.

        Both PACE and vLLM use the same OpenAI-native fields.
        PACE additionally accepts ``mlperf_mode`` for raw token-ID streaming.
        """
        payload = {
            "model": self.model_path,
            "prompt": prompt_text,
            "stream": stream,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
        }

        if self.mlperf_mode:
            payload["mlperf_mode"] = True

        if query_id is not None and hasattr(self.data_object, "input_lens"):
            payload["input_length"] = self.data_object.input_lens[query_id]

        return payload

    def _tokenize_response(self, text):
        """Encode text to token IDs (fallback for SSE text-based responses)."""
        if not text:
            return []
        return self.tokenizer.encode(text, add_special_tokens=False)

    def flush_queries(self):
        if self.pending_tasks and self.event_loop:
            for task in self.pending_tasks:
                try:
                    task.result()
                except Exception:
                    pass
            self.pending_tasks.clear()


# -----------------------------------------------------------------------------
# SUTServer (Server scenario, streaming)
# -----------------------------------------------------------------------------


class SUTServer(SUT):
    def __init__(
        self,
        model_path=None,
        dtype="bfloat16",
        total_sample_count=13368,
        dataset_path=None,
        endpoint_url="http://localhost:8080/v1/completions",
        timeout_sec=60.0,
        max_tokens=128,
        server_type="pace",
    ):
        super().__init__(
            model_path=model_path,
            dtype=dtype,
            total_sample_count=total_sample_count,
            dataset_path=dataset_path,
            endpoint_url=endpoint_url,
            timeout_sec=timeout_sec,
            max_tokens=max_tokens,
            server_type=server_type,
        )

    async def _make_streaming_request_async(
        self, prompt_text, query_id=None, max_retries=3
    ):
        payload = self._format_request_payload(
            prompt_text, stream=True, query_id=query_id
        )

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
        }

        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self.session.post(
                    self.endpoint_url,
                    json=payload,
                    headers=headers,
                )
                return response
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 0.1 * (2**attempt)
                    await asyncio.sleep(wait_time)
        log.error("Request failed after %d retries: %s", max_retries, last_error)
        raise last_error

    async def _process_server_query_async(self, query_sample):
        query_id = query_sample.index
        prompt_text = self.data_object.source_encoded_input_ids[query_id]

        response = None
        try:
            collected_token_ids = []

            response = await self._make_streaming_request_async(
                prompt_text, query_id=query_id
            )

            if self.mlperf_mode:
                await self._process_mlperf_stream(
                    response, query_sample, collected_token_ids
                )
            else:
                full_response = await self._process_sse_stream(response, query_sample)
                collected_token_ids = self._tokenize_response(full_response)

            n_tokens = len(collected_token_ids)
            response_array = array.array(
                "B", np.array(collected_token_ids, np.int32).tobytes()
            )
            bi = response_array.buffer_info()
            response_obj = [
                lg.QuerySampleResponse(query_sample.id, bi[0], bi[1], n_tokens)
            ]
            lg.QuerySamplesComplete(response_obj)

        except Exception as e:
            log.error("Query %s failed: %s", query_sample.id, e)
            response_array = array.array("B", b"")
            bi = response_array.buffer_info()
            response_obj = [lg.QuerySampleResponse(query_sample.id, bi[0], bi[1], 0)]
            lg.QuerySamplesComplete(response_obj)
        finally:
            if response is not None:
                response.close()

    async def _process_mlperf_stream(self, response, query_sample, collected_token_ids):
        """Process mlperf_mode streaming: raw token-ID lines + [DONE].

        PACE-specific optimisation. Each line is a plain integer token ID,
        terminated by a ``[DONE]`` sentinel. No SSE framing, no JSON.
        """
        first_token_sent = False

        async for chunk in response.content.iter_any():
            if not chunk:
                continue
            for line in chunk.decode("utf-8").split("\n"):
                line = line.strip()
                if not line:
                    continue
                if line == "[DONE]":
                    return
                try:
                    token_id = int(line)
                except ValueError:
                    continue

                collected_token_ids.append(token_id)

                if not first_token_sent:
                    response_data = array.array(
                        "B", np.array([token_id], np.int32).tobytes()
                    )
                    bi = response_data.buffer_info()
                    lg.FirstTokenComplete(
                        [lg.QuerySampleResponse(query_sample.id, bi[0], bi[1])]
                    )
                    first_token_sent = True

    async def _process_sse_stream(self, response, query_sample):
        """Process standard OpenAI SSE streaming (works for both PACE and vLLM).

        Parses ``data: {json}`` lines from the SSE stream, extracts
        ``choices[0].text`` (completions format), accumulates text, and
        returns the full response for post-hoc tokenization.
        """
        first_token_sent = False
        full_response = ""

        async for chunk in response.content.iter_any():
            if not chunk:
                continue
            for line in chunk.decode("utf-8").split("\n"):
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue

                data_part = line[6:]
                if data_part.strip() == "[DONE]":
                    return full_response

                try:
                    chunk_data = json.loads(data_part)
                except json.JSONDecodeError:
                    continue

                choices = chunk_data.get("choices", [])
                if not choices:
                    continue
                token_text = choices[0].get("text", "")

                if token_text and not first_token_sent:
                    first_tokens = self._tokenize_response(token_text)
                    response_data = array.array(
                        "B", np.array(first_tokens, np.int32).tobytes()
                    )
                    bi = response_data.buffer_info()
                    lg.FirstTokenComplete(
                        [lg.QuerySampleResponse(query_sample.id, bi[0], bi[1])]
                    )
                    first_token_sent = True

                full_response += token_text

        return full_response

    def issue_queries(self, query_samples):
        if not self.event_loop or not self.session:
            raise RuntimeError("SUT not started. Call start() first.")
        if len(query_samples) > 1:
            log.warning(
                "issue_queries called with len=%d (expected 1 for Server scenario); processing first sample only",
                len(query_samples),
            )
        task = asyncio.run_coroutine_threadsafe(
            self._process_server_query_async(query_samples[0]), self.event_loop
        )
        self.pending_tasks.append(task)
