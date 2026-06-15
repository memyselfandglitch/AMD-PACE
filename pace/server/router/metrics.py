# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Request-lifecycle metrics and server-wide statistics for the scheduler.

Owns TTFT/TPOT tracking, interval merging, and the periodic metrics loop.
The Scheduler base class delegates all stats bookkeeping here.
"""

import asyncio
from typing import Dict, List, Tuple

from pace.server.router.utils import Request, ttft_histogram, tpot_histogram
from pace.utils.logging import PACE_INFO, PACE_WARNING


class SchedulerMetrics:
    """Tracks per-request and session-wide inference metrics."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.ttft_intervals: List[Tuple[float, float]] = []
        self.processing_intervals: List[Tuple[float, float]] = []
        self.active_ttft_time: float = 0.0
        self.active_processing_time: float = 0.0
        self.session_ttft: float = 0.0
        self.session_tpot: float = 0.0
        self.session_requests_served_per_second: float = 0.0
        self.total_generated_tokens: int = 0
        self.active_time: float = 0.0
        self.ttft_active_time: float = 0.0
        self.total_completed_requests: int = 0

    async def cleanup_request(
        self,
        req: Request,
        active_requests: Dict[str, Request],
    ):
        """Finalize a completed request: remove from active set, compute stats."""
        request_id = req.request_id
        if request_id not in active_requests:
            return

        del active_requests[request_id]
        finished_at = req.req_stats.get("finished_at")
        prefill_finished_at = req.req_stats.get("prefill_finished_at")
        created_at = req.req_stats.get("created_at")
        gen_tokens = max(0, req.req_stats.get("generated_tokens_count", 0))

        PACE_INFO(f"Request {request_id} finished at {finished_at:.6f}")

        if gen_tokens > 0:
            req.req_stats["TPOT"] = (finished_at - created_at) / gen_tokens
        if prefill_finished_at:
            req.req_stats["TTFT"] = prefill_finished_at - created_at

        if req.req_stats["TTFT"]:
            PACE_INFO(
                f"Time to first token (TTFT) for request {request_id}: "
                f"{req.req_stats['TTFT']:.6f}s"
            )
            ttft_histogram.observe(req.req_stats["TTFT"])
        if req.req_stats["TPOT"]:
            PACE_INFO(
                f"Per-token time (TPOT) for request {request_id}: "
                f"{req.req_stats['TPOT']:.6f}s"
            )
            tpot_histogram.observe(req.req_stats["TPOT"])

        PACE_INFO(f"scheduler metrics enabled: {self.enabled}")
        if self.enabled:
            PACE_INFO("Updating scheduler metrics...")
            if req.req_stats["TTFT"] and prefill_finished_at:
                self.ttft_intervals.append((created_at, prefill_finished_at))
            if gen_tokens > 0:
                self.processing_intervals.append((created_at, finished_at))
            self.active_ttft_time = self._calculate_active_time(self.ttft_intervals)
            self.active_processing_time = self._calculate_active_time(
                self.processing_intervals
            )

        self.total_generated_tokens += gen_tokens
        self.total_completed_requests += 1
        PACE_INFO(
            f"Request spent {req.req_stats['end_wait_time'] - req.req_stats['created_at']} seconds Waiting."
        )
        PACE_INFO(f"request generated {gen_tokens} tokens")
        PACE_INFO(f"Cleaned up request {request_id}")

    async def server_metrics_loop(self, is_running_fn, get_queue_size_fn):
        """Periodically compute server session metrics based on completed requests."""
        while is_running_fn():
            try:
                await asyncio.sleep(30)
                if get_queue_size_fn() == 0:
                    PACE_INFO("Calculating server metrics...")
                    self._calculate_server_metrics()
                    self.ttft_intervals.clear()
                    self.processing_intervals.clear()
            except asyncio.CancelledError:
                break
            except Exception as e:
                PACE_WARNING(f"Error in server metrics loop: {str(e)}")

    def _calculate_server_metrics(self):
        """Compute active-time based TTFT and TPOT across completed requests."""
        self.ttft_active_time = self.ttft_active_time + (
            self._calculate_active_time(self.ttft_intervals)
            if self.ttft_intervals
            else 0.0
        )
        self.session_ttft = (
            self.ttft_active_time / self.total_completed_requests
            if self.total_completed_requests
            else 0.0
        )

        self.active_time = self.active_time + (
            self._calculate_active_time(self.processing_intervals)
            if self.processing_intervals
            else 0.0
        )
        self.session_tpot = (
            (self.active_time / self.total_generated_tokens)
            if self.total_generated_tokens > 0
            else 0.0
        )

        self.session_requests_served_per_second = (
            self.total_completed_requests / self.active_time
            if self.active_time > 0
            else 0.0
        )

        PACE_INFO(
            f"Updated server metrics: session_ttft={self.session_ttft:.6f}, "
            f"session_tpot={self.session_tpot:.6f}"
        )

    def _calculate_active_time(self, intervals: List[Tuple[float, float]]) -> float:
        """Merge overlapping intervals and return cumulative active time."""
        if not intervals:
            return 0.0

        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged = [sorted_intervals[0]]

        for current_start, current_end in sorted_intervals[1:]:
            last_start, last_end = merged[-1]
            if current_start <= last_end:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))

        return sum(end - start for start, end in merged)

    def server_metrics_snapshot(self) -> Dict[str, float]:
        """Return current server session metrics."""
        return {
            "sched_session_ttft": self.session_ttft,
            "sched_session_tpot": self.session_tpot,
            "sched_active_ttft_time": self.active_ttft_time,
            "sched_requests_served_per_second": self.session_requests_served_per_second,
            "sched_total_generated_tokens": self.total_generated_tokens,
        }
