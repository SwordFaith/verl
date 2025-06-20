# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

import json
import threading
import time
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, Tuple, TypeVar
from uuid import uuid4

import ray
import ray.actor

from verl.tools.utils.search_r1_like_utils import perform_single_search_batch

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

T = TypeVar("T")


# Adapted from verl/tools/sandbox_fusion_tools.py
class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class SearchExecutionWorker:
    """Worker for executing search operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """Execute function with optional rate limiting."""
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception:
                    # TODO we should make this available to the tool caller
                    # Error logging is handled by BaseTool universal logging
                    pass
        else:
            return fn(*fn_args, **fn_kwargs)


def init_search_execution_pool(num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode):
    """Initialize search execution pool."""
    if mode == PoolMode.ThreadMode:
        return ray.remote(SearchExecutionWorker).options(max_concurrency=num_workers).remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class SearchTool(BaseTool):
    """Search tool for retrieving information using external retrieval services.

    This tool provides search functionality with rate limiting and concurrent execution
    support through Ray. It integrates with external retrieval services to perform
    semantic search operations.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the search tool
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize SearchTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Example tool_schema:
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Searches for relevant information based on queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_list": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of search queries"
                            }
                        },
                        "required": ["query_list"]
                    }
                }
            }
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 120)
        self.rate_limit = config.get("rate_limit", 120)
        self.timeout = config.get("timeout", 30)

        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_search_execution_pool(num_workers=self.num_workers, enable_global_rate_limit=self.enable_global_rate_limit, rate_limit=self.rate_limit, mode=PoolMode.ThreadMode)

        # Retrieval service configuration
        self.retrieval_service_url = config.get("retrieval_service_url")
        assert self.retrieval_service_url, "Configuration must include 'retrieval_service_url'"
        self.topk = config.get("topk", 3)
        if self.retrieval_service_url == "":
            raise ValueError("retrieval_service_url is not set")

        if self.tool_logger:
            self.tool_logger.info(f"Initialized SearchTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id

    def execute_search(self, instance_id: str, query_list: list, retrieval_service_url: str, topk: int, timeout: int):
        """Execute search operation using retrieval service.

        Args:
            instance_id: Tool instance ID
            query_list: List of search queries
            retrieval_service_url: URL of the retrieval service
            topk: Number of top results to return
            timeout: Request timeout in seconds

        Returns:
            Tuple of (result_text, metadata)
        """
        result_text, metadata = perform_single_search_batch(
            retrieval_service_url=retrieval_service_url,
            query_list=query_list,
            topk=topk,
            concurrent_semaphore=None,  # Ray handles concurrency control
            timeout=timeout,
        )
        if self.tool_logger:
            self.tool_logger.debug(f"Search result for instance {instance_id}: {result_text}")
        return result_text, metadata

    async def _execute_impl(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, bool, dict[str, Any]]:
        """Execute the search tool with enhanced metrics collection.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query_list and optional timeout

        Returns: tool_response, tool_reward_score, success, tool_specific_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            success: Whether the execution was successful.
            tool_specific_metrics: Enhanced search performance metrics.
        """
        timeout = self.timeout
        query_list_from_params = parameters.get("query_list")

        if not query_list_from_params or not isinstance(query_list_from_params, list):
            error_msg = "Error: 'query_list' is missing, empty, or not a list in parameters."
            if self.tool_logger:
                self.tool_logger.error(f"[SearchTool] {error_msg} Received parameters: {parameters}")

            # Enhanced error metrics
            specific_metrics = {"error_type": "invalid_parameters", "query_count": 0, "total_results": 0, "search_latency_ms": 0.0, "api_status": "parameter_error", "queries": [], "query_complexity_scores": [], "relevance_scores": []}
            return json.dumps({"result": error_msg}), 0.0, False, specific_metrics

        # Execute search using Ray execution pool with timing
        search_start_time = time.time()
        try:
            result_text, metadata = await self.execution_pool.execute.remote(self.execute_search, instance_id, query_list_from_params, self.retrieval_service_url, self.topk, timeout)
            search_end_time = time.time()

            # Store results in instance dictionary
            self._instance_dict[instance_id]["reward"].append(result_text.strip())

            # Enhanced tool-specific metrics for search performance
            specific_metrics = {
                # Basic search metrics
                "query_count": metadata.get("query_count", len(query_list_from_params)),
                "total_results": metadata.get("total_results", 0),
                "api_status": metadata.get("status", "unknown"),
                "queries": query_list_from_params,
                # Performance metrics
                "search_latency_ms": (search_end_time - search_start_time) * 1000,
                "avg_latency_per_query_ms": ((search_end_time - search_start_time) * 1000) / len(query_list_from_params) if query_list_from_params else 0,
                "timeout_used": timeout,
                "rate_limit_enabled": self.enable_global_rate_limit,
                # Query analysis metrics
                "query_complexity_scores": [self._calculate_query_complexity(q) for q in query_list_from_params],
                "avg_query_length": sum(len(q) for q in query_list_from_params) / len(query_list_from_params) if query_list_from_params else 0,
                "unique_query_ratio": len(set(query_list_from_params)) / len(query_list_from_params) if query_list_from_params else 0,
                # Results analysis
                "results_per_query": metadata.get("total_results", 0) / len(query_list_from_params) if query_list_from_params else 0,
                "relevance_scores": metadata.get("relevance_scores", []),
                "result_char_length": len(result_text) if result_text else 0,
                # Error tracking
                "error_type": None,
            }

            # Determine success based on API status and results
            success = metadata.get("status") == "success" and metadata.get("total_results", 0) > 0

            return result_text, 0.0, success, specific_metrics

        except Exception as e:
            search_end_time = time.time()
            error_result = json.dumps({"result": f"Search execution failed: {e}"})

            if self.tool_logger:
                self.tool_logger.error(f"[SearchTool] Execution failed: {e}")

            # Enhanced error metrics
            specific_metrics = {
                "error_type": self._classify_search_error(e),
                "exception_details": str(e),
                "query_count": len(query_list_from_params) if query_list_from_params else 0,
                "total_results": 0,
                "search_latency_ms": (search_end_time - search_start_time) * 1000,
                "api_status": "execution_error",
                "queries": query_list_from_params if query_list_from_params else [],
                "query_complexity_scores": [],
                "relevance_scores": [],
            }
            return error_result, 0.0, False, specific_metrics

    def _calculate_query_complexity(self, query: str) -> float:
        """Calculate complexity score for a search query."""
        if not query:
            return 0.0

        # Simple complexity scoring based on query characteristics
        complexity_score = 0.0

        # Length factor (normalized)
        complexity_score += min(len(query) / 100.0, 1.0) * 0.3

        # Word count factor
        word_count = len(query.split())
        complexity_score += min(word_count / 20.0, 1.0) * 0.3

        # Special characters/operators factor
        special_chars = ['"', "'", "(", ")", "AND", "OR", "NOT", "+", "-"]
        special_count = sum(1 for char in special_chars if char in query.upper())
        complexity_score += min(special_count / 5.0, 1.0) * 0.4

        return min(complexity_score, 1.0)  # Cap at 1.0

    def _classify_search_error(self, exception: Exception) -> str:
        """Classify the type of search error."""
        error_str = str(exception).lower()

        if "timeout" in error_str or "time" in error_str:
            return "timeout_error"
        elif "connection" in error_str or "network" in error_str:
            return "network_error"
        elif "rate" in error_str or "limit" in error_str:
            return "rate_limit_error"
        elif "permission" in error_str or "auth" in error_str:
            return "authentication_error"
        elif "not found" in error_str or "404" in error_str:
            return "service_not_found"
        else:
            return "unknown_execution_error"

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
