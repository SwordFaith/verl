# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, Tuple, TypeVar
from uuid import uuid4

import ray
import requests

from verl.utils.reward_score.sandbox_fusion.utils import _process_single_case

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

T = TypeVar("T")


MAX_THREAD_POOL_WORKERS = 1024


class PoolMode(Enum):
    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        # this only used for observalability
        self.current_count = 0
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        return self.current_count


class ExecutionWorker:
    def __init__(self, enable_global_rate_limit=True, rate_limit=10, max_workers=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._max_workers = max_workers

    def _init_rate_limit(self, rate_limit):
        # TODO validation for rate_limit
        # A Singleton Rate Limitor
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        with ExitStack() as stack:
            stack.callback(self.rate_limit_worker.release.remote)
            ray.get(self.rate_limit_worker.acquire.remote())
            try:
                # Submit the task to the thread pool
                future = self._thread_pool.submit(fn, *fn_args, **fn_kwargs)
                return future.result()
            except Exception as e:
                if self.tool_logger:
                    self.tool_logger.warning(f"Error when executing code: {e}")
                raise

    def shutdown(self):
        """Shutdown the thread pool gracefully"""
        self._thread_pool.shutdown(wait=True)


def init_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(ExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(
                enable_global_rate_limit=enable_global_rate_limit,
                rate_limit=rate_limit,
                max_workers=min(num_workers, MAX_THREAD_POOL_WORKERS),  # Limit max workers to avoid resource exhaustion
            )
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class SandboxFusionTool(BaseTool):
    """A tool for executing the code using sanbox fusion image.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "description": "A tool for execute code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "code needs to be execute and grad",
                        },
                    },
                    "required": ["code"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        # TODO: better documentation for the config
        self.num_workers = min(config.get("num_workers", 10), MAX_THREAD_POOL_WORKERS)  # Limit max workers
        self.rate_limit = min(config.get("rate_limit", 10), MAX_THREAD_POOL_WORKERS)  # Limit rate limit
        self.default_timeout = config.get("default_timeout", 30)
        self.cell_timeout = config.get("cell_timeout", 10)
        self.default_language = config.get("default_language", "python")
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )
        self.sandbox_fusion_url = config.get("sandbox_fusion_url", "")
        self.memory_limit_mb = config.get("memory_limit_mb", 1024)
        if self.sandbox_fusion_url == "":
            raise ValueError("sandbox_fusion_url is not set")
        self.mode = config.get("mode", "run_code")
        if self.mode == "run_jupyter":
            self.sandbox_fusion_url = self.sandbox_fusion_url.rstrip("/") + "/run_jupyter"
        else:
            self.sandbox_fusion_url = self.sandbox_fusion_url.rstrip("/") + "/run_code"
        if self.tool_logger:
            self.tool_logger.info(f"Init SandboxFusionTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": [],
            "cells": [],
        }
        return instance_id

    async def _execute_impl(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> Tuple[str, float, bool, dict[str, Any]]:
        code = parameters.get("code", "")
        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)
        if not isinstance(code, str):
            code = str(code)

        # Normalize code
        code = code.rstrip("\n ")
        if len(code) > 0:
            if self.mode in ["run_jupyter", "sim_jupyter"]:
                self._instance_dict[instance_id]["cells"].append(code)
        else:
            if self.tool_logger:
                self.tool_logger.error(f"no code parsed, instance_id: {instance_id}, parameters: {parameters}")

            # Essential metrics for empty code case
            specific_metrics = {
                # === Primary Metrics ===
                "lines_of_code": 0,
                "execution_time_ms": 0.0,
                "return_code": 1,  # Error case
                "execution_status": "parameter_error",
                "execution_mode": self.mode,
                # === Secondary Metrics ===
                "has_stdout": False,
                "has_stderr": False,
                "has_timeout": False,
                "display_object_count": 0,
                # === Error Classification ===
                "error_type": "empty_code",
            }
            return "no code parsed", 0.0, False, specific_metrics

        # Execute code based on mode and get both result, success status, and extracted API metrics
        if self.mode == "run_jupyter":
            result, success, extracted_api_metrics = await self.execution_pool.execute.remote(
                self.get_jupyter_mode_result, instance_id, timeout
            )
        elif self.mode == "run_code":
            result, success, extracted_api_metrics = await self.execution_pool.execute.remote(
                self.execute_code, instance_id, code, timeout, language
            )
        elif self.mode == "sim_jupyter":
            result, success, extracted_api_metrics = await self.execution_pool.execute.remote(
                self.get_sim_jupyter_mode_result, instance_id, timeout
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Use extracted API metrics directly
        execution_metadata = extracted_api_metrics or {}

        # Calculate detailed code metrics (prioritize lines_of_code as primary metric)
        code_lines = [line for line in code.split("\n") if line.strip()]
        effective_lines_count = len(code_lines)
        # total_lines_count = len(code.split("\n"))

        # Essential metrics for sandbox fusion tool (focus on training-critical data)
        error_msg = "" if success else str(result)
        api_status = execution_metadata.get("api_execution_status", "unknown")

        specific_metrics = {
            # === Primary Metrics (training critical) ===
            "lines_of_code": effective_lines_count,  # PRIMARY: code complexity indicator
            "execution_time_ms": execution_metadata.get("api_execution_time_ms", 0.0),  # Real API timing
            "return_code": execution_metadata.get("api_return_code", 0 if success else 1),  # Exit status
            "execution_status": api_status,  # API status
            "execution_mode": self.mode,  # run_code/run_jupyter/sim_jupyter
            # === Secondary Metrics (analytics) ===
            "has_stdout": bool(execution_metadata.get("has_stdout", result and success)),
            "has_stderr": bool(execution_metadata.get("has_stderr", not success)),
            "has_timeout": execution_metadata.get("has_timeout", False),  # Timeout detection
            # === Jupyter-specific (when applicable) ===
            "display_object_count": execution_metadata.get("display_object_count", 0),
            # === Fine-grained Error Classification ===
            "error_type": self._classify_error_type(error_msg, api_status) if not success else None,
        }

        # Calculate reward based on execution success
        reward = 0.001 if success else -0.01

        # Accumulate reward in instance dictionary
        if instance_id in self._instance_dict:
            if isinstance(self._instance_dict[instance_id]["reward"], list):
                self._instance_dict[instance_id]["reward"].append(reward)
            else:
                # Convert to list if it's not already
                self._instance_dict[instance_id]["reward"] = [self._instance_dict[instance_id]["reward"], reward]

        return result, reward, success, specific_metrics

    def _format_execution_result(self, raw_response: str, success: bool, api_metadata: dict) -> str:
        """Format execution result for better model comprehension using full API data"""
        if success:
            # Success: Return clean output, prioritizing API stdout
            if api_metadata.get("stdout"):
                return api_metadata["stdout"].strip()
            elif raw_response and raw_response.strip():
                return raw_response.strip()
            else:
                return "Execution completed successfully"
        else:
            # Error: Clean error message + actionable suggestion
            if api_metadata.get("stderr"):
                error_msg = api_metadata["stderr"].strip()
            else:
                error_msg = raw_response.strip() if raw_response else "Execution failed"

            suggestion = self._get_python_error_suggestion(error_msg)
            return f"{error_msg}\n{suggestion}" if suggestion else error_msg

    def _get_python_error_suggestion(self, error_msg: str) -> str:
        """Provide actionable suggestions for common Python errors"""
        if "NameError" in error_msg:
            return "Suggestion: Check variable name and ensure it's defined before use"
        elif "TypeError" in error_msg:
            return "Suggestion: Verify data types and function arguments"
        elif "ZeroDivisionError" in error_msg:
            return "Suggestion: Add check for division by zero"
        elif "SyntaxError" in error_msg:
            return "Suggestion: Check Python syntax and indentation"
        elif "IndentationError" in error_msg:
            return "Suggestion: Fix code indentation (use 4 spaces per level)"
        elif "ModuleNotFoundError" in error_msg:
            return "Suggestion: Check if required module is installed"
        elif "ImportError" in error_msg:
            return "Suggestion: Verify module installation and import statement"
        elif "AttributeError" in error_msg:
            return "Suggestion: Check object attributes and method names"
        elif "KeyError" in error_msg:
            return "Suggestion: Verify dictionary keys exist before accessing"
        elif "IndexError" in error_msg:
            return "Suggestion: Check list/array bounds before indexing"
        elif "ValueError" in error_msg:
            return "Suggestion: Check input values and data conversion"
        return ""

    def _classify_error_type(self, error_msg: str, api_status: str) -> str:
        """Classify error type with fine-grained categories for training analytics"""
        if not error_msg:
            return "unknown_error"

        # API-level errors
        if api_status == "SandboxError":
            return "sandbox_error"
        elif "TimeLimitExceeded" in error_msg or "time limit exceeded" in error_msg.lower():
            return "timeout_error"
        elif "Gateway Timeout" in error_msg:
            return "api_timeout_error"

        # Python runtime errors (fine-grained)
        if "NameError" in error_msg:
            return "name_error"  # Variable not defined
        elif "TypeError" in error_msg:
            return "type_error"  # Wrong data type
        elif "ZeroDivisionError" in error_msg:
            return "zero_division_error"  # Division by zero
        elif "SyntaxError" in error_msg:
            return "syntax_error"  # Invalid Python syntax
        elif "IndentationError" in error_msg:
            return "indentation_error"  # Indentation issues
        elif "ModuleNotFoundError" in error_msg:
            return "module_not_found_error"  # Missing module
        elif "ImportError" in error_msg:
            return "import_error"  # Import issues
        elif "AttributeError" in error_msg:
            return "attribute_error"  # Missing attribute/method
        elif "KeyError" in error_msg:
            return "key_error"  # Dictionary key missing
        elif "IndexError" in error_msg:
            return "index_error"  # List/array out of bounds
        elif "ValueError" in error_msg:
            return "value_error"  # Invalid value for operation
        elif "FileNotFoundError" in error_msg:
            return "file_not_found_error"  # File system error
        elif "PermissionError" in error_msg:
            return "permission_error"  # Permission issues
        elif "MemoryError" in error_msg:
            return "memory_error"  # Out of memory
        elif "RecursionError" in error_msg:
            return "recursion_error"  # Stack overflow
        elif "KeyboardInterrupt" in error_msg:
            return "keyboard_interrupt"  # User interruption

        # Compilation errors
        elif "compilation failed" in error_msg.lower() or "compile" in error_msg.lower():
            return "compilation_error"

        # Generic categories
        elif "error" in error_msg.lower():
            return "runtime_error"  # Generic runtime error
        else:
            return "execution_failure"  # Non-error execution failure

    def execute_code(self, instance_id, code, timeout=30, language="python"):
        """Execute code and return (result, success, api_response_data) for enhanced metrics"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result_status, metadata = _process_single_case(
                    0, None, None, self.sandbox_fusion_url, code, timeout, language
                )

                # Check for API request errors first
                if metadata.get("api_request_error"):
                    error_msg = metadata["api_request_error"]
                    # Log payload and response when API request fails
                    payload = metadata.get("payload", "N/A")
                    response_text = metadata.get("response_text", "N/A")
                    if self.tool_logger:
                        self.tool_logger.error(
                            f"API request error for instance {instance_id}: {error_msg}, "
                            f"payload: {payload}, response: {response_text}"
                        )

                    # Check if this is a retryable error (like Gateway Timeout)
                    if "Gateway Timeout" in error_msg and attempt < max_retries - 1:
                        delay = random.uniform(1, 5)
                        if self.tool_logger:
                            self.tool_logger.warning(
                                f"API request failed with Gateway Timeout, retrying in {delay:.2f}s "
                                f"(attempt {attempt + 1}/{max_retries})"
                            )
                        time.sleep(delay)
                        continue
                    return f"Error in calling code interpreter: {error_msg}", False, {}

                # Check API status
                api_status = metadata.get("api_status")
                if api_status == "SandboxError":
                    error_msg = "Sandbox error occurred"
                    if metadata.get("api_response"):
                        error_msg += f": {metadata['api_response']}"

                    # Log payload and response for SandboxError
                    payload = metadata.get("payload", "N/A")
                    response_text = metadata.get("response_text", "N/A")
                    if self.tool_logger:
                        self.tool_logger.error(
                            f"Sandbox error for instance {instance_id}: {error_msg}, "
                            f"payload: {payload}, response: {response_text}"
                        )
                    return f"Error in calling code interpreter: {error_msg}", False, {}

                elif api_status == "Failed":
                    # Handle compile errors
                    compile_status = metadata.get("compile_status")
                    if compile_status in ["Error", "TimeLimitExceeded"] or (
                        compile_status == "Finished" and metadata.get("compile_stderr")
                    ):
                        if compile_status == "TimeLimitExceeded":
                            error_msg = (
                                f"Compilation time limit exceeded, time: {metadata.get('compile_duration', 'unknown')}"
                            )
                        else:
                            error_msg = "Compilation failed"

                        api_metadata = {"stderr": metadata.get("compile_stderr", error_msg)}
                        formatted_result = self._format_execution_result("", False, api_metadata)
                        if self.tool_logger:
                            self.tool_logger.warning(f"Compile error for instance {instance_id}: {formatted_result}")
                        return formatted_result, False, {}

                    # Handle runtime errors
                    run_status = metadata.get("run_status")
                    if run_status == "TimeLimitExceeded":
                        api_metadata = {
                            "stderr": (
                                f"Execution time limit exceeded, time: {metadata.get('duration', 'unknown')}, "
                                f"timeout: {timeout}"
                            ),
                            "stdout": metadata.get("stdout"),
                        }
                        extracted_api_metrics = {
                            "api_execution_time_ms": (metadata.get("duration", 0) or 0) * 1000,
                            "api_execution_status": "TimeLimitExceeded",
                            "has_timeout": True,
                            "has_stdout": bool(metadata.get("stdout")),
                            "has_stderr": bool(metadata.get("stderr")),
                        }
                        formatted_result = self._format_execution_result("", False, api_metadata)
                        if self.tool_logger:
                            self.tool_logger.warning(f"Runtime timeout for instance {instance_id}: {formatted_result}")
                        return formatted_result, False, extracted_api_metrics

                    elif run_status == "Error" or (run_status == "Finished" and metadata.get("exit_code") != 0):
                        # Runtime error - use enhanced formatting with API data
                        api_metadata = {
                            "stdout": metadata.get("stdout"),
                            "stderr": metadata.get("stderr"),
                            "exit_code": metadata.get("exit_code"),
                        }
                        # Extract API metrics for failed execution
                        extracted_api_metrics = {
                            "api_execution_time_ms": (metadata.get("duration", 0) or 0) * 1000,
                            "api_return_code": metadata.get("exit_code"),
                            "api_execution_status": run_status,
                            "has_stdout": bool(metadata.get("stdout")),
                            "has_stderr": bool(metadata.get("stderr")),
                        }
                        formatted_result = self._format_execution_result("", False, api_metadata)
                        if self.tool_logger:
                            self.tool_logger.warning(f"Runtime error for instance {instance_id}: {formatted_result}")
                        return formatted_result, False, extracted_api_metrics

                    # Unknown failure state
                    if self.tool_logger:
                        self.tool_logger.warning(f"Unknown failure state for instance {instance_id}: {metadata}")
                    return f"Unknown execution failure: {metadata.get('status', 'unknown')}", False, {}

                elif api_status == "Success":
                    # Handle successful execution
                    if metadata.get("run_status") == "Finished":
                        # Determine success based on exit code
                        exit_code = metadata.get("exit_code", 0)
                        is_success = exit_code == 0

                        # Prepare API metadata for formatting and metrics
                        api_metadata = {
                            "stdout": metadata.get("stdout"),
                            "stderr": metadata.get("stderr"),
                            "exit_code": exit_code,
                        }

                        # Extract specific API metrics for tool metrics
                        extracted_api_metrics = {
                            "api_execution_time_ms": (metadata.get("duration", 0) or 0) * 1000,
                            "api_return_code": exit_code,
                            "api_execution_status": "Finished",
                            "has_stdout": bool(metadata.get("stdout")),
                            "has_stderr": bool(metadata.get("stderr")),
                        }

                        # Use enhanced formatting
                        formatted_result = self._format_execution_result("", is_success, api_metadata)
                        if self.tool_logger:
                            self.tool_logger.debug(
                                f"Execution for instance {instance_id} (success={is_success}): {formatted_result}"
                            )
                        return formatted_result, is_success, extracted_api_metrics
                    else:
                        if self.tool_logger:
                            self.tool_logger.warning(
                                f"Unexpected success state for instance {instance_id}: "
                                f"run_status={metadata.get('run_status')}"
                            )
                        return f"Unexpected execution state: {metadata.get('run_status')}", False, {}

                else:
                    # Unknown API status
                    if self.tool_logger:
                        self.tool_logger.error(f"Unknown API status for instance {instance_id}: {api_status}")
                    return f"Unknown API status: {api_status}", False, {}

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = random.uniform(1, 5)
                    if self.tool_logger:
                        self.tool_logger.warning(
                            f"Request failed with error: {e}, retrying in {delay:.2f}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                    time.sleep(delay)
                    continue
                if self.tool_logger:
                    self.tool_logger.error(f"Error in execute_code after {max_retries} attempts: {e}")
                return f"Error in calling code interpreter: {e}", False, {}

        # Should not reach here
        return "Error in calling code interpreter: Maximum retries exceeded", False, {}

    def get_jupyter_mode_result(self, instance_id, timeout: Optional[int] = None):
        # Create a new payload for each request with all cells
        payload = {
            "cells": self._instance_dict[instance_id]["cells"],
            "cell_timeout": self.cell_timeout,
            "total_timeout": timeout if timeout is not None else self.default_timeout,
            "kernel": "python3",
            "files": {},
            "fetch_files": [],
        }
        try:
            response = requests.request("POST", self.sandbox_fusion_url, json=payload)
        except Exception as e:
            if self.tool_logger:
                self.tool_logger.error(f"Error in get_jupyter_mode_result: {e}\npayload: {payload}")
            return f"Error in calling code interpreter: {e}", False, {}
        if response.status_code != 200:
            if self.tool_logger:
                self.tool_logger.error(
                    f"Error in get_jupyter_mode_result: {response.status_code}\n"
                    f"payload: {payload}\nresponse: {response.text}"
                )
            try:
                response_json = response.json()
                error_message = response_json["error_message"]
                return error_message, False, {}
            except Exception:
                return f"Error in calling code interpreter: {response.text}", False, {}
        try:
            response_json = response.json()
            status = response_json["status"]
            if status == "Success":
                # Check if there are errors in the cell execution
                last_cell = response_json["cells"][-1]
                has_errors = last_cell.get("error") is not None and len(last_cell["error"]) > 0

                # Success if no errors, even if there are warnings (stderr)
                is_success = not has_errors

                # Format response using enhanced formatter
                api_metadata = {"stdout": last_cell.get("stdout"), "stderr": last_cell.get("stderr")}
                formatted_result = self._format_execution_result("", is_success, api_metadata)

                # Extract specific API metrics from jupyter response with driver timing
                driver_info = response_json.get("driver", {})
                extracted_api_metrics = {
                    "api_execution_time_ms": (driver_info.get("execution_time", 0) or 0) * 1000,
                    "api_execution_status": driver_info.get("status", "unknown"),
                    "has_display_objects": bool(last_cell.get("display")),
                    "display_object_count": len(last_cell.get("display", [])),
                    "has_cell_errors": bool(last_cell.get("error")),
                    "cell_error_count": len(last_cell.get("error", [])),
                    "has_stdout": bool(last_cell.get("stdout")),
                    "has_stderr": bool(last_cell.get("stderr")),
                }

                return formatted_result, is_success, extracted_api_metrics

            elif status == "Failed":
                execution_status = response_json["driver"]["status"]
                if execution_status == "TimeLimitExceeded":
                    # Extract basic metrics even for timeout
                    extracted_api_metrics = {"api_execution_status": "TimeLimitExceeded", "has_timeout": True}
                    return "Execution time limit exceeded", False, extracted_api_metrics
                else:
                    error_msg = f"Unknown execution status: {execution_status}"
                    if self.tool_logger:
                        self.tool_logger.error(f"{error_msg}\nresponse: {response.text}")
                    return error_msg, False, {}
            else:
                error_msg = f"Unknown response status: {status}"
                if self.tool_logger:
                    self.tool_logger.error(f"{error_msg}\nresponse: {response.text}")
                return error_msg, False, {}
        except Exception as e:
            if self.tool_logger:
                self.tool_logger.error(
                    f"Error in get_jupyter_mode_result: {e}\npayload: {payload}\nresponse: {response.text}"
                )
            return f"Error in calling code interpreter: {response.text}", False, {}

    def get_sim_jupyter_mode_result(self, instance_id, timeout: Optional[int] = None):
        """Get sim jupyter execution result and return (result, success, extracted_api_metrics)"""
        if len(self._instance_dict[instance_id]["cells"]) == 0:
            return "no code parsed", False, {}
        elif len(self._instance_dict[instance_id]["cells"]) == 1:
            prev_cells = []
        else:
            prev_cells = self._instance_dict[instance_id]["cells"][:-1]

        def fix_jupyter_style_cell_code(cell_code: str) -> str:
            """jupyter style code use varaiable without print, and reference code in other fields"""
            cell_code_lines = cell_code.split("\n")
            last_line = cell_code_lines[-1]
            if "print(" in last_line:
                return cell_code
            elif "=" in last_line:
                variables = last_line.split("=")[0].strip()
                if "," in variables:
                    variables = variables.split(",")
                else:
                    variables = [variables]
                for variable in variables:
                    last_line += f"print('{variable}:', {variable})"
            else:
                last_line = f"print('{last_line}:', {last_line})"
            cell_code_lines[-1] = last_line
            return "\n".join(cell_code_lines)

        cur_cell = self._instance_dict[instance_id]["cells"][-1]
        cur_cell = fix_jupyter_style_cell_code(cur_cell)
        assembled_cells = prev_cells + [cur_cell]
        assembled_code = "\n".join(assembled_cells)
        payload = {
            "run_timeout": timeout if timeout is not None else self.default_timeout,
            "code": assembled_code,
            "language": "python",
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.request("POST", self.sandbox_fusion_url, json=payload)
                if response.status_code == 200:
                    break
                if attempt < max_retries - 1:
                    delay = random.uniform(1, 5)
                    if self.tool_logger:
                        self.tool_logger.warning(
                            f"Request failed with status {response.status_code}, retrying in {delay:.2f}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                    time.sleep(delay)
                    continue
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = random.uniform(1, 5)
                    if self.tool_logger:
                        self.tool_logger.warning(
                            f"Request failed with error: {e}, retrying in {delay:.2f}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                    time.sleep(delay)
                    continue
                if self.tool_logger:
                    self.tool_logger.error(
                        f"Error in get_sim_jupyter_mode_result after {max_retries} attempts: {e}\npayload: {payload}"
                    )
                return f"Error in calling code interpreter: {e}", False, {}

        if response.status_code != 200:
            if self.tool_logger:
                self.tool_logger.error(
                    f"Error in get_sim_jupyter_mode_result: {response.status_code}\n"
                    f"payload: {payload}\nresponse: {response.text}"
                )
            try:
                response_json = response.json()
                error_message = response_json["error_message"]
                return error_message, False, {}
            except Exception:
                return f"Error in calling code interpreter: {response.text}", False, {}
        try:
            response_json = response.json()
            status = response_json["status"]
            if status == "Success":
                execution_status = response_json["run_result"]["status"]
                if execution_status == "Finished":
                    # Determine success based on return code
                    return_code = response_json["run_result"].get("return_code", 0)
                    is_success = return_code == 0

                    # Use enhanced formatting and extract API metrics
                    api_metadata = {
                        "stdout": response_json["run_result"].get("stdout"),
                        "stderr": response_json["run_result"].get("stderr"),
                    }
                    formatted_result = self._format_execution_result("", is_success, api_metadata)

                    # Extract API metrics from sim jupyter response
                    extracted_api_metrics = {
                        "api_execution_time_ms": (response_json["run_result"].get("execution_time", 0) or 0) * 1000,
                        "api_return_code": return_code,
                        "api_execution_status": execution_status,
                        "has_stdout": bool(response_json["run_result"].get("stdout")),
                        "has_stderr": bool(response_json["run_result"].get("stderr")),
                    }

                    return formatted_result, is_success, extracted_api_metrics
                else:
                    error_msg = f"Unknown execution status: {execution_status}"
                    if self.tool_logger:
                        self.tool_logger.error(f"{error_msg}\nresponse: {response.text}")
                    return error_msg, False, {}
            elif status == "Failed":
                execution_status = response_json["run_result"]["status"]
                # Drop last cell if failed, to avoid keep failed in further execution
                self._instance_dict[instance_id]["cells"] = self._instance_dict[instance_id]["cells"][:-1]
                if execution_status == "TimeLimitExceeded":
                    api_metadata = {
                        "stderr": (
                            f"Execution time limit exceeded, time: {response_json['run_result']['execution_time']}, "
                            f"timeout: {payload['run_timeout']}"
                        ),
                        "stdout": response_json["run_result"].get("stdout"),
                        "execution_time": response_json["run_result"].get("execution_time", 0),
                    }
                    formatted_result = self._format_execution_result("", False, api_metadata)
                    extracted_api_metrics = {
                        "api_execution_time_ms": (response_json["run_result"].get("execution_time", 0) or 0) * 1000,
                        "api_execution_status": "TimeLimitExceeded",
                        "has_timeout": True,
                        "has_stdout": bool(response_json["run_result"].get("stdout")),
                        "has_stderr": bool(response_json["run_result"].get("stderr")),
                    }
                    if self.tool_logger:
                        self.tool_logger.warning(
                            f"Execution time limit exceeded, time: {response_json['run_result']['execution_time']},"
                            f"payload: {payload}, response: {response.text}"
                        )
                    return formatted_result, False, extracted_api_metrics
                elif execution_status == "Finished":
                    # Failed status with Finished execution means non-zero return code
                    return_code = response_json["run_result"].get("return_code", 1)
                    api_metadata = {
                        "stderr": response_json["run_result"].get("stderr"),
                        "stdout": response_json["run_result"].get("stdout"),
                        "exit_code": return_code,
                    }
                    formatted_result = self._format_execution_result("", False, api_metadata)
                    extracted_api_metrics = {
                        "api_execution_time_ms": (response_json["run_result"].get("execution_time", 0) or 0) * 1000,
                        "api_return_code": return_code,
                        "api_execution_status": execution_status,
                        "has_stdout": bool(response_json["run_result"].get("stdout")),
                        "has_stderr": bool(response_json["run_result"].get("stderr")),
                    }
                    return formatted_result, False, extracted_api_metrics
                else:
                    error_msg = f"Unknown execution status: {execution_status}"
                    if self.tool_logger:
                        self.tool_logger.error(f"{error_msg}\nresponse: {response.text}")
                    return error_msg, False, {}
            elif status == "SandboxError":
                error_msg = f"Sandbox error: {response.text}"
                if self.tool_logger:
                    self.tool_logger.error(error_msg)
                return error_msg, False, {}
            else:
                error_msg = f"Unknown response status: {status}"
                if self.tool_logger:
                    self.tool_logger.error(f"{error_msg}\nresponse: {response.text}")
                return error_msg, False, {}
        except Exception as e:
            if self.tool_logger:
                self.tool_logger.error(
                    f"Error in get_sim_jupyter_mode_result: {e}\npayload: {payload}\nresponse: {response.text}"
                )
            return f"Error in calling code interpreter: {response.text}", False, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # Return accumulated reward from all successful tool executions
        reward_list = self._instance_dict[instance_id]["reward"]
        return sum(reward_list) if isinstance(reward_list, list) else reward_list

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
