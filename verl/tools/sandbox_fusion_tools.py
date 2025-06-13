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

import logging
import os
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

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

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
                logger.warning(f"Error when executing code: {e}")
                raise

    def shutdown(self):
        """Shutdown the thread pool gracefully"""
        self._thread_pool.shutdown(wait=True)


def init_execution_pool(num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode):
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
        self.execution_pool = init_execution_pool(num_workers=self.num_workers, enable_global_rate_limit=self.enable_global_rate_limit, rate_limit=self.rate_limit, mode=PoolMode.ThreadMode)
        self.sandbox_fusion_url = config.get("sandbox_fusion_url", "")
        if self.sandbox_fusion_url == "":
            raise ValueError("sandbox_fusion_url is not set")
        self.mode = config.get("mode", "run_code")
        if self.mode == "run_jupyter":
            self.sandbox_fusion_url = self.sandbox_fusion_url.rstrip("/") + "/run_jupyter"
        else:
            self.sandbox_fusion_url = self.sandbox_fusion_url.rstrip("/") + "/run_code"
        log_msg = f"Init SandboxFusionTool with config: {config}"
        logger.info(log_msg)

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

    async def _execute_impl(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, bool, dict[str, Any]]:
        code = parameters.get("code", "")
        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)
        if not isinstance(code, str):
            code = str(code)

        # TODO: better documentation for the code
        code = code.rstrip("\n ")
        if len(code) > 0:
            if self.mode in ["run_jupyter", "sim_jupyter"]:
                self._instance_dict[instance_id]["cells"].append(code)
        else:
            logger.error(f"no code parsed, instance_id: {instance_id}, parameters: {parameters}")
            return "no code parsed", 0.0, False, {}

        # Execute code based on mode and get both result and success status
        if self.mode == "run_jupyter":
            result, success = await self.execution_pool.execute.remote(self.get_jupyter_mode_result, instance_id, timeout)
        elif self.mode == "run_code":
            result, success = await self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language)
        elif self.mode == "sim_jupyter":
            result, success = await self.execution_pool.execute.remote(self.get_sim_jupyter_mode_result, instance_id, timeout)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Tool-specific metrics
        specific_metrics = {"code_length": len(code), "execution_mode": self.mode, "language": language if "language" in locals() else "python", "timeout_used": timeout if "timeout" in locals() else 30}

        return result, 0.0, success, specific_metrics

    def execute_code(self, instance_id, code, timeout=30, language="python"):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result_status, metadata = _process_single_case(0, None, None, self.sandbox_fusion_url, code, timeout, language)

                # Check for API request errors first
                if metadata.get("api_request_error"):
                    error_msg = metadata["api_request_error"]
                    # Check if this is a retryable error (like Gateway Timeout)
                    if "Gateway Timeout" in error_msg and attempt < max_retries - 1:
                        delay = random.uniform(1, 5)
                        logger.warning(f"API request failed with Gateway Timeout, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue

                    logger.error(f"API request error for instance {instance_id}: {error_msg}")
                    return f"Error in calling code interpreter: {error_msg}", False

                # Check API status
                api_status = metadata.get("api_status")
                if api_status == "SandboxError":
                    error_msg = "Sandbox error occurred"
                    if metadata.get("api_response"):
                        error_msg += f": {metadata['api_response']}"
                    logger.error(f"Sandbox error for instance {instance_id}: {error_msg}")
                    return f"Error in calling code interpreter: {error_msg}", False

                elif api_status == "Failed":
                    # Handle compile errors
                    compile_status = metadata.get("compile_status")
                    if compile_status in ["Error", "TimeLimitExceeded"] or (compile_status == "Finished" and metadata.get("compile_stderr")):
                        ret_str = "Compilation failed"
                        if compile_status == "TimeLimitExceeded":
                            ret_str = f"Compilation time limit exceeded, time: {metadata.get('compile_duration', 'unknown')}"
                        if metadata.get("compile_stderr"):
                            ret_str += f"\ncompile_stderr: {metadata['compile_stderr']}"
                        logger.warning(f"Compile error for instance {instance_id}: {ret_str}")
                        return ret_str, False

                    # Handle runtime errors
                    run_status = metadata.get("run_status")
                    if run_status == "TimeLimitExceeded":
                        ret_str = f"Execution time limit exceeded, time: {metadata.get('duration', 'unknown')}, timeout: {timeout}"
                        if metadata.get("stdout"):
                            ret_str += f"\nstdout: {metadata['stdout']}"
                        if metadata.get("stderr"):
                            ret_str += f"\nstderr: {metadata['stderr']}"
                        logger.warning(f"Runtime timeout for instance {instance_id}: {ret_str}")
                        return ret_str, False

                    elif run_status == "Error" or (run_status == "Finished" and metadata.get("exit_code") != 0):
                        ret_str = ""
                        if metadata.get("exit_code") is not None and metadata["exit_code"] != 0:
                            ret_str += f"return_code: {metadata['exit_code']}\n"
                        if metadata.get("stdout"):
                            ret_str += f"stdout: {metadata['stdout']}\n"
                        if metadata.get("stderr"):
                            ret_str += f"stderr: {metadata['stderr']}\n"
                        logger.warning(f"Runtime error for instance {instance_id}: {ret_str}")
                        return ret_str, False

                    # Unknown failure state
                    logger.warning(f"Unknown failure state for instance {instance_id}: {metadata}")
                    return f"Unknown execution failure: {metadata.get('status', 'unknown')}", False

                elif api_status == "Success":
                    # Handle successful execution
                    if metadata.get("run_status") == "Finished":
                        # Determine success based on exit code
                        exit_code = metadata.get("exit_code", 0)
                        is_success = exit_code == 0

                        ret_str = ""
                        if metadata.get("exit_code") is not None and metadata["exit_code"] != 0:
                            ret_str += f"return_code: {metadata['exit_code']}\n"
                        if metadata.get("stdout"):
                            ret_str += f"stdout: {metadata['stdout']}\n"
                        if metadata.get("stderr"):
                            ret_str += f"stderr: {metadata['stderr']}\n"

                        # Return just stdout if successful and no errors, otherwise return structured output
                        if is_success and not metadata.get("stderr"):
                            actual_output = metadata["stdout"] if metadata["stdout"] is not None else ""
                            logger.debug(f"Successful execution for instance {instance_id}: {actual_output}")
                            return actual_output, True
                        else:
                            logger.debug(f"Execution completed for instance {instance_id} with success={is_success}: {ret_str}")
                            return ret_str, is_success
                    else:
                        logger.warning(f"Unexpected success state for instance {instance_id}: run_status={metadata.get('run_status')}")
                        return f"Unexpected execution state: {metadata.get('run_status')}", False

                else:
                    # Unknown API status
                    logger.error(f"Unknown API status for instance {instance_id}: {api_status}")
                    return f"Unknown API status: {api_status}", False

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = random.uniform(1, 5)
                    logger.warning(f"Request failed with error: {e}, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                logger.error(f"Error in execute_code after {max_retries} attempts: {e}")
                return f"Error in calling code interpreter: {e}", False

        # Should not reach here
        return "Error in calling code interpreter: Maximum retries exceeded", False

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
            logger.error(f"Error in get_jupyter_mode_result: {e}\npayload: {payload}")
            return f"Error in calling code interpreter: {e}", False
        if response.status_code != 200:
            logger.error(f"Error in get_jupyter_mode_result: {response.status_code}\npayload: {payload}\nresponse: {response.text}")
            try:
                response_json = response.json()
                error_message = response_json["error_message"]
                return error_message, False
            except Exception:
                return f"Error in calling code interpreter: {response.text}", False
        try:
            response_json = response.json()
            status = response_json["status"]
            if status == "Success":
                # Check if there are errors in the cell execution
                last_cell = response_json["cells"][-1]
                has_errors = last_cell.get("error") is not None and len(last_cell["error"]) > 0

                ret_str = ""
                if last_cell["stdout"] is not None and len(last_cell["stdout"]) > 0:
                    ret_str += f"stdout: {last_cell['stdout']}\n"
                if last_cell["display"] is not None and len(last_cell["display"]) > 0:
                    ret_str += f"displays: {last_cell['display']}\n"
                if last_cell["stderr"] is not None and len(last_cell["stderr"]) > 0:
                    ret_str += f"stderr: {last_cell['stderr']}\n"
                if has_errors:
                    ret_str += f"errors: {last_cell['error']}\n"

                # Success if no errors, even if there are warnings (stderr)
                is_success = not has_errors
                return ret_str, is_success

            elif status == "Failed":
                execution_status = response_json["driver"]["status"]
                if execution_status == "TimeLimitExceeded":
                    return "Execution time limit exceeded", False
                else:
                    error_msg = f"Unknown execution status: {execution_status}"
                    logger.error(f"{error_msg}\nresponse: {response.text}")
                    return error_msg, False
            else:
                error_msg = f"Unknown response status: {status}"
                logger.error(f"{error_msg}\nresponse: {response.text}")
                return error_msg, False
        except Exception as e:
            logger.error(f"Error in get_jupyter_mode_result: {e}\npayload: {payload}\nresponse: {response.text}")
            return f"Error in calling code interpreter: {response.text}", False

    def get_sim_jupyter_mode_result(self, instance_id, timeout: Optional[int] = None):
        if len(self._instance_dict[instance_id]["cells"]) == 0:
            return "no code parsed", False
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
                    logger.warning(f"Request failed with status {response.status_code}, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = random.uniform(1, 5)
                    logger.warning(f"Request failed with error: {e}, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                logger.error(f"Error in get_sim_jupyter_mode_result after {max_retries} attempts: {e}\npayload: {payload}")
                return f"Error in calling code interpreter: {e}", False

        if response.status_code != 200:
            logger.error(f"Error in get_sim_jupyter_mode_result: {response.status_code}\npayload: {payload}\nresponse: {response.text}")
            try:
                response_json = response.json()
                error_message = response_json["error_message"]
                return error_message, False
            except Exception:
                return f"Error in calling code interpreter: {response.text}", False
        try:
            response_json = response.json()
            status = response_json["status"]
            if status == "Success":
                execution_status = response_json["run_result"]["status"]
                if execution_status == "Finished":
                    # Determine success based on return code
                    return_code = response_json["run_result"].get("return_code", 0)
                    is_success = return_code == 0

                    ret_str = ""
                    if return_code is not None and return_code != 0:
                        ret_str += f"return_code: {return_code}\n"
                    if response_json["run_result"]["stdout"] is not None and len(response_json["run_result"]["stdout"]) > 0:
                        ret_str += f"stdout: {response_json['run_result']['stdout']}\n"
                    if response_json["run_result"]["stderr"] is not None and len(response_json["run_result"]["stderr"]) > 0:
                        ret_str += f"stderr: {response_json['run_result']['stderr']}\n"
                    return ret_str, is_success
                else:
                    error_msg = f"Unknown execution status: {execution_status}"
                    logger.error(f"{error_msg}\nresponse: {response.text}")
                    return error_msg, False
            elif status == "Failed":
                execution_status = response_json["run_result"]["status"]
                # Drop last cell if failed, to avoid keep failed in further execution
                self._instance_dict[instance_id]["cells"] = self._instance_dict[instance_id]["cells"][:-1]
                if execution_status == "TimeLimitExceeded":
                    ret_str = f"Execution time limit exceeded, time: {response_json['run_result']['execution_time']}, timeout: {payload['run_timeout']}"
                    if response_json["run_result"]["stdout"] is not None and len(response_json["run_result"]["stdout"]) > 0:
                        ret_str += f"stdout: {response_json['run_result']['stdout']}\n"
                    if response_json["run_result"]["stderr"] is not None and len(response_json["run_result"]["stderr"]) > 0:
                        ret_str += f"stderr: {response_json['run_result']['stderr']}\n"
                    logger.warning(f"Execution time limit exceeded, time: {response_json['run_result']['execution_time']}, payload: {payload}, response: {response.text}")
                    return ret_str, False
                elif execution_status == "Finished":
                    # Failed status with Finished execution means non-zero return code
                    return_code = response_json["run_result"].get("return_code", 1)
                    ret_str = ""
                    if return_code is not None and return_code != 0:
                        ret_str += f"return_code: {return_code}\n"
                    if response_json["run_result"]["stdout"] is not None and len(response_json["run_result"]["stdout"]) > 0:
                        ret_str += f"stdout: {response_json['run_result']['stdout']}\n"
                    if response_json["run_result"]["stderr"] is not None and len(response_json["run_result"]["stderr"]) > 0:
                        ret_str += f"stderr: {response_json['run_result']['stderr']}\n"
                    return ret_str, False
                else:
                    error_msg = f"Unknown execution status: {execution_status}"
                    logger.error(f"{error_msg}\nresponse: {response.text}")
                    return error_msg, False
            elif status == "SandboxError":
                error_msg = f"Sandbox error: {response.text}"
                logger.error(error_msg)
                return error_msg, False
            else:
                error_msg = f"Unknown response status: {status}"
                logger.error(f"{error_msg}\nresponse: {response.text}")
                return error_msg, False
        except Exception as e:
            logger.error(f"Error in get_sim_jupyter_mode_result: {e}\npayload: {payload}\nresponse: {response.text}")
            return f"Error in calling code interpreter: {response.text}", False

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
