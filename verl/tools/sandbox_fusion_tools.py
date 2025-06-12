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

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
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
        elif self.mode == "run_code":
            logger.error(f"no code parsed, instance_id: {instance_id}, parameters: {parameters}")
            return "no code parsed", 0.0, {}
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if self.mode == "run_jupyter":
            result = await self.execution_pool.execute.remote(self.get_jupyter_mode_result, instance_id, timeout)
        elif self.mode == "run_code":
            result = await self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language)
        elif self.mode == "sim_jupyter":
            result = await self.execution_pool.execute.remote(self.get_sim_jupyter_mode_result, instance_id, timeout)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return result, 0.0, {}

    def execute_code(self, instance_id, code, timeout=30, language="python"):
        result_status, metadata = _process_single_case(0, None, None, self.sandbox_fusion_url, code, timeout, language)
        # we should always expect this since we don't have correct answer
        if metadata["run_status"] == "Finished":
            actual_output = metadata["stdout"] if metadata["stdout"] is not None else ""
            logger.debug(f"actual_output from sandbox fusion: {actual_output},{instance_id}")
            return actual_output
        else:
            return "no stdout here"

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
            return f"Error in calling code interpreter: {e}"
        if response.status_code != 200:
            logger.error(f"Error in get_jupyter_mode_result: {response.status_code}\npayload: {payload}\nresponse: {response.text}")
            try:
                response_json = response.json()
                error_message = response_json["error_message"]
                return error_message
            except Exception:
                return f"Error in calling code interpreter: {response.text}"
        try:
            response_json = response.json()
            status = response_json["status"]
            if status == "Success":
                ret_str = ""
                if response_json["cells"][-1]["stdout"] is not None and len(response_json["cells"][-1]["stdout"]) > 0:
                    ret_str += f"stdout: {response_json['cells'][-1]['stdout']}\n"
                if response_json["cells"][-1]["display"] is not None and len(response_json["cells"][-1]["display"]) > 0:
                    ret_str += f"displays: {response_json['cells'][-1]['display']}\n"
                if response_json["cells"][-1]["stderr"] is not None and len(response_json["cells"][-1]["stderr"]) > 0:
                    ret_str += f"stderr: {response_json['cells'][-1]['stderr']}\n"
                if response_json["cells"][-1]["error"] is not None and len(response_json["cells"][-1]["error"]) > 0:
                    ret_str += f"errors: {response_json['cells'][-1]['error']}\n"
                return ret_str
            elif status == "Failed":
                execution_status = response_json["driver"]["status"]
                if execution_status == "TimeLimitExceeded":
                    return "Execution time limit exceeded"
                else:
                    raise ValueError(f"Unknown execution status: {execution_status}\nresponse: {response.text}")
            else:
                raise ValueError(f"Unknown response status: {status}\nresponse: {response.text}")
        except Exception as e:
            logger.error(f"Error in get_jupyter_mode_result: {e}\npayload: {payload}\nresponse: {response.text}")
            return f"Error in calling code interpreter: {response.text}"

    def get_sim_jupyter_mode_result(self, instance_id, timeout: Optional[int] = None):
        if len(self._instance_dict[instance_id]["cells"]) == 0:
            return "no code parsed"
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
                return f"Error in calling code interpreter: {e}"

        if response.status_code != 200:
            logger.error(f"Error in get_sim_jupyter_mode_result: {response.status_code}\npayload: {payload}\nresponse: {response.text}")
            try:
                response_json = response.json()
                error_message = response_json["error_message"]
                return error_message
            except Exception:
                return f"Error in calling code interpreter: {response.text}"
        try:
            response_json = response.json()
            status = response_json["status"]
            if status == "Success":
                execution_status = response_json["run_result"]["status"]
                if execution_status == "Finished":
                    ret_str = ""
                    if response_json["run_result"]["return_code"] is not None and response_json["run_result"]["return_code"] != 0:
                        ret_str += f"return_code: {response_json['run_result']['return_code']}\n"
                    if response_json["run_result"]["stdout"] is not None and len(response_json["run_result"]["stdout"]) > 0:
                        ret_str += f"stdout: {response_json['run_result']['stdout']}\n"
                    if response_json["run_result"]["stderr"] is not None and len(response_json["run_result"]["stderr"]) > 0:
                        ret_str += f"stderr: {response_json['run_result']['stderr']}\n"
                    return ret_str
                else:
                    raise ValueError(f"Unknown execution status: {execution_status}\nresponse: {response.text}")
            elif status == "Failed":
                execution_status = response_json["run_result"]["status"]
                # Drop last cell if failed, to avoid keep failed in further execution
                self._instance_dict[instance_id]["cells"].pop(-1)
                if execution_status == "TimeLimitExceeded":
                    ret_str = f"Execution time limit exceeded, time: {response_json['run_result']['execution_time']}, timeout: {payload['run_timeout']}"
                    if response_json["run_result"]["stdout"] is not None and len(response_json["run_result"]["stdout"]) > 0:
                        ret_str += f"stdout: {response_json['run_result']['stdout']}\n"
                    if response_json["run_result"]["stderr"] is not None and len(response_json["run_result"]["stderr"]) > 0:
                        ret_str += f"stderr: {response_json['run_result']['stderr']}\n"
                    logger.warning(f"Execution time limit exceeded, time: {response_json['run_result']['execution_time']}, timeout: {payload['run_timeout']}, response: {response.text}")
                    return ret_str
                elif execution_status == "Finished":
                    ret_str = ""
                    if response_json["run_result"]["return_code"] is not None and response_json["run_result"]["return_code"] != 0:
                        ret_str += f"return_code: {response_json['run_result']['return_code']}\n"
                    if response_json["run_result"]["stdout"] is not None and len(response_json["run_result"]["stdout"]) > 0:
                        ret_str += f"stdout: {response_json['run_result']['stdout']}\n"
                    if response_json["run_result"]["stderr"] is not None and len(response_json["run_result"]["stderr"]) > 0:
                        ret_str += f"stderr: {response_json['run_result']['stderr']}\n"
                    return ret_str
                else:
                    raise ValueError(f"Unknown execution status: {execution_status}\nresponse: {response.text}")
            elif status == "SandboxError":
                raise ValueError(f"Sandbox error: {response.text}")
            else:
                raise ValueError(f"Unknown response status: {status}\nresponse: {response.text}")
        except Exception as e:
            logger.error(f"Error in get_sim_jupyter_mode_result: {e}\npayload: {payload}\nresponse: {response.text}")
            return f"Error in calling code interpreter: {response.text}"

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
