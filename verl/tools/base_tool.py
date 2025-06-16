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
import copy
import importlib
import importlib.util
import json
import logging
import os
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import torch.distributed as dist
from omegaconf import OmegaConf

from .schemas import OpenAIFunctionToolSchema


class AdvancedNoiseFilter(logging.Filter):
    """Advanced noise filter that supports different actions for different output targets."""

    def __init__(self, noise_filters: dict, output_target: str = "console"):
        super().__init__()
        self.noise_filters = noise_filters
        self.output_target = output_target  # "console" or "file"

    def filter(self, record):
        message = record.getMessage().lower()

        for filter_name, filter_config in self.noise_filters.items():
            patterns = filter_config.get("patterns", [])

            # Check if message matches any pattern
            if any(pattern.lower() in message for pattern in patterns):
                # Get action for current output target
                action_key = f"{self.output_target}_action"
                action = filter_config.get(action_key, filter_config.get("action", "allow"))

                if action == "suppress":
                    return False  # Block the record
                elif action == "downgrade":
                    # Downgrade log level
                    downgrade_to = filter_config.get("downgrade_to", "DEBUG")
                    record.levelno = getattr(logging, downgrade_to)
                    record.levelname = downgrade_to
                # action == 'allow' does nothing
                break

        return True


class ToolLogger:
    """Universal tool logger with race condition protection and logging compatibility.

    Features:
    - Rank-level isolation to prevent multi-process conflicts
    - Instance-level uniqueness for same-process multiple tool instances
    - Thread-safe operations with internal locking
    - JSON structured logging with optional plain text compatibility
    """

    def __init__(self, tool_name: str, rank_info: dict, config: dict):
        self.tool_name = tool_name
        self.rank_info = rank_info
        self.config = self._apply_default_config(config)
        self.logger = None

        # Generate unique instance ID to prevent same-process conflicts
        self.instance_id = str(uuid4())[:8]

        # Thread-safe lock for logger operations
        self._lock = threading.RLock()

        # Apply test mode if enabled
        if self.config.get("test_mode", {}).get("enable", False):
            self._apply_test_mode()

        if self.config.get("enable", True):
            self._setup_logger()

    def _apply_default_config(self, user_config: dict) -> dict:
        """Apply default configuration and merge with user config."""
        default_config = {"enable": True, "separate_by_rank": True, "outputs": {"console": {"enable": True, "level": "ERROR", "apply_filters": True}, "file": {"enable": True, "level": "WARNING", "apply_filters": False}}, "noise_filters": {}, "test_mode": {"enable": False}, "tool_specific": {}}

        merged = copy.deepcopy(default_config)
        self._deep_update(merged, user_config)
        return merged

    def _deep_update(self, base_dict: dict, update_dict: dict):
        """Deep merge two dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def _apply_test_mode(self):
        """Apply test mode configuration overrides."""
        test_config = self.config.get("test_mode", {})

        if test_config.get("console_silent", False):
            # Make console completely silent
            self.config.setdefault("outputs", {})
            self.config["outputs"]["console"] = {"enable": False, "level": "CRITICAL", "apply_filters": True}

        if test_config.get("file_verbose", False):
            # Enable verbose file logging
            self.config.setdefault("outputs", {})
            self.config["outputs"]["file"] = {
                "enable": True,
                "level": "DEBUG",
                "apply_filters": False,  # Don't filter, preserve all info
            }

    def _setup_logger(self):
        """Setup the logger with handlers ensuring complete isolation."""
        with self._lock:
            # Create unique logger name with rank, process, and instance isolation
            if self.config.get("separate_by_rank", True):
                logger_name = f"tool_{self.tool_name}_rank_{self.rank_info['global_rank']}_pid_{os.getpid()}_{self.instance_id}"
            else:
                logger_name = f"tool_{self.tool_name}_pid_{os.getpid()}_{self.instance_id}"

            self.logger = logging.getLogger(logger_name)
            self.logger.setLevel(logging.DEBUG)

            # Always add handlers for unique loggers (no handler reuse)
            if not self.logger.handlers:
                self._add_handlers()

            self.logger.propagate = False  # Prevent propagation to root logger

    def _add_handlers(self):
        """Add file and console handlers based on configuration."""
        outputs_config = self.config.get("outputs", {})

        # File handler
        file_config = outputs_config.get("file", {})
        if file_config.get("enable", True) and self.config.get("log_dir"):
            self._add_file_handler(file_config)

        # Console handler
        console_config = outputs_config.get("console", {})
        if console_config.get("enable", True):
            self._add_console_handler(console_config)

    def _add_file_handler(self, file_config: dict):
        """Add rotating file handler with race condition protection."""
        log_dir = self.config["log_dir"]
        if self.config.get("separate_by_rank", True):
            log_dir = os.path.join(log_dir, f"rank_{self.rank_info['global_rank']}")

        os.makedirs(log_dir, exist_ok=True)

        # Parse file size
        max_size = self._parse_size(file_config.get("max_size", "50MB"))
        backup_count = file_config.get("backup_count", 3)

        # Create unique log file per instance to prevent file conflicts
        log_filename = f"{self.tool_name}_pid_{os.getpid()}_{self.instance_id}.log"
        log_path = os.path.join(log_dir, log_filename)

        file_handler = RotatingFileHandler(log_path, maxBytes=max_size, backupCount=backup_count)

        file_handler.setLevel(getattr(logging, file_config.get("level", "WARNING")))
        file_handler.setFormatter(self._get_formatter())

        # Apply noise filter if configured
        if file_config.get("apply_filters", False):
            noise_filters = self.config.get("noise_filters", {})
            file_handler.addFilter(AdvancedNoiseFilter(noise_filters, "file"))

        self.logger.addHandler(file_handler)

    def _add_console_handler(self, console_config: dict):
        """Add console handler."""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, console_config.get("level", "ERROR")))
        console_handler.setFormatter(self._get_formatter())

        # Apply noise filter if configured
        if console_config.get("apply_filters", True):
            noise_filters = self.config.get("noise_filters", {})
            console_handler.addFilter(AdvancedNoiseFilter(noise_filters, "console"))

        self.logger.addHandler(console_handler)

    def _parse_size(self, size_str: str) -> int:
        """Parse file size string to bytes."""
        size_str = size_str.upper()
        if size_str.endswith("MB"):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith("KB"):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith("GB"):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)

    def _get_formatter(self):
        """Get log formatter with complete traceability information."""
        format_str = f"%(asctime)s - {self.tool_name} - %(levelname)s - [Rank:{self.rank_info['global_rank']},PID:{os.getpid()},Instance:{self.instance_id}] - %(message)s"
        return logging.Formatter(format_str)

    def log(self, level: str, message: str, extra_data: dict = None):
        """Log a message with optional extra data (JSON structured format)."""
        if not self.logger:
            return

        with self._lock:
            log_data = {"timestamp": time.time(), "rank": self.rank_info["global_rank"], "tool": self.tool_name, "instance": self.instance_id, "message": message, **(extra_data or {})}
            getattr(self.logger, level.lower())(json.dumps(log_data))

    def debug(self, message: str, extra_data: dict = None):
        """Log a debug message with JSON format."""
        self.log("debug", message, extra_data)

    def info(self, message: str, extra_data: dict = None):
        """Log an info message with JSON format."""
        self.log("info", message, extra_data)

    def warning(self, message: str, extra_data: dict = None):
        """Log a warning message with JSON format."""
        self.log("warning", message, extra_data)

    def error(self, message: str, extra_data: dict = None):
        """Log an error message with JSON format."""
        self.log("error", message, extra_data)

    def critical(self, message: str, extra_data: dict = None):
        """Log a critical message with JSON format."""
        self.log("critical", message, extra_data)

    # Standard logging interface compatibility (plain text format)
    def raw_log(self, level: int, message: str):
        """Log with standard logging level (plain text format)."""
        if not self.logger:
            return
        with self._lock:
            self.logger.log(level, message)

    def debug_plain(self, message: str):
        """Standard debug logging (plain text)."""
        self.raw_log(logging.DEBUG, message)

    def info_plain(self, message: str):
        """Standard info logging (plain text)."""
        self.raw_log(logging.INFO, message)

    def warning_plain(self, message: str):
        """Standard warning logging (plain text)."""
        self.raw_log(logging.WARNING, message)

    def error_plain(self, message: str):
        """Standard error logging (plain text)."""
        self.raw_log(logging.ERROR, message)

    def critical_plain(self, message: str):
        """Standard critical logging (plain text)."""
        self.raw_log(logging.CRITICAL, message)

    def exception(self, message: str):
        """Log an exception with traceback."""
        if not self.logger:
            return
        with self._lock:
            self.logger.exception(message)

    # Property to expose underlying logger for advanced usage
    @property
    def base_logger(self):
        """Access to underlying Python logger for direct standard logging usage.

        Warning: Direct usage bypasses thread safety. Use with caution.
        """
        return self.logger


class BaseTool:
    """Base class for tools with enhanced universal logging capabilities.

    A tool should support the following methods:

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        self.config = config
        self.tool_schema = tool_schema or self.get_openai_tool_schema()
        assert self.tool_schema is not None, "Tool schema is not set!"
        self.name = self.tool_schema.function.name

        # Initialize universal tool logger
        tool_logging_config = config.get("tool_logging_config", {})
        if tool_logging_config.get("enable", True):
            rank_info = self._get_rank_info()

            # Apply tool-specific configuration overrides
            tool_specific_config = tool_logging_config.get("tool_specific", {}).get(self.__class__.__name__, {})
            merged_config = copy.deepcopy(tool_logging_config)
            if tool_specific_config:
                self._deep_update(merged_config, tool_specific_config)

            # Set log directory if not specified
            if not merged_config.get("log_dir"):
                default_local_dir = merged_config.get("default_local_dir", "checkpoints")
                merged_config["log_dir"] = os.path.join(default_local_dir, "tool_logs")

            self.tool_logger = ToolLogger(self.name, rank_info, merged_config)
        else:
            self.tool_logger = None

        if self.tool_logger:
            self.tool_logger.debug("Tool initialized", {"tool_class": self.__class__.__name__, "tool_name": self.name, "schema": self.tool_schema.model_dump(exclude_unset=True, exclude_none=True)})
        else:
            print(json.dumps(self.tool_schema.model_dump(exclude_unset=True, exclude_none=True), indent=2))

    def _get_rank_info(self) -> dict:
        """Get distributed environment rank information."""
        rank_info = {"global_rank": 0, "local_rank": 0, "world_size": 1, "node_id": "unknown", "worker_id": "unknown"}

        # PyTorch distributed environment
        if dist.is_initialized():
            rank_info["global_rank"] = dist.get_rank()
            rank_info["world_size"] = dist.get_world_size()
            rank_info["local_rank"] = int(os.environ.get("LOCAL_RANK", 0))

        # Use torchrun/distributed environment variables (more stable than Ray API)
        rank_info["node_id"] = os.environ.get("GROUP_RANK", os.environ.get("NODE_RANK", "unknown"))
        rank_info["worker_id"] = os.environ.get("RANK", str(rank_info["global_rank"]))

        # Additional torchrun environment info for better identification
        job_id = os.environ.get("JOB_ID", "unknown")
        if job_id != "unknown":
            rank_info["node_id"] = f"{job_id}_{rank_info['node_id']}"

        # Fallback: use process ID as identifier
        if rank_info["global_rank"] == 0 and rank_info["node_id"] == "unknown":
            rank_info["global_rank"] = os.getpid() % 1000
            rank_info["node_id"] = f"pid_{os.getpid()}"

        return rank_info

    def _deep_update(self, base_dict: dict, update_dict: dict):
        """Deep merge two dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
        """
        if instance_id is None:
            return str(uuid4())
        else:
            return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, Dict[str, Any]]:
        """Enhanced execute method with integrated tool logging.

        Args:
            instance_id: The instance id of the tool.
            parameters: The json string of the parameters of the tool.

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        start_time = time.time()

        # Log execution start
        if self.tool_logger:
            self.tool_logger.info("EXECUTION_START", {"instance_id": instance_id, "parameters_size": len(str(parameters)), "tool_class": self.__class__.__name__})

        try:
            response, reward, success, specific_metrics = await self._execute_impl(instance_id, parameters, **kwargs)
            end_time = time.time()

            # Collect base tool metrics
            base_metrics = {
                "latency_ms": (end_time - start_time) * 1000,
                "response_char_length": len(response) if response else 0,
                "success": success,
                "tool_name": self.name,
            }

            # Combine base metrics with tool-specific metrics
            combined_metrics = {"base_metrics": base_metrics, "specific_metrics": specific_metrics or {}}

            # Log execution result
            if self.tool_logger:
                log_level = "info" if success else "warning"

                # Check if this is a noise error (timeout, etc.)
                if not success and self._is_noise_error(response, specific_metrics):
                    log_level = "debug"  # Will be handled by noise filter

                self.tool_logger.log(log_level, "EXECUTION_COMPLETE", {"instance_id": instance_id, "success": success, "latency_ms": base_metrics["latency_ms"], "response_preview": response[:200] if response else None, "metrics": combined_metrics})

            return response, reward, combined_metrics

        except Exception as e:
            end_time = time.time()

            # Log exception
            if self.tool_logger:
                self.tool_logger.error("EXECUTION_EXCEPTION", {"instance_id": instance_id, "exception": str(e), "exception_type": type(e).__name__, "latency_ms": (end_time - start_time) * 1000})

            raise

    def _is_noise_error(self, response: str, metrics: dict) -> bool:
        """Check if this is a noise error (timeout, retryable error, etc.)."""
        if not response:
            return False

        noise_indicators = ["TimeLimitExceeded", "time limit exceed", "timeout", "Gateway Timeout", "Connection refused", "retry"]

        response_lower = response.lower()
        return any(indicator.lower() in response_lower for indicator in noise_indicators)

    async def _execute_impl(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Implementation method for tool execution. Override this in subclasses.

        Args:
            instance_id: The instance id of the tool.
            parameters: The parameters of the tool.

        Returns:
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            success: Whether the execution was successful.
            specific_metrics: Tool-specific metrics dictionary.
        """
        return "Updated the tool state.", 0.0, True, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward of the tool.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The reward of the tool.
        """
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance.

        Args:
            instance_id: The instance id of the tool.
        """
        pass


def initialize_tools_from_config(tools_config_file, tool_logging_config: dict = None) -> List[BaseTool]:
    """Initialize tools from config file with optional tool logging configuration.

    Args:
        tools_config_file: The config file of the tools.
        tool_logging_config: Optional tool logging configuration to inject.

    Returns:
        A list of tools.
    """
    tools_config = OmegaConf.load(tools_config_file)

    tool_list = []
    for tool_config in tools_config.tools:
        cls_name = tool_config.class_name
        module_name, class_name = cls_name.rsplit(".", 1)

        if module_name not in sys.modules:
            spec = importlib.util.find_spec(module_name)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        else:
            module = sys.modules[module_name]

        tool_cls = getattr(module, class_name)

        if tool_config.get("tool_schema", None) is None:
            tool_schema = None
        else:
            tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
            tool_schema = OpenAIFunctionToolSchema.parse_obj(tool_schema_dict)

        # Prepare tool configuration with optional logging config
        tool_config_dict = OmegaConf.to_container(tool_config.config, resolve=True)
        if tool_logging_config:
            tool_config_dict["tool_logging_config"] = tool_logging_config

        tool = tool_cls(config=tool_config_dict, tool_schema=tool_schema)
        tool_list.append(tool)

    return tool_list
