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
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel, model_validator
from transformers import PreTrainedTokenizer

from verl.tools.schemas import OpenAIFunctionToolCall, OpenAIFunctionToolSchema
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

BASE_CHAT_HISTORY = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "I am a user."}]


class FinishReasonTypeEnum(str, Enum):
    """The enum for finish reason type."""

    LENGTH = "length"
    STOP = "stop"
    TOOL_CALL = "tool_calls"

    @classmethod
    def from_str(cls, value: str) -> "FinishReasonTypeEnum":
        if value == "stop":
            return cls.STOP
        elif value == "length":
            return cls.LENGTH
        elif value == "tool_calls":
            return cls.TOOL_CALL
        else:
            raise ValueError(f"Unsupported finish reason type: {value}")


class Message(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[OpenAIFunctionToolCall]] = None


class AsyncRolloutRequestStateEnum(str, Enum):
    """The enum for async rollout request state."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TOOL_CALLING = "tool_calling"


class AsyncRolloutRequest(BaseModel):
    """The data model for async rollout."""

    batch_data_id: int = 0
    rollout_offset: int = 0
    request_id: str
    state: AsyncRolloutRequestStateEnum
    messages: List[Message]
    tool_schemas: Optional[List[OpenAIFunctionToolSchema]] = None
    tools_kwargs: Dict[str, Any] = {}
    input_ids: List[int]
    prompt_ids: List[int]
    response_ids: List[int]
    attention_mask: List[int]
    prompt_attention_mask: List[int]
    response_attention_mask: List[int]
    position_ids: List[int]
    prompt_position_ids: List[int]
    response_position_ids: List[int]
    loss_mask: List[int]
    prompt_loss_mask: List[int]
    response_loss_mask: List[int]
    reward_scores: Dict[str, float]
    max_prompt_len: int
    max_response_len: int = 8192
    max_model_len: int = 32768
    metrics: Dict[str, List[Any]] = {}

    use_inference_chat_template: bool
    enable_tokenization_sanity_check: bool
    generation_prompt_ids: List[int]
    base_conv_wo_gen_prompt_end_pos: int
    base_conv_with_gen_prompt_end_pos: int

    turn_stats: Dict[str, Any] = {}
    turn_stats_list: List[Dict[str, Any]] = []

    # Enhanced conversation tracking
    conversation_metadata: Dict[str, Any] = {}
    termination_reason: Optional[str] = None
    termination_metadata: Dict[str, Any] = {}
    turn_tool_calls_detail: List[int] = []

    @model_validator(mode="before")
    @classmethod
    def initialize_request(cls, values):
        if not (messages := values.get("messages")):
            raise ValueError("messages is required for AsyncRolloutRequest initialization")
        if not (max_prompt_len := values.get("max_prompt_len")):
            raise ValueError("max_prompt_len is required for AsyncRolloutRequest initialization")
        if not (tokenizer := values.pop("tokenizer", None)):
            raise ValueError("tokenizer is required for AsyncRolloutRequest initialization")

        values["messages"] = [Message.model_validate(msg) for msg in messages]

        tools = [tool.model_dump() for tool in tool_schemas] if (tool_schemas := values.get("tool_schemas", [])) else None
        tokens_without_prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=False, tokenize=True)
        if not values.get("input_ids") or not values.get("attention_mask"):
            tokenization_dict_with_prompt = tokenizer.apply_chat_template(messages, tools=[tool.model_dump() for tool in tool_schemas], add_generation_prompt=True, tokenize=True, return_dict=True)
            values["input_ids"], values["attention_mask"] = tokenization_dict_with_prompt["input_ids"], tokenization_dict_with_prompt["attention_mask"]
            if len(values["input_ids"]) > max_prompt_len:
                # Only log the warning to avoid truncating in the middle of generation prompt. Consider raising an error for this case in the future.
                logger.warning(f"Prompt {values['batch_data_id']} length {len(values['input_ids'])} greater than max_prompt_len {max_prompt_len} after applied chat template with tools.")

        values["prompt_ids"], values["prompt_attention_mask"] = values["input_ids"], values["attention_mask"]
        values["position_ids"] = values["prompt_position_ids"] = compute_position_id_with_mask(torch.tensor(values["attention_mask"])).tolist()
        values["loss_mask"] = values["prompt_loss_mask"] = [0] * len(values["input_ids"])
        values["generation_prompt_ids"] = values["input_ids"][len(tokens_without_prompt) :]
        values["base_conv_wo_gen_prompt_end_pos"] = len(tokenizer.apply_chat_template(BASE_CHAT_HISTORY, tools=tools, add_generation_prompt=False, tokenize=False))
        values["base_conv_with_gen_prompt_end_pos"] = len(tokenizer.apply_chat_template(BASE_CHAT_HISTORY, tools=tools, add_generation_prompt=True, tokenize=False))

        # Initialize enhanced conversation tracking
        values["conversation_metadata"] = {
            "total_turns": 0,
            "role_turns": {"assistant": 0, "user": 0, "tool": 0, "system": 0},
            "role_tokens": {"assistant": 0, "user": 0, "tool": 0, "system": 0},
            "role_chars": {"assistant": 0, "user": 0, "tool": 0, "system": 0},
            "turn_token_lengths": [],
            "turn_char_lengths": [],
            "assistant_tool_calls_total": 0,
            "turns_with_tools_count": 0,
        }

        values["termination_metadata"] = {"final_turn_count": 0, "final_token_count": 0}

        values["turn_tool_calls_detail"] = []

        return values

    def _update_input_ids(self, new_input_ids: List[int], attention_mask: bool, loss_mask: bool) -> None:
        """
        Update the input_ids, attention_mask, position_ids, and loss_mask of the request in additive manner.
        """
        self.input_ids += new_input_ids
        attention_mask = [int(attention_mask)] * len(new_input_ids)
        self.attention_mask += attention_mask
        self.loss_mask += [int(loss_mask)] * len(new_input_ids)
        self.position_ids += (compute_position_id_with_mask(torch.tensor(attention_mask)) + (self.position_ids[-1] + 1)).tolist()

        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Request {self.request_id} has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    def get_generation_prompt_ids(self, tokenizer: PreTrainedTokenizer) -> list[int]:
        generation_prompt_ids = [] if self.input_ids[-len(self.generation_prompt_ids) :] == self.generation_prompt_ids else self.generation_prompt_ids
        if generation_prompt_ids:
            self._update_input_ids(generation_prompt_ids, attention_mask=True, loss_mask=False)

        if self.use_inference_chat_template:
            return tokenizer.apply_chat_template([msg.model_dump() for msg in self.messages], tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None), add_generation_prompt=True, tokenize=True)
        else:
            return self.input_ids

    def add_assistant_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        tool_calls: Optional[List[OpenAIFunctionToolCall]] = None,
    ) -> Dict[str, Any]:
        self.messages.append(Message(role="assistant", content=content, tool_calls=tool_calls))
        full_content = tokenizer.apply_chat_template([*BASE_CHAT_HISTORY, self.messages[-1]], tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None), add_generation_prompt=False, tokenize=False)
        content_ids = tokenizer.encode(full_content[self.base_conv_with_gen_prompt_end_pos :], add_special_tokens=False)
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=True)

        # 返回该轮次的统计信息
        return {"role": "assistant", "token_count": len(content_ids), "char_count": len(content), "has_tool_calls": tool_calls is not None, "tool_calls_count": len(tool_calls) if tool_calls else 0}

    def add_tool_response_messages(self, tokenizer: PreTrainedTokenizer, contents: list[str]) -> Dict[str, Any]:
        if not contents:
            return {"role": "tool", "token_count": 0, "char_count": 0, "tool_count": 0}

        self.messages.extend([Message(role="tool", content=content) for content in contents])
        full_content = tokenizer.apply_chat_template([*BASE_CHAT_HISTORY, *self.messages[-len(contents) :]], tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None), add_generation_prompt=False, tokenize=False)
        content_ids = tokenizer.encode(full_content[self.base_conv_wo_gen_prompt_end_pos :], add_special_tokens=False)
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=False)

        # 计算统计信息
        total_chars = sum(len(content) for content in contents)

        return {"role": "tool", "token_count": len(content_ids), "char_count": total_chars, "tool_count": len(contents)}

    def check_if_tool_response_messages_overlong(self, tokenizer: PreTrainedTokenizer, contents: list[str]) -> bool:
        if not contents:
            return False
        content = tokenizer.apply_chat_template([*BASE_CHAT_HISTORY, *self.messages[-len(contents) :]], tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None), add_generation_prompt=False, tokenize=False)
        content_ids = tokenizer.encode(content[self.base_conv_wo_gen_prompt_end_pos :], add_special_tokens=False)
        return (len(self.input_ids) + len(content_ids)) > self.max_model_len

    def update_metrics(self, metrics: Any, tool_id: str) -> None:
        """
        metrics: should be a dict of tools_name -> Any
        """
        if self.metrics.get(tool_id) is None:
            self.metrics[tool_id] = []
        self.metrics[tool_id].append(metrics)

    def get_aggregated_tool_metrics(self) -> Dict[str, Any]:
        """Get aggregated tool metrics for this request.

        This method processes the metrics collected during tool execution and aggregates them into:
        1. Shared metrics: Common metrics across all tools (latency, success rate, etc.)
        2. Tool-specific metrics: Metrics unique to each tool type

        Data structure in self.metrics:
        {
            "search": [metrics1, metrics2, ...],           # search tool execution records
            "code_interpreter": [metrics3, metrics4, ...], # code tool execution records
            "turn_stats": [...],                           # conversation turn statistics
            "token_stats": [...]                           # token usage statistics
        }

        Returns:
            Dict containing:
            - shared_metrics: Aggregated metrics common to all tools
            - tool_specific_metrics: Per-tool aggregated metrics with tool-specific data
        """
        # Shared metrics aggregated across all tool types
        shared_metrics = {
            "tool_invocation_count": 0,
            "tool_response_char_length_total": 0,
            "tool_response_char_length_avg": 0.0,
            "tool_latency_ms_total": 0.0,
            "tool_latency_ms_min": float("inf"),
            "tool_latency_ms_max": 0.0,
            "tool_latency_ms_avg": 0.0,
            "tool_success_count": 0,
            "tool_success_rate": 0.0,
            "tools_used": set(),
        }

        # Tool-specific metrics organized by tool name
        tool_specific_metrics = {}

        total_calls = 0
        total_char_length = 0

        # Filter out non-tool metric keys to get only actual tool names
        # self.metrics contains both tool execution data (keyed by tool name)
        # and other metadata (turn_stats, token_stats)
        tool_metrics_keys = [k for k in self.metrics.keys() if k not in ["turn_stats", "token_stats"]]

        for tool_id in tool_metrics_keys:
            metrics_list = self.metrics[tool_id]
            shared_metrics["tools_used"].add(tool_id)

            # Initialize tool-specific metrics for this tool
            tool_specific_metrics[tool_id] = {
                "invocation_count": 0,
                "success_count": 0,
                "success_rate": 0.0,
                "total_latency_ms": 0.0,
                "avg_latency_ms": 0.0,
                "specific_metrics": {},  # Store tool-unique metrics
            }

            for metrics in metrics_list:
                if isinstance(metrics, dict):
                    total_calls += 1
                    tool_specific_metrics[tool_id]["invocation_count"] += 1

                    # Extract base metrics and tool-specific metrics
                    base_metrics = metrics.get("base_metrics", {})
                    specific_metrics = metrics.get("specific_metrics", {})

                    # Process base metrics for shared aggregation
                    if "latency_ms" in base_metrics:
                        latency = base_metrics["latency_ms"]
                        shared_metrics["tool_latency_ms_total"] += latency
                        shared_metrics["tool_latency_ms_min"] = min(shared_metrics["tool_latency_ms_min"], latency)
                        shared_metrics["tool_latency_ms_max"] = max(shared_metrics["tool_latency_ms_max"], latency)
                        tool_specific_metrics[tool_id]["total_latency_ms"] += latency

                    if "response_char_length" in base_metrics:
                        char_length = base_metrics["response_char_length"]
                        total_char_length += char_length

                    if "success" in base_metrics and base_metrics["success"]:
                        shared_metrics["tool_success_count"] += 1
                        tool_specific_metrics[tool_id]["success_count"] += 1

                    # Aggregate tool-specific metrics
                    for key, value in specific_metrics.items():
                        if key not in tool_specific_metrics[tool_id]["specific_metrics"]:
                            tool_specific_metrics[tool_id]["specific_metrics"][key] = []
                        tool_specific_metrics[tool_id]["specific_metrics"][key].append(value)

        # Calculate shared metric averages
        shared_metrics["tool_invocation_count"] = total_calls
        shared_metrics["tool_response_char_length_total"] = total_char_length

        if total_calls > 0:
            shared_metrics["tool_latency_ms_avg"] = shared_metrics["tool_latency_ms_total"] / total_calls
            shared_metrics["tool_response_char_length_avg"] = total_char_length / total_calls
            shared_metrics["tool_success_rate"] = shared_metrics["tool_success_count"] / total_calls

        # Handle edge case for min latency
        if shared_metrics["tool_latency_ms_min"] == float("inf"):
            shared_metrics["tool_latency_ms_min"] = 0.0

        # Calculate per-tool specific metric averages
        for tool_id, tool_stats in tool_specific_metrics.items():
            invocation_count = tool_stats["invocation_count"]

            if invocation_count > 0:
                tool_stats["success_rate"] = tool_stats["success_count"] / invocation_count
                tool_stats["avg_latency_ms"] = tool_stats["total_latency_ms"] / invocation_count

            # Aggregate tool-specific metrics
            aggregated_specific = {}
            for metric_name, values in tool_stats["specific_metrics"].items():
                if not values:
                    continue

                # Apply different aggregation strategies based on value types
                if all(isinstance(v, (int, float)) for v in values):
                    # Numeric types: calculate mean, min, max, total
                    aggregated_specific[f"{metric_name}_avg"] = sum(values) / len(values)
                    aggregated_specific[f"{metric_name}_min"] = min(values)
                    aggregated_specific[f"{metric_name}_max"] = max(values)
                    aggregated_specific[f"{metric_name}_total"] = sum(values)
                elif all(isinstance(v, bool) for v in values):
                    # Boolean types: calculate true rate and count
                    aggregated_specific[f"{metric_name}_rate"] = sum(values) / len(values)
                    aggregated_specific[f"{metric_name}_count"] = sum(values)
                elif all(isinstance(v, str) for v in values):
                    # String types: calculate unique values
                    unique_values = list(set(values))
                    aggregated_specific[f"{metric_name}_unique_count"] = len(unique_values)
                    if len(unique_values) <= 10:  # Avoid too many unique values
                        aggregated_specific[f"{metric_name}_unique_values"] = unique_values
                else:
                    # Other types: keep sample values
                    aggregated_specific[f"{metric_name}_samples"] = values[:5] if len(values) > 5 else values

            tool_stats["specific_metrics"] = aggregated_specific

        # Convert set to list for JSON serialization
        shared_metrics["tools_used"] = list(shared_metrics["tools_used"])

        return {"shared_metrics": shared_metrics, "tool_specific_metrics": tool_specific_metrics}

    def track_conversation_turn_from_stats(self, role: str, turn_stats: Dict[str, Any]):
        """根据现有的 turn_stats 追踪对话轮次信息"""
        tokens = turn_stats.get("token_count", 0)
        chars = turn_stats.get("char_count", 0)
        tool_calls_count = turn_stats.get("tool_calls_count", 0)

        # 确保角色存在于字典中
        if role not in self.conversation_metadata["role_turns"]:
            self.conversation_metadata["role_turns"][role] = 0
            self.conversation_metadata["role_tokens"][role] = 0
            self.conversation_metadata["role_chars"][role] = 0

        # 更新角色统计
        self.conversation_metadata["role_turns"][role] += 1
        self.conversation_metadata["role_tokens"][role] += tokens
        self.conversation_metadata["role_chars"][role] += chars

        # 更新总轮次
        self.conversation_metadata["total_turns"] += 1

        # 记录轮次长度
        self.conversation_metadata["turn_token_lengths"].append(tokens)
        self.conversation_metadata["turn_char_lengths"].append(chars)

        # 记录工具调用
        if tool_calls_count > 0:
            self.conversation_metadata["turns_with_tools_count"] += 1
            if role == "assistant":
                self.conversation_metadata["assistant_tool_calls_total"] += tool_calls_count

        # 记录turn-level工具调用详情
        turn_index = self.conversation_metadata["total_turns"] - 1
        while len(self.turn_tool_calls_detail) <= turn_index:
            self.turn_tool_calls_detail.append(0)
        self.turn_tool_calls_detail[turn_index] = tool_calls_count

    def initialize_conversation_from_prompt(self, messages: List[Message], tokenizer):
        """从初始 prompt 中的 messages 初始化对话统计"""
        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                tokens = len(tokenizer.encode(msg.content))
                chars = len(msg.content)
                role = msg.role

                # 确保角色存在于字典中
                if role not in self.conversation_metadata["role_turns"]:
                    self.conversation_metadata["role_turns"][role] = 0
                    self.conversation_metadata["role_tokens"][role] = 0
                    self.conversation_metadata["role_chars"][role] = 0

                # 更新统计
                self.conversation_metadata["role_turns"][role] += 1
                self.conversation_metadata["role_tokens"][role] += tokens
                self.conversation_metadata["role_chars"][role] += chars
                self.conversation_metadata["total_turns"] += 1
                self.conversation_metadata["turn_token_lengths"].append(tokens)
                self.conversation_metadata["turn_char_lengths"].append(chars)

                # 检查工具调用
                tool_calls_count = 0
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls_count = len(msg.tool_calls)
                    if role == "assistant":
                        self.conversation_metadata["assistant_tool_calls_total"] += tool_calls_count
                    self.conversation_metadata["turns_with_tools_count"] += 1

                # 记录turn-level工具调用
                turn_index = self.conversation_metadata["total_turns"] - 1
                while len(self.turn_tool_calls_detail) <= turn_index:
                    self.turn_tool_calls_detail.append(0)
                self.turn_tool_calls_detail[turn_index] = tool_calls_count

    def set_termination_reason(self, reason: str):
        """设置终止原因"""
        self.termination_reason = reason
        self.termination_metadata["final_turn_count"] = self.conversation_metadata["total_turns"]
        self.termination_metadata["final_token_count"] = sum(self.conversation_metadata["role_tokens"].values())

    def get_enhanced_tool_metrics(self) -> Dict[str, Any]:
        """获取增强的工具指标，包括 per-tool specific metrics"""

        # 1. Trajectory-level 工具调用统计
        trajectory_tool_calls = 0
        turn_tool_calls_all = []  # 所有轮次的工具调用数

        # 2. 收集所有工具指标的原始值
        all_latencies = []
        all_response_lengths = []
        all_success_rates = []

        # 3. Per-tool 指标收集
        tool_specific_raw_data = {}

        # 处理 turn-level 工具调用统计
        for turn_calls in self.turn_tool_calls_detail:
            if turn_calls > 0:
                turn_tool_calls_all.append(turn_calls)

        # 过滤掉非工具指标的键
        tool_metrics_keys = [k for k in self.metrics.keys() if k not in ["turn_stats", "token_stats"]]

        for tool_name in tool_metrics_keys:
            metrics_list = self.metrics[tool_name]
            tool_specific_raw_data[tool_name] = {
                "calls_per_trajectory": len(metrics_list),
                "calls_per_turn_values": [],
                "latency_ms_values": [],
                "success_rate_values": [],
                "code_char_len_values": [],
                "response_char_len_values": [],
            }

            trajectory_tool_calls += len(metrics_list)

            for metrics in metrics_list:
                if isinstance(metrics, dict):
                    base_metrics = metrics.get("base_metrics", {})
                    specific_metrics = metrics.get("specific_metrics", {})

                    # 收集基础指标
                    if "latency_ms" in base_metrics:
                        latency = base_metrics["latency_ms"]
                        all_latencies.append(latency)
                        tool_specific_raw_data[tool_name]["latency_ms_values"].append(latency)

                    if "response_char_length" in base_metrics:
                        resp_len = base_metrics["response_char_length"]
                        all_response_lengths.append(resp_len)
                        tool_specific_raw_data[tool_name]["response_char_len_values"].append(resp_len)

                    if "success" in base_metrics:
                        success = base_metrics["success"]
                        success_rate = 1.0 if success else 0.0
                        all_success_rates.append(success_rate)
                        tool_specific_raw_data[tool_name]["success_rate_values"].append(success_rate)

                    # 收集工具特定指标
                    if "code_char_len" in specific_metrics:
                        code_len = specific_metrics["code_char_len"]
                        tool_specific_raw_data[tool_name]["code_char_len_values"].append(code_len)

        # 4. 构建聚合指标
        aggregated_metrics = {"tool_metrics": {}, "tool_specific_metrics": {}}

        # Trajectory-level 工具调用统计
        aggregated_metrics["tool_metrics"]["tool_calls_per_trajectory"] = trajectory_tool_calls

        # Turn-level 工具调用统计
        if turn_tool_calls_all:
            aggregated_metrics["tool_metrics"]["tool_calls_per_turn_min"] = min(turn_tool_calls_all)
            aggregated_metrics["tool_metrics"]["tool_calls_per_turn_avg"] = sum(turn_tool_calls_all) / len(turn_tool_calls_all)
            aggregated_metrics["tool_metrics"]["tool_calls_per_turn_max"] = max(turn_tool_calls_all)
            # 保存原始值用于批次聚合
            aggregated_metrics["tool_metrics"]["tool_calls_per_turn_values"] = turn_tool_calls_all

        # 全局工具指标
        if all_latencies:
            aggregated_metrics["tool_metrics"]["latency_ms_min"] = min(all_latencies)
            aggregated_metrics["tool_metrics"]["latency_ms_avg"] = sum(all_latencies) / len(all_latencies)
            aggregated_metrics["tool_metrics"]["latency_ms_max"] = max(all_latencies)
            aggregated_metrics["tool_metrics"]["latency_ms_values"] = all_latencies

        if all_response_lengths:
            aggregated_metrics["tool_metrics"]["response_char_len_min"] = min(all_response_lengths)
            aggregated_metrics["tool_metrics"]["response_char_len_avg"] = sum(all_response_lengths) / len(all_response_lengths)
            aggregated_metrics["tool_metrics"]["response_char_len_max"] = max(all_response_lengths)
            aggregated_metrics["tool_metrics"]["response_char_len_values"] = all_response_lengths

        if all_success_rates:
            trajectory_success_rate = sum(all_success_rates) / len(all_success_rates)
            aggregated_metrics["tool_metrics"]["success_rate"] = trajectory_success_rate

        # Per-tool specific metrics
        for tool_name, raw_data in tool_specific_raw_data.items():
            tool_aggregated = {}

            # Calls per trajectory and turn
            tool_aggregated["calls_per_trajectory"] = raw_data["calls_per_trajectory"]

            # 延迟指标
            if raw_data["latency_ms_values"]:
                values = raw_data["latency_ms_values"]
                tool_aggregated["latency_ms_min"] = min(values)
                tool_aggregated["latency_ms_avg"] = sum(values) / len(values)
                tool_aggregated["latency_ms_max"] = max(values)
                tool_aggregated["latency_ms_values"] = values  # 原始值

            # 成功率指标
            if raw_data["success_rate_values"]:
                values = raw_data["success_rate_values"]
                tool_aggregated["success_rate_min"] = min(values)
                tool_aggregated["success_rate_avg"] = sum(values) / len(values)
                tool_aggregated["success_rate_max"] = max(values)
                tool_aggregated["success_rate_values"] = values

            # 代码长度指标
            if raw_data["code_char_len_values"]:
                values = raw_data["code_char_len_values"]
                tool_aggregated["code_char_len_min"] = min(values)
                tool_aggregated["code_char_len_avg"] = sum(values) / len(values)
                tool_aggregated["code_char_len_max"] = max(values)
                tool_aggregated["code_char_len_values"] = values

            # 响应长度指标
            if raw_data["response_char_len_values"]:
                values = raw_data["response_char_len_values"]
                tool_aggregated["response_char_len_min"] = min(values)
                tool_aggregated["response_char_len_avg"] = sum(values) / len(values)
                tool_aggregated["response_char_len_max"] = max(values)
                tool_aggregated["response_char_len_values"] = values

            aggregated_metrics["tool_specific_metrics"][tool_name] = tool_aggregated

        return aggregated_metrics

    def get_conversation_metrics(self) -> Dict[str, Any]:
        """获取对话指标"""
        conv_meta = self.conversation_metadata

        # 计算比例
        total_turns = conv_meta["total_turns"]
        tool_turn_ratio = conv_meta["turns_with_tools_count"] / total_turns if total_turns > 0 else 0
        assistant_turn_ratio = conv_meta["role_turns"]["assistant"] / total_turns if total_turns > 0 else 0

        return {
            # 基础对话统计
            "total_turns": total_turns,
            "total_tokens": sum(conv_meta["role_tokens"].values()),
            "total_chars": sum(conv_meta["role_chars"].values()),
            # 按角色统计
            "assistant_turns": conv_meta["role_turns"].get("assistant", 0),
            "user_turns": conv_meta["role_turns"].get("user", 0),
            "tool_turns": conv_meta["role_turns"].get("tool", 0),
            "system_turns": conv_meta["role_turns"].get("system", 0),
            "assistant_tokens": conv_meta["role_tokens"].get("assistant", 0),
            "user_tokens": conv_meta["role_tokens"].get("user", 0),
            "tool_tokens": conv_meta["role_tokens"].get("tool", 0),
            "system_tokens": conv_meta["role_tokens"].get("system", 0),
            "assistant_chars": conv_meta["role_chars"].get("assistant", 0),
            "user_chars": conv_meta["role_chars"].get("user", 0),
            "tool_chars": conv_meta["role_chars"].get("tool", 0),
            "system_chars": conv_meta["role_chars"].get("system", 0),
            # 工具使用统计
            "assistant_tool_calls": conv_meta["assistant_tool_calls_total"],
            "turns_with_tools": conv_meta["turns_with_tools_count"],
            # 轮次长度统计 (原始值，用于批次聚合)
            "turn_token_lengths": conv_meta["turn_token_lengths"],
            "turn_char_lengths": conv_meta["turn_char_lengths"],
            # 比例统计
            "tool_turn_ratio": tool_turn_ratio,
            "assistant_turn_ratio": assistant_turn_ratio,
        }

    def get_termination_metrics(self) -> Dict[str, Any]:
        """获取终止指标"""
        return {"reason": self.termination_reason, "final_turn_count": self.termination_metadata["final_turn_count"], "final_token_count": self.termination_metadata["final_token_count"]}

    def finalize(
        self,
        tokenizer: PreTrainedTokenizer,
        reward_scores: Dict[str, float],
        finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP,
    ) -> None:
        self.state = AsyncRolloutRequestStateEnum.COMPLETED
        self.reward_scores = reward_scores
        if self.enable_tokenization_sanity_check:
            full_tokens = tokenizer.apply_chat_template([msg.model_dump() for msg in self.messages], tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None), add_generation_prompt=False, tokenize=True)
            if self.input_ids != full_tokens:
                logger.warning("Inconsistent training and inference tokenization detected. This may lead to unexpected behavior during training. Please review your chat template to determine if this is intentional. For more information, refer to the multiturn README.md.")
                logger.info(f"Inference tokenization result:\n{tokenizer.decode(full_tokens, skip_special_tokens=False)}\ntraining content:\n{tokenizer.decode(self.input_ids, skip_special_tokens=False)}")

        # In case we failed to generate the assistant message and the generation prompt ids were already added to input_ids, remove them from the end of input_ids
        if self.input_ids[-len(self.generation_prompt_ids) :] == self.generation_prompt_ids:
            self.input_ids = self.input_ids[: -len(self.generation_prompt_ids)]
            self.attention_mask = self.attention_mask[: -len(self.generation_prompt_ids)]
            self.position_ids = self.position_ids[: -len(self.generation_prompt_ids)]
            self.loss_mask = self.loss_mask[: -len(self.generation_prompt_ids)]

        self.response_ids = self.input_ids[len(self.prompt_ids) :]
        if finish_reason_type == FinishReasonTypeEnum.STOP:
            pass
        elif finish_reason_type == FinishReasonTypeEnum.LENGTH:
            pass
        else:
            raise ValueError(f"Unsupported finalize finish reason type: {finish_reason_type}")
        self.truncate_output_ids(tokenizer)
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Request {self.request_id} has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    def truncate_output_ids(self, tokenizer: PreTrainedTokenizer) -> None:
        self.input_ids = self.input_ids[: self.max_model_len]
        self.attention_mask = self.attention_mask[: self.max_model_len]
        self.position_ids = self.position_ids[: self.max_model_len]
        self.loss_mask = self.loss_mask[: self.max_model_len]
        self.response_ids = self.input_ids[len(self.prompt_ids) :][: self.max_response_len]
        self.response_attention_mask = self.attention_mask[len(self.prompt_attention_mask) :][: self.max_response_len]
        self.response_position_ids = self.position_ids[len(self.prompt_position_ids) :][: self.max_response_len]
        self.response_loss_mask = self.loss_mask[len(self.prompt_loss_mask) :][: self.max_response_len]
