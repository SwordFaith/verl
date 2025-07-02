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
import difflib
import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch
from pydantic import BaseModel, model_validator
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin

from verl.tools.schemas import OpenAIFunctionToolCall, OpenAIFunctionToolSchema
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

BASE_CHAT_HISTORY = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


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
    content: str | Dict[str, Any] | List[Dict[str, Any]]
    tool_calls: Optional[List[OpenAIFunctionToolCall]] = None


class AsyncRolloutRequestStateEnum(str, Enum):
    """The enum for async rollout request state."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TOOL_CALLING = "tool_calling"
    INTERACTING = "interacting"


class TokenizationSanityCheckModeEnum(str, Enum):
    """The enum for tokenization sanity check mode."""

    OFF = "off"
    STRICT = "strict"
    IGNORE_STRIPPABLE = "ignore_strippable"


class AsyncRolloutRequest(BaseModel):
    """The data model for async rollout."""

    batch_data_id: int = 0
    rollout_offset: int = 0
    request_id: str
    uid: str | None = ""
    state: AsyncRolloutRequestStateEnum
    messages: List[Message]
    multi_modal_keys: Optional[List[str]] = None
    multi_modal_data: Optional[Dict[str, Any]] = None
    tool_schemas: Optional[List[OpenAIFunctionToolSchema]] = None
    tools_kwargs: Dict[str, Any] = {}
    interaction_kwargs: Dict[str, Any] = {}
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
    tokenization_sanity_check_mode: TokenizationSanityCheckModeEnum
    generation_prompt_ids: List[int]
    base_conv_wo_gen_prompt_end_pos: int
    base_conv_with_gen_prompt_end_pos: int

    turn_stats: Dict[str, Any] = {}
    turn_stats_list: List[Dict[str, Any]] = []

    # Unified conversation tracking
    turn_details: List[Dict[str, Any]] = []  # Single source of turn data
    termination_reason: Optional[str] = None
    turn_tool_calls_detail: List[int] = []
    tool_truncation_metrics: List[Dict[str, Any]] = []

    # Turn-level and tool rewards for multi-turn training
    turn_level_rewards: List[
        float
    ] = []  # List[float] - rewards for each message turn from tool execute() calls, 0.0 for non-tool turns
    tool_rewards: Dict[
        str, float
    ] = {}  # Dict[str, float] - final calc_reward() scores for each tool name used in trajectory

    @model_validator(mode="before")
    @classmethod
    def initialize_request(cls, values):
        if not (messages := values.get("messages")):
            raise ValueError("messages is required for AsyncRolloutRequest initialization")
        if not (max_prompt_len := values.get("max_prompt_len")):
            raise ValueError("max_prompt_len is required for AsyncRolloutRequest initialization")
        if not (processing_class := values.pop("processing_class", None)):
            raise ValueError("processing_class is required for AsyncRolloutRequest initialization")

        values["messages"] = [Message.model_validate(msg) for msg in messages]

        # If there is no multi_modal_keys, we assume the multi-modal data is image and video.
        if not values.get("multi_modal_keys"):
            values["multi_modal_keys"] = ["image", "video"]
        if not values.get("multi_modal_data"):
            values["multi_modal_data"] = {key: [] for key in values["multi_modal_keys"]}
        else:
            # check if all multi_modal_keys are in multi_modal_data
            for key in values["multi_modal_keys"]:
                if key not in values["multi_modal_data"]:
                    values["multi_modal_data"][key] = []

        tools = (
            [tool.model_dump() for tool in tool_schemas] if (tool_schemas := values.get("tool_schemas", [])) else None
        )

        multi_modal_data = values["multi_modal_data"]
        tokens_without_prompt = cls._handle_apply_chat_template(
            processing_class,
            messages,
            multi_modal_data=multi_modal_data,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
        )
        if not values.get("input_ids") or not values.get("attention_mask"):
            tokenization_dict_with_prompt = cls._handle_apply_chat_template(
                processing_class,
                messages,
                multi_modal_data=multi_modal_data,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
            )

            values["input_ids"], values["attention_mask"] = (
                tokenization_dict_with_prompt["input_ids"],
                tokenization_dict_with_prompt["attention_mask"],
            )
            if len(values["input_ids"]) > max_prompt_len:
                # Only log the warning to avoid truncating in the middle of generation prompt. Consider raising an
                # error for this case in the future.
                logger.warning(
                    f"Prompt {values['batch_data_id']} length {len(values['input_ids'])} greater than max_prompt_len "
                    f"{max_prompt_len} after applied chat template with tools."
                )

        values["prompt_ids"], values["prompt_attention_mask"] = values["input_ids"], values["attention_mask"]
        values["position_ids"] = values["prompt_position_ids"] = compute_position_id_with_mask(
            torch.tensor(values["attention_mask"])
        ).tolist()
        values["loss_mask"] = values["prompt_loss_mask"] = [0] * len(values["input_ids"])
        values["generation_prompt_ids"] = values["input_ids"][len(tokens_without_prompt) :]
        # Initialize unified conversation tracking
        values["base_conv_wo_gen_prompt_end_pos"] = len(
            cls._handle_apply_chat_template(
                processing_class,
                BASE_CHAT_HISTORY,
                multi_modal_data=multi_modal_data,
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
            )
        )
        values["base_conv_with_gen_prompt_end_pos"] = len(
            cls._handle_apply_chat_template(
                processing_class,
                BASE_CHAT_HISTORY,
                multi_modal_data=multi_modal_data,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
            )
        )
        values["turn_details"] = []
        values["turn_tool_calls_detail"] = []

        return values

    @staticmethod
    def _handle_apply_chat_template(
        processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin],
        messages: List[Message],
        multi_modal_data: Dict[str, Any],
        tools: Optional[List[OpenAIFunctionToolSchema]] = None,
        add_generation_prompt: bool = False,
        tokenize: bool = False,
        return_dict: bool = False,
    ):
        if isinstance(processing_class, PreTrainedTokenizer) or isinstance(processing_class, PreTrainedTokenizerFast):
            if any(len(values) > 0 for values in multi_modal_data.values()):
                logger.warning(
                    "There is multi_modal_data but you are not using a processor. Multi-modal data will be ignored."
                )
            return processing_class.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                return_dict=return_dict,
            )
        elif isinstance(processing_class, ProcessorMixin):
            raw_prompt = processing_class.apply_chat_template(
                messages, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False
            )
            if not tokenize:
                return raw_prompt

            # When we update multi_model_keys, we also need to update this logic
            images = images if len(images := multi_modal_data.get("image", [])) > 0 else None
            videos = videos if len(videos := multi_modal_data.get("video", [])) > 0 else None
            model_inputs = processing_class(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
            assert model_inputs["input_ids"].shape[0] == 1, "input_ids should be a 1D array"
            model_inputs = {k: v[0].tolist() if hasattr(v, "tolist") else v for k, v in model_inputs.items()}
            if return_dict:
                return model_inputs
            else:
                return model_inputs["input_ids"]
        else:
            raise ValueError(f"Unsupported processing class type: {type(processing_class)}")

    def _update_input_ids(self, new_input_ids: List[int], attention_mask: bool, loss_mask: bool) -> None:
        """
        Update the input_ids, attention_mask, position_ids, and loss_mask of the request in additive manner.
        """
        self.input_ids += new_input_ids
        attention_mask = [int(attention_mask)] * len(new_input_ids)
        self.attention_mask += attention_mask
        self.loss_mask += [int(loss_mask)] * len(new_input_ids)
        self.position_ids += (
            compute_position_id_with_mask(torch.tensor(attention_mask)) + (self.position_ids[-1] + 1)
        ).tolist()

        assert (
            len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask)
        ), f"""Request {self.request_id} has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    def get_generation_prompt_ids(
        self, processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin]
    ) -> list[int]:
        generation_prompt_ids = (
            []
            if self.input_ids[-len(self.generation_prompt_ids) :] == self.generation_prompt_ids
            else self.generation_prompt_ids
        )
        if generation_prompt_ids:
            self._update_input_ids(generation_prompt_ids, attention_mask=True, loss_mask=False)

        if self.use_inference_chat_template:
            messages = [msg.model_dump() for msg in self.messages]
            tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None
            generation_prompt_ids = self._handle_apply_chat_template(
                processing_class,
                messages,
                multi_modal_data=self.multi_modal_data,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
            )
            return generation_prompt_ids
        else:
            return self.input_ids

    def add_user_message(
        self,
        processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin],
        content: str,
    ) -> None:
        self.messages.append(Message(role="user", content=content))
        messages = [*BASE_CHAT_HISTORY, self.messages[-1]]
        tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None

        # We don't need to pass multi_modal_data here because we don't have any multi-modal data from Engine
        # Inference, it is pure text.
        content_ids = self._handle_apply_chat_template(
            processing_class, messages, multi_modal_data={}, tools=tools, add_generation_prompt=False, tokenize=True
        )[self.base_conv_wo_gen_prompt_end_pos :]
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=False)

    def add_assistant_message(
        self,
        processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin],
        content: str,
        tool_calls: Optional[List[OpenAIFunctionToolCall]] = None,
    ) -> Dict[str, Any]:
        self.messages.append(Message(role="assistant", content=content, tool_calls=tool_calls))
        messages = [*BASE_CHAT_HISTORY, self.messages[-1]]
        tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None

        # We don't need to pass multi_modal_data here because we don't have any multi-modal data from Engine
        # Inference, it is pure text.
        content_ids = self._handle_apply_chat_template(
            processing_class, messages, multi_modal_data={}, tools=tools, add_generation_prompt=False, tokenize=True
        )[self.base_conv_with_gen_prompt_end_pos :]
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=True)

        # Return turn-level statistics
        return {
            "role": "assistant",
            "token_count": len(content_ids),
            "char_count": len(content),
            "has_tool_calls": tool_calls is not None,
            "tool_calls_count": len(tool_calls) if tool_calls else 0,
        }

    def add_tool_response_messages(
        self, processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin], contents: list[str]
    ) -> Dict[str, Any]:
        if not contents:
            return {"role": "tool", "token_count": 0, "char_count": 0, "tool_count": 0}

        self.messages.extend([Message(role="tool", content=content) for content in contents])

        messages = [*BASE_CHAT_HISTORY, *self.messages[-len(contents) :]]
        tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None

        # Currently we don't support tool creates multi-modal data
        content_ids = self._handle_apply_chat_template(
            processing_class, messages, multi_modal_data={}, tools=tools, add_generation_prompt=False, tokenize=True
        )[self.base_conv_wo_gen_prompt_end_pos :]
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=False)

        return {
            "role": "tool",
            "token_count": len(content_ids),
            "char_count": sum(len(content) for content in contents),
            "tool_count": len(contents),
        }

    def check_if_tool_response_messages_overlong(
        self, tokenizer: PreTrainedTokenizer, contents: list[str]
    ) -> tuple[bool, int]:
        if not contents:
            return False
        content = tokenizer.apply_chat_template(
            [*BASE_CHAT_HISTORY, *self.messages[-len(contents) :]],
            tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None),
            add_generation_prompt=False,
            tokenize=False,
        )
        content_ids = tokenizer.encode(content[self.base_conv_wo_gen_prompt_end_pos :], add_special_tokens=False)
        return (len(self.input_ids) + len(content_ids)) > self.max_model_len, len(content_ids)

    def update_metrics(self, metrics: Any, tool_id: str) -> None:
        """
        metrics: should be a dict of tools_name -> Any
        """
        if self.metrics.get(tool_id) is None:
            self.metrics[tool_id] = []
        self.metrics[tool_id].append(metrics)

    def track_turn(self, role: str, turn_stats: Dict[str, Any]):
        """Unified turn tracking (replaces separate conversation + turn tracking)"""
        import time

        # Extract basic turn information
        tokens = turn_stats.get("token_count", 0)
        chars = turn_stats.get("char_count", 0)
        tool_calls_count = turn_stats.get("tool_calls_count", 0)

        # Create unified turn record
        turn_record = {
            "turn_index": len(self.turn_details),
            "timestamp": time.time(),
            "role": role,
            "token_count": tokens,
            "char_count": chars,
            "tool_calls_count": tool_calls_count,
            "has_tool_calls": tool_calls_count > 0,
            "request_id": self.request_id,
            "batch_data_id": self.batch_data_id,
            "rollout_offset": self.rollout_offset,
            **turn_stats,  # Include any additional turn statistics
        }

        # Add to unified turn details
        self.turn_details.append(turn_record)

        # Update turn-level tool calls detail (for backward compatibility)
        turn_index = len(self.turn_details) - 1
        while len(self.turn_tool_calls_detail) <= turn_index:
            self.turn_tool_calls_detail.append(0)
        self.turn_tool_calls_detail[turn_index] = tool_calls_count

    def initialize_conversation_from_prompt(self, messages: List[Message], tokenizer):
        """Initialize conversation from initial prompt messages using unified tracking"""
        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                tokens = len(tokenizer.encode(msg.content))
                chars = len(msg.content)
                tool_calls_count = 0

                # Check for tool calls
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls_count = len(msg.tool_calls)

                # Use unified turn tracking
                turn_stats = {
                    "token_count": tokens,
                    "char_count": chars,
                    "tool_calls_count": tool_calls_count,
                    "source": "initial_prompt",
                }
                self.track_turn(msg.role, turn_stats)

    def set_termination_reason(self, reason: str):
        """Set termination reason (unified)"""
        self.termination_reason = reason

    def get_tool_metrics(self) -> List[Dict[str, Any]]:
        """Extract individual tool execution metrics from stored data.

        Returns:
            List of individual tool execution records, each compatible with ToolMetrics schema.
            Empty list if no tool executions occurred.
        """
        individual_tool_metrics = []

        if not self.metrics:
            return individual_tool_metrics

        # Extract all individual tool execution records
        for tool_name, metrics_list in self.metrics.items():
            if tool_name not in ["turn_stats", "token_stats", "conversation_stats"]:
                # Validate metrics_list structure
                if not isinstance(metrics_list, list):
                    logger.warning(f"Invalid metrics_list type for tool {tool_name}: {type(metrics_list)}")
                    continue

                # Each metrics_list contains tool execution records created by _assemble_tool_metrics()
                for metrics_dict in metrics_list:
                    # Validate individual metrics_dict
                    if not isinstance(metrics_dict, dict) or not metrics_dict:
                        logger.warning(f"Invalid metrics_dict for tool {tool_name}: {type(metrics_dict)}")
                        continue

                    # Add tool truncation metrics if present (at request level)
                    if self.tool_truncation_metrics:
                        metrics_dict = metrics_dict.copy()  # Avoid modifying original
                        metrics_dict["tool_truncation_events"] = self.tool_truncation_metrics

                    individual_tool_metrics.append(metrics_dict)

        return individual_tool_metrics

    def get_conversation_metrics(self) -> Dict[str, Any]:
        """Generate unified conversation metrics (includes turns + termination)"""
        from collections import defaultdict

        total_turns = len(self.turn_details)
        total_tokens = sum(t.get("token_count", 0) for t in self.turn_details)
        total_chars = sum(t.get("char_count", 0) for t in self.turn_details)

        # Role-based aggregation
        role_stats = defaultdict(lambda: {"turns": 0, "tokens": 0, "chars": 0})
        for turn in self.turn_details:
            role = turn.get("role", "unknown")
            role_stats[role]["turns"] += 1
            role_stats[role]["tokens"] += turn.get("token_count", 0)
            role_stats[role]["chars"] += turn.get("char_count", 0)

        # Tool usage patterns
        assistant_tool_calls = sum(
            t.get("tool_calls_count", 0) for t in self.turn_details if t.get("role") == "assistant"
        )
        turns_with_tools = sum(1 for t in self.turn_details if t.get("tool_calls_count", 0) > 0)

        # Calculate ratios
        tool_turn_ratio = turns_with_tools / total_turns if total_turns > 0 else 0
        assistant_turn_ratio = role_stats["assistant"]["turns"] / total_turns if total_turns > 0 else 0

        return {
            # Request context
            "request_id": self.request_id,
            "batch_data_id": self.batch_data_id,
            "rollout_offset": self.rollout_offset,
            "timestamp": self.turn_details[-1].get("timestamp", 0) if self.turn_details else 0,
            # Basic conversation statistics
            "total_turns": total_turns,
            "total_tokens": total_tokens,
            "total_chars": total_chars,
            # Role distribution
            "assistant_turns": role_stats["assistant"]["turns"],
            "user_turns": role_stats["user"]["turns"],
            "tool_turns": role_stats["tool"]["turns"],
            "system_turns": role_stats["system"]["turns"],
            "assistant_tokens": role_stats["assistant"]["tokens"],
            "user_tokens": role_stats["user"]["tokens"],
            "tool_tokens": role_stats["tool"]["tokens"],
            "system_tokens": role_stats["system"]["tokens"],
            "assistant_chars": role_stats["assistant"]["chars"],
            "user_chars": role_stats["user"]["chars"],
            "tool_chars": role_stats["tool"]["chars"],
            "system_chars": role_stats["system"]["chars"],
            # Tool usage patterns
            "assistant_tool_calls": assistant_tool_calls,
            "turns_with_tools": turns_with_tools,
            "tool_turn_ratio": tool_turn_ratio,
            "assistant_turn_ratio": assistant_turn_ratio,
            # Termination context (unified)
            "termination_reason": self.termination_reason or "unknown",
            "termination_turn_count": total_turns,
            "termination_token_count": total_tokens,
            # Raw data for batch aggregation (unified)
            "turn_details": self.turn_details,
            "turn_token_lengths": [t.get("token_count", 0) for t in self.turn_details],
            "turn_char_lengths": [t.get("char_count", 0) for t in self.turn_details],
        }

    def _get_prompt_diffs(
        self,
        processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin],
        full_prompt_ids: List[int],
        current_prompt_ids: List[int],
        diff_surrounding_chars: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get differences between full prompt and current prompt with surrounding context.

        This function helps debug tokenization mismatches by showing the differences between
        full prompt and current prompt with surrounding context. Instead of just showing
        the exact diff, it includes additional tokens before and after to help locate
        the issue in the chat template.

        For example, if the actual diff is a newline change from "\n\n" to "\n", with
        diff_surrounding_chars the output might look like:

        full_prompt_chunk:    "<|im_start|>assistant\n\nI think..."
        current_prompt_chunk: "<|im_start|>assistant\nI think..."

        This context makes it much easier to identify where in the chat template the
        mismatch occurs.

        Args:
            processing_class: The processing class to use for decoding the token IDs
            full_prompt_ids: Token IDs from applying chat template to all messages at once
            current_prompt_ids: Token IDs from incremental chat template application
            diff_surrounding_chars: Number of surrounding characters to include for context (default: 10)

        Returns:
            List of dicts containing the differing chunks with context and their indices
        """
        full_prompt = processing_class.decode(full_prompt_ids, skip_special_tokens=False)
        current_prompt = processing_class.decode(current_prompt_ids, skip_special_tokens=False)
        s = difflib.SequenceMatcher(None, full_prompt, current_prompt, autojunk=False)
        diffs = []
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            if tag == "equal":
                continue

            # Get the surrounding context for better readability
            start_i = max(0, i1 - diff_surrounding_chars)
            end_i = min(len(full_prompt), i2 + diff_surrounding_chars)
            start_j = max(0, j1 - diff_surrounding_chars)
            end_j = min(len(current_prompt), j2 + diff_surrounding_chars)

            diffs.append(
                {
                    "full_prompt_chunk": full_prompt[start_i:end_i],
                    "current_prompt_chunk": current_prompt[start_j:end_j],
                    "indices": (start_i, end_i, start_j, end_j),
                }
            )
        return diffs

    def finalize(
        self,
        processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin],
        reward_scores: Dict[str, List[float]],
        finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP,
    ) -> None:
        self.state = AsyncRolloutRequestStateEnum.COMPLETED
        self.reward_scores = reward_scores
        if self.tokenization_sanity_check_mode != TokenizationSanityCheckModeEnum.OFF:
            # When there is a diff, we log the diffs with diff_surrounding_chars context
            diff_surrounding_chars = 10

            messages = [msg.model_dump() for msg in self.messages]
            tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None
            full_prompt_ids = self._handle_apply_chat_template(
                processing_class,
                messages,
                multi_modal_data=self.multi_modal_data,
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
            )

            if diffs := self._get_prompt_diffs(
                processing_class, full_prompt_ids, self.input_ids, diff_surrounding_chars=diff_surrounding_chars
            ):
                log_warning = False
                if self.tokenization_sanity_check_mode == TokenizationSanityCheckModeEnum.STRICT:
                    log_warning = True
                elif self.tokenization_sanity_check_mode == TokenizationSanityCheckModeEnum.IGNORE_STRIPPABLE:
                    non_strippable_diffs_exist = any(
                        d["full_prompt_chunk"].strip() or d["current_prompt_chunk"].strip() for d in diffs
                    )
                    if non_strippable_diffs_exist:
                        log_warning = True

                if log_warning:
                    mode_str = f" ({self.tokenization_sanity_check_mode.value})"
                    logger.warning(
                        f"Inconsistent training and inference tokenization detected{mode_str}. This may lead to "
                        f"unexpected behavior during training. Please review your chat template to determine if this "
                        f"is intentional. For more information, refer to the multiturn README.md."
                    )
                    logger.warning(
                        f"Showing {diff_surrounding_chars} characters before and after the diffs for context and "
                        f"better readability."
                    )
                    diff_details_list = []
                    for d in diffs:
                        i1, i2, j1, j2 = d["indices"]
                        diff_details_list.append(
                            f"idx {i1}:{i2} -> {j1}:{j2} | full_prompt_chunk: {repr(d['full_prompt_chunk'])} | "
                            f"current_prompt_chunk: {repr(d['current_prompt_chunk'])}"
                        )
                    diff_details = "\n".join(diff_details_list)
                    logger.warning(f"Found differences:\n{diff_details}")

        # In case we failed to generate the assistant message and the generation prompt ids were already added to
        # input_ids, remove them from the end of input_ids
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
        self.truncate_output_ids(processing_class)
        assert (
            len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask)
        ), f"""Request {self.request_id} has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    def truncate_output_ids(
        self, processing_class: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin]
    ) -> None:
        self.input_ids = self.input_ids[: self.max_model_len]
        self.attention_mask = self.attention_mask[: self.max_model_len]
        self.position_ids = self.position_ids[: self.max_model_len]
        self.loss_mask = self.loss_mask[: self.max_model_len]
        self.response_ids = self.input_ids[len(self.prompt_ids) :][: self.max_response_len]
        self.response_attention_mask = self.attention_mask[len(self.prompt_attention_mask) :][: self.max_response_len]
        self.response_position_ids = self.position_ids[len(self.prompt_position_ids) :][: self.max_response_len]
        self.response_loss_mask = self.loss_mask[len(self.prompt_loss_mask) :][: self.max_response_len]
