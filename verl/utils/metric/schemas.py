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
"""
Standardized Pydantic schemas for metrics aggregation.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class BaseMetrics(BaseModel):
    """Base class for all metrics with common fields."""

    request_id: str = Field(..., description="Unique request identifier")
    batch_data_id: int = Field(..., description="Batch data index")
    rollout_offset: int = Field(..., description="Rollout offset within batch")
    timestamp: float = Field(..., description="Timestamp of metric collection")


class ToolMetrics(BaseMetrics):
    """Standardized tool execution metrics."""

    # Tool execution context
    tool_name: str = Field(..., description="Name of the executed tool")

    # Base execution metrics
    latency_ms: float = Field(..., ge=0, description="Tool execution latency in milliseconds")
    success: bool = Field(..., description="Whether tool execution succeeded")
    response_char_length: int = Field(..., ge=0, description="Length of tool response in characters")

    # Optional metrics
    error_type: Optional[str] = Field(None, description="Error type if execution failed")
    tool_specific_metrics: Dict[str, Any] = Field(default_factory=dict, description="Tool-specific additional metrics")

    # Tool chain context
    tool_calls_per_trajectory: int = Field(..., ge=0, description="Number of tool calls in this trajectory")
    tool_calls_per_turn: int = Field(..., ge=0, description="Number of tool calls in this turn")


class ConversationMetrics(BaseMetrics):
    """Unified conversation flow metrics (consolidates turn/conversation/termination)."""

    # === Basic conversation statistics ===
    total_turns: int = Field(..., ge=0, description="Total number of turns")
    total_tokens: int = Field(..., ge=0, description="Total tokens in conversation")
    total_chars: int = Field(..., ge=0, description="Total characters in conversation")

    # === Role-specific statistics ===
    assistant_turns: int = Field(..., ge=0, description="Number of assistant turns")
    user_turns: int = Field(..., ge=0, description="Number of user turns")
    tool_turns: int = Field(..., ge=0, description="Number of tool turns")
    system_turns: int = Field(..., ge=0, description="Number of system turns")

    assistant_tokens: int = Field(..., ge=0, description="Tokens in assistant turns")
    user_tokens: int = Field(..., ge=0, description="Tokens in user turns")
    tool_tokens: int = Field(..., ge=0, description="Tokens in tool turns")
    system_tokens: int = Field(..., ge=0, description="Tokens in system turns")

    assistant_chars: int = Field(..., ge=0, description="Characters in assistant turns")
    user_chars: int = Field(..., ge=0, description="Characters in user turns")
    tool_chars: int = Field(..., ge=0, description="Characters in tool turns")
    system_chars: int = Field(..., ge=0, description="Characters in system turns")

    # === Tool usage statistics ===
    assistant_tool_calls: int = Field(..., ge=0, description="Number of tool calls by assistant")
    turns_with_tools: int = Field(..., ge=0, description="Number of turns containing tool calls")

    # === Derived ratios ===
    tool_turn_ratio: float = Field(..., ge=0, le=1, description="Ratio of turns with tools")
    assistant_turn_ratio: float = Field(..., ge=0, le=1, description="Ratio of assistant turns")

    # === Termination information (merged from TerminationMetrics) ===
    termination_reason: str = Field(..., description="Termination reason")
    termination_turn_count: int = Field(..., ge=0, description="Number of turns at termination")
    termination_token_count: int = Field(..., ge=0, description="Number of tokens at termination")

    # === Raw data for batch aggregation (merged from TurnStats) ===
    turn_details: List[Dict[str, Any]] = Field(default_factory=list, description="Detailed per-turn statistics")
    turn_token_lengths: List[int] = Field(default_factory=list, description="Token count per turn")
    turn_char_lengths: List[int] = Field(default_factory=list, description="Character count per turn")

    @validator("tool_turn_ratio", "assistant_turn_ratio")
    def validate_ratio(cls, v):
        """Ensure ratios are between 0 and 1."""
        if not (0 <= v <= 1):
            raise ValueError("Ratio must be between 0 and 1")
        return v

    @validator("termination_reason")
    def validate_termination_reason(cls, v):
        """Validate termination reason."""
        valid_reasons = {"eos_token", "max_tokens", "max_assistant_turns", "unknown"}
        if v not in valid_reasons:
            # Allow any string but warn about unknown values
            return v
        return v


class AggregatedNumericMetrics(BaseModel):
    """Base class for aggregated metrics results."""

    # Statistical aggregations
    count: int = Field(..., ge=0, description="Number of samples")
    min: float = Field(..., description="Minimum value")
    max: float = Field(..., description="Maximum value")
    avg: float = Field(..., description="Average value")
    std: float = Field(..., description="Standard deviation value")
    sum: float = Field(..., description="Sum value")


class AggregatedCategoricalMetrics(BaseModel):
    """Aggregated categorical metrics."""

    # Categorical aggregations
    value_counts: Dict[str, int] = Field(default_factory=dict, description="Count of each value")
    value_ratios: Dict[str, float] = Field(default_factory=dict, description="Ratio of each value")


class AggregatedToolMetrics(BaseModel):
    """Aggregated tool execution metrics."""

    # Tool-specific aggregations
    tool_calls_per_trajectory_stats: AggregatedNumericMetrics
    tool_calls_per_turn_stats: AggregatedNumericMetrics
    latency_ms_stats: AggregatedNumericMetrics
    response_char_length_stats: AggregatedNumericMetrics
    success_rate: float = Field(..., ge=0, le=1, description="Overall success rate")
    tool_names_metrics: AggregatedCategoricalMetrics = Field(..., description="Tool names distribution")

    # Optional categorical metrics (only present if data contains these fields)
    error_type_metrics: Optional[AggregatedCategoricalMetrics] = Field(None, description="Error type distribution (if present)")

    # Per-tool breakdown
    tool_specific_stats: Dict[str, AggregatedNumericMetrics | AggregatedCategoricalMetrics] = Field(default_factory=dict)


class AggregatedConversationMetrics(BaseModel):
    """Aggregated conversation flow metrics (unified with termination)."""

    # Conversation flow aggregations
    total_turns_stats: AggregatedNumericMetrics
    total_tokens_stats: AggregatedNumericMetrics
    total_chars_stats: AggregatedNumericMetrics

    # Role-specific aggregations
    assistant_turns_stats: AggregatedNumericMetrics
    user_turns_stats: AggregatedNumericMetrics
    tool_turns_stats: AggregatedNumericMetrics

    # Tool usage aggregations
    assistant_tool_calls_stats: AggregatedNumericMetrics
    turns_with_tools_stats: AggregatedNumericMetrics
    tool_turn_ratio_stats: AggregatedNumericMetrics
    assistant_turn_ratio_stats: AggregatedNumericMetrics

    # Turn length aggregations
    turn_token_lengths_stats: AggregatedNumericMetrics
    turn_char_lengths_stats: AggregatedNumericMetrics

    # Termination aggregations (unified with AggregatedCategoricalMetrics)
    termination_reason_metrics: AggregatedCategoricalMetrics
    termination_turn_count_stats: AggregatedNumericMetrics
    termination_token_count_stats: AggregatedNumericMetrics


class BatchMetrics(BaseModel):
    """Complete batch-level metrics aggregation (dual-layer architecture)."""

    # Dual-layer architecture: Tool + Conversation (unified)
    tool_metrics: Optional[AggregatedToolMetrics] = None
    conversation_metrics: Optional[AggregatedConversationMetrics] = None  # Includes termination + turn data

    # Batch metadata
    batch_size: int = Field(..., ge=0, description="Number of requests in batch")
    processing_time_ms: float = Field(..., ge=0, description="Time taken to process batch")

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        extra = "forbid"
