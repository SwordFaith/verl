# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
Unified metrics aggregation utilities with shared logic.
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, TypeVar, Union

import numpy as np

from .schemas import (
    AggregatedConversationMetrics,
    AggregatedMetrics,
    AggregatedToolMetrics,
    BatchMetrics,
    ConversationMetrics,
    ToolMetrics,
)

T = TypeVar("T")


class MetricsAggregator:
    """Shared aggregation logic for all metric types."""

    @staticmethod
    def compute_basic_stats(values: List[Union[int, float]]) -> AggregatedMetrics:
        """Compute basic statistical aggregations for a list of values."""
        if not values:
            return AggregatedMetrics(count=0, min_value=0.0, max_value=0.0, avg_value=0.0, total_value=0.0, raw_values=[])

        values_array = np.array(values, dtype=float)
        return AggregatedMetrics(count=len(values), min_value=float(np.min(values_array)), max_value=float(np.max(values_array)), avg_value=float(np.mean(values_array)), total_value=float(np.sum(values_array)), raw_values=values.copy())

    @staticmethod
    def compute_success_rate(success_list: List[bool]) -> float:
        """Compute success rate from list of boolean values."""
        if not success_list:
            return 0.0
        return sum(success_list) / len(success_list)

    @staticmethod
    def compute_distribution(items: List[str]) -> Dict[str, float]:
        """Compute distribution of categorical items as ratios."""
        if not items:
            return {}

        counts = defaultdict(int)
        for item in items:
            counts[item] += 1

        total = len(items)
        return {item: count / total for item, count in counts.items()}

    @staticmethod
    def extract_values(metrics_list: List[T], field_name: str) -> List[Any]:
        """Extract values of a specific field from list of metrics objects."""
        values = []
        for metric in metrics_list:
            if hasattr(metric, field_name):
                value = getattr(metric, field_name)
                if value is not None:
                    values.append(value)
            elif isinstance(metric, dict) and field_name in metric:
                value = metric[field_name]
                if value is not None:
                    values.append(value)
        return values

    @staticmethod
    def group_by_field(metrics_list: List[T], field_name: str) -> Dict[Any, List[T]]:
        """Group metrics by a specific field value."""
        groups = defaultdict(list)
        for metric in metrics_list:
            if hasattr(metric, field_name):
                key = getattr(metric, field_name)
            elif isinstance(metric, dict) and field_name in metric:
                key = metric[field_name]
            else:
                key = "unknown"
            groups[key].append(metric)
        return dict(groups)

    @staticmethod
    def aggregate_categorical_fields(data_list: List[Dict[str, Any]], prefix: str = "") -> Dict[str, Any]:
        """
        Aggregate categorical (string/boolean) fields into count/ratio statistics for logging compatibility.

        Converts non-numeric fields into numeric format:
        - Boolean fields: true_count, true_ratio, false_count, false_ratio
        - String fields: top-5 values with count/ratio for each + summary stats

        Args:
            data_list: List of metric dictionaries containing mixed data types
            prefix: Prefix for metric names (e.g., "tools_", "conversations_")

        Returns:
            Dictionary with numeric-only metrics suitable for WandB/TensorBoard
        """
        aggregated = {}
        if not data_list:
            return aggregated

        # Filter out empty/invalid data
        valid_data = [data for data in data_list if data and isinstance(data, dict)]
        if not valid_data:
            return aggregated

        # Collect all categorical fields (non-numeric types)
        categorical_fields = set()
        for data in valid_data:
            for key, value in data.items():
                # Identify categorical fields: strings, booleans (but not numeric types)
                if isinstance(value, (str, bool)) and not isinstance(value, (int, float)):
                    categorical_fields.add(key)

        # Process each categorical field
        for field in categorical_fields:
            # Extract non-null values for this field
            values = [data.get(field) for data in valid_data if field in data and data.get(field) is not None]
            total_count = len(values)

            if total_count > 0:
                # Boolean type: convert to true/false counts and ratios
                if all(isinstance(v, bool) for v in values):
                    true_count = sum(1 for v in values if v is True)
                    false_count = total_count - true_count

                    aggregated[f"{prefix}{field}_true_count"] = true_count
                    aggregated[f"{prefix}{field}_true_ratio"] = true_count / total_count if total_count > 0 else 0.0
                    aggregated[f"{prefix}{field}_false_count"] = false_count
                    aggregated[f"{prefix}{field}_false_ratio"] = false_count / total_count if total_count > 0 else 0.0

                # String type: convert to top-K value counts and ratios
                else:
                    # Count occurrences of each value
                    value_counts = {}
                    for v in values:
                        v_str = str(v)
                        value_counts[v_str] = value_counts.get(v_str, 0) + 1

                    # Top-5 most frequent values to prevent metric explosion
                    top_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:5]

                    for value_name, count in top_values:
                        # Create safe metric name (replace special characters)
                        safe_name = value_name.replace("/", "_").replace(" ", "_").replace("-", "_")
                        safe_name = safe_name.replace(".", "_").replace(":", "_")[:20]  # Limit length
                        safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")  # Keep only safe chars

                        if safe_name:  # Only add if name is valid after cleaning
                            aggregated[f"{prefix}{field}_{safe_name}_count"] = count
                            aggregated[f"{prefix}{field}_{safe_name}_ratio"] = count / total_count

                    # Summary statistics for the field
                    aggregated[f"{prefix}{field}_total_samples"] = total_count
                    aggregated[f"{prefix}{field}_unique_values"] = len(value_counts)

        return aggregated

    @staticmethod
    def filter_numeric_metrics(metrics_dict: Dict[str, Any], exclude_fields: set = None) -> Dict[str, Any]:
        """
        Filter metrics dictionary to include only valid numeric values suitable for logging.

        Args:
            metrics_dict: Dictionary of mixed metric types
            exclude_fields: Set of field names to exclude from filtering

        Returns:
            Dictionary containing only valid numeric metrics
        """
        if exclude_fields is None:
            exclude_fields = {"count", "min_value", "max_value", "avg_value", "total_value", "raw_values"}

        filtered = {}
        for key, value in metrics_dict.items():
            if key not in exclude_fields and isinstance(value, (int, float)):
                # Validate numeric values (exclude NaN/Inf)
                if not (math.isnan(value) or math.isinf(value)):
                    filtered[key] = value

        return filtered


def aggregate_tool_metrics(metrics_list: List[Union[ToolMetrics, Dict]]) -> AggregatedToolMetrics:
    """
    Aggregate tool execution metrics with standardized structure.

    Args:
        metrics_list: List of ToolMetrics objects or compatible dictionaries

    Returns:
        AggregatedToolMetrics with comprehensive tool statistics
    """
    if not metrics_list:
        return AggregatedToolMetrics(
            count=0,
            min_value=0.0,
            max_value=0.0,
            avg_value=0.0,
            total_value=0.0,
            tool_calls_per_trajectory_stats=AggregatedMetrics(count=0, min_value=0.0, max_value=0.0, avg_value=0.0, total_value=0.0),
            tool_calls_per_turn_stats=AggregatedMetrics(count=0, min_value=0.0, max_value=0.0, avg_value=0.0, total_value=0.0),
            latency_ms_stats=AggregatedMetrics(count=0, min_value=0.0, max_value=0.0, avg_value=0.0, total_value=0.0),
            response_char_length_stats=AggregatedMetrics(count=0, min_value=0.0, max_value=0.0, avg_value=0.0, total_value=0.0),
            success_rate=0.0,
        )

    aggregator = MetricsAggregator()

    # Extract basic metrics
    tool_calls_per_trajectory = aggregator.extract_values(metrics_list, "tool_calls_per_trajectory")
    tool_calls_per_turn = aggregator.extract_values(metrics_list, "tool_calls_per_turn")
    latency_ms = aggregator.extract_values(metrics_list, "latency_ms")
    response_char_length = aggregator.extract_values(metrics_list, "response_char_length")
    success_values = aggregator.extract_values(metrics_list, "success")
    tool_names = aggregator.extract_values(metrics_list, "tool_name")

    # Compute aggregations
    result = AggregatedToolMetrics(
        count=len(metrics_list),
        min_value=0.0,  # Will be computed based on primary metric
        max_value=0.0,
        avg_value=0.0,
        total_value=0.0,
        tool_calls_per_trajectory_stats=aggregator.compute_basic_stats(tool_calls_per_trajectory),
        tool_calls_per_turn_stats=aggregator.compute_basic_stats(tool_calls_per_turn),
        latency_ms_stats=aggregator.compute_basic_stats(latency_ms),
        response_char_length_stats=aggregator.compute_basic_stats(response_char_length),
        success_rate=aggregator.compute_success_rate(success_values),
        tools_used=list(set(tool_names)) if tool_names else [],
    )

    # Compute per-tool statistics
    tool_groups = aggregator.group_by_field(metrics_list, "tool_name")
    tool_specific_stats = {}

    for tool_name, tool_metrics in tool_groups.items():
        tool_latencies = aggregator.extract_values(tool_metrics, "latency_ms")
        tool_specific_stats[tool_name] = aggregator.compute_basic_stats(tool_latencies)

    result.tool_specific_stats = tool_specific_stats

    # Set primary metric stats (using latency as primary)
    if latency_ms:
        result.min_value = result.latency_ms_stats.min_value
        result.max_value = result.latency_ms_stats.max_value
        result.avg_value = result.latency_ms_stats.avg_value
        result.total_value = result.latency_ms_stats.total_value

    return result


def aggregate_conversation_metrics(metrics_list: List[Union[ConversationMetrics, Dict]]) -> AggregatedConversationMetrics:
    """
    Aggregate unified conversation flow metrics (includes termination + turn data).

    Args:
        metrics_list: List of ConversationMetrics objects or compatible dictionaries

    Returns:
        AggregatedConversationMetrics with comprehensive conversation and termination statistics
    """
    if not metrics_list:
        empty_stats = AggregatedMetrics(count=0, min_value=0.0, max_value=0.0, avg_value=0.0, total_value=0.0)
        return AggregatedConversationMetrics(
            count=0,
            min_value=0.0,
            max_value=0.0,
            avg_value=0.0,
            total_value=0.0,
            total_turns_stats=empty_stats,
            total_tokens_stats=empty_stats,
            total_chars_stats=empty_stats,
            assistant_turns_stats=empty_stats,
            user_turns_stats=empty_stats,
            tool_turns_stats=empty_stats,
            assistant_tool_calls_stats=empty_stats,
            turns_with_tools_stats=empty_stats,
            tool_turn_ratio_stats=empty_stats,
            assistant_turn_ratio_stats=empty_stats,
            turn_token_lengths_stats=empty_stats,
            turn_char_lengths_stats=empty_stats,
            termination_reason_distribution={},
            termination_turn_count_stats=empty_stats,
            termination_token_count_stats=empty_stats,
        )

    aggregator = MetricsAggregator()

    # Extract conversation metrics
    total_turns = aggregator.extract_values(metrics_list, "total_turns")
    total_tokens = aggregator.extract_values(metrics_list, "total_tokens")
    total_chars = aggregator.extract_values(metrics_list, "total_chars")

    assistant_turns = aggregator.extract_values(metrics_list, "assistant_turns")
    user_turns = aggregator.extract_values(metrics_list, "user_turns")
    tool_turns = aggregator.extract_values(metrics_list, "tool_turns")

    assistant_tool_calls = aggregator.extract_values(metrics_list, "assistant_tool_calls")
    turns_with_tools = aggregator.extract_values(metrics_list, "turns_with_tools")

    tool_turn_ratio = aggregator.extract_values(metrics_list, "tool_turn_ratio")
    assistant_turn_ratio = aggregator.extract_values(metrics_list, "assistant_turn_ratio")

    # Extract termination metrics (unified)
    termination_reasons = aggregator.extract_values(metrics_list, "termination_reason")
    termination_turn_counts = aggregator.extract_values(metrics_list, "termination_turn_count")
    termination_token_counts = aggregator.extract_values(metrics_list, "termination_token_count")

    # Flatten turn length arrays
    all_turn_token_lengths = []
    all_turn_char_lengths = []
    for metric in metrics_list:
        if hasattr(metric, "turn_token_lengths"):
            all_turn_token_lengths.extend(metric.turn_token_lengths)
        elif isinstance(metric, dict) and "turn_token_lengths" in metric:
            all_turn_token_lengths.extend(metric["turn_token_lengths"])

        if hasattr(metric, "turn_char_lengths"):
            all_turn_char_lengths.extend(metric.turn_char_lengths)
        elif isinstance(metric, dict) and "turn_char_lengths" in metric:
            all_turn_char_lengths.extend(metric["turn_char_lengths"])

    # Compute aggregations
    result = AggregatedConversationMetrics(
        count=len(metrics_list),
        min_value=0.0,
        max_value=0.0,
        avg_value=0.0,
        total_value=0.0,
        # Conversation flow aggregations
        total_turns_stats=aggregator.compute_basic_stats(total_turns),
        total_tokens_stats=aggregator.compute_basic_stats(total_tokens),
        total_chars_stats=aggregator.compute_basic_stats(total_chars),
        # Role-specific aggregations
        assistant_turns_stats=aggregator.compute_basic_stats(assistant_turns),
        user_turns_stats=aggregator.compute_basic_stats(user_turns),
        tool_turns_stats=aggregator.compute_basic_stats(tool_turns),
        # Tool usage aggregations
        assistant_tool_calls_stats=aggregator.compute_basic_stats(assistant_tool_calls),
        turns_with_tools_stats=aggregator.compute_basic_stats(turns_with_tools),
        tool_turn_ratio_stats=aggregator.compute_basic_stats(tool_turn_ratio),
        assistant_turn_ratio_stats=aggregator.compute_basic_stats(assistant_turn_ratio),
        # Turn length aggregations
        turn_token_lengths_stats=aggregator.compute_basic_stats(all_turn_token_lengths),
        turn_char_lengths_stats=aggregator.compute_basic_stats(all_turn_char_lengths),
        # Termination aggregations (unified)
        termination_reason_distribution=aggregator.compute_distribution(termination_reasons),
        termination_turn_count_stats=aggregator.compute_basic_stats(termination_turn_counts),
        termination_token_count_stats=aggregator.compute_basic_stats(termination_token_counts),
    )

    # Set primary metric stats (using total_turns as primary)
    if total_turns:
        result.min_value = result.total_turns_stats.min_value
        result.max_value = result.total_turns_stats.max_value
        result.avg_value = result.total_turns_stats.avg_value
        result.total_value = result.total_turns_stats.total_value

    return result


def aggregate_batch_metrics(tool_metrics_list: Optional[List[Union[ToolMetrics, Dict]]] = None, conversation_metrics_list: Optional[List[Union[ConversationMetrics, Dict]]] = None, processing_time_ms: float = 0.0) -> BatchMetrics:
    """
    Aggregate dual-layer metrics into a single batch result.

    Args:
        tool_metrics_list: List of tool execution metrics
        conversation_metrics_list: List of unified conversation metrics (includes termination + turn data)
        processing_time_ms: Time taken to process the batch

    Returns:
        BatchMetrics with dual-layer aggregated results
    """
    batch_size = 0
    if tool_metrics_list:
        batch_size = len(tool_metrics_list)
    elif conversation_metrics_list:
        batch_size = len(conversation_metrics_list)

    return BatchMetrics(tool_metrics=aggregate_tool_metrics(tool_metrics_list) if tool_metrics_list else None, conversation_metrics=aggregate_conversation_metrics(conversation_metrics_list) if conversation_metrics_list else None, batch_size=batch_size, processing_time_ms=processing_time_ms)
