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
    AggregatedCategoricalMetrics,
    AggregatedConversationMetrics,
    AggregatedNumericMetrics,
    AggregatedToolMetrics,
    BatchMetrics,
    ConversationMetrics,
    ToolMetrics,
)

T = TypeVar("T")


class MetricsAggregator:
    """Shared aggregation logic for all metric types."""

    @staticmethod
    def compute_basic_stats(values: List[Union[int, float]]) -> AggregatedNumericMetrics:
        """Compute basic statistical aggregations for a list of values."""
        if not values:
            return AggregatedNumericMetrics(count=0, min=0.0, max=0.0, avg=0.0, std=0.0, sum=0.0)

        values_array = np.array(values, dtype=float)
        return AggregatedNumericMetrics(count=len(values), min=float(np.min(values_array)), max=float(np.max(values_array)), avg=float(np.mean(values_array)), std=float(np.std(values_array)), sum=float(np.sum(values_array)))

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
    def compute_categorical_metrics(items: List[str]) -> AggregatedCategoricalMetrics:
        """Compute categorical metrics aggregation from list of string values."""
        if not items:
            return AggregatedCategoricalMetrics(value_counts={}, value_ratios={})

        counts = defaultdict(int)
        for item in items:
            if item is not None:
                counts[str(item)] += 1

        total = len([item for item in items if item is not None])
        if total == 0:
            return AggregatedCategoricalMetrics(value_counts={}, value_ratios={})

        ratios = {item: count / total for item, count in counts.items()}
        return AggregatedCategoricalMetrics(value_counts=dict(counts), value_ratios=ratios)

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
    def aggregate_categorical_fields(data_list: List[Dict[str, Any]], exclude_fields: set = None) -> Dict[str, AggregatedCategoricalMetrics]:
        """
        Aggregate categorical (string/boolean) fields into AggregatedCategoricalMetrics objects.

        Args:
            data_list: List of metric dictionaries containing mixed data types
            exclude_fields: Set of field names to exclude from categorical aggregation.
                          If None, defaults to common unique identifiers.

        Returns:
            Dictionary mapping field names to AggregatedCategoricalMetrics objects
        """
        result = {}
        if not data_list:
            return result

        # Default exclude fields: common unique identifiers that produce meaningless statistics
        if exclude_fields is None:
            exclude_fields = {"request_id", "instance_id", "batch_data_id", "rollout_offset", "timestamp"}

        # Filter out empty/invalid data and apply field exclusion
        valid_data = []
        for data in data_list:
            if data and isinstance(data, dict):
                # Exclude unique identifiers that produce meaningless statistics
                filtered_item = {k: v for k, v in data.items() if k not in exclude_fields}
                if filtered_item:  # Only add if there are remaining fields after filtering
                    valid_data.append(filtered_item)

        if not valid_data:
            return result

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

            if values:
                # Convert all values to strings for consistent handling
                if all(isinstance(v, bool) for v in values):
                    # Convert boolean to string
                    str_values = ["true" if v else "false" for v in values]
                else:
                    # Convert all to string
                    str_values = [str(v) for v in values]

                result[field] = MetricsAggregator.compute_categorical_metrics(str_values)

        return result

    @staticmethod
    def filter_numeric_metrics(metrics_dict: Dict[str, Any], exclude_fields: set = None) -> Dict[str, Any]:
        """
        Filter metrics dictionary to include only valid numeric values suitable for logging.

        Args:
            metrics_dict: Dictionary of mixed metric types
            exclude_fields: Set of field names to exclude from filtering (optional)

        Returns:
            Dictionary containing only valid numeric metrics
        """
        if exclude_fields is None:
            # No fields excluded by default since raw_values was removed from schema
            exclude_fields = set()

        filtered = {}
        for key, value in metrics_dict.items():
            if key not in exclude_fields and isinstance(value, (int, float)):
                # Validate numeric values (exclude NaN/Inf)
                if not (math.isnan(value) or math.isinf(value)):
                    filtered[key] = value

        return filtered

    @staticmethod
    def flatten_categorical_metrics(categorical_metrics: Dict[str, AggregatedCategoricalMetrics], prefix: str = "") -> Dict[str, Any]:
        """
        Flatten AggregatedCategoricalMetrics to monitoring-compatible format.

        Args:
            categorical_metrics: Dictionary mapping field names to AggregatedCategoricalMetrics
            prefix: Prefix for metric names (e.g., "tools_", "conversations_")

        Returns:
            Dictionary with flattened categorical metrics suitable for monitoring systems
        """
        flattened = {}

        for field_name, cat_metrics in categorical_metrics.items():
            # Add counts
            for value, count in cat_metrics.value_counts.items():
                # Create safe metric name
                safe_value = value.replace("/", "_").replace(" ", "_").replace("-", "_")
                safe_value = safe_value.replace(".", "_").replace(":", "_")[:20]
                safe_value = "".join(c for c in safe_value if c.isalnum() or c == "_")

                if safe_value:
                    flattened[f"{prefix}{field_name}_{safe_value}_count"] = count

            # Add ratios
            for value, ratio in cat_metrics.value_ratios.items():
                # Create safe metric name
                safe_value = value.replace("/", "_").replace(" ", "_").replace("-", "_")
                safe_value = safe_value.replace(".", "_").replace(":", "_")[:20]
                safe_value = "".join(c for c in safe_value if c.isalnum() or c == "_")

                if safe_value:
                    flattened[f"{prefix}{field_name}_{safe_value}_ratio"] = ratio

            # Add summary statistics
            total_samples = sum(cat_metrics.value_counts.values())
            unique_values = len(cat_metrics.value_counts)
            flattened[f"{prefix}{field_name}_total_samples"] = total_samples
            flattened[f"{prefix}{field_name}_unique_values"] = unique_values

        return flattened


def aggregate_tool_metrics(metrics_list: List[Union[ToolMetrics, Dict]]) -> AggregatedToolMetrics:
    """
    Aggregate tool execution metrics with standardized structure.

    Args:
        metrics_list: List of ToolMetrics objects or compatible dictionaries

    Returns:
        AggregatedToolMetrics with comprehensive tool statistics
    """
    if not metrics_list:
        empty_stats = AggregatedNumericMetrics(count=0, min=0.0, max=0.0, avg=0.0, std=0.0, sum=0.0)
        empty_categorical = AggregatedCategoricalMetrics(value_counts={}, value_ratios={})
        return AggregatedToolMetrics(
            tool_calls_per_trajectory_stats=empty_stats,
            tool_calls_per_turn_stats=empty_stats,
            latency_ms_stats=empty_stats,
            response_char_length_stats=empty_stats,
            success_rate=0.0,
            tool_names_metrics=empty_categorical,
        )

    aggregator = MetricsAggregator()

    # Extract basic metrics
    tool_calls_per_trajectory = aggregator.extract_values(metrics_list, "tool_calls_per_trajectory")
    tool_calls_per_turn = aggregator.extract_values(metrics_list, "tool_calls_per_turn")
    latency_ms = aggregator.extract_values(metrics_list, "latency_ms")
    response_char_length = aggregator.extract_values(metrics_list, "response_char_length")
    success_values = aggregator.extract_values(metrics_list, "success")
    tool_names = aggregator.extract_values(metrics_list, "tool_name")

    # Extract optional categorical metrics
    error_types = aggregator.extract_values(metrics_list, "error_type")
    # Filter out None values for error_type
    error_types = [et for et in error_types if et is not None]

    # Compute aggregations
    result = AggregatedToolMetrics(
        tool_calls_per_trajectory_stats=aggregator.compute_basic_stats(tool_calls_per_trajectory),
        tool_calls_per_turn_stats=aggregator.compute_basic_stats(tool_calls_per_turn),
        latency_ms_stats=aggregator.compute_basic_stats(latency_ms),
        response_char_length_stats=aggregator.compute_basic_stats(response_char_length),
        success_rate=aggregator.compute_success_rate(success_values),
        tool_names_metrics=aggregator.compute_categorical_metrics(tool_names),
        error_type_metrics=aggregator.compute_categorical_metrics(error_types) if error_types else None,
    )

    # Compute per-tool statistics (both numeric and categorical)
    tool_groups = aggregator.group_by_field(metrics_list, "tool_name")
    tool_specific_stats = {}

    for tool_name, tool_metrics in tool_groups.items():
        tool_latencies = aggregator.extract_values(tool_metrics, "latency_ms")
        tool_specific_stats[tool_name] = aggregator.compute_basic_stats(tool_latencies)

        # Add categorical metrics for this tool
        tool_dicts = []
        for metric in tool_metrics:
            if hasattr(metric, "__dict__"):
                tool_dicts.append(metric.__dict__)
            elif isinstance(metric, dict):
                tool_dicts.append(metric)

        if tool_dicts:
            categorical_metrics = aggregator.aggregate_categorical_fields(tool_dicts)
            for field_name, cat_metrics in categorical_metrics.items():
                tool_specific_stats[f"{tool_name}_{field_name}"] = cat_metrics

    result.tool_specific_stats = tool_specific_stats

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
        empty_stats = AggregatedNumericMetrics(count=0, min=0.0, max=0.0, avg=0.0, std=0.0, sum=0.0)
        empty_categorical = AggregatedCategoricalMetrics(value_counts={}, value_ratios={})
        return AggregatedConversationMetrics(
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
            termination_reason_metrics=empty_categorical,
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
        termination_reason_metrics=aggregator.compute_categorical_metrics(termination_reasons),
        termination_turn_count_stats=aggregator.compute_basic_stats(termination_turn_counts),
        termination_token_count_stats=aggregator.compute_basic_stats(termination_token_counts),
    )

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
