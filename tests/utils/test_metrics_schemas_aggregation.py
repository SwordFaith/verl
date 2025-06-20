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
Tests for metrics schemas validation and aggregation functions.

This module tests the dual-layer metrics architecture's aggregation layer:
- Pydantic schema validation (ToolMetrics, ConversationMetrics)
- Aggregation functions (aggregate_tool_metrics, aggregate_conversation_metrics, aggregate_batch_metrics)
- MetricsAggregator utility functions
- Dual-layer batch aggregation
"""

import pytest

from verl.utils.metric import (
    ConversationMetrics,
    MetricsAggregator,
    ToolMetrics,
    aggregate_batch_metrics,
    aggregate_conversation_metrics,
    aggregate_tool_metrics,
)


class TestMetricsSchemasAggregation:
    """Test suite for metrics schemas validation and aggregation functions."""

    def test_conversation_metrics_schema_validation(self):
        """Test ConversationMetrics Pydantic schema validation."""
        # Valid unified conversation metrics data
        conv_data = {
            "request_id": "test-123",
            "batch_data_id": 0,
            "rollout_offset": 0,
            "timestamp": 1234567890.0,
            "total_turns": 4,
            "total_tokens": 200,
            "total_chars": 800,
            "assistant_turns": 2,
            "user_turns": 2,
            "tool_turns": 0,
            "system_turns": 0,
            "assistant_tokens": 100,
            "user_tokens": 100,
            "tool_tokens": 0,
            "system_tokens": 0,
            "assistant_chars": 400,
            "user_chars": 400,
            "tool_chars": 0,
            "system_chars": 0,
            "assistant_tool_calls": 1,
            "turns_with_tools": 1,
            "tool_turn_ratio": 0.25,
            "assistant_turn_ratio": 0.5,
            "termination_reason": "eos_token",
            "termination_turn_count": 4,
            "termination_token_count": 200,
            "turn_details": [
                {"turn_index": 0, "role": "user", "token_count": 50, "char_count": 200, "tool_calls_count": 0},
                {"turn_index": 1, "role": "assistant", "token_count": 50, "char_count": 200, "tool_calls_count": 1},
                {"turn_index": 2, "role": "user", "token_count": 50, "char_count": 200, "tool_calls_count": 0},
                {"turn_index": 3, "role": "assistant", "token_count": 50, "char_count": 200, "tool_calls_count": 0},
            ],
            "turn_token_lengths": [50, 50, 50, 50],
            "turn_char_lengths": [200, 200, 200, 200],
        }

        # Should validate successfully
        conv_metrics = ConversationMetrics(**conv_data)
        assert conv_metrics.total_turns == 4
        assert conv_metrics.termination_reason == "eos_token"
        assert len(conv_metrics.turn_details) == 4

    def test_conversation_metrics_validation_errors(self):
        """Test ConversationMetrics validation edge cases."""
        # Invalid ratio (> 1.0)
        with pytest.raises(ValueError):
            ConversationMetrics(
                request_id="test",
                batch_data_id=0,
                rollout_offset=0,
                timestamp=1234567890.0,
                total_turns=4,
                total_tokens=200,
                total_chars=800,
                assistant_turns=2,
                user_turns=2,
                tool_turns=0,
                system_turns=0,
                assistant_tokens=100,
                user_tokens=100,
                tool_tokens=0,
                system_tokens=0,
                assistant_chars=400,
                user_chars=400,
                tool_chars=0,
                system_chars=0,
                assistant_tool_calls=1,
                turns_with_tools=1,
                tool_turn_ratio=1.5,  # Invalid: > 1.0
                assistant_turn_ratio=0.5,
                termination_reason="eos_token",
                termination_turn_count=4,
                termination_token_count=200,
                turn_details=[],
                turn_token_lengths=[50, 50, 50, 50],
                turn_char_lengths=[200, 200, 200, 200],
            )

    def test_tool_metrics_schema_validation(self):
        """Test ToolMetrics Pydantic schema validation."""
        tool_data = {
            "request_id": "test-456",
            "batch_data_id": 1,
            "rollout_offset": 0,
            "timestamp": 1234567890.0,
            "tool_name": "sandbox_fusion",
            "instance_id": "sf-001",
            "latency_ms": 1500.5,
            "success": True,
            "response_char_length": 234,
            "error_type": None,
            "tool_specific_metrics": {"code_char_len": 150},
            "tool_calls_per_trajectory": 3,
            "tool_calls_per_turn": 1,
        }

        tool_metrics = ToolMetrics(**tool_data)
        assert tool_metrics.tool_name == "sandbox_fusion"
        assert tool_metrics.latency_ms == 1500.5
        assert tool_metrics.success is True

    def test_conversation_metrics_aggregation(self):
        """Test unified conversation metrics aggregation."""
        conv_metrics_list = [
            {
                "request_id": "test-1",
                "batch_data_id": 0,
                "rollout_offset": 0,
                "timestamp": 1234567890.0,
                "total_turns": 4,
                "total_tokens": 200,
                "total_chars": 800,
                "assistant_turns": 2,
                "user_turns": 2,
                "tool_turns": 0,
                "system_turns": 0,
                "assistant_tokens": 100,
                "user_tokens": 100,
                "tool_tokens": 0,
                "system_tokens": 0,
                "assistant_chars": 400,
                "user_chars": 400,
                "tool_chars": 0,
                "system_chars": 0,
                "assistant_tool_calls": 1,
                "turns_with_tools": 1,
                "tool_turn_ratio": 0.25,
                "assistant_turn_ratio": 0.5,
                "termination_reason": "eos_token",
                "termination_turn_count": 4,
                "termination_token_count": 200,
                "turn_details": [],
                "turn_token_lengths": [50, 50, 50, 50],
                "turn_char_lengths": [200, 200, 200, 200],
            },
            {
                "request_id": "test-2",
                "batch_data_id": 1,
                "rollout_offset": 0,
                "timestamp": 1234567891.0,
                "total_turns": 6,
                "total_tokens": 300,
                "total_chars": 1200,
                "assistant_turns": 3,
                "user_turns": 3,
                "tool_turns": 0,
                "system_turns": 0,
                "assistant_tokens": 150,
                "user_tokens": 150,
                "tool_tokens": 0,
                "system_tokens": 0,
                "assistant_chars": 600,
                "user_chars": 600,
                "tool_chars": 0,
                "system_chars": 0,
                "assistant_tool_calls": 2,
                "turns_with_tools": 2,
                "tool_turn_ratio": 0.33,
                "assistant_turn_ratio": 0.5,
                "termination_reason": "max_tokens",
                "termination_turn_count": 6,
                "termination_token_count": 300,
                "turn_details": [],
                "turn_token_lengths": [50, 50, 50, 50, 50, 50],
                "turn_char_lengths": [200, 200, 200, 200, 200, 200],
            },
        ]

        # Test aggregation
        aggregated = aggregate_conversation_metrics(conv_metrics_list)

        # Verify basic aggregations
        assert aggregated.count == 2
        assert aggregated.total_turns_stats.min_value == 4
        assert aggregated.total_turns_stats.max_value == 6
        assert aggregated.total_turns_stats.avg_value == 5.0

        # Verify termination metrics (unified)
        assert "eos_token" in aggregated.termination_reason_distribution
        assert "max_tokens" in aggregated.termination_reason_distribution
        assert aggregated.termination_reason_distribution["eos_token"] == 0.5
        assert aggregated.termination_reason_distribution["max_tokens"] == 0.5

    def test_tool_metrics_aggregation(self):
        """Test tool metrics aggregation."""
        tool_metrics_list = [
            {"request_id": "test-1", "batch_data_id": 0, "rollout_offset": 0, "timestamp": 1234567890.0, "tool_name": "sandbox_fusion", "instance_id": "sf-001", "latency_ms": 1000.0, "success": True, "response_char_length": 200, "tool_calls_per_trajectory": 2, "tool_calls_per_turn": 1},
            {"request_id": "test-1", "batch_data_id": 0, "rollout_offset": 0, "timestamp": 1234567891.0, "tool_name": "search_tool", "instance_id": "st-001", "latency_ms": 500.0, "success": True, "response_char_length": 150, "tool_calls_per_trajectory": 2, "tool_calls_per_turn": 1},
        ]

        aggregated = aggregate_tool_metrics(tool_metrics_list)

        # Verify aggregations
        assert aggregated.count == 2
        assert aggregated.latency_ms_stats.min_value == 500.0
        assert aggregated.latency_ms_stats.max_value == 1000.0
        assert aggregated.latency_ms_stats.avg_value == 750.0
        assert aggregated.success_rate == 1.0
        assert set(aggregated.tools_used) == {"sandbox_fusion", "search_tool"}

    def test_dual_layer_batch_aggregation(self):
        """Test dual-layer batch metrics aggregation."""
        tool_metrics_list = [
            {"request_id": "test-1", "batch_data_id": 0, "rollout_offset": 0, "timestamp": 1234567890.0, "tool_name": "sandbox_fusion", "instance_id": "sf-001", "latency_ms": 1000.0, "success": True, "response_char_length": 200, "tool_calls_per_trajectory": 1, "tool_calls_per_turn": 1}
        ]

        conversation_metrics_list = [
            {
                "request_id": "test-1",
                "batch_data_id": 0,
                "rollout_offset": 0,
                "timestamp": 1234567890.0,
                "total_turns": 4,
                "total_tokens": 200,
                "total_chars": 800,
                "assistant_turns": 2,
                "user_turns": 2,
                "tool_turns": 0,
                "system_turns": 0,
                "assistant_tokens": 100,
                "user_tokens": 100,
                "tool_tokens": 0,
                "system_tokens": 0,
                "assistant_chars": 400,
                "user_chars": 400,
                "tool_chars": 0,
                "system_chars": 0,
                "assistant_tool_calls": 1,
                "turns_with_tools": 1,
                "tool_turn_ratio": 0.25,
                "assistant_turn_ratio": 0.5,
                "termination_reason": "eos_token",
                "termination_turn_count": 4,
                "termination_token_count": 200,
                "turn_details": [],
                "turn_token_lengths": [50, 50, 50, 50],
                "turn_char_lengths": [200, 200, 200, 200],
            }
        ]

        # Test dual-layer aggregation
        batch_metrics = aggregate_batch_metrics(tool_metrics_list=tool_metrics_list, conversation_metrics_list=conversation_metrics_list, processing_time_ms=100.0)

        # Verify dual-layer structure
        assert batch_metrics.tool_metrics is not None
        assert batch_metrics.conversation_metrics is not None
        assert batch_metrics.batch_size == 1
        assert batch_metrics.processing_time_ms == 100.0

        # Verify tool metrics
        assert batch_metrics.tool_metrics.count == 1
        assert batch_metrics.tool_metrics.latency_ms_stats.avg_value == 1000.0

        # Verify conversation metrics (now includes termination)
        assert batch_metrics.conversation_metrics.count == 1
        assert batch_metrics.conversation_metrics.total_turns_stats.avg_value == 4.0
        assert batch_metrics.conversation_metrics.termination_reason_distribution["eos_token"] == 1.0

    def test_empty_metrics_aggregation(self):
        """Test aggregation with empty metric lists."""
        # Empty tool metrics
        tool_aggregated = aggregate_tool_metrics([])
        assert tool_aggregated.count == 0
        assert tool_aggregated.success_rate == 0.0

        # Empty conversation metrics
        conv_aggregated = aggregate_conversation_metrics([])
        assert conv_aggregated.count == 0
        assert conv_aggregated.termination_reason_distribution == {}

        # Empty batch metrics
        batch_metrics = aggregate_batch_metrics(tool_metrics_list=None, conversation_metrics_list=None, processing_time_ms=50.0)
        assert batch_metrics.tool_metrics is None
        assert batch_metrics.conversation_metrics is None
        assert batch_metrics.batch_size == 0

    def test_metrics_aggregator_utilities(self):
        """Test shared MetricsAggregator utility functions."""
        aggregator = MetricsAggregator()

        # Test basic stats computation
        values = [10, 20, 30, 40, 50]
        stats = aggregator.compute_basic_stats(values)
        assert stats.count == 5
        assert stats.min_value == 10
        assert stats.max_value == 50
        assert stats.avg_value == 30
        assert stats.total_value == 150

        # Test success rate computation
        success_list = [True, True, False, True, False]
        success_rate = aggregator.compute_success_rate(success_list)
        assert success_rate == 0.6

        # Test distribution computation
        items = ["eos_token", "max_tokens", "eos_token", "max_assistant_turns"]
        distribution = aggregator.compute_distribution(items)
        assert distribution["eos_token"] == 0.5
        assert distribution["max_tokens"] == 0.25
        assert distribution["max_assistant_turns"] == 0.25

        # Test value extraction
        metrics_list = [{"tool_name": "sandbox_fusion", "latency_ms": 1000}, {"tool_name": "search_tool", "latency_ms": 500}]
        tool_names = aggregator.extract_values(metrics_list, "tool_name")
        latencies = aggregator.extract_values(metrics_list, "latency_ms")
        assert tool_names == ["sandbox_fusion", "search_tool"]
        assert latencies == [1000, 500]


if __name__ == "__main__":
    pytest.main([__file__])
