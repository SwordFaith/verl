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
Tests for metrics collection in AsyncRolloutRequest.

This module tests the data collection layer of the metrics architecture:
- AsyncRolloutRequest unified conversation tracking
- Turn-level metrics collection and tracking
- Conversation metrics generation from collected data
- Integration compatibility with aggregation layer
"""

from typing import Dict, List
from unittest.mock import Mock

import pytest

from verl.workers.rollout.schemas import AsyncRolloutRequest, Message


class TestAsyncRolloutRequestMetrics:
    """Test suite for metrics collection in AsyncRolloutRequest."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        self.mock_tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 10 tokens

        # Base messages for testing
        self.base_messages = [{"role": "user", "content": "Hello, can you help me with coding?"}, {"role": "assistant", "content": "Of course! I'd be happy to help you with coding."}]

    def create_test_request(self, messages: List[Dict] = None) -> AsyncRolloutRequest:
        """Create a test AsyncRolloutRequest with unified metrics."""
        if messages is None:
            messages = self.base_messages

        return AsyncRolloutRequest(request_id="test-123", batch_data_id=0, rollout_offset=0, state="completed", messages=messages, max_prompt_len=512, tokenizer=self.mock_tokenizer)

    def test_unified_turn_tracking(self):
        """Test unified turn tracking functionality."""
        request = self.create_test_request()

        # Track a new turn with tool calls
        turn_stats = {"token_count": 45, "char_count": 180, "tool_calls_count": 1, "additional_data": "test"}
        request.track_turn("assistant", turn_stats)

        # Verify turn was tracked
        assert len(request.turn_details) == 1
        turn = request.turn_details[0]
        assert turn["turn_index"] == 0
        assert turn["role"] == "assistant"
        assert turn["token_count"] == 45
        assert turn["char_count"] == 180
        assert turn["tool_calls_count"] == 1
        assert turn["has_tool_calls"] is True
        assert turn["request_id"] == "test-123"
        assert turn["batch_data_id"] == 0
        assert turn["rollout_offset"] == 0
        assert turn["additional_data"] == "test"

        # Track another turn without tools
        turn_stats_2 = {"token_count": 30, "char_count": 120, "tool_calls_count": 0}
        request.track_turn("user", turn_stats_2)

        # Verify second turn
        assert len(request.turn_details) == 2
        turn_2 = request.turn_details[1]
        assert turn_2["turn_index"] == 1
        assert turn_2["role"] == "user"
        assert turn_2["has_tool_calls"] is False

    def test_unified_conversation_metrics_generation(self):
        """Test unified conversation metrics generation."""
        request = self.create_test_request()

        # Add several turns with mixed content
        turns_data = [
            {"role": "user", "token_count": 50, "char_count": 200, "tool_calls_count": 0},
            {"role": "assistant", "token_count": 60, "char_count": 240, "tool_calls_count": 2},
            {"role": "tool", "token_count": 30, "char_count": 120, "tool_calls_count": 0},
            {"role": "assistant", "token_count": 40, "char_count": 160, "tool_calls_count": 0},
            {"role": "user", "token_count": 35, "char_count": 140, "tool_calls_count": 0},
        ]

        for turn_data in turns_data:
            role = turn_data.pop("role")
            request.track_turn(role, turn_data)

        # Set termination reason
        request.set_termination_reason("eos_token")

        # Get unified conversation metrics
        conv_metrics = request.get_conversation_metrics()

        # Verify basic statistics
        assert conv_metrics["total_turns"] == 5
        assert conv_metrics["total_tokens"] == 215  # 50+60+30+40+35
        assert conv_metrics["total_chars"] == 860  # 200+240+120+160+140

        # Verify role distribution
        assert conv_metrics["user_turns"] == 2
        assert conv_metrics["assistant_turns"] == 2
        assert conv_metrics["tool_turns"] == 1
        assert conv_metrics["system_turns"] == 0

        assert conv_metrics["user_tokens"] == 85  # 50+35
        assert conv_metrics["assistant_tokens"] == 100  # 60+40
        assert conv_metrics["tool_tokens"] == 30

        # Verify tool usage patterns
        assert conv_metrics["assistant_tool_calls"] == 2
        assert conv_metrics["turns_with_tools"] == 1
        assert conv_metrics["tool_turn_ratio"] == 0.2  # 1/5
        assert conv_metrics["assistant_turn_ratio"] == 0.4  # 2/5

        # Verify termination context (unified)
        assert conv_metrics["termination_reason"] == "eos_token"
        assert conv_metrics["termination_turn_count"] == 5
        assert conv_metrics["termination_token_count"] == 215

        # Verify raw data for aggregation
        assert len(conv_metrics["turn_details"]) == 5
        assert conv_metrics["turn_token_lengths"] == [50, 60, 30, 40, 35]
        assert conv_metrics["turn_char_lengths"] == [200, 240, 120, 160, 140]

        # Verify request context
        assert conv_metrics["request_id"] == "test-123"
        assert conv_metrics["batch_data_id"] == 0
        assert conv_metrics["rollout_offset"] == 0

    def test_conversation_metrics_with_no_turns(self):
        """Test conversation metrics when no turns have been tracked."""
        request = self.create_test_request()
        request.set_termination_reason("max_tokens")

        conv_metrics = request.get_conversation_metrics()

        # Should have empty/zero values
        assert conv_metrics["total_turns"] == 0
        assert conv_metrics["total_tokens"] == 0
        assert conv_metrics["total_chars"] == 0
        assert conv_metrics["assistant_tool_calls"] == 0
        assert conv_metrics["turns_with_tools"] == 0
        assert conv_metrics["tool_turn_ratio"] == 0
        assert conv_metrics["assistant_turn_ratio"] == 0
        assert conv_metrics["termination_reason"] == "max_tokens"
        assert conv_metrics["termination_turn_count"] == 0
        assert conv_metrics["termination_token_count"] == 0

    def test_conversation_metrics_edge_cases(self):
        """Test conversation metrics with edge cases."""
        request = self.create_test_request()

        # Test with only assistant turns
        assistant_turns = [
            {"token_count": 50, "char_count": 200, "tool_calls_count": 1},
            {"token_count": 60, "char_count": 240, "tool_calls_count": 0},
        ]

        for turn_data in assistant_turns:
            request.track_turn("assistant", turn_data)

        conv_metrics = request.get_conversation_metrics()

        # All turns are assistant turns
        assert conv_metrics["assistant_turn_ratio"] == 1.0
        assert conv_metrics["user_turns"] == 0
        assert conv_metrics["tool_turn_ratio"] == 0.5  # 1 out of 2 turns has tools

    def test_initialize_conversation_from_prompt(self):
        """Test initialization of conversation tracking from prompt messages."""
        messages = [Message(role="system", content="You are a helpful assistant."), Message(role="user", content="Hello world"), Message(role="assistant", content="Hello! How can I help you?", tool_calls=None)]

        request = AsyncRolloutRequest(request_id="test-456", batch_data_id=1, rollout_offset=0, state="pending", messages=messages, max_prompt_len=512, tokenizer=self.mock_tokenizer)

        # Initialize conversation from the prompt
        request.initialize_conversation_from_prompt(messages, self.mock_tokenizer)

        # Verify turns were tracked
        assert len(request.turn_details) == 3

        # Check system turn
        system_turn = request.turn_details[0]
        assert system_turn["role"] == "system"
        assert system_turn["source"] == "initial_prompt"

        # Check user turn
        user_turn = request.turn_details[1]
        assert user_turn["role"] == "user"
        assert user_turn["source"] == "initial_prompt"

        # Check assistant turn
        assistant_turn = request.turn_details[2]
        assert assistant_turn["role"] == "assistant"
        assert assistant_turn["source"] == "initial_prompt"

    def test_tool_calls_detail_tracking(self):
        """Test tool calls detail tracking (backward compatibility)."""
        request = self.create_test_request()

        # Track turns with different tool call counts
        turn_data = [
            {"token_count": 50, "char_count": 200, "tool_calls_count": 0},  # No tools
            {"token_count": 60, "char_count": 240, "tool_calls_count": 2},  # 2 tools
            {"token_count": 40, "char_count": 160, "tool_calls_count": 1},  # 1 tool
        ]

        for i, data in enumerate(turn_data):
            request.track_turn(f"role_{i}", data)

        # Verify tool_calls_detail tracking
        assert len(request.turn_tool_calls_detail) == 3
        assert request.turn_tool_calls_detail == [0, 2, 1]

    def test_termination_reason_setting(self):
        """Test termination reason setting."""
        request = self.create_test_request()

        # Test various termination reasons
        reasons = ["eos_token", "max_tokens", "max_assistant_turns", "custom_reason"]

        for reason in reasons:
            request.set_termination_reason(reason)
            assert request.termination_reason == reason


class TestRolloutRequestMetricsIntegration:
    """Integration tests for rollout request metrics with aggregation layer."""

    def test_metrics_data_structure_compatibility(self):
        """Test that metrics data structures are compatible with aggregation."""
        # Mock a complete rollout request with metrics
        request = AsyncRolloutRequest(request_id="integration-test", batch_data_id=0, rollout_offset=0, state="completed", messages=[{"role": "user", "content": "Test message"}, {"role": "assistant", "content": "Test response"}], max_prompt_len=512, tokenizer=Mock())

        # Mock tokenizer behavior
        request.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        request.tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Track some turns
        request.track_turn("user", {"token_count": 50, "char_count": 200, "tool_calls_count": 0})
        request.track_turn("assistant", {"token_count": 75, "char_count": 300, "tool_calls_count": 1})
        request.set_termination_reason("eos_token")

        # Get conversation metrics
        conv_metrics = request.get_conversation_metrics()

        # Test that the structure is compatible with ConversationMetrics schema
        from verl.utils.metric import ConversationMetrics

        # Should be able to create ConversationMetrics object
        conversation_obj = ConversationMetrics(**conv_metrics)
        assert conversation_obj.total_turns == 2
        assert conversation_obj.termination_reason == "eos_token"

    def test_batch_metrics_aggregation_integration(self):
        """Test integration with batch metrics aggregation."""
        from verl.utils.metric import aggregate_batch_metrics, aggregate_conversation_metrics

        # Create multiple mock requests
        requests_data = []
        for i in range(3):
            conv_metrics = {
                "request_id": f"test-{i}",
                "batch_data_id": i,
                "rollout_offset": 0,
                "timestamp": 1234567890.0 + i,
                "total_turns": 4 + i,
                "total_tokens": 200 + i * 50,
                "total_chars": 800 + i * 200,
                "assistant_turns": 2,
                "user_turns": 2 + i,
                "tool_turns": 0,
                "system_turns": 0,
                "assistant_tokens": 100,
                "user_tokens": 100 + i * 50,
                "tool_tokens": 0,
                "system_tokens": 0,
                "assistant_chars": 400,
                "user_chars": 400 + i * 200,
                "tool_chars": 0,
                "system_chars": 0,
                "assistant_tool_calls": i,
                "turns_with_tools": min(i, 1),
                "tool_turn_ratio": 0.25 if i > 0 else 0,
                "assistant_turn_ratio": 0.5,
                "termination_reason": "eos_token" if i % 2 == 0 else "max_tokens",
                "termination_turn_count": 4 + i,
                "termination_token_count": 200 + i * 50,
                "turn_details": [],
                "turn_token_lengths": [50] * (4 + i),
                "turn_char_lengths": [200] * (4 + i),
            }
            requests_data.append(conv_metrics)

        # Test conversation aggregation
        conv_aggregated = aggregate_conversation_metrics(requests_data)
        assert conv_aggregated.count == 3
        assert conv_aggregated.total_turns_stats.min_value == 4
        assert conv_aggregated.total_turns_stats.max_value == 6

        # Test batch aggregation
        batch_metrics = aggregate_batch_metrics(tool_metrics_list=None, conversation_metrics_list=requests_data, processing_time_ms=150.0)

        assert batch_metrics.conversation_metrics is not None
        assert batch_metrics.tool_metrics is None
        assert batch_metrics.batch_size == 3
        assert batch_metrics.processing_time_ms == 150.0

    def test_ppo_trainer_compatibility(self):
        """Test compatibility with PPO trainer metrics processing."""
        # Mock gen_batch_output structure that PPO trainer expects
        gen_batch_output = Mock()
        gen_batch_output.non_tensor_batch = {
            "conversation_metrics": [
                {
                    "request_id": "ppo-test-1",
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
            ],
            "tool_metrics": [],
        }

        # Test that data can be processed by PPO trainer-style aggregation
        from verl.utils.metric import aggregate_batch_metrics

        conversation_metrics_data = gen_batch_output.non_tensor_batch.get("conversation_metrics", [])
        tool_metrics_data = gen_batch_output.non_tensor_batch.get("tool_metrics", [])

        batch_metrics = aggregate_batch_metrics(tool_metrics_list=tool_metrics_data if tool_metrics_data else None, conversation_metrics_list=conversation_metrics_data if conversation_metrics_data else None, processing_time_ms=100.0)

        # Should work without errors and produce expected structure
        assert batch_metrics.conversation_metrics is not None
        assert batch_metrics.conversation_metrics.count == 1
        assert batch_metrics.conversation_metrics.total_turns_stats.avg_value == 4.0

        # Should include termination data in conversation metrics
        assert "eos_token" in batch_metrics.conversation_metrics.termination_reason_distribution
        assert batch_metrics.conversation_metrics.termination_reason_distribution["eos_token"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__])
