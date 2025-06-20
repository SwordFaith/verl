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

from .aggregation import (
    MetricsAggregator,
    aggregate_batch_metrics,
    aggregate_conversation_metrics,
    aggregate_tool_metrics,
)
from .schemas import (
    AggregatedConversationMetrics,
    AggregatedMetrics,
    AggregatedToolMetrics,
    BaseMetrics,
    BatchMetrics,
    ConversationMetrics,
    ToolMetrics,
)
from .utils import reduce_metrics

__all__ = [
    # Legacy utility
    "reduce_metrics",
    # Dual-layer aggregation functions
    "aggregate_tool_metrics",
    "aggregate_conversation_metrics",  # Now includes termination + turn data
    "aggregate_batch_metrics",
    "MetricsAggregator",
    # Dual-layer Pydantic schemas
    "BaseMetrics",
    "ToolMetrics",
    "ConversationMetrics",  # Now unified with termination + turn data
    "AggregatedMetrics",
    "AggregatedToolMetrics",
    "AggregatedConversationMetrics",  # Now includes termination aggregations
    "BatchMetrics",
]
