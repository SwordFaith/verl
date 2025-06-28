# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from typing import Any, Dict, List, Optional

from .math_dapo import verify as math_dapo_verify


def compute_retool_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    messages: Optional[List[Dict[str, Any]]] = None,
    turn_level_rewards: Optional[List[float]] = None,
    tool_rewards: Optional[Dict[str, float]] = None,
    enable_tool_exploration_bonus: bool = True,
    tool_success_bonus: float = 0.001,
) -> Dict[str, Any]:
    """Compute ReTool reward score incorporating turn-level tool rewards.

    Args:
        data_source: Source of the data (e.g., 'retool_prompt_dapo')
        solution_str: The generated solution string
        ground_truth: The ground truth answer
        messages: List[Dict[str, Any]] - messages style trajectory
        turn_level_rewards: List[float] - rewards for each message turn from tool execute() calls, 0.0 for non-tool turns
        tool_rewards: Dict[str, float] - final calc_reward() scores for each tool type used in trajectory
        enable_tool_exploration_bonus: Whether to give exploration bonus for failed trajectories
        tool_success_bonus: Bonus per successful tool execution (0.001) for failed trajectories

    Returns:
        Dictionary containing score, accuracy, prediction and turn-level details
    """
    if messages is not None:
        for message in messages[::-1]:
            if message["role"] == "assistant":
                _solution_str = message["content"]
                break
    else:
        _solution_str = solution_str

    is_correct, normalized_pred = math_dapo_verify(_solution_str[-300:], ground_truth)

    if is_correct:
        reward = 1.0
    else:
        reward = -1.0

    if tool_rewards is not None:
        for tool_name, tool_reward in tool_rewards.items():
            if tool_reward > 0:
                reward += tool_reward
            else:
                reward -= tool_reward
    elif turn_level_rewards is not None:
        reward += sum(turn_level_rewards)

    # Detailed metrics for analysis
    result = {
        "score": reward,
        "acc": is_correct,
        "pred": normalized_pred,
    }

    return result


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Temporary interface for compatibility with existing reward managers.

    This function provides the standard interface expected by reward managers,
    while the main logic is in compute_retool_score.

    Args:
        data_source: Source identifier
        solution_str: Generated solution
        ground_truth: Expected answer
        extra_info: Contains turn_level_rewards (List[float]) and tool_rewards (Dict[str, float])
    """

    # Extract turn-level and tool reward information from extra_info
    messages = extra_info.get("messages")
    turn_level_rewards = None
    tool_rewards = None

    if extra_info:
        messages = extra_info.get("messages")  # List[Dict[str, Any]]
        turn_level_rewards = extra_info.get("turn_level_rewards")  # List[float] from execute() calls
        tool_rewards = extra_info.get("tool_rewards")  # Dict[str, float] from calc_reward() calls

    return compute_retool_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        messages=messages,
        turn_level_rewards=turn_level_rewards,
        tool_rewards=tool_rewards,
    )
