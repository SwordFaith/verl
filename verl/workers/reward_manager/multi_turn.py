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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Optional

import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register

logger = logging.getLogger(__name__)


def compute_single_reward(compute_score_func, data_source, response_str, ground_truth, extra_info):
    """Single reward computation for ProcessPool"""
    try:
        return compute_score_func(data_source=data_source, solution_str=response_str, ground_truth=ground_truth, extra_info=extra_info)
    except Exception as e:
        logger.warning(f"Reward computation failed: {e}")
        return 0.0


def parallel_compute_rewards(compute_score_func, tasks_data, num_processes=32, timeout=300.0):
    """Parallel reward computation using ProcessPool"""
    scores = []

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all tasks
        futures = []
        for task_data in tasks_data:
            future = executor.submit(compute_single_reward, compute_score_func, task_data["data_source"], task_data["response_str"], task_data["ground_truth"], task_data["extra_info"])
            futures.append(future)

        # Collect results
        for future in futures:
            try:
                result = future.result(timeout=timeout)
                if isinstance(result, dict):
                    scores.append(result.get("score", 0.0))
                elif isinstance(result, (int, float, bool)):
                    scores.append(float(result))
                else:
                    scores.append(0.0)
            except Exception as e:
                logger.warning(f"Task failed: {e}")
                scores.append(0.0)

    return scores


@register("multi_turn")
class MultiTurnRewardManager:
    """Multi-turn reward manager that incorporates turn-level tool rewards."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        num_processes: int = 32,
        timeout: float = 300.0,
    ) -> None:
        """
        Initialize the MultiTurnRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, default_compute_score will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to "data_source".
            num_processes: Number of processes for parallel reward computation. Defaults to 32.
            timeout: Timeout in seconds for each reward computation task. Defaults to 300.0.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.num_processes = num_processes
        self.timeout = timeout

    def __call__(self, data: DataProto, return_dict=False):
        """Enhanced reward calculation incorporating turn-level rewards with ProcessPool optimization."""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        # Prepare task data for parallel processing
        tasks_data = []
        valid_response_lengths = []

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            # Extract turn-level rewards and tool rewards from rollout
            messages = None
            turn_level_rewards = None
            tool_rewards = None

            if "messages" in data_item.non_tensor_batch:
                messages = data_item.non_tensor_batch["messages"]

            if "turn_level_rewards" in data_item.non_tensor_batch:
                turn_level_rewards = data_item.non_tensor_batch["turn_level_rewards"]
                if hasattr(turn_level_rewards, "tolist"):
                    turn_level_rewards = turn_level_rewards.tolist()

            if "tool_rewards" in data_item.non_tensor_batch:
                tool_rewards = data_item.non_tensor_batch["tool_rewards"]
                if hasattr(tool_rewards, "item"):
                    tool_rewards = tool_rewards.item()

            # Prepare task data
            task_data = {
                "data_source": data_source,
                "response_str": response_str,
                "ground_truth": ground_truth,
                "extra_info": {
                    "messages": messages["messages"] if messages is not None else None,
                    "turn_level_rewards": turn_level_rewards,
                    "tool_rewards": tool_rewards,
                },
            }

            tasks_data.append(task_data)
            valid_response_lengths.append(valid_response_length)

        print("tasks_data[0]:", tasks_data[0])
        # Parallel computation using configured parameters
        try:
            scores = parallel_compute_rewards(self.compute_score, tasks_data, num_processes=self.num_processes, timeout=self.timeout)
        except Exception as e:
            logger.error(f"Parallel reward computation failed: {e}")
            scores = [0.0] * len(tasks_data)

        # Apply scores to reward tensor and debug logging
        for i, (task_data, score, valid_response_length) in enumerate(zip(tasks_data, scores, valid_response_lengths)):
            reward_tensor[i, valid_response_length - 1] = score

            data_source = task_data["data_source"]

            # Debug logging
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[response]", task_data["response_str"])
                print("[ground_truth]", task_data["ground_truth"])

                extra_info = task_data["extra_info"]
                if extra_info.get("turn_level_rewards", None) is not None:
                    print("[turn_level_rewards]", sum(extra_info["turn_level_rewards"][:20]))
                if extra_info.get("tool_rewards", None) is not None:
                    print("[tool_rewards]", extra_info["tool_rewards"])
                print("[final_reward]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
