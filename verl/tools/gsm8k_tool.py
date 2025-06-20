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

from typing import Any, Optional, Tuple
from uuid import uuid4

from verl.utils.reward_score import gsm8k

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema


class Gsm8kTool(BaseTool):
    """A demo tool for calculating the reward of gsm8k.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "calc_gsm8k_reward",
                "description": "A tool for calculating the reward of gsm8k",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The answer to the question",
                        },
                    },
                    "required": ["answer"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id

    async def _execute_impl(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, bool, dict[str, Any]]:
        answer = parameters.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)

        # Format answer with required prefix
        formatted_answer = answer if answer.startswith("#### ") else "#### " + answer
        self._instance_dict[instance_id]["response"] = formatted_answer

        # Calculate reward and improvement
        new_reward = await self.calc_reward(instance_id)
        previous_reward = self._instance_dict[instance_id]["reward"]
        answer_improved = new_reward > previous_reward

        # Apply penalty for non-improved answer submission
        tool_reward = 0.0 if answer_improved else -0.05

        # Update stored reward
        self._instance_dict[instance_id]["reward"] = new_reward

        # Enhanced tool-specific metrics for mathematical problem solving
        specific_metrics = {
            # Basic answer tracking
            "parsed_answer": answer,
            "formatted_answer": formatted_answer,
            "ground_truth": self._instance_dict[instance_id]["ground_truth"],
            # Reward and improvement tracking
            "current_reward": new_reward,
            "previous_reward": previous_reward,
            "answer_improvement": answer_improved,
            "reward_delta": new_reward - previous_reward,
            # Mathematical analysis metrics
            "solution_step_count": len(answer.split("\n")) if answer else 0,
            "answer_char_length": len(answer),
            "contains_calculation": any(op in answer for op in ["+", "-", "*", "/", "=", "$"]),
            "answer_format_correct": answer.startswith("#### ") or formatted_answer.startswith("#### "),
            # Error classification if not successful
            "error_type": None if new_reward > 0 else self._classify_answer_error(answer, self._instance_dict[instance_id]["ground_truth"]),
        }

        success = new_reward > 0  # Consider successful if reward is positive
        response_text = f"Current parsed answer='{answer}' reward={new_reward} improved={answer_improved}"

        return response_text, tool_reward, success, specific_metrics

    def _classify_answer_error(self, answer: str, ground_truth: str) -> str:
        """Classify the type of error in the mathematical answer."""
        if not answer or answer.strip() == "":
            return "empty_answer"

        if not answer.startswith("#### "):
            return "format_error"

        # Extract numerical answer
        try:
            answer_num = answer.replace("#### ", "").strip()
            if not answer_num:
                return "no_numerical_answer"

            # Try to parse as number
            try:
                float(answer_num.replace(",", "").replace("$", ""))
                return "wrong_calculation"  # Format is correct but value is wrong
            except ValueError:
                return "non_numerical_answer"  # Contains non-numeric content

        except Exception:
            return "parsing_error"

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return gsm8k.compute_score(
            self._instance_dict[instance_id]["response"],
            self._instance_dict[instance_id]["ground_truth"],
            method="flexible",
            format_score=0.0,
            score=1.0,
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
