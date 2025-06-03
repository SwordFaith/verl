# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess the DAPO-Math-17k dataset to multiturn format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs

PROMBLEM_PREFIX = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/retool_dapo")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_path = "/user/longxiang1/data/BytedTsinghua-SIA/DAPO-Math-17k"
    dataset = datasets.load_dataset(data_path, "default")

    train_dataset = dataset["train"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            prompt = example["prompt"]
            prompt_content = prompt[0]["content"]
            problem = prompt_content.split(PROMBLEM_PREFIX)[1]
            system_prompt = "You are a helpful assistant that can solve math problems with interaction Code Interpreter by Python code."
            user_prompt_template = (
                "Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process.\n\n"
                "**user question:**\n"
                "{problem}"
                "\nLet me compute that step by step using code to ensure accuracy."
            )
            user_prompt = user_prompt_template.format(problem=problem)
            example["prompt"] = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ]
            orig_extra_info = example.pop("extra_info")
            extra_info = orig_extra_info.copy()
            extra_info["need_tools_kwargs"] = True
            extra_info["tools_kwargs"] = {
                "code_interpreter": {
                    "create_kwargs": {
                        "ground_truth": example["reward_model"]["ground_truth"],
                    },
                },
            }
            example["extra_info"] = extra_info
            return example

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
