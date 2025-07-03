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

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/retool_aime2024_25")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    aime_2024_data_path = "Maxwell-Jia/AIME_2024"
    aime_2025_data_path = "yentinglin/aime_2025"
    aime_2024_dataset = datasets.load_dataset(aime_2024_data_path, "default")
    aime_2025_dataset = datasets.load_dataset(aime_2025_data_path, "default")

    aime_2024_dataset = aime_2024_dataset["train"]
    aime_2025_dataset = aime_2025_dataset["train"]

    # add a row to each data item that represents a unique id
    def make_map_fn_2024(split):
        def process_fn(example, idx):
            question = example["Problem"]
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that can solve math problems with interaction "
                        "Code Interpreter by Python code."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Solve the following problem step by step. You now have the ability to selectively "
                        "write executable Python code to enhance your reasoning process.\n\n"
                        f"**user question:**\n{question}\n\n"
                        "Remember to place the final answer in the last part using the format: "
                        "\n<answer>\n\\boxed{'The final answer goes here.'}\n</answer>"
                    ),
                },
            ]
            reward_model = {"ground_truth": str(example["Answer"]), "style": "rule-lighteval/MATH_v2"}
            extra_info = {
                "index": idx,
                "raw_prompt": question,
            }
            extra_info["need_tools_kwargs"] = True
            extra_info["tools_kwargs"] = {
                "code_interpreter": {
                    "create_kwargs": {
                        "ground_truth": reward_model["ground_truth"],
                    },
                },
            }
            res = {
                "data_source": "retool_aime2024",
                "prompt": prompt,
                "ability": "MATH",
                "reward_model": reward_model,
                "extra_info": extra_info,
            }
            return res

        return process_fn

    def make_map_fn_2025(split):
        def process_fn(example, idx):
            question = example.pop("problem")
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that can solve math problems with interaction "
                        "Code Interpreter by Python code."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Solve the following problem step by step. You now have the ability to selectively "
                        "write executable Python code to enhance your reasoning process.\n\n"
                        f"**user question:**\n{question}\n\n"
                        "Remember to place the final answer in the last part using the format: "
                        "\n<answer>\n\\boxed{'The final answer goes here.'}\n</answer>"
                    ),
                },
            ]
            reward_model = {"ground_truth": str(example["answer"]), "style": "rule-lighteval/MATH_v2"}
            extra_info = {
                "index": idx,
                "raw_prompt": question,
            }
            extra_info["need_tools_kwargs"] = True
            extra_info["tools_kwargs"] = {
                "code_interpreter": {
                    "create_kwargs": {
                        "ground_truth": reward_model["ground_truth"],
                    },
                },
            }
            res = {
                "data_source": "retool_aime2025",
                "prompt": prompt,
                "ability": "MATH",
                "reward_model": reward_model,
                "extra_info": extra_info,
            }
            return res

        return process_fn

    aime_2024_train_dataset = aime_2024_dataset.map(
        function=make_map_fn_2024("train"), with_indices=True, remove_columns=aime_2024_dataset.column_names
    )
    aime_2025_train_dataset = aime_2025_dataset.map(
        function=make_map_fn_2025("train"), with_indices=True, remove_columns=aime_2025_dataset.column_names
    )

    train_dataset = datasets.concatenate_datasets([aime_2024_train_dataset, aime_2025_train_dataset])

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
