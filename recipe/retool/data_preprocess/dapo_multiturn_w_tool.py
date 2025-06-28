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
    parser.add_argument("--local_dir", default="~/data/retool_dapo")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_path = "BytedTsinghua-SIA/DAPO-Math-17k"
    dataset = datasets.load_dataset(data_path, "default")

    train_dataset = dataset["train"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            orig_extra_info = example.pop("extra_info")
            prompt = example.pop("prompt")
            question = "\n\n".join(prompt[0]["content"].split("\n\n")[1:-1])
            prompt = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can solve math problems with interaction Code Interpreter by Python code.",
                },
                {
                    "role": "user",
                    "content": (
                        "Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process.\n\n"
                        f"**user question:**\n{question}\n\n"
                        "Remember to place the final answer in the last part using the format: \n<answer>\n\\boxed{'The final answer goes here.'}\n</answer>"
                    ),
                },
            ]
            extra_info = orig_extra_info.copy()
            extra_info["need_tools_kwargs"] = True
            extra_info["tools_kwargs"] = {
                "code_interpreter": {
                    "create_kwargs": {
                        "ground_truth": example["reward_model"]["ground_truth"],
                    },
                },
            }
            example["data_source"] = "retool_dapo"
            example["prompt"] = prompt
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
