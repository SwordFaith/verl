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
usage: torchrun --standalone --nnodes=1 \
    --nproc_per_node=2 $(which pytest) \
    -s test_sglang_async_rollout_w_tools.py
"""

import numpy as np
import torch
from tensordict import TensorDict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from utils_sglang import (
    are_lists_similar,
    clean_torchelastic_env,
    generate_hf_output,
    get_rollout_config,
    initialize_global_process_group,
    load_tokenizer_and_model,
    prepare_inputs,
)

from verl import DataProto
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout
from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager


def test_async_sglang_rollout_w_tool():
    assert torch.cuda.device_count() >= 2
    initialize_global_process_group()
    clean_torchelastic_env()

    max_prompt_length = 1024
    max_response_length = 1024
    dtype = "bfloat16"
    tensor_parallel_size = 2
    local_model_path = "/user/longxiang1/models/swordfaith/ReTool-Qwen3-4B-SFT-cold-started"

    tokenizer, actor_model = load_tokenizer_and_model(local_model_path)

    preencode_prompts = [
        [{"role": "system", "content": "You are a helpful assistant that can solve math problems with interaction Code Interpreter by Python code.", "tool_calls": None},
            {"role": "user", "content": "Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. \n\n**user question:**\nThere are 152 students at Dala High School. Assume the following:  \n- 100 students take a Math class  \n- 94 students take a Science class  \n- 57 students take an English class  \n- 73 students take a Math class and a Science class  \n- 24 students take a Math class and an English class  \n- 27 students take a Science class and an English class  \n- 22 students take a Math class and a Science class and an English class\n  \nHow many students take neither a Math class nor a Science class nor an Eglish class?\n\nRemember to place the final answer in the last part using the format: \n<answer>\n\x08oxed{'The final answer goes here.'}\n</answer>", "tool_calls": None}]
    ]
    prompts = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in preencode_prompts]
    input_ids, attention_mask, position_ids = prepare_inputs(tokenizer, prompts, max_prompt_length)

    # hf_response_tokens = generate_hf_output(actor_model, input_ids, attention_mask, tokenizer, max_response_length)
    hf_response_tokens = ['<|im_start|>assistant\n<think>\nOkay, let\'s tackle this problem step by step. We need to find out how many students at Dala High School are not taking any of the three classes: Math, Science, or English. The total number of students is 152. \n\nFirst, let\'s list out all the given information:\n\n- Total students: 152\n- Math class: 100 students\n- Science class: 94 students\n- English class: 57 students\n- Math and Science: 73 students\n- Math and English: 24 students\n- Science and English: 27 students\n- Math, Science, and English: 22 students\n\nHmm, this seems like a classic inclusion-exclusion problem. I remember that when dealing with overlapping sets, we need to account for the overlaps to avoid double-counting. The formula for three sets is:\n\nTotal = (A + B + C) - (AB + AC + BC) + ABC\n\nWhere:\n- A, B, C are the individual sets (Math, Science, English)\n- AB, AC, BC are the pairwise overlaps\n- ABC is the triple overlap\n\nSo, plugging in the numbers:\n\nTotal = (100 + 94 + 57) - (73 + 24 + 27) + 22\n\nLet me compute each part step by step. First, the sum of the individual classes:\n\n100 (Math) + 94 (Science) + 57 (English) = 251\n\nNext, the sum of the pairwise overlaps:\n\n73 (Math & Science) + 24 (Math & English) + 27 (Science & English) = 124\n\nThen, the triple overlap is 22.\n\nSo, applying the formula:\n\nTotal = 251 - 124 + 22\n\nLet me calculate that. 251 minus 124 is 127, and then adding 22 gives 149.\n\nWait, so according to this, 149 students are taking at least one of the three classes. Therefore, the number of students taking none of the classes would be the total students minus 149.\n\nTotal students: 152\n\nSo, 152 - 149 = 3\n\nBut let me verify this with code to ensure accuracy.\n<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "total_students = 152\\nmath = 100\\nscience = 94\\nenglish = 57\\nmath_science = 73\\nmath_english = 24\\nscience_english = 27\\nall_three = 22\\n\\n# Calculate the number of students taking at least one class\\nstudents_with_classes = (math + science + english) - (math_science + math_english + science_english) + all_three\\nstudents_neither = total_students - students_with_classes\\nprint(students_neither)", "execution_count": 1, "language": "python"}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n3\n\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\nOkay, the code confirms that the number of students taking none of the classes is 3. That matches our manual calculation. So, the answer should be 3.\n\nBut just to make sure I didn\'t make a mistake in the formula. Let me recall: when you add the individual sets, you\'ve counted the overlaps twice, so you subtract them once. Then, the triple overlap was subtracted three times in the pairwise overlaps, so you add it back once. That seems right.\n\nAlternatively, maybe I can use a Venn diagram approach. Let\'s visualize three overlapping circles: Math, Science, English. The intersections are given. The triple overlap is 22. So, the number of students taking only Math and Science would be 73 - 22 = 51. Similarly, only Math and English: 24 - 22 = 2. Only Science and English: 27 - 22 = 5. Then, the number of students taking only Math: 100 - (51 + 2 + 22) = 25. Only Science: 94 - (51 + 5 + 22) = 16. Only English: 57 - (2 + 5 + 22) = 28. Then, the total students taking at least one class would be 25 + 16 + 28 + 51 + 2 + 5 + 22. Let\'s compute that:\n\n25 + 16 = 41\n\n41 + 28 = 69\n\n69 + 51 = ']

    fsdp_device_mesh = init_device_mesh("cuda", mesh_shape=(tensor_parallel_size,), mesh_dim_names=("fsdp",))
    inference_device_mesh_cpu = init_device_mesh("cpu", mesh_shape=(1, tensor_parallel_size, 1), mesh_dim_names=("dp", "infer_tp", "pp"))

    fsdp_model = FSDP(
        actor_model,
        use_orig_params=True,
        device_id=fsdp_device_mesh["fsdp"].get_local_rank(),
        mixed_precision=MixedPrecision(param_dtype=getattr(torch, dtype)),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_mesh=fsdp_device_mesh,
    )

    rollout_config = get_rollout_config(max_response_length, max_prompt_length, dtype, tensor_parallel_size, tool_config_path="examples/sglang_multiturn/config/tool_config/sandbox_fusion_tool_config.yaml")
    rollout = SGLangRollout(actor_module=local_model_path, config=rollout_config, tokenizer=tokenizer, model_hf_config=actor_model.config)

    rollout_sharding_manager = FSDPSGLangShardingManager(
        module=fsdp_model,
        inference_engine=rollout._engine,
        model_config=actor_model.config,
        full_params=True,
        device_mesh=inference_device_mesh_cpu,
    )

    with rollout_sharding_manager:
        prompt_dict = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0],
        )
        print(f"preprocessed {input_ids.shape=}")

        messages = np.asarray(preencode_prompts)
        non_tensor_batch = {"raw_prompt": messages, "tools_kwargs": np.array([{"code_interpreter": {}}])}
        prompts = DataProto(batch=prompt_dict, non_tensor_batch=non_tensor_batch)

        prompts.meta_info.update(
            {
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            }
        )

        prompts = rollout_sharding_manager.preprocess_data(prompts)
        # log_gpu_memory_usage("Before generating sequences", logger=None)
        output = rollout.generate_sequences(prompts=prompts)
        print(f"generated {output.batch['responses'].shape=}")
        # log_gpu_memory_usage("After generating sequences", logger=None)
        output = rollout_sharding_manager.postprocess_data(output)
        print(f"postprocessed {output.batch['responses'].shape=}")
        sglang_output = output.to("cpu")

    sglang_response_tokens = tokenizer.batch_decode(sglang_output.batch["responses"])

    print(f"hf response: {hf_response_tokens}")
    print(f"sglang response: {sglang_response_tokens}")
    assert are_lists_similar(hf_response_tokens, sglang_response_tokens, threshold=10)
    print("SGLang w tool Test Passed!")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    test_async_sglang_rollout_w_tool()
