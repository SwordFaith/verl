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
from __future__ import annotations

import warnings
from omegaconf import DictConfig
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout


class AsyncSGLangRollout(SGLangRollout):
    def __init__(
        self,
        actor_module: nn.Module | str,
        config: DictConfig,
        tokenizer,
        model_hf_config,
        port=None,
        trust_remote_code: bool = False,
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ):
        warnings.warn(
            "`AsyncSGLangRollout` is deprecated and will be removed in a future release. "
            "Please use `SGLangRollout` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            actor_module=actor_module,
            config=config,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            port=port,
            trust_remote_code=trust_remote_code,
            device_mesh=device_mesh,
            **kwargs,
        )