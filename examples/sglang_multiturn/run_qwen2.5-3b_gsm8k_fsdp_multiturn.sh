#!/bin/bash
set -x

python3 -m verl.trainer.main_ppo \
    --config-path=/root/verl/examples/sglang_multiturn/config \
    --config-name='gsm8k_multiturn_grpo'
