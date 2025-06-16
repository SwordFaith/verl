#!/bin/bash
set -x

export PYTHONUNBUFFERED=1
export RUST_BACKTRACE=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export WANDB_DIR=/data/tensorboard/

ulimit -n 65535

EXPERIMENT_NAME=retool-multiturn-sft-qwen2.5-32b-sp8

HOME=/wuxi_gpfs/user/longxiang1
torchrun \
     --nnodes=${WORLD_SIZE} \
     --nproc_per_node=8 \
     --node_rank=${RANK} \
     --master_addr=${MASTER_ADDR} \
     --master_port=${MASTER_PORT} \
     -m verl.trainer.fsdp_sft_trainer \
     data.max_length=16384 \
     data.train_batch_size=128 \
     data.micro_batch_size_per_gpu=2 \
     data.train_files=$HOME/data/retool_multi_turn_sft_preprocessed/train.parquet \
     data.val_files=$HOME/data/retool_multi_turn_sft_preprocessed/test.parquet \
     data.multiturn.enable=true \
     data.multiturn.messages_key=messages \
     data.multiturn.tools_key=tools \
     model.partial_pretrain=$HOME/models/Qwen/Qwen2.5-32B-Instruct \
     model.trust_remote_code=true \
     model.fsdp_config.cpu_offload=true \
     model.fsdp_config.offload_params=true \
     model.fsdp_config.model_dtype=bf16 \
     optim.lr=1e-6 \
     trainer.default_local_dir=$HOME/checkpoints/retool-multiturn-sft/$EXPERIMENT_NAME \
     trainer.project_name=retool-multiturn-sft \
     trainer.experiment_name=$EXPERIMENT_NAME \
     trainer.logger=['console','wandb'] \
     trainer.total_epochs=12 \
     trainer.default_hdfs_dir=null $@ \
     trainer.save_freq=14 \
     ulysses_sequence_parallel_size=8 \
     use_remove_padding=true
