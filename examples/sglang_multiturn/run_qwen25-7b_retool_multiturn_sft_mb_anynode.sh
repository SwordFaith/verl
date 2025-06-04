# run on 8xH100
# make sure your current working directory is the root of the project

set -x

export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=offline
export WANDB_DIR=/data/tensorboard/
export RUST_BACKTRACE=1
export HYDRA_FULL_ERROR=1
export PIP_INDEX_URL=https://swnexus.thuwayinfo.com/repository/group-pypi/simple
export http_proxy=http://whitelist-proxy.cybertron.svc.cluster.local:7891
export https_proxy=http://whitelist-proxy.cybertron.svc.cluster.local:7891
export no_proxy=127.0.0.1,localhost,code-sandbox.ali-dev.modelbest.co,swnexus.thuwayinfo.com

ulimit -n 65535

PROJECT_DIR="$(pwd)"
RESPONSE_LENGTH=${RESPONSE_LENGTH:-$((1024 * 16))}
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
WANDB_PROJECT=retool_async_rl
WANDB_EXPERIMENT_BASE=qwen2.5-7b_function_rm-retool-async-sgl-sft-bsz128-n8-${RESPONSE_LENGTH}
CKPT_PATH=/user/${USER_ID}/checkpoints/${WANDB_PROJECT}/${WANDB_EXPERIMENT_BASE}
WANDB_EXPERIMENT=${WANDB_EXPERIMENT_BASE}-v$(date +"%y%m%d%H%M")-${WORLD_SIZE}xnode-${JOB_ID}
ACTOR_SP=${ACTOR_SP:-4}
ROLLOUT_TP=${ROLLOUT_TP:-2}
OFFLOAD=True

ulimit -n 65535

ray stop --force

if [ $WORLD_SIZE -gt 1 ]; then
    # Start ray cluster
    MASTER_IP=$(getent hosts $MASTER_ADDR | awk '{print $1}')
    RAY_PORT=$(($MASTER_PORT + 1))

    export RAY_ADDRESS="${MASTER_IP}:${RAY_PORT}"
    if [ $RANK -eq 0 ]; then
        # 启动head节点
        ray start --head \
        --port=$RAY_PORT \
        --num-gpus=8 \
        --num-cpus=80 \
        --include-dashboard=false 
        sleep 10
        # 等待head节点就绪
        echo "Head is ready"
    else
        # worker节点等待head节点就绪
        # 启动worker节点
        ray start --address="${RAY_ADDRESS}" \
        --num-gpus=8 \
        --num-cpus=80 \
        --block
        echo "Worker ${RANK} is connected"
    fi
fi

# Train over 4 nodes, 8 H100-80GB GPUs per node.
if [ $RANK -eq 0 ]; then
    python3 -m verl.trainer.main_ppo \
        --config-path="$CONFIG_PATH" \
        --config-name='retool_multiturn_grpo' \
        algorithm.adv_estimator=grpo \
        data.train_batch_size=128 \
        data.max_prompt_length=2048 \
        data.max_response_length=$RESPONSE_LENGTH \
        data.filter_overlong_prompts=False \
        data.truncation='error' \
        data.return_raw_chat=True \
        data.train_files=/user/longxiang1/data/retool_prompt_dapo/train.parquet \
        data.val_files=/user/longxiang1/data/retool_prompt_aime2024/train.parquet \
        actor_rollout_ref.model.path=/user/longxiang1/checkpoints/retool-multiturn-sft/retool-multiturn-sft-qwen2.5-7b-sp4-mb/global_step_28 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.use_liger=False \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        +actor_rollout_ref.model.enable_activation_offloading=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=128 \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((1024 * 18)) \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD} \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD} \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ACTOR_SP} \
        +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
        actor_rollout_ref.rollout.name=sglang_async \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/sandbox_fusion_retool_config_mb.yaml" \
        actor_rollout_ref.ref.fsdp_config.param_offload=${OFFLOAD} \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name=${WANDB_PROJECT} \
        trainer.experiment_name=${WANDB_EXPERIMENT} \
        trainer.val_before_train=True \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=${WORLD_SIZE} \
        trainer.save_freq=20 \
        trainer.test_freq=20 \
        trainer.total_training_steps=500 \
        trainer.default_local_dir=${CKPT_PATH} \
        trainer.total_epochs=1 $@
fi
