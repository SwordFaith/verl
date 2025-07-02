# run on 8xH100
# make sure your current working directory is the root of the project

set -x


HOME=/veRL/veRL
WORLD_SIZE=4
JOB_ID=13

train_prompt_bsz=${TRAIN_BATCH_SIZE:-512}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-16}
train_prompt_mini_bsz=$((train_prompt_bsz / 1))

PROJECT_DIR="$(pwd)"
RESPONSE_LENGTH=${RESPONSE_LENGTH:-$((1024 * 16))}
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-30}
CONFIG_PATH="$PROJECT_DIR/recipe/retool/config"
WANDB_PROJECT=retool_async_rl
WANDB_EXPERIMENT_BASE=qwen2.5-32b_function_rm-retool-async-sgl-no-mask-turn-level-clip-higher-sft_28step-grpo-ve-bsz${train_prompt_bsz}-n${n_resp_per_prompt}-${RESPONSE_LENGTH}-${MAX_ASSISTANT_TURNS}turns
CKPT_PATH=$HOME/checkpoints/${WANDB_PROJECT}/${WANDB_EXPERIMENT_BASE}
WANDB_EXPERIMENT=${WANDB_EXPERIMENT_BASE}-v$(date +"%y%m%d%H%M")-${WORLD_SIZE}xnode-${JOB_ID}
ACTOR_SP=${ACTOR_SP:-8}
ROLLOUT_TP=${ROLLOUT_TP:-4}
OFFLOAD=True

# Train over 4 nodes, 8 H100-80GB GPUs per node.
# Construct the command to be executed by ray job
# This ensures that ulimit settings are applied within the job's environment
# and all shell variables are correctly expanded before submission.
CMD="python3 -m verl.trainer.main_ppo \
--config-path=$CONFIG_PATH \
--config-name='retool_multiturn_grpo' \
algorithm.adv_estimator=grpo \
data.train_batch_size=${train_prompt_bsz} \
data.max_prompt_length=2048 \
data.max_response_length=$RESPONSE_LENGTH \
data.filter_overlong_prompts=False \
data.truncation=error \
data.return_raw_chat=True \
data.train_files=$HOME/dataset/retool_src_dapo/train.parquet \
data.val_files=$HOME/dataset/retool_src_aime2024_25/train.parquet \
actor_rollout_ref.model.path=$HOME/model/Qwen2.5-32B-instruct-retool-sft-28step \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.model.use_liger=False \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
+actor_rollout_ref.model.enable_activation_offloading=True \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.actor.optim.weight_decay=0.01 \
actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((1024 * 18)) \
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=0.0 \
actor_rollout_ref.actor.kl_loss_type=low_var_kl \
actor_rollout_ref.actor.entropy_coeff=0 \
actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD} \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD} \
actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ACTOR_SP} \
actor_rollout_ref.actor.clip_ratio_high=0.28 \
actor_rollout_ref.actor.strategy=fsdp2 \
+actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
actor_rollout_ref.rollout.name=sglang \
actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_ASSISTANT_TURNS} \
actor_rollout_ref.rollout.multi_turn.tool_config_path=$PROJECT_DIR/recipe/retool/config/tool_config/sandbox_fusion_retool_config_vefaas.yaml \
actor_rollout_ref.ref.fsdp_config.param_offload=${OFFLOAD} \
actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
actor_rollout_ref.rollout.val_kwargs.temperature=0.95 \
actor_rollout_ref.rollout.val_kwargs.n=16 \
algorithm.use_kl_in_reward=False \
reward_model.reward_manager=multi_turn \
trainer.critic_warmup=0 \
trainer.logger=[console,wandb] \
trainer.project_name=${WANDB_PROJECT} \
trainer.experiment_name=${WANDB_EXPERIMENT} \
trainer.val_before_train=True \
trainer.n_gpus_per_node=8 \
trainer.nnodes=${WORLD_SIZE} \
trainer.save_freq=20 \
trainer.test_freq=5 \
trainer.total_training_steps=500 \
trainer.default_local_dir=${CKPT_PATH} \
+trainer.wandb_proxy=http://100.68.175.150:3128 \
trainer.log_val_generations=0 \
trainer.total_epochs=1 $@"

# Submit the command as a job to the Ray cluster.
# The --working-dir ensures that the current project files are available to the job.
ray job submit --working-dir . \
  --runtime-env-json='{
    "env_vars": {
       "PYTHONUNBUFFERED": "1",
       "RAY_DEDUP_LOGS": "0",
       "RUST_BACKTRACE": "1",
       "HYDRA_FULL_ERROR": "1",
       "WANDB_API_KEY": "33001a6e20b2abe823e2d618ecc9c6a0b01fb825"
    }
  }' \
  -- bash -c "$CMD"

