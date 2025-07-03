# run on multi-node H100 cluster
# make sure your current working directory is the root of the project

set -x

# ================= data/model/tool =================
HOME=/veRL/veRL
WORLD_SIZE=${WORLD_SIZE:-4}
JOB_ID=${JOB_ID:-12}

HDFS_ROOT=${HDFS_ROOT:-/veRL/veRL}
DATA_ROOT=${DATA_ROOT:-/veRL/veRL}

dapo_math_17k=$DATA_ROOT/dataset/retool_src_dapo
aime_2024_25=$DATA_ROOT/dataset/retool_src_aime2024_25
model_path=$HDFS_ROOT/model/Qwen2.5-32B-instruct-retool-sft-28step

train_files="['$dapo_math_17k']"
test_files="['$aime_2024_25']"

# tool
tool_config_path=recipe/retool/sandbox_fusion_tool_config.yaml

# wandb
project_name=retool_async_rl
experiment_name=qwen2.5-32b_sft28step_agent_loop_token_out_$(date +"%Y%m%d")-v$(date +"%y%m%d%H%M")-${WORLD_SIZE}xnode-${JOB_ID}
default_local_dir=$DATA_ROOT/checkpoints/$project_name/$experiment_name

# ================= algorithm =================
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_turns=8
max_prompt_length=2048
max_response_length=16384
actor_lr=1e-6

train_batch_size=${TRAIN_BATCH_SIZE:-512}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-16}
n_resp_per_prompt_val=16

# ================= performance =================
infer_tp=${ROLLOUT_TP:-4} # vllm
train_sp=${ACTOR_SP:-8} # train
offload=${OFFLOAD:-True}

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 4 ))

# Construct the command to be executed by ray job
# This ensures that ulimit settings are applied within the job's environment
# and all shell variables are correctly expanded before submission.
CMD="python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files=\"$train_files\" \
    data.val_files=\"$test_files\" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.custom_cls.path=recipe/retool/retool.py \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path=recipe/retool/retool.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.95 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    trainer.logger=[console,wandb] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=True \
    trainer.log_val_generations=100 \
    trainer.nnodes=$WORLD_SIZE \
    trainer.save_freq=30 \
    trainer.default_local_dir=$default_local_dir \
    trainer.test_freq=5 \
    +trainer.wandb_proxy=http://100.68.175.150:3128 \
    trainer.total_epochs=1 \$@"

# Submit the command as a job to the Ray cluster.
# The --working-dir ensures that the current project files are available to the job.
ray job submit --working-dir . \
  --runtime-env-json='{
    "env_vars": {
       "VLLM_USE_V1": "1",
       "PYTHONUNBUFFERED": "1",
       "RAY_DEDUP_LOGS": "0",
       "RUST_BACKTRACE": "1",
       "HYDRA_FULL_ERROR": "1",
       "WANDB_API_KEY": "33001a6e20b2abe823e2d618ecc9c6a0b01fb825"
    }
  }' \
  -- bash -c "$CMD"
