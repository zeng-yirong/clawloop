#!/usr/bin/env bash
set -euo pipefail

# Portable launcher. Hardware, model, and data settings belong to the VERL
# command line; the historical cluster-specific script is kept separately as
# train_half_turn.cluster.sh for provenance.
: "${VERL_ROOT:?Set VERL_ROOT to a compatible VERL checkout}"
: "${BASE_TASKS:?Set BASE_TASKS to a directory restored from tasks.jsonl}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RELEASE_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "$VERL_ROOT"
export PYTHONPATH="${RELEASE_ROOT}/recipe:${PYTHONPATH:-}"
TOOL_CONFIG=${NANOCLAW_TOOL_CONFIG:-${SCRIPT_DIR}/nanoclaw_tool_config.yaml}

# Defaults mirror the validated 9B/27B experiment profiles. Every value can
# be overridden through an environment variable or an extra Hydra argument.
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to a Qwen3.5 checkpoint}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-22768}
MAX_ASSISTANT_RESPONSE_LENGTH=${MAX_ASSISTANT_RESPONSE_LENGTH:-16384}
MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-8192}
MAX_TURNS=${MAX_TURNS:-35}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-8}
ACTOR_LR=${ACTOR_LR:-1e-6}
NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
PROJECT_NAME=${PROJECT_NAME:-nanoclaw}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen35_nanoclaw_grpo}
LOCAL_DIR=${LOCAL_DIR:-${VERL_ROOT}/checkpoints/${EXPERIMENT_NAME}}
export NANOCLAW_MASK_LOOPING_RESPONSES=${NANOCLAW_MASK_LOOPING_RESPONSES:-True}
export NANOCLAW_MASK_ONLY_POSITIVE_ADVANTAGE=${NANOCLAW_MASK_ONLY_POSITIVE_ADVANTAGE:-True}
export NANOCLAW_MASK_BUDGET_EXHAUSTED_LAST_TURN=${NANOCLAW_MASK_BUDGET_EXHAUSTED_LAST_TURN:-True}
export NANOCLAW_MASK_DUPLICATE_TOOL_RESULT_TURNS=${NANOCLAW_MASK_DUPLICATE_TOOL_RESULT_TURNS:-True}
export NANOCLAW_MASK_ERROR_TOOL_RESULT_TURNS=${NANOCLAW_MASK_ERROR_TOOL_RESULT_TURNS:-True}

exec python3 -m verl.trainer.main_ppo \
  "data.train_files=['${BASE_TASKS}']" \
  "data.val_files=['${VAL_TASKS:-${BASE_TASKS}}']" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.prompt_key=prompt \
  data.return_raw_chat=True \
  data.custom_cls.path=pkg://nanoclaw_recipe.nanoclaw \
  data.custom_cls.name=CustomRLHFDataset \
  data.tool_config_path="${TOOL_CONFIG}" \
  +data.nanoclaw_task_glob="${NANOCLAW_TASK_GLOB:-data_*}" \
  +data.nanoclaw_temp_root="${NANOCLAW_TEMP_ROOT:-/tmp/verl_nanoclaw_workspaces}" \
  +data.nanoclaw_cleanup_workspaces="${NANOCLAW_CLEANUP_WORKSPACES:-True}" \
  +data.nanoclaw_keep_failed_workspaces="${NANOCLAW_KEEP_FAILED_WORKSPACES:-False}" \
  +data.nanoclaw_env_builder_timeout="${NANOCLAW_ENV_BUILDER_TIMEOUT:-120}" \
  +data.nanoclaw_verifier_timeout="${NANOCLAW_VERIFIER_TIMEOUT:-300}" \
  +data.nanoclaw_reward_score_mode="${NANOCLAW_REWARD_SCORE_MODE:-ratio}" \
  +data.nanoclaw_allow_bash="${NANOCLAW_ALLOW_BASH:-True}" \
  reward.custom_reward_function.path=pkg://nanoclaw_recipe.nanoclaw \
  reward.custom_reward_function.name=compute_score \
  "+reward.custom_reward_function.reward_kwargs.cleanup_workspaces=${NANOCLAW_CLEANUP_WORKSPACES:-True}" \
  "+reward.custom_reward_function.reward_kwargs.keep_failed_workspaces=${NANOCLAW_KEEP_FAILED_WORKSPACES:-False}" \
  "+reward.custom_reward_function.reward_kwargs.verifier_timeout=${NANOCLAW_VERIFIER_TIMEOUT:-300}" \
  "+reward.custom_reward_function.reward_kwargs.reward_score_mode=${NANOCLAW_REWARD_SCORE_MODE:-ratio}" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=True \
  algorithm.kl_penalty=low_var_kl \
  algorithm.kl_ctrl.type=fixed \
  algorithm.kl_ctrl.kl_coef=0.001 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
  actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${INFER_TP:-1}" \
  actor_rollout_ref.rollout.max_model_len="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_TURNS}" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_TURNS}" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_response_length="${MAX_ASSISTANT_RESPONSE_LENGTH}" \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length="${MAX_TOOL_RESPONSE_LENGTH}" \
  actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
  actor_rollout_ref.rollout.n="${N_RESP_PER_PROMPT}" \
  trainer.nnodes="${NNODES}" \
  trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${LOCAL_DIR}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  "$@"
