#!/usr/bin/env bash
set -euo pipefail

# Portable Qwen3.5-27B profile distilled from the original 27b.sh experiment.
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-27B}"
export PROJECT_NAME="${PROJECT_NAME:-qwen3.5-27b_nanoclaw}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3.5-27b_nanoclaw_grpo_aam}"
export NNODES="${NNODES:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export INFER_TP="${INFER_TP:-4}"
exec "$(dirname "${BASH_SOURCE[0]}")/train_half_turn.sh" "$@"

