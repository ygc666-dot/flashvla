#!/usr/bin/env bash
# Single-process launcher for a training run that has `fsdp.enable: true`.
#
# `flashvla/train.py` asserts on `torch.distributed.get_backend_config()`, so the
# process group must exist even with one rank. Two launchers do *not* provide it:
#   * `train/train.sh cfg 1`  -> plain python, no init_process_group at all
#   * `accelerate launch --num_processes=1` -> accelerate's simple_launcher starts
#     a bare subprocess and never initializes a group
# "Default process group has not been initialized" is exactly that failure (S0).
#
# Faking the launcher's environment is enough: accelerate then treats the run as
# distributed, initializes the process group itself with the backend coming from
# train.py's `InitProcessGroupKwargs` ("cuda:nccl,cpu:gloo"), which is what the
# FSDP2 checkpoint path demands. Verified on GPU4 with one rank (S0).
#
# Usage:  bash scripts/p3/launch_train.sh --config_path=<yaml> [--override ...]
set -uo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"
export ACCELERATE_USE_FSDP=true
export FSDP_STATE_DICT_TYPE=SHARDED_STATE_DICT
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29531}"

cd "$FLASHVLA_ROOT" || exit 1
exec "$FV_PY" -m flashvla.train "$@"
