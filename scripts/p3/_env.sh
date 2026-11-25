#!/usr/bin/env bash
# Shared constants for the P3 scripts (`docs/sdn_flashvla/flashvla_simpler/3.4_todo.md`).
#
# Sourced, not executed. Everything is overridable from the environment so the same
# scripts can be re-run on GPU2 / a different data root without editing files.

FLASHVLA_ROOT="${FLASHVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

P3_LOG="${P3_LOG:-$FLASHVLA_ROOT/logs/p3_overnight}"
P3_GATES="$P3_LOG/gates"
GPU="${P3_GPU:-4}"
DATA_ROOT="${P3_DATA_ROOT:-/data/zjr/simpler_lerobot}"

# The conda envs were moved here from /home/zjr/StreamingVLA, so the `bin/*`
# entry points of the envs still carry a stale shebang. Always go through `python`.
ENVS="${P3_ENVS:-/home/zjr/workspace_tj/StreamingVLA/miniconda_ygc/envs}"
FV_PY="${FV_PY:-$ENVS/flashvla/bin/python}"
SDN_PY="${SDN_PY:-$ENVS/sdn/bin/python}"

SMOKE_CFG="$FLASHVLA_ROOT/train/configs/pi05/simpler/_smoke.yaml"
OVERNIGHT_CFG="$FLASHVLA_ROOT/train/configs/pi05/simpler/_overnight.generated.yaml"
BASE_CKPT="$FLASHVLA_ROOT/ckpt/pi05_base"

# T0 output root. `evaluate.py` appends `/vanilla[/seeds=A-B]/<task>`.
T0_ROOT="${P3_T0_ROOT:-$FLASHVLA_ROOT/third_party/SDN/eval_logs/p3_baseline}"
T0_TASKS=(
  google_robot_close_drawer
  google_robot_move_near
  google_robot_open_drawer
  google_robot_pick_coke_can
  google_robot_place_apple_in_closed_top_drawer
  widowx_carrot_on_plate
  widowx_put_eggplant_in_basket
  widowx_spoon_on_towel
  widowx_stack_cube
)
T0_N_TRAJS="${P3_T0_N_TRAJS:-50}"
T0_START_SEED="${P3_T0_START_SEED:-2000}"
T0_TASK_TIMEOUT="${P3_T0_TASK_TIMEOUT:-2400}"   # 40 min per attempt, kills hangs

# Effective batch target for every run (`3.4_todo.md` decision 5).
EQUIV_BATCH="${P3_EQUIV_BATCH:-32}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$P3_GATES"
