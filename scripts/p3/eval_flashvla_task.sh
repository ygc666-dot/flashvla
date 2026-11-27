#!/usr/bin/env bash
# Evaluate ONE `pi0-flashvla` checkpoint on ONE SIMPLER task (F7 / F1).
#
# Starts the policy server (py3.12 `flashvla` env) on port `12400+GPU`, waits for
# the socket, then runs `evaluate.py --method flashvla` (py3.10 `sdn` env) for
# `N` episodes starting at `SEED`.  This is the exact script that produced
# `eval_logs/f7_f1*` on 2026-11-25 (≈22 min for 50 close_drawer episodes).
#
# Usage:
#   scripts/p3/eval_flashvla_task.sh \
#     --gpu 4 \
#     --ckpt  /abs/path/runs/f6_s5_3_fractal8000/checkpoints/008000/pretrained_model \
#     --out   /abs/path/third_party/SDN/eval_logs/f7_f1 \
#     [--task google_robot_close_drawer] [--n 50] [--seed 2000] [--tag run]
#
# Notes (all learned the hard way, see docs/.../f7_next_steps.md §8):
#  * `--ckpt` and `--out` must be absolute; the script cd-s into third_party/SDN.
#  * `--ckpt` is verified by the client against the served checkpoint, so a stale
#    server on the same port fails loudly instead of silently scoring the wrong model.
#  * server load is ≈130 s; the readiness probe is a real `listen()`, not a sleep.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
ROOT=$(pwd)
ENVS=${F7_ENVS:-/home/zjr/workspace_tj/StreamingVLA/miniconda_ygc/envs}
LOGDIR=${F7_LOGDIR:-$ROOT/logs}

GPU=4
CKPT=""
OUT=""
TASK=google_robot_close_drawer
N=50
SEED=2000
TAG=run

while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --ckpt) CKPT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done
[ -n "$CKPT" ] && [ -n "$OUT" ] || { echo "usage: $0 --gpu N --ckpt ABS --out ABS [--task T] [--n N] [--seed S] [--tag TAG]"; exit 2; }
case "$CKPT" in /*) ;; *) echo "--ckpt must be absolute"; exit 2 ;; esac
case "$OUT" in /*) ;; *) echo "--out must be absolute"; exit 2 ;; esac

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH BASH_ENV

PORT=$((12400 + GPU))
mkdir -p "$OUT" "$LOGDIR"
SL="$LOGDIR/f7_srv_${TAG}_gpu${GPU}.log"
EL="$LOGDIR/f7_${TASK}_${TAG}_gpu${GPU}.log"

echo "[eval] $(date) ckpt=$CKPT task=$TASK n=$N seed=$SEED gpu=$GPU port=$PORT" | tee "$EL"
CUDA_VISIBLE_DEVICES="$GPU" "$ENVS/flashvla/bin/python" scripts/p3/flashvla_server.py \
  --ckpt "$CKPT" --port "$PORT" >> "$SL" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT INT TERM

deadline=$((SECONDS + 600))
until (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; do
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "[eval][fail] server never listened on $PORT; see $SL" | tee -a "$EL"; exit 1
  fi
  sleep 5
done
echo "[eval] server ready after ${SECONDS}s" | tee -a "$EL"

cd "$ROOT/third_party/SDN" || exit 1
CUDA_VISIBLE_DEVICES="$GPU" "$ENVS/sdn/bin/python" -m policy.pizero.evaluate \
  --method flashvla --task "$TASK" --checkpoint "$CKPT" \
  --num-gpus 1 --n-trajs "$N" --start-seed "$SEED" \
  --reload-every 0 --no-video --result-root "$OUT" >> "$EL" 2>&1
rc=$?
echo "[eval] rc=$rc $(date)" | tee -a "$EL"
find "$OUT" -name "000_success_*.log" -newermt '-6 hours' -print | sed 's/^/[eval] result: /' | tee -a "$EL"
exit $rc
