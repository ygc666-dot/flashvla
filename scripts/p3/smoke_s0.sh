#!/usr/bin/env bash
# S0 gate: which single-card recipe can actually train this 4.1 B policy?
#
# `3.4_todo.md` S0 asks one question ("can FSDP2 run with one process?"), but the
# measurements widened it: FSDP2 in bf16 mode keeps fp32 master weights (+16.6 GB)
# and fp32 gradients (+13.4 GB) on the card, so the memory ceiling is NOT the
# optimizer state. That makes a *no-FSDP + adafactor* recipe worth measuring on
# the same footing: nothing to shard on one rank, and adafactor removes the 30 GB
# AdamW state that originally forced FSDP.
#
# Variants (same body, different defaults):
#   smoke_s0.sh                                   fsdp2 + cpu_offload + adamw
#   smoke_s0_adafactor.sh                         fsdp2 + cpu_offload + adafactor
#   S0_TAG=x S0_OPTIMIZER=adafactor S0_CPU_OFFLOAD=false bash smoke_s0.sh
#                                                 fsdp2, optimizer state on the card
#   S0_TAG=y S0_OPTIMIZER=adafactor S0_FSDP=false bash smoke_s0.sh
#                                                 plain single process, adafactor
#
# Pass criterion: exit code 0 and a finite loss. Throughput and peak memory land in
# $P3_GATES/S0_<tag>.json; `sweep_bs.sh` then sweeps batch_size for the winner.
#
# Env overrides: S0_TAG, S0_OPTIMIZER, S0_BS, S0_STEPS, S0_CPU_OFFLOAD, S0_FSDP
set -uo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_env.sh"

TAG="${S0_TAG:-cpu_offload_adamw}"
OPT="${S0_OPTIMIZER:-adamw}"
BS="${S0_BS:-2}"
STEPS="${S0_STEPS:-20}"
OFFLOAD="${S0_CPU_OFFLOAD:-true}"
FSDP_ENABLE="${S0_FSDP:-true}"

OUT="$P3_LOG/s0/$TAG"
LOG="$P3_LOG/P3_s0_$TAG.log"
# Do NOT pre-create `$OUT`: train.py refuses an existing output_dir that holds no
# resumable checkpoint, and we always want a fresh measurement run here.
rm -rf "$OUT"

echo "== S0[$TAG] $(date '+%F %T') gpu=$GPU bs=$BS steps=$STEPS optimizer=$OPT fsdp=$FSDP_ENABLE cpu_offload=$OFFLOAD =="

if [ "$FSDP_ENABLE" = "true" ]; then
  # See launch_train.sh for why this is not `accelerate launch --num_processes=1`.
  LAUNCHER=(bash "$FLASHVLA_ROOT/scripts/p3/launch_train.sh")
else
  LAUNCHER=("$FV_PY" -m flashvla.train)
fi

CUDA_VISIBLE_DEVICES="$GPU" "${LAUNCHER[@]}" \
  --config_path="$SMOKE_CFG" \
  --fsdp.enable="$FSDP_ENABLE" --fsdp.cpu_offload="$OFFLOAD" --fsdp.mixed_precision=bf16 \
  --optimizer.type="$OPT" \
  --batch_size="$BS" --grad_accum_steps=1 --steps="$STEPS" \
  --save_checkpoint=false --num_workers=0 --log_freq="$STEPS" \
  --wandb.enable=false --output_dir="$OUT" \
  > "$LOG" 2>&1
rc=$?

"$FV_PY" "$FLASHVLA_ROOT/scripts/p3/parse_train_log.py" "$LOG" \
  --bs "$BS" --accum 1 --tag "$TAG" --rc "$rc" \
  --optimizer "$OPT" --cpu-offload "$OFFLOAD" --fsdp "$FSDP_ENABLE" \
  > "$P3_GATES/S0_$TAG.json"

echo "-- S0[$TAG] rc=$rc --"
tail -3 "$P3_GATES/S0_$TAG.json"
[ "$rc" -eq 0 ] || { echo "S0[$TAG] FAILED, tail of $LOG:"; tail -20 "$LOG"; }
exit "$rc"
