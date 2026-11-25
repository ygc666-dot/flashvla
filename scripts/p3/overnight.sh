#!/usr/bin/env bash
# P0-P7 orchestrator (`3.4_todo.md` "通宵流水线"). Every stage is idempotent:
# `gates/<stage>.done` marks completion, so the same command can be re-run after
# any interruption, and `bash overnight.sh P0 P1` runs only those two stages.
#
#   P0 env self-check (human present)          P4 S1-S4 pipeline smoke
#   P1 download pi05_base (human present)      P5 S5 batch-size sweep
#   P2 T0 pi0* baseline, 9 tasks x 50 seeds    P6 start bridge 4,000 steps
#   P3 S0 single-process FSDP2 route           P7 P3_REPORT.md + P3_STATUS.txt
#
# P1 is the last stage that needs the network: start the script with the proxy you
# have already verified exported (`export HTTPS_PROXY=http://127.0.0.1:7890`).
set -uo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_env.sh"

cd "$FLASHVLA_ROOT" || exit 1
LOG="$P3_LOG"
mkdir -p "$LOG/gates"

stage() { echo "[$(date '+%m-%d %H:%M')] == $*" | tee -a "$LOG/main.log"; }
note()  { echo "[$(date '+%m-%d %H:%M')]    $*" | tee -a "$LOG/main.log"; }
done_m() { [ -f "$P3_GATES/$1.done" ]; }
mark()   { date > "$P3_GATES/$1.done"; }

STAGES=("$@")
if [ "${#STAGES[@]}" -eq 0 ]; then
  STAGES=(P0 P1 P2 P3 P4 P5 P6 P7)
fi

for name in "${STAGES[@]}"; do
  case "$name" in

  P0)
    done_m P0 && { note "P0 already done"; continue; }
    stage "P0 环境自检"
    bash scripts/p3/probe_env.sh > "$LOG/P0_env.log" 2>&1 \
      || { stage "P0 FAILED - 见 P0_env.log（人还在，直接修）"; exit 1; }
    mark P0
    ;;

  P1)
    done_m P1 && { note "P1 already done"; continue; }
    stage "P1 下载 pi05_base（需要代理，下载稳定后再离开）"
    if [ ! -s "$BASE_CKPT/model.safetensors" ]; then
      # HF_HUB_OFFLINE=1 is exported globally for the offline stages; drop it here.
      HF_HUB_OFFLINE=0 "$FV_PY" -c "
from huggingface_hub import snapshot_download
print(snapshot_download('lerobot/pi05_base', local_dir='$BASE_CKPT'))
" > "$LOG/P1_download.log" 2>&1 \
        || { stage "P1 FAILED - 不回退，停在原地（见 P1_download.log）"; exit 1; }
    fi
    for f in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
      [ -s "$BASE_CKPT/$f" ] || { stage "P1 缺少 $f，停"; exit 1; }
    done
    sz=$(du -sm "$BASE_CKPT" | cut -f1)
    [ "$sz" -ge 7000 ] || { stage "P1 只有 ${sz} MB (<7 GB)，停"; exit 1; }
    "$FV_PY" -c "import json;print('type =', json.load(open('$BASE_CKPT/config.json'))['type'])"
    mark P1
    ;;

  P2)
    done_m P2 && { note "P2 already done"; continue; }
    stage "P2 T0 基线（9 任务 x ${T0_N_TRAJS} seeds，幂等跳过已完成任务）"
    bash scripts/p3/run_t0.sh > "$LOG/P2_t0.log" 2>&1
    mark P2
    "$FV_PY" -c "
import json;d=json.load(open('$P3_GATES/P2_t0.json'));print('macro', d['macro'], 'failed', d['failed'])
" >> "$LOG/main.log" 2>&1 || true
    ;;

  P3)
    done_m P3 && { note "P3 already done"; continue; }
    stage "P3 S0：单进程 FSDP2 三条路线（cpu_offload+adamw / adafactor+offload / adafactor 纯卡）"
    bash scripts/p3/smoke_s0.sh           > "$LOG/P3_s0.log" 2>&1 \
      && note "S0 cpu_offload+adamw OK" || note "S0 cpu_offload+adamw 失败"
    bash scripts/p3/smoke_s0_adafactor.sh > "$LOG/P3_s0_adafactor.log" 2>&1 \
      && note "S0 adafactor(+offload) OK" || note "S0 adafactor(+offload) 失败"
    S0_TAG=adafactor_no_offload S0_OPTIMIZER=adafactor S0_CPU_OFFLOAD=false \
      bash scripts/p3/smoke_s0.sh > "$LOG/P3_s0_adafactor_no_offload.log" 2>&1 \
      && note "S0 adafactor 纯卡 OK" || note "S0 adafactor 纯卡 失败"
    mark P3
    ;;

  P4)
    done_m P4 && { note "P4 already done"; continue; }
    stage "P4 S1-S4 冒烟"
    bash scripts/p3/smoke_pipeline.sh > "$LOG/P4_smoke.log" 2>&1
    note "P4 $(tail -1 "$P3_GATES/P4_pipeline.json" 2>/dev/null | tr -d ' ')"
    mark P4
    ;;

  P5)
    done_m P5 && { note "P5 already done"; continue; }
    stage "P5 S5 bs 扫描（bs ∈ ${P3_S5_BS_LIST:-1 2 4})"
    bash scripts/p3/sweep_bs.sh > "$LOG/P5_sweep.log" 2>&1
    mark P5
    ;;

  P6)
    done_m P6 && { note "P6 already done"; continue; }
    stage "P6 起正式训练（gate 未过就不起）"
    if ! grep -q '"trainable": true' "$P3_GATES/decision.json" 2>/dev/null; then
      stage "P6 跳过：decision.trainable != true"
      mark P6
      continue
    fi
    free_mib=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    if [ "${free_mib:-0}" -lt 40960 ]; then
      stage "P6 跳过：GPU$GPU free=${free_mib}MiB < 40 GB（不等待、不轮询）"
      mark P6
      continue
    fi
    bash scripts/p3/render_cfg.sh >> "$LOG/main.log" 2>&1 \
      || { stage "P6 render_cfg 失败"; mark P6; continue; }
    # The launcher follows `decision.fsdp`: with FSDP2 we must fake the accelerate
    # env (see launch_train.sh), without it a plain `python -m flashvla.train` is
    # both simpler and faster on one rank.
    if grep -q '"fsdp": true' "$P3_GATES/decision.json"; then
      LAUNCHER=(bash scripts/p3/launch_train.sh)
    else
      LAUNCHER=("$FV_PY" -m flashvla.train)
    fi
    setsid nohup env CUDA_VISIBLE_DEVICES="$GPU" "${LAUNCHER[@]}" \
      --config_path="$OVERNIGHT_CFG" > "$LOG/P6_train.log" 2>&1 &
    echo $! > "$LOG/P6_train.pid"
    stage "P6 已启动 pid=$(cat "$LOG/P6_train.pid")，配置 $OVERNIGHT_CFG"
    mark P6
    ;;

  P7)
    lastdone=$(ls -1 "$P3_GATES"/*.done 2>/dev/null | wc -l)
    stage "P7 报告（已完成 $lastdone 个阶段）"
    "$FV_PY" scripts/p3/make_report.py --log-dir "$LOG" --out "$FLASHVLA_ROOT/P3_REPORT.md" \
      > "$LOG/P7_report.log" 2>&1 || stage "P7 失败，见 P7_report.log"
    mark P7
    ;;

  *)
    stage "未知阶段 $name（可用：P0 P1 P2 P3 P4 P5 P6 P7）"
    exit 2
    ;;
  esac
done

stage "all requested stages finished"
