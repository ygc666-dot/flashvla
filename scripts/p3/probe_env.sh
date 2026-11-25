#!/usr/bin/env bash
# P0 environment self-check (`3.4_todo.md` P0). A human is present: any failure
# must stop the pipeline here, while the failure is still cheap to fix.
#
# Checks (in order):
#   1. GPU<GPU> free memory >= 40 GB
#   2. >= 20 GB free on the repo filesystem
#   3. the 18 LeRobot leaves of the P2 conversion are readable
#   4. third_party/SDN/pretrained/open-pi-zero has the two pi0* checkpoints (T0)
#   5. one `curl -I https://huggingface.co` with whatever proxy is already exported
#      -- recorded, never fatal (P1 is watched by a human anyway)
#
# Exit code: 0 only if 1-4 pass.
set -uo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_env.sh"

rc=0
say() { printf '%-34s %s\n' "$1" "$2"; }

echo "== P0 $(date '+%F %T') on $(hostname) =="
echo "root=$FLASHVLA_ROOT gpu=$GPU log=$P3_LOG"

# 1) GPU
free_mib=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
if [ -n "${free_mib:-}" ] && [ "$free_mib" -ge 40960 ]; then
  say "GPU$GPU free" "OK   ${free_mib} MiB"
else
  say "GPU$GPU free" "FAIL ${free_mib:-?} MiB (need >= 40960)"; rc=1
fi

# 2) disk
avail_gb=$(df -Pk "$FLASHVLA_ROOT" | awk 'NR==2{printf "%d", $4/1048576}')
if [ "${avail_gb:-0}" -ge 20 ]; then
  say "disk free" "OK   ${avail_gb} GB"
else
  say "disk free" "FAIL ${avail_gb:-?} GB (need >= 20)"; rc=1
fi

# 3) the 18 LeRobot leaves
leaves=0
for sub in success failure; do
  for d in "$DATA_ROOT"/fractal/*/"$sub" "$DATA_ROOT"/bridge/*/"$sub"; do
    [ -f "$d/meta/info.json" ] && leaves=$((leaves + 1))
  done
done
if [ "$leaves" -eq 18 ]; then
  say "dataset leaves" "OK   $leaves (fractal 10 + bridge 8)"
else
  say "dataset leaves" "FAIL $leaves != 18 under $DATA_ROOT"; rc=1
fi

# 4) T0 weights
n_pt=$(ls -1 "$FLASHVLA_ROOT"/third_party/SDN/pretrained/open-pi-zero/*.pt 2>/dev/null | wc -l)
if [ "$n_pt" -ge 2 ]; then
  say "open-pi-zero" "OK   $n_pt checkpoints"
else
  say "open-pi-zero" "FAIL $n_pt < 2 checkpoints"; rc=1
fi

# 5) network: informational only
echo "-- proxy in this environment: HTTPS_PROXY=${HTTPS_PROXY:-<none>} HTTP_PROXY=${HTTP_PROXY:-<none>}"
curl -sS -I --max-time 10 https://huggingface.co 2>&1 | head -3 || true

echo "== P0 $([ "$rc" -eq 0 ] && echo PASS || echo FAIL) =="
exit "$rc"
