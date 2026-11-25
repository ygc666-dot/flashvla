#!/usr/bin/env bash
# P5 / S5: how large can `batch_size` go on one card, and what does it cost?
#
# For bs in {1,2,4}: run 10 optimizer steps with the single-process FSDP2 recipe
# that won S0, sample `nvidia-smi` memory.used on GPU<GPU> every 2 s (that is the
# *device* peak, CUDA context included), and read samples/s out of the trainer log.
#
# Outputs:
#   logs/p3_overnight/S5_sweep.csv        bs / optimizer / peak / samples/s / s_step
#   logs/p3_overnight/gates/S5_bs<N>.json per-run raw numbers
#   logs/p3_overnight/gates/decision.json what `render_cfg.sh` injects
#
# Decision rule (`3.4_todo.md`): the largest bs whose peak fits in
# (free - 2 GB); `grad_accum_steps = 32 // bs` keeps the equivalent batch at 32.
# If nothing fits, bs=1 and `trainable=false` (overnight training is not started).
set -uo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_env.sh"

STEPS="${S5_STEPS:-10}"
BS_LIST="${S5_BS_LIST:-1 2 4}"
S5_DIR="$P3_LOG/s5"
CSV="$P3_LOG/S5_sweep.csv"
mkdir -p "$S5_DIR"

# ---- which single-card recipe survived S0 (highest samples/s among ok runs) ----
# `S5_FORCE_RECIPE="<optimizer> <cpu_offload> <fsdp>"` skips the S0 winner, for
# re-measuring one recipe by hand.
if [ -n "${S5_FORCE_RECIPE:-}" ]; then
  read -r OPT OFFLOAD FSDP_ENABLE <<< "$S5_FORCE_RECIPE"
else
  read -r OPT OFFLOAD FSDP_ENABLE < <("$FV_PY" - "$P3_GATES" <<'PY'
import glob, json, sys
best = None
for path in sorted(glob.glob(sys.argv[1] + "/S0_*.json")):
    try:
        run = json.load(open(path))
    except Exception:
        continue
    if not run.get("ok"):
        continue
    if best is None or run.get("samples_per_s", 0) > best.get("samples_per_s", 0):
        best = run
if best is None:
    print("none none none")
else:
    print(f"{best.get('optimizer') or 'adamw'} {best.get('cpu_offload') or 'true'} "
          f"{best.get('fsdp') or 'true'}")
PY
)
fi

if [ "${OPT}" = "none" ]; then
  echo "no S0 variant succeeded -> single-card training is not viable"
  cat > "$P3_GATES/decision.json" <<JSON
{
  "trainable": false,
  "reason": "S0: neither cpu_offload+adamw nor adafactor finished 20 steps on one card",
  "bs": 1,
  "grad_accum_steps": $EQUIV_BATCH,
  "optimizer": "adamw",
  "cpu_offload": true,
  "throughput": 0.0
}
JSON
  exit 0
fi

free_mib=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
budget_mib=$((free_mib - 2048))
echo "== S5 $(date '+%F %T') gpu=$GPU optimizer=$OPT fsdp=$FSDP_ENABLE cpu_offload=$OFFLOAD free=${free_mib}MiB budget=${budget_mib}MiB =="

if [ "$FSDP_ENABLE" = "true" ]; then
  LAUNCHER=(bash "$FLASHVLA_ROOT/scripts/p3/launch_train.sh")
else
  LAUNCHER=("$FV_PY" -m flashvla.train)
fi

echo "bs,optimizer,cpu_offload,peak_mib,device_peak_mib,samples_per_s,s_per_step,loss,ok" > "$CSV"

for bs in $BS_LIST; do
  OUT="$S5_DIR/bs$bs"
  LOG="$P3_LOG/P5_bs$bs.log"
  # train.py refuses a pre-existing output_dir, so only remove it.
  rm -rf "$OUT"
  echo "-- bs=$bs $(date '+%T')"

  CUDA_VISIBLE_DEVICES="$GPU" "${LAUNCHER[@]}" \
    --config_path="$SMOKE_CFG" \
    --fsdp.enable="$FSDP_ENABLE" --fsdp.cpu_offload="$OFFLOAD" --fsdp.mixed_precision=bf16 \
    --optimizer.type="$OPT" \
    --batch_size="$bs" --grad_accum_steps=1 --steps="$STEPS" \
    --save_checkpoint=false --num_workers=0 --log_freq="$STEPS" \
    --wandb.enable=false --output_dir="$OUT" \
    > "$LOG" 2>&1 &
  pid=$!

  # Peak memory is attributed to THIS process tree, not to the whole device:
  # GPU4 is shared with other people's jobs tonight, and a device-wide reading
  # would silently push the batch-size decision down (or make every bs look
  # identical). `launch_train.sh` execs python, so $pid is the trainer itself.
  peak=0
  device_peak=0
  while kill -0 "$pid" 2>/dev/null; do
    mine=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
           | awk -F', *' -v p="$pid" '$1 == p {sum += $2} END {print sum + 0}')
    used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    case "$mine" in ''|*[!0-9]*) mine=0 ;; esac
    case "$used" in ''|*[!0-9]*) used=0 ;; esac
    [ "$mine" -gt "$peak" ] && peak=$mine
    [ "$used" -gt "$device_peak" ] && device_peak=$used
    sleep 1
  done
  wait "$pid"; rc=$?
  echo "   peak(this run)=${peak}MiB device_peak=${device_peak}MiB"

  "$FV_PY" "$FLASHVLA_ROOT/scripts/p3/parse_train_log.py" "$LOG" \
    --bs "$bs" --accum 1 --tag "s5_bs$bs" --rc "$rc" \
    --optimizer "$OPT" --cpu-offload "$OFFLOAD" --fsdp "$FSDP_ENABLE" \
    > "$P3_GATES/S5_bs$bs.json"

  "$FV_PY" - "$P3_GATES/S5_bs$bs.json" "$peak" "$device_peak" "$CSV" <<'PY'
import json, sys

run, peak, device_peak, csv_path = (json.load(open(sys.argv[1])), int(sys.argv[2]),
                                    int(sys.argv[3]), sys.argv[4])
row = [str(run["batch_size"]), run.get("optimizer", ""), str(run.get("cpu_offload", "")),
       str(peak), str(device_peak), f"{run.get('samples_per_s', 0):.4f}",
       f"{run.get('s_per_step', 0):.4f}", str(run.get("loss")), str(bool(run["ok"]))]
with open(csv_path, "a") as f:
    f.write(",".join(row) + "\n")
print("   row:", ",".join(row))
PY

  if [ "$rc" -ne 0 ]; then
    echo "   bs=$bs OOM/failure (peak=${peak}MiB), tail:"
    grep -iE "out of memory|error|Traceback" "$LOG" | tail -3
  fi
done

# ---- decision ----
"$FV_PY" - "$P3_GATES" "$CSV" "$budget_mib" "$EQUIV_BATCH" > "$P3_GATES/decision.json" <<'PY'
import csv, glob, json, sys

gates, csv_path, budget, equiv = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

runs = []
for path in glob.glob(f"{gates}/S5_bs*.json"):
    run = json.load(open(path))
    rows = [r for r in csv.DictReader(open(csv_path))
            if r["bs"] == str(run["batch_size"])]
    peak = int(rows[-1]["peak_mib"]) if rows else 0
    run["peak_mib"] = peak
    runs.append(run)

fits = [r for r in runs if r["ok"] and r["peak_mib"] <= budget]
best = max(fits, key=lambda r: r["batch_size"]) if fits else None

decision = {
    "trainable": best is not None,
    "bs": int(best["batch_size"]) if best else 1,
    "grad_accum_steps": equiv // int(best["batch_size"]) if best else equiv,
    "optimizer": (best or runs[0]).get("optimizer", "adamw") if runs else "adamw",
    "cpu_offload": (best or runs[0]).get("cpu_offload", "true") if runs else "true",
    "fsdp": (best or runs[0]).get("fsdp", "true") if runs else "true",
    "throughput": round(best.get("samples_per_s", 0.0), 4) if best else 0.0,
    "peak_mib": best["peak_mib"] if best else 0,
    "budget_mib": budget,
    "equiv_batch": equiv,
    "runs": [{k: r.get(k) for k in ("batch_size", "ok", "peak_mib", "samples_per_s", "s_per_step", "loss")}
             for r in sorted(runs, key=lambda r: r["batch_size"])],
}
if best is None:
    decision["reason"] = "no batch_size fitted the memory budget (free - 2 GB)"
elif best.get("samples_per_s", 0) < 0.5:
    decision["trainable"] = False
    decision["reason"] = f"best throughput {best['samples_per_s']:.3f} samples/s < 0.5"

print(json.dumps(decision, indent=2))
PY

echo "-- decision.json --"
cat "$P3_GATES/decision.json"
