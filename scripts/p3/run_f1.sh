#!/usr/bin/env bash
# F2 gate driver: 5 `google_robot_*` tasks x 50 seeds (2000-2049) for the F7/P0
# checkpoint (`MEAN_STD` + 三路 + 8000 步).  `f7_next_steps.md` §4.1 / §4.2.
#
# Difference vs `run_g3.sh`: the inner unit is `eval_flashvla_task.sh` (one
# self-contained server+client per task) instead of a shared server + raw
# `evaluate.py` loop.  Costs ~2.5 min extra per task (server reload) but keeps a
# single code path with the already-validated F1 runs.
#
#   scripts/p3/run_f1.sh full     # 5 tasks x 50 seeds x 1 ckpt  (~2 h on 1 GPU)
#   scripts/p3/run_f1.sh smoke    # 1 task x 10 seeds (2100-2109)  (~8 min)
#   scripts/p3/run_f1.sh report   # aggregate only, no eval
#
# Env: P3_F1_GPUS (default "2"), P3_F1_CKPTS (default "008000"),
#      P3_F1_N_TRAJS / P3_F1_START_SEED / P3_F1_TIMEOUT / P3_F1_LOGDIR
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
ROOT=$(pwd)

LOGDIR=${P3_F1_LOGDIR:-$ROOT/logs/05_f7_deploy/f2_gate}
OUT_ROOT=${P3_F1_OUT_ROOT:-$ROOT/third_party/SDN/eval_logs/f7_f2}
CKPT_ROOT=${P3_F1_CKPT_ROOT:-$ROOT/runs/f7_s1_fractal8000_meanstd/checkpoints}
GPUS=${P3_F1_GPUS:-2}
CKPTS_STR=${P3_F1_CKPTS:-008000}
TIMEOUT_S=${P3_F1_TIMEOUT:-5400}

# Order is fixed: it must match T0_REFS line-for-line (§4.1 warning).
F1_TASKS=(
  google_robot_pick_coke_can
  google_robot_move_near
  google_robot_close_drawer
  google_robot_open_drawer
  google_robot_place_apple_in_closed_top_drawer
)
# T0 = in-process pi0*, seeds 2000-2049 (gates/P2_t0.json). Macro 0.564 -> gate 0.544.
T0_REFS="0.86 0.74 0.68 0.36 0.18"

MODE="${1:-full}"
case "$MODE" in
  smoke)
    TASKS=(google_robot_pick_coke_can)
    N=10; SEED=2100 ;;
  full)
    TASKS=("${F1_TASKS[@]}")
    N=${P3_F1_N_TRAJS:-50}; SEED=${P3_F1_START_SEED:-2000} ;;
  report)
    TASKS=("${F1_TASKS[@]}")
    N=${P3_F1_N_TRAJS:-50}; SEED=${P3_F1_START_SEED:-2000} ;;
  *) echo "usage: $0 [smoke|full|report]"; exit 2 ;;
esac
read -r -a CKPTS <<< "$CKPTS_STR"
mkdir -p "$LOGDIR" "$OUT_ROOT"

LOCK=${P3_F1_LOCK:-$ROOT/logs/05_f7_deploy/f2_gate/.run_f1.lock}
exec 9>"$LOCK"
if ! flock -n 9; then echo "[fail] another run_f1.sh holds $LOCK"; exit 3; fi

echo "== F1 $MODE $(date '+%F %T')"
echo "   ckpts=${CKPTS[*]} tasks=${#TASKS[@]} seeds=$SEED..$((SEED + N - 1)) gpus=$GPUS"
echo "   out=$OUT_ROOT logs=$LOGDIR"

if [ "$MODE" != "report" ]; then
  for step in "${CKPTS[@]}"; do
    CK="$CKPT_ROOT/$step/pretrained_model"
    if [ ! -f "$CK/model.safetensors" ]; then
      echo "[fail] missing checkpoint $CK"; continue
    fi
    for task in "${TASKS[@]}"; do
      if compgen -G "$OUT_ROOT/$step/flashvla/*/$task/000_success_*.log" > /dev/null; then
        echo "[skip] $step/$task  <- $(ls -1 "$OUT_ROOT/$step"/flashvla/*/$task/000_success_*.log | head -1 | xargs basename)"
        continue
      fi
      ok=0
      for attempt in 1 2; do
        echo "[run ] $step/$task attempt $attempt $(date '+%T')"
        for gpu in $GPUS; do
          F7_LOGDIR="$LOGDIR" timeout "$TIMEOUT_S" \
            scripts/p3/eval_flashvla_task.sh \
              --gpu "$gpu" --ckpt "$CK" --out "$OUT_ROOT/$step" \
              --task "$task" --n "$N" --seed "$SEED" --tag "f1_${step}" \
              > "$LOGDIR/${step}_${task}.attempt${attempt}.log" 2>&1
          rc=$?
          [ "$rc" -eq 0 ] && break
          echo "[warn] $step/$task gpu=$gpu attempt $attempt rc=$rc"
        done
        [ "$rc" -eq 0 ] && { ok=1; break; }
      done
      [ "$ok" -eq 1 ] || echo "[fail] $step/$task -> kept as missing in the aggregate"
    done
  done
fi

echo "== aggregating =="
# System python3 here is 3.4 (no f-strings); use the flashvla env interpreter
# exactly like `run_g3.sh` does.
PY=${F7_PY:-${F7_ENVS:-/home/zjr/workspace_tj/StreamingVLA/miniconda_ygc/envs}/flashvla/bin/python}
"$PY" - "$OUT_ROOT" "$LOGDIR" "$MODE" "$SEED" "$((SEED + N - 1))" "$N" "$T0_REFS" "${CKPTS[@]}" -- "${TASKS[@]}" <<'PY'
import glob, json, math, os, re, sys

argv = sys.argv[1:]
root, logdir, mode, start, last, n, refs_s = argv[:7]
ckpts = argv[7:argv.index('--')]
tasks = argv[argv.index('--') + 1:]
refs = [float(v) for v in refs_s.split()]
t0 = dict(zip(tasks, refs))
t0_macro = sum(refs) / len(refs)

rep = {"mode": mode, "seeds": f"{start}-{last}", "n_per_task": int(n),
       "protocol": "flashvla / 5 google_robot tasks / seeds 2000-2049",
       "t0_refs": t0, "t0_macro": round(t0_macro, 4),
       "gate": round(t0_macro - 0.02, 4), "ckpts": {}}

for ck in ckpts:
    rates, missing = {}, []
    for t in tasks:
        hits = (sorted(glob.glob(f"{root}/{ck}/flashvla/*/{t}/000_success_*.log"))
                or sorted(glob.glob(f"{root}/{ck}/flashvla/{t}/000_success_*.log")))
        if not hits:
            missing.append(t); rates[t] = None; continue
        rates[t] = float(re.search(r"000_success_([0-9.]+)\.log$", hits[-1]).group(1))
    vals = [v for v in rates.values() if v is not None]
    block = {"rates": rates, "missing": missing,
             "se": {t: (None if v is None else round(math.sqrt(v * (1 - v) / int(n)), 4))
                    for t, v in rates.items()},
             "macro": (sum(vals) / len(vals)) if vals else None}
    if vals:
        block["delta_vs_t0_pp"] = round(block["macro"] - t0_macro, 4)
        block["gate_pass"] = bool(block["macro"] >= t0_macro - 0.02)
        block["per_task_delta_pp"] = {t: (None if rates[t] is None else round(rates[t] - t0[t], 4))
                                      for t in tasks}
    rep["ckpts"][ck] = block

out = os.path.join(logdir, f"F2_{mode}_gate.json")
with open(out, "w") as f:
    json.dump(rep, f, indent=2)

print(f"T0 macro={t0_macro * 100:.2f}%  gate(macro-2pp)={(t0_macro - 0.02) * 100:.2f}%")
for ck, b in rep["ckpts"].items():
    print(f"  ckpt {ck}")
    for t in tasks:
        v = b["rates"][t]
        se = b["se"][t]
        print(f"    {t:45s} {'--' if v is None else f'{v * 100:5.1f}%'} "
              f"(T0 {t0[t] * 100:5.1f}%, SE {'' if se is None else f'{se * 100:.1f}pp'})")
    if b["macro"] is not None:
        print(f"    macro={b['macro'] * 100:.2f}%  delta={b['delta_vs_t0_pp'] * 100:+.2f}pp  "
              f"gate_pass={b['gate_pass']}")
    else:
        print(f"    macro=None missing={b['missing']}")
print(f"-> {out}")
PY

echo "== F1 $MODE done $(date '+%F %T') =="
