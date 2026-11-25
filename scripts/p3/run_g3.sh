#!/usr/bin/env bash
# G3: evaluate a `pi05-flashvla` checkpoint on SIMPLER -- 4 widowx tasks x 50 seeds
# (seeds 2000-2049, the same protocol as T0).  `3_5_p3_execution_report.md` §10.3.
#
# Layout: one policy server per GPU (py3.12 flashvla env) + one `evaluate.py`
# worker per GPU (py3.10 sdn env). `evaluate.py --num-gpus N` splits the episode
# seeds across workers; each worker connects to `port_base + <its GPU id>`.
#
#   ./run_g3.sh smoke   # 1 task x 10 seeds (2100-2109), one ckpt, ~10 min
#   ./run_g3.sh full    # 4 tasks x 50 seeds (2000-2049), both ckpts, ~2 h
#
# Notes:
#  * only the 4 `widowx_*` tasks exist in this checkpoint's training data
#    (`/data/zjr/simpler_lerobot/bridge` = the bridge group, 4 leaves). The
#    google_robot tasks T0 also covers have no training coverage here, so G3's
#    denominator is the widowx subset of T0 (80.0 %), not the 9-task 66.89 %.
#  * smoke seeds start at 2100 so they never collide with the G3 seeds, and land
#    in `$G3_ROOT/smoke/<step>/` so they cannot be mistaken for G3 results.
#  * each task is retried once and `timeout`-killed, as in run_t0.sh: a hung
#    evaluator (a neighbour job on the same card) must not eat the whole run.
set -uo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_env.sh"

MODE="${1:-full}"
G3_ROOT="${P3_G3_ROOT:-$FLASHVLA_ROOT/third_party/SDN/eval_logs/p3_g3}"

# Two concurrent copies would both bind the policy-server ports (the loser's
# servers die, its clients silently attach to the winner's server and interleave
# episodes on one shared action buffer) and race on the same result dirs. The
# lock is global rather than per-G3_ROOT because the ports are global.
RUN_LOCK="${P3_G3_LOCK:-$P3_LOG/G3/.run_g3.lock}"
mkdir -p "$G3_ROOT" "$(dirname "$RUN_LOCK")"
exec 9>"$RUN_LOCK"
if ! flock -n 9; then
  echo "[fail] another run_g3.sh holds $RUN_LOCK"
  exit 3
fi
PORT_BASE="${P3_G3_PORT_BASE:-12400}"
# GPU4 first (the card this project has been using), GPU0 second.
G3_GPUS="${P3_G3_GPUS:-4 0}"
G3_TASKS=(
  widowx_carrot_on_plate
  widowx_put_eggplant_in_basket
  widowx_spoon_on_towel
  widowx_stack_cube
)
# T0 (vanilla / seeds 2000-2049) on the same 4 tasks, from gates/P2_t0.json
# (rates are fractions, matching the 000_success_<rate>.log filenames).
# Order must follow G3_TASKS.
T0_WIDOWX_REFS="0.68 0.88 0.88 0.76"

CKPT_ROOT="${P3_CKPT_ROOT:-/data/zjr/simpler_lerobot/runs/pi05_flashvla_bridge_overnight/checkpoints}"

case "$MODE" in
  smoke)
    CKPTS=(${P3_G3_CKPTS:-004000})
    TASKS=(widowx_stack_cube)
    N_TRAJS="${P3_G3_N_TRAJS:-10}"
    START_SEED="${P3_G3_START_SEED:-2100}"
    TASK_TIMEOUT="${P3_G3_TASK_TIMEOUT:-2400}"
    ;;
  full|report)
    CKPTS=(${P3_G3_CKPTS:-004000 002750})
    TASKS=("${G3_TASKS[@]}")
    N_TRAJS="${P3_G3_N_TRAJS:-50}"
    START_SEED="${P3_G3_START_SEED:-2000}"
    TASK_TIMEOUT="${P3_G3_TASK_TIMEOUT:-5400}"
    ;;
  *) echo "usage: $0 [smoke|full|report]"; exit 2 ;;
esac
LAST_SEED=$((START_SEED + N_TRAJS - 1))

mkdir -p "$G3_ROOT" "$P3_LOG/G3"
cd "$FLASHVLA_ROOT/third_party/SDN" || exit 1

GPU_CSV=$(echo $G3_GPUS | tr ' ' ',')
N_GPU=$(echo $G3_GPUS | wc -w)
echo "== G3 $MODE $(date '+%F %T')"
echo "   ckpts=${CKPTS[*]} tasks=${#TASKS[@]} seeds=$START_SEED..$LAST_SEED"
echo "   gpus=$GPU_CSV ports=$PORT_BASE+<gpu> timeout=${TASK_TIMEOUT}s"

SERVER_PIDS=()
stop_servers() {
  for pid in ${SERVER_PIDS[@]:-}; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null
  done
  SERVER_PIDS=()
  sleep 2
}
trap 'stop_servers' EXIT INT TERM

start_servers() {
  local ckpt_dir="$1" tag="$2"
  stop_servers
  for gpu in $G3_GPUS; do
    local port=$((PORT_BASE + gpu))
    local log="$P3_LOG/G3/server_${tag}_gpu${gpu}.log"
    # Absolute path: this script has already `cd`-ed into third_party/SDN.
    CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false CODEBUDDY_SAFE_DELETE_ENABLED=0 \
      nohup "$FV_PY" "$FLASHVLA_ROOT/scripts/p3/flashvla_server.py" \
        --ckpt "$ckpt_dir" --port "$port" ${P3_G3_SERVER_ARGS:-} >> "$log" 2>&1 &
    SERVER_PIDS+=($!)
    echo "   [server] gpu=$gpu port=$port pid=$! log=$log"
  done
}

wait_ports() {
  # The server only calls listen() after the checkpoint is on the GPU (~135 s),
  # so a port check is a true readiness signal and saves a 300 s client retry.
  local gpu port deadline=$((SECONDS + 420))
  for gpu in $G3_GPUS; do
    port=$((PORT_BASE + gpu))
    while [ "$SECONDS" -lt "$deadline" ]; do
      if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
        echo "   [ready] gpu=$gpu port=$port after $((SECONDS - start_s))s"
        break
      fi
      sleep 5
    done
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "[fail] policy server on port $port never came up; see $P3_LOG/G3/"
      return 1
    fi
  done
  return 0
}

[ "$MODE" = "report" ] && SKIP_EVAL=1 || SKIP_EVAL=0

for step in "${CKPTS[@]}"; do
  [ "$SKIP_EVAL" -eq 1 ] && continue
  ckpt_dir="$CKPT_ROOT/$step/pretrained_model"
  if [ ! -f "$ckpt_dir/model.safetensors" ]; then
    echo "[fail] missing checkpoint $ckpt_dir"; continue
  fi
  if [ "$MODE" = "smoke" ]; then
    OUT_ROOT="$G3_ROOT/smoke/$step"
    rm -rf "$OUT_ROOT"        # smoke must always re-run
  else
    OUT_ROOT="$G3_ROOT/$step"
  fi
  mkdir -p "$OUT_ROOT"
  echo "== checkpoint $step  out=$OUT_ROOT  ($(date '+%T'))"
  start_s=$SECONDS
  start_servers "$ckpt_dir" "$step"
  if ! wait_ports; then continue; fi

  for task in "${TASKS[@]}"; do
    done_log=$(ls -1 "$OUT_ROOT"/flashvla/*/"$task"/000_success_*.log \
                    "$OUT_ROOT"/flashvla/"$task"/000_success_*.log 2>/dev/null | head -1)
    if [ -n "${done_log:-}" ]; then
      echo "[skip] $step/$task  <- $(basename "$done_log")"
      continue
    fi

    ok=0
    for attempt in 1 2; do
      echo "[run ] $step/$task attempt $attempt $(date '+%T')"
      # No --opts: defaults (exec_steps 4, port = 12400 + gpu) keep the result
      # dir at <root>/flashvla/seeds=A-B/<task>, exactly like T0's layout.
      CUDA_VISIBLE_DEVICES="$GPU_CSV" timeout "$TASK_TIMEOUT" \
        "$SDN_PY" -m policy.pizero.evaluate \
        --method flashvla --task "$task" \
        --checkpoint "$ckpt_dir" \
        --num-gpus "$N_GPU" --n-trajs "$N_TRAJS" --start-seed "$START_SEED" \
        --reload-every 0 \
        --result-root "$OUT_ROOT" --no-video \
        > "$P3_LOG/G3/${step}_${task}.attempt$attempt.log" 2>&1
      rc=$?
      if [ "$rc" -eq 0 ]; then ok=1; break; fi
      echo "[warn] $step/$task attempt $attempt rc=$rc, see G3/${step}_${task}.attempt$attempt.log"
    done
    [ "$ok" -eq 1 ] || echo "[fail] $step/$task -> rate=0 (kept in the denominator)"
  done
  stop_servers
done

echo "== aggregating =="
"$FV_PY" - "$G3_ROOT" "$P3_GATES" "$MODE" "$START_SEED" "$LAST_SEED" \
    "$T0_WIDOWX_REFS" "${CKPTS[@]}" -- "${TASKS[@]}" <<'PY'
import glob, json, os, re, sys

argv = sys.argv[1:]
root, gates, mode, start_seed, last_seed, t0_refs = argv[:6]
ckpts = argv[6:argv.index('--')]
tasks = argv[argv.index('--') + 1:]
refs = [float(v) for v in t0_refs.split()]
t0_macro = sum(refs) / len(refs)
t0_by_task = dict(zip(tasks, refs))

out_root = os.path.join(root, "smoke") if mode == "smoke" else root
report = {"mode": mode, "seeds": f"{start_seed}-{last_seed}",
          "protocol": "flashvla / 4 widowx tasks (the ckpt's training tasks)",
          "t0_widowx_refs": t0_by_task, "t0_macro": round(t0_macro, 4),
          "gate": round(t0_macro - 0.02, 4), "ckpts": {}}

for ckpt in ckpts:
    rates, missing = {}, []
    for task in tasks:
        hits = (sorted(glob.glob(f"{out_root}/{ckpt}/flashvla/*/{task}/000_success_*.log"))
                or sorted(glob.glob(f"{out_root}/{ckpt}/flashvla/{task}/000_success_*.log")))
        if not hits:
            missing.append(task)
            rates[task] = None
            continue
        rates[task] = float(re.search(r"000_success_([0-9.]+)\.log$", hits[-1]).group(1))
    vals = [v for v in rates.values() if v is not None]
    macro = sum(vals) / len(vals) if vals else None
    block = {"rates": rates, "missing": missing, "macro": macro}
    if macro is not None:
        block["delta_vs_t0_pp"] = round(macro - t0_macro, 4)
        block["gate_pass"] = bool(macro >= t0_macro - 0.02)
        block["per_task_delta_pp"] = {
            t: (None if rates[t] is None else round(rates[t] - t0_by_task[t], 2)) for t in tasks}
    report["ckpts"][ckpt] = block

out = os.path.join(gates, f"G3_{mode}_{os.path.basename(root.rstrip('/'))}.json")
with open(out, "w") as f:
    json.dump(report, f, indent=2)

print(f"T0 (widowx subset, seeds 2000-2049) macro = {t0_macro * 100:.2f}%  "
      f"-> G3 gate = {(t0_macro - 0.02) * 100:.2f}%")
for ckpt, b in report["ckpts"].items():
    r = ", ".join(f"{t}={'-' if v is None else f'{v * 100:.0f}%'}"
                  for t, v in b["rates"].items())
    print(f"  {ckpt}: {r}")
    if b["macro"] is not None:
        print(f"     macro={b['macro'] * 100:.2f}%  delta={b['delta_vs_t0_pp'] * 100:+.2f}pp  "
              f"pass={b['gate_pass']}")
    else:
        print(f"     macro=None missing={b['missing']}")
print(f"-> {out}")
PY

echo "== G3 $MODE done $(date '+%F %T') =="
exit 0
