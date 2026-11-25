#!/usr/bin/env bash
# P2 / B1: pi0* baseline (T0) on GPU4 -- 9 tasks x 50 seeds, seeds 2000-2049.
#
# Reasoning behind the layout (`3.4_todo.md` §3):
#   * seeds 2000-2049 are disjoint from train (1000-1499) and the R1 eval (0-49)
#   * a task that already has `000_success_*.log` is skipped, so the script is
#     idempotent and an interrupted night can be resumed with the same command
#   * one retry per task; if it still fails the task keeps `rate = 0` and stays in
#     the 9-task denominator (dropping it would inflate the G3 macro)
#   * macro outside 55-70 % is only flagged (`warn`), never fatal
#
# Outputs:
#   $T0_ROOT/vanilla[/seeds=2000-2049]/<task>/000_success_<rate>.log
#   $P3_GATES/P2_t0.json  {rates, macro, failed, warn}
#
# Always exits 0 unless the evaluator itself cannot even start.
set -uo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_env.sh"

mkdir -p "$P3_LOG/P2_t0" "$T0_ROOT"
cd "$FLASHVLA_ROOT/third_party/SDN" || exit 1

echo "== T0 $(date '+%F %T') gpu=$GPU seeds=$T0_START_SEED..$((T0_START_SEED + T0_N_TRAJS - 1)) =="

for task in "${T0_TASKS[@]}"; do
  done_log=$(ls -1 "$T0_ROOT"/vanilla/*/"$task"/000_success_*.log \
                   "$T0_ROOT"/vanilla/"$task"/000_success_*.log 2>/dev/null | head -1)
  if [ -n "${done_log:-}" ]; then
    echo "[skip] $task  <- $(basename "$done_log")"
    continue
  fi

  ok=0
  for attempt in 1 2; do
    echo "[run ] $task  attempt $attempt  $(date '+%T')"
    # `timeout` is not optional here: on 2026-11-22 the first attempt hung for
    # 20+ min with 0% GPU utilisation (a second evaluation job was running on the
    # same card) and would have eaten the whole night. 40 min is ~2x the expected
    # 20 min per task; a kill counts as a failed attempt.
    CUDA_VISIBLE_DEVICES="$GPU" timeout "$T0_TASK_TIMEOUT" \
      "$SDN_PY" -m policy.pizero.evaluate \
      --method vanilla --task "$task" \
      --n-trajs "$T0_N_TRAJS" --start-seed "$T0_START_SEED" \
      --result-root "$T0_ROOT" --no-video \
      > "$P3_LOG/P2_t0/$task.attempt$attempt.log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then ok=1; break; fi
    echo "[warn] $task attempt $attempt failed (rc=$rc), see P2_t0/$task.attempt$attempt.log"
  done
  [ "$ok" -eq 1 ] || echo "[fail] $task -> rate=0 (kept in the denominator)"
done

echo "== aggregating =="
"$FV_PY" - "$T0_ROOT" "$P3_GATES/P2_t0.json" "${T0_TASKS[@]}" <<'PY'
import glob, json, re, sys

root, out, tasks = sys.argv[1], sys.argv[2], sys.argv[3:]
rates, failed = {}, []
for task in tasks:
    hits = (sorted(glob.glob(f"{root}/vanilla/*/{task}/000_success_*.log"))
            or sorted(glob.glob(f"{root}/vanilla/{task}/000_success_*.log")))
    if not hits:
        rates[task] = 0.0
        failed.append(task)
        continue
    rate = float(re.search(r"000_success_([0-9.]+)\.log$", hits[-1]).group(1))
    rates[task] = float(rate)

macro = sum(rates.values()) / len(tasks)
warn = []
if not (0.55 <= macro <= 0.70):
    warn.append(f"macro {macro:.4f} outside the expected 55-70% band")
for task in failed:
    warn.append(f"{task}: no 000_success_*.log, counted as 0")

report = {"rates": rates, "macro": round(macro, 4), "failed": failed, "warn": warn,
          "n_trajs": None, "protocol": "vanilla / seeds=2000-2049"}
with open(out, "w") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
print(f"macro={macro:.4f} ({len(tasks) - len(failed)}/{len(tasks)} tasks ok)")
PY

echo "== T0 done $(date '+%F %T') =="
exit 0
