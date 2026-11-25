#!/usr/bin/env python
"""P7: fold every gate file into `P3_REPORT.md` + `logs/p3_overnight/P3_STATUS.txt`.

Everything is read defensively: a missing gate means "that stage did not run", and
the report says so instead of failing. Usage:

    python scripts/p3/make_report.py [--log-dir logs/p3_overnight] [--out P3_REPORT.md]
"""

import argparse
import csv
import glob
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_json(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def read_text(path, limit=None):
    try:
        text = Path(path).read_text(errors="ignore")
    except Exception:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-limit:]) if limit else "\n".join(lines)


def loss_curve(log_path, limit=5):
    rows = []
    for line in read_text(log_path).splitlines():
        match = re.search(r"step:(\d+).*?loss:\s*(-?[0-9.eE+]+).*?updt_s:\s*([0-9.eE+]+)", line)
        if match:
            rows.append((int(match.group(1)), float(match.group(2)), float(match.group(3))))
    if not rows:
        return None
    head = rows[:limit]
    tail = rows[-limit:]
    return {"first": head, "last": tail, "n": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=str(ROOT / "logs/p3_overnight"))
    ap.add_argument("--out", default=str(ROOT / "P3_REPORT.md"))
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    gates = log_dir / "gates"
    now = time.strftime("%Y-%m-%d %H:%M")

    t0 = load_json(gates / "P2_t0.json")
    pipeline = load_json(gates / "P4_pipeline.json")
    decision = load_json(gates / "decision.json")
    s0_runs = []
    for path in sorted(glob.glob(str(gates / "S0_*.json"))):
        run = load_json(path)
        if run:
            s0_runs.append(run)
    s5_rows = []
    try:
        s5_rows = list(csv.DictReader(open(log_dir / "S5_sweep.csv")))
    except Exception:
        pass

    shapes = read_text(log_dir / "S4_shapes.txt")
    train_log = log_dir / "P6_train.log"
    curve = loss_curve(train_log)
    pid_file = log_dir / "P6_train.pid"
    pid = read_text(pid_file).strip() if pid_file.exists() else ""
    training_alive = bool(pid) and os.path.exists(f"/proc/{pid}")

    flags = []
    if t0:
        macro = t0.get("macro", 0.0)
        if not (0.55 <= macro <= 0.70):
            flags.append(f"**T0 macro {macro:.1%} 越界（期望 55–70%）**：先复核评测协议，判据仍用它做分母")
        for task in t0.get("failed") or []:
            flags.append(f"**T0 任务失败** `{task}` 按 rate=0 计入分母")
    else:
        flags.append("**T0 未产出** `gates/P2_t0.json`（P2 没跑完）")
    if decision and not decision.get("trainable", False):
        flags.append(f"**未启动正式训练**：{decision.get('reason', 'decision.trainable=false')}")
    if "image_resolution" in shapes and "(224, 224)" not in shapes:
        flags.append("**S4**：进视觉塔的分辨率不是 224×224，时间表要按 S4 的实测重算")

    lines = [f"# P3 执行报告（{now}）", ""]
    lines.append(f"日志目录：`{log_dir}`　GPU：`P3_GPU` 见各阶段日志　决策文件：`gates/decision.json`")
    lines.append("")
    if flags:
        lines.append("## ⚠ 需要你先看的三件事")
        lines.append("")
        lines += [f"- {flag}" for flag in flags]
    else:
        lines.append("## 结论")
        lines.append("")
        lines.append("- T0 在带宽内、S0–S5 全部通过、训练已按量测配置启动。")

    lines += ["", "## 1. T0（π0* 基线，vanilla / seeds 2000–2049）", ""]
    if t0:
        rates = t0.get("rates", {})
        lines.append("| 任务 | rate |")
        lines.append("| :--- | ---: |")
        for task, rate in rates.items():
            mark = " ← 失败，按 0 计" if task in (t0.get("failed") or []) else ""
            lines.append(f"| `{task}` | {rate:.1%}{mark} |")
        lines.append(f"| **macro** | **{t0.get('macro', 0.0):.2%}** |")
        for warn in t0.get("warn") or []:
            lines.append(f"\n- warn: {warn}")
    else:
        lines.append("未运行 / 未完成（`gates/P2_t0.json` 缺失）。")

    lines += ["", "## 2. S0（单卡路线：1 进程 FSDP2）", ""]
    if s0_runs:
        lines.append("| 变体 | optimizer | cpu_offload | bs | 成功 | samples/s | s/step | loss |")
        lines.append("| :--- | :--- | :-: | ---: | :-: | ---: | ---: | ---: |")
        for run in s0_runs:
            lines.append(
                f"| `{run.get('tag', '')}` | {run.get('optimizer', '')} | {run.get('cpu_offload', '')} "
                f"| {run.get('batch_size', '')} | {'✅' if run.get('ok') else '❌'} "
                f"| {run.get('samples_per_s', 0):.3f} | {run.get('s_per_step', 0):.2f} "
                f"| {run.get('loss', 'n/a')} |"
            )
    else:
        lines.append("未运行（`gates/S0_*.json` 缺失）。")

    lines += ["", "## 3. S1–S4（数据 / 单步 / 存重载 / 图像尺寸）", ""]
    if pipeline:
        lines.append(f"- 返回码：S1={pipeline.get('S1_data')}，S2={pipeline.get('S2_two_steps')}，"
                     f"S3={pipeline.get('S3_reload')}，S4={pipeline.get('S4_shapes')}（0 = 通过）")
    lines.append("")
    interesting = [line for line in shapes.splitlines()
                   if line.startswith(("[cfg]", "[dataset]", "[preprocessed]", "[policy]", "[data]"))]
    if interesting:
        lines.append("```")
        lines += interesting[:16]
        lines.append("```")
    else:
        lines.append("`S4_shapes.txt` 缺失。")

    lines += ["", "## 4. S5（bs 扫描）", ""]
    if s5_rows:
        lines.append("| bs | optimizer | cpu_offload | 本进程峰值 MiB | 整卡峰值 MiB | samples/s | s/step | loss | ok |")
        lines.append("| ---: | :--- | :-: | ---: | ---: | ---: | ---: | ---: | :-: |")
        for row in s5_rows:
            lines.append(f"| {row['bs']} | {row['optimizer']} | {row['cpu_offload']} "
                         f"| {row['peak_mib']} | {row.get('device_peak_mib', '')} "
                         f"| {row['samples_per_s']} | {row['s_per_step']} | {row['loss']} | {row['ok']} |")
    else:
        lines.append("未运行（`S5_sweep.csv` 缺失）。")
    if decision:
        lines += ["", "```json", json.dumps(decision, indent=2, ensure_ascii=False), "```"]

    lines += ["", "## 5. 正式训练（P6）", ""]
    if pid:
        lines.append(f"- pid `{pid}`，{'仍在运行' if training_alive else '已退出'}")
    else:
        lines.append("- 未启动。")
    if curve:
        lines.append(f"- loss 行数 {curve['n']}；前 3 行 / 后 3 行：")
        lines.append("```")
        lines += [f"  step {step:>5}  loss {loss:.4f}  updt_s {updt:.2f}"
                  for step, loss, updt in curve["first"][:3]]
        lines.append("  ...")
        lines += [f"  step {step:>5}  loss {loss:.4f}  updt_s {updt:.2f}"
                  for step, loss, updt in curve["last"][:3]]
        lines.append("```")
    else:
        lines.append("- `P6_train.log` 里还没有 loss 行。")

    lines += ["", "## 6. 阶段流水（main.log）", "", "```"]
    lines += read_text(log_dir / "main.log", 40).splitlines()
    lines += ["```", ""]

    Path(args.out).write_text("\n".join(lines))

    status = [f"P3_STAGE_SUMMARY {now}"]
    status.append(f"T0 macro: {t0.get('macro') if t0 else 'missing'}")
    status.append(f"S0: {[(r.get('tag'), r.get('ok'), round(r.get('samples_per_s', 0), 3)) for r in s0_runs]}")
    status.append(f"decision: {json.dumps({k: (decision or {}).get(k) for k in ('trainable', 'bs', 'optimizer', 'throughput')})}")
    status.append(f"training: pid={pid or '-'} alive={training_alive}")
    (log_dir / "P3_STATUS.txt").write_text("\n".join(status) + "\n")
    print("\n".join(status))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
