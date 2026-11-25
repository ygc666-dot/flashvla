#!/usr/bin/env python
"""Append the T0 (pi0* baseline) table to `docs/sdn_flashvla/flashvla_simpler/3_3.md`.

`3.4_todo.md` §3 asks for exactly this after the baseline run: the nine rates are
the denominator of the G3 test, so they belong next to the P2/P3 numbers instead of
only in a JSON gate. Idempotent: the section is skipped when the marker is present.

Usage: python scripts/p3/append_t0_doc.py [--gate ...] [--doc ...]
"""

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
MARKER = "## T0：π0* 基线（vanilla，seeds 2000–2049）"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default=str(ROOT / "logs/p3_overnight/gates/P2_t0.json"))
    ap.add_argument("--doc", default=str(ROOT / "docs/sdn_flashvla/flashvla_simpler/3_3.md"))
    args = ap.parse_args()

    gate, doc = pathlib.Path(args.gate), pathlib.Path(args.doc)
    if not gate.exists():
        print(f"no gate file at {gate}, nothing to append")
        return
    if MARKER in doc.read_text(errors="ignore"):
        print("T0 section already in the doc")
        return

    data = json.loads(gate.read_text())
    lines = [
        "",
        "---",
        "",
        MARKER,
        "",
        "协议：`policy.pizero.evaluate --method vanilla --n-trajs 50 --start-seed 2000 --no-video`，",
        "权重 `third_party/SDN/pretrained/open-pi-zero`，9 任务 × 50 seeds（与训练集 1000–1499、",
        "R1 评测集 0–49 三方不相交）。",
        "",
        "| 任务 | rate |",
        "| :--- | ---: |",
    ]
    failed = set(data.get("failed") or [])
    for task, rate in data["rates"].items():
        tag = " ← 失败，按 0 计入分母" if task in failed else ""
        lines.append(f"| `{task}` | {rate:.1%}{tag} |")
    lines.append(f"| **macro** | **{data['macro']:.2%}** |")
    for warn in data.get("warn") or []:
        lines.append(f"\n- warn: {warn}")
    lines.append("")

    with open(doc, "a") as f:
        f.write("\n".join(lines))
    print(f"appended T0 section to {doc} (macro {data['macro']:.2%})")


if __name__ == "__main__":
    main()
