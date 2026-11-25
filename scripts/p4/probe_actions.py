#!/usr/bin/env python
"""Offline action probe: does a served checkpoint predict the dataset's actions?

Why: the M=1 eval (S7) of `pi0_flashvla_bridge_success/004000` came out at
8 / 36 / 6 / 0 % where pi05 scored 42 / 78 / 62 / 54 %. Before spending another
2.4 h of GPU on a retrain we need to know whether the *model* is bad or the
*serving path* is. This script replays the exact deployed path
(`scripts/p3/flashvla_server.FlashVLAPolicyServer.act`, including the cold-start
burst and the checkpoint's own postprocessor) on dataset frames and scores the
returned 4-action chunk against the ground-truth actions.

Usage (one checkpoint at a time; ~10 GB of GPU):
  HF_HUB_OFFLINE=1 python scripts/p4/probe_actions.py \
    --ckpt /data/.../pi0_flashvla_bridge_success/checkpoints/004000/pretrained_model \
    --leaf /data/zjr/simpler_lerobot/bridge/widowx_stack_cube/success --samples 24
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.p3.flashvla_server import FlashVLAPolicyServer  # noqa: E402


def load_frames(leaf: str, samples: int, chunk: int, seed: int) -> list[dict]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=f"local/{Path(leaf).name}", root=leaf)
    total = ds.meta.total_frames
    rng = np.random.default_rng(seed)
    idxs = np.sort(rng.choice(np.arange(chunk, total - chunk), size=samples, replace=False))
    out = []
    for i in idxs:
        item = ds[int(i)]
        img = item["observation.images.top"]
        img_u8 = (img.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
        gt = torch.stack([ds[int(i + k)]["action"] for k in range(chunk)]).float().numpy()
        out.append(
            {
                "frame": int(i),
                "image": img_u8,
                "state": item["observation.state"].float().numpy(),
                "gt": gt,
                "task": item.get("task", ""),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--leaf", required=True)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    frames = load_frames(args.leaf, args.samples, args.chunk, args.seed)
    server = FlashVLAPolicyServer(args.ckpt, device=args.device, warmup=False)

    preds, gts = [], []
    for k, fr in enumerate(frames):
        server.reset(seed=k, instruction=str(fr["task"]))
        pred = np.asarray(server.act(fr["image"], fr["state"], str(fr["task"]))["actions"], dtype=np.float32)
        preds.append(pred[: args.chunk, :7])
        gts.append(fr["gt"])
    pred = np.stack(preds)  # [S, chunk, 7]
    gt = np.stack(gts)      # [S, chunk, 7]

    dims = ["dx", "dy", "dz", "rx", "ry", "rz", "grip"]
    abs_err = np.abs(pred - gt).mean(axis=(0, 1))
    gt_abs = np.abs(gt).mean(axis=(0, 1))
    pr_abs = np.abs(pred).mean(axis=(0, 1))
    # Baseline "predict the dataset mean action": its MAE is a useful floor.
    mean_action = gt.mean(axis=(0, 1))
    mean_err = np.abs(gt - mean_action).mean(axis=(0, 1))
    corr = []
    for d in range(7):
        a = pred[:, :, d].ravel()
        b = gt[:, :, d].ravel()
        corr.append(float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-8 and b.std() > 1e-8 else float("nan"))

    # F6/S5 判据 also wants the gripper as a binary decision, not a regression:
    # the dumper stores it as exactly {-1, +1} (`simpler_adapter.py:194-199`), and
    # the server thresholds at 0 before sending it to SIMPLER.
    grip_pred = np.where(pred[:, :, -1] > 0.0, 1.0, -1.0)
    grip_gt = np.where(gt[:, :, -1] > 0.0, 1.0, -1.0)
    grip_acc = float((grip_pred == grip_gt).mean())
    # Accuracy of the trivial "always the majority class" answer, for reference.
    majority = float(max((grip_gt > 0).mean(), (grip_gt < 0).mean()))

    report = {
        "ckpt": args.ckpt,
        "leaf": args.leaf,
        "samples": len(frames),
        "mae_overall": float(np.abs(pred - gt).mean()),
        "mae_vs_mean_baseline": float(mean_err.mean()),
        "grip_accuracy": round(grip_acc, 4),
        "grip_accuracy_majority": round(majority, 4),
        "per_dim": {
            d: {
                "mae": round(float(abs_err[i]), 5),
                "mae_mean_baseline": round(float(mean_err[i]), 5),
                "gt_abs_mean": round(float(gt_abs[i]), 5),
                "pred_abs_mean": round(float(pr_abs[i]), 5),
                "corr": None if np.isnan(corr[i]) else round(corr[i], 3),
                **(
                    {"accuracy": round(grip_acc, 4), "accuracy_majority": round(majority, 4)}
                    if d == "grip"
                    else {}
                ),
            }
            for i, d in enumerate(dims)
        },
        "gt_stats": {
            d: {"mean": round(float(gt[:, :, i].mean()), 4), "std": round(float(gt[:, :, i].std()), 4)}
            for i, d in enumerate(dims)
        },
    }
    print(json.dumps(report, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
