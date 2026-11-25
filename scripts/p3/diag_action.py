#!/usr/bin/env python
"""In-distribution diagnostic for the G3 harness: does the policy reproduce the
training targets it was fit to?

Motivation (`3_5_p3_execution_report.md` §10.2): the 10-seed SIMPLER smoke came
back 0/10 with `speed` 0.0051 m/step vs T0's 0.0109, i.e. the policy moves at
half the pi0* rate and never closes the gripper. Before touching the harness it
has to be clear whether the *model* is off or the *wiring* is off. This script
removes the simulator and the TCP hop from the picture:

  for a handful of real frames from the training set,
    feed the frame 4 times (3 cold-start queries + 1 steady query, which is
    exactly the training-time "one observation, N buffer configs" setup),
    and compare the resulting physical action chunk with the recorded action.

Read it as: |pred| / |gt| ~ 1 and a small per-dim MAE means the checkpoint and
the unnormalization are right, and a bad sim rollout is a harness/domain issue.
A ratio near 0.5 means the model regresses to the mean (under-trained or
mismatched normalization), and no amount of plumbing will fix it.

Usage:
  CUDA_VISIBLE_DEVICES=4 <flashvla>/bin/python scripts/p3/diag_action.py \
      --config train/configs/pi05/simpler/_overnight.generated.yaml \
      --ckpt /data/.../checkpoints/004000/pretrained_model --n-samples 8
"""
import argparse
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import draccus
import numpy as np
import torch

import flashvla.configs  # noqa: F401  registers "pi05-flashvla"
from lerobot.configs.policies import PreTrainedConfig
from flashvla.configs.train_config import FlashVLATrainConfig
from flashvla.datasets.flashvla_dataset import flashvla_collate_fn
from flashvla.policies.factory import make_pre_post_processors
from flashvla.policies.pi05.modeling_pi05_flashvla import PI05FlashVLAPolicy
from flashvla.train import make_flashvla_dataset


def describe(name, arr):
    return {
        "absmean": round(float(np.abs(arr).mean()), 6),
        "per_dim_absmean": np.abs(arr).mean(axis=0).round(5).tolist(),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--stride", type=int, default=997, help="sample spacing inside the dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = draccus.parse(FlashVLATrainConfig, args=[], config_path=args.config)
    dataset = make_flashvla_dataset(cfg)
    print(f"[data] frames={dataset.num_frames} episodes={dataset.num_episodes} "
          f"leaves={len(dataset.subsets)}")

    policy_cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    policy_cfg.compile_model = False
    policy_cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = PI05FlashVLAPolicy.from_pretrained(args.ckpt, config=policy_cfg)
    policy.to(policy_cfg.device).eval()
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.ckpt)
    n_slots = int(policy.config.num_buffer_slots)
    print(f"[policy] {policy_cfg.type} device={policy_cfg.device} slots={n_slots} "
          f"chunk={policy.config.chunk_size} exec={policy.config.n_action_steps}")

    rows = []
    for i in range(args.n_samples):
        idx = (i * args.stride) % dataset.num_frames
        raw = dataset[idx]
        batch = pre(flashvla_collate_fn([raw]))

        torch.manual_seed(args.seed)
        policy.reset()
        for _ in range(n_slots - 1):            # cold start: same observation
            policy.predict_action_chunk(batch)
        chunk = policy.predict_action_chunk(batch)
        pred = post(chunk)[0].float().cpu().numpy()          # [C, 7] physical

        # Ground truth: the recorded command, first `C` steps of the 16-step chunk
        # (steady streaming executes buffer slot 0 = the earliest C actions).
        gt = np.asarray(raw["action"][: pred.shape[0]], dtype=np.float32)
        if gt.shape[1] > pred.shape[1]:
            gt = gt[:, : pred.shape[1]]

        rows.append({"index": int(idx), "pred": pred, "gt": gt})
        err = np.abs(pred - gt).mean()
        print(f"[{i}] idx={idx:6d} pred|a|={np.abs(pred).mean():.4f} "
              f"gt|a|={np.abs(gt).mean():.4f} MAE={err:.4f}")
        print(f"     pred={pred[0].round(3).tolist()}")
        print(f"     gt  ={gt[0].round(3).tolist()}")

    preds = np.stack([r["pred"] for r in rows])
    gts = np.stack([r["gt"] for r in rows])
    ratio = float(np.abs(preds).mean() / max(np.abs(gts).mean(), 1e-9))
    # Which way does the prediction lean? Take the sign agreement on the gripper.
    gripper_agree = float((np.sign(preds[..., -1]) == np.sign(gts[..., -1])).mean())
    report = {
        "ckpt": args.ckpt,
        "n_samples": args.n_samples,
        "pred": describe("pred", preds),
        "gt": describe("gt", gts),
        "abs_ratio_pred_over_gt": round(ratio, 4),
        "mae_per_dim": np.abs(preds - gts).mean(axis=(0, 1)).round(5).tolist(),
        "mae_overall": round(float(np.abs(preds - gts).mean()), 5),
        "gripper_sign_agreement": round(gripper_agree, 3),
    }
    print("\n[summary] " + json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
