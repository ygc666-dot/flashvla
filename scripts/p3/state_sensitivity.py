"""Measure how much `observation.state` actually changes the action chunk.

The `state_cond` FiLM path (`suffix_embedder.state_*`) is zero-initialized, so a
checkpoint can look perfectly healthy while ignoring its proprioception
completely. `3_5_p3_execution_report.md` §11.3 measured a relative action change
of 0.000 (max |delta| 6.6e-5) on `004000` -- i.e. effectively blind. This script
is the A/B that decides whether a training change (e.g. `state_path_lr_mult`)
fixed it.

Both arms get the same seed, so the sampled noise is identical and the only
difference is the state vector.

Usage:
  CUDA_VISIBLE_DEVICES=0 <flashvla>/bin/python scripts/p3/state_sensitivity.py \
      --ckpt <run>/checkpoints/000500/pretrained_model
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig

from flashvla.policies.pi05.modeling_pi05_flashvla import PI05FlashVLAPolicy
from flashvla.policies.factory import make_pre_post_processors


def build(cfg, image_u8, state, instruction, pre):
    img = torch.from_numpy(np.ascontiguousarray(image_u8, dtype=np.float32) / 255.0)
    img = img.permute(2, 0, 1).unsqueeze(0)
    batch = {
        "observation.images.top": img,
        "observation.state": torch.as_tensor(state, dtype=torch.float32).reshape(1, -1),
        "task": [instruction],
    }
    return pre(batch)


def chunk(policy, batch, seed):
    # reset() clears the 4-slot buffer and re-seeds torch, so the sampled noise
    # is identical across arms and the steady-state query is comparable.
    torch.manual_seed(seed)
    policy.reset()
    with torch.inference_mode():
        out = None
        for _ in range(int(policy.config.num_buffer_slots)):
            out = policy.predict_action_chunk(batch)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--state-dim", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    policy_cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    policy_cfg.compile_model = False
    policy_cfg.device = args.device
    policy = PI05FlashVLAPolicy.from_pretrained(args.ckpt, config=policy_cfg)
    policy.to(args.device).eval()
    # The saved pipeline pins `device_processor.device` to whatever the run used,
    # so a CPU comparison has to override it or the pipeline cannot even build.
    pre, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    rng = np.random.default_rng(0)
    image = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
    state_a = np.zeros(args.state_dim, dtype=np.float32)
    state_b = np.linspace(-1.0, 1.0, args.state_dim, dtype=np.float32)
    instruction = "put carrot on plate"

    act_a = chunk(policy, build(policy.config, image, state_a, instruction, pre), args.seed)
    act_b = chunk(policy, build(policy.config, image, state_b, instruction, pre), args.seed)

    diff = (act_a - act_b).abs()
    scale = act_a.abs().mean().item()
    report = {
        "ckpt": args.ckpt,
        "state_a": state_a.tolist(),
        "state_b": state_b.tolist(),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "action_abs_mean": scale,
        "relative": float(diff.mean() / scale) if scale else float("inf"),
        # Anything above ~1e-2 counts as "the state is actually being used".
        "state_used": bool(diff.mean() / scale > 1e-2) if scale else True,
        "state_mlp_out_weight_absmean": float(
            policy.model.suffix_embedder.state_mlp_out.weight.abs().mean()
        ),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
