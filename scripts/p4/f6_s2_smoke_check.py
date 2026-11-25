#!/usr/bin/env python
"""Reload an F6/S2 checkpoint and drive the three-path inference end to end.

Covers what the 2-step smoke cannot: that the *saved* three-stream checkpoint
reloads, and that `sample_actions` -> `_cold_start` / `_steady_streaming` ->
`denoise_step` actually runs with the proprio stream in place (N cold-start
steps, then one steady step).

Usage:
  CUDA_VISIBLE_DEVICES=4 <env>/bin/python scripts/p4/f6_s2_smoke_check.py \
      --ckpt runs/f6_s2_smoke_fractal/checkpoints/000002/pretrained_model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flashvla.policies.pi0.modeling_pi0_flashvla import PI0FlashVLAPolicy  # noqa: E402
from flashvla.policies.pi0.utils import build_flashvla_attention_mask_and_position_ids  # noqa: E402,F401


def check_preprocessor(ckpt: str, leaf: str, device: str, samples: int = 8) -> None:
    """Run the checkpoint's own preprocessor on real dataset frames.

    This is the only place that exercises the *saved* 8-D -> 7-D layout step and
    the injected 7-D stat set together, i.e. the S4 acceptance item "the encoder
    sees the layout it was trained on, in the right units".
    """
    import numpy as np
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    import flashvla.configs  # noqa: F401  (registers the pi0-flashvla config type)
    from flashvla.policies.factory import make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.device = device
    pre, _ = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    ds = LeRobotDataset(repo_id=f"local/{Path(leaf).name}", root=leaf)
    idxs = np.linspace(0, ds.meta.total_frames - 1, samples).astype(int)
    out = []
    for i in idxs:
        item = ds[int(i)]
        batch = pre(
            {
                "observation.images.top": item["observation.images.top"].unsqueeze(0),
                "observation.state": item["observation.state"].unsqueeze(0),
                "task": [item.get("task", "")],
            }
        )
        out.append(batch["observation.state"])
    stacked = torch.cat(out, dim=0).float()
    print(
        f"preprocessor state: shape={tuple(stacked.shape)} "
        f"min={stacked.min().item():+.3f} max={stacked.max().item():+.3f} "
        f"|x|>1.5 count={int((stacked.abs() > 1.5).sum())}"
    )
    per_dim = stacked.abs().max(dim=0).values
    print("  per-dim abs max: " + " ".join(f"{v:.2f}" for v in per_dim.tolist()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--text-len", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--leaf",
        default=None,
        help="also push real frames through the checkpoint's preprocessor (F6/S4)",
    )
    args = ap.parse_args()

    policy = PI0FlashVLAPolicy.from_pretrained(args.ckpt).eval().to(args.device)
    model = policy.model
    print(f"use_proprio_stack = {model.use_proprio_stack}")
    print(f"proprio_in_proj   = {hasattr(model.suffix_embedder, 'proprio_in_proj')}")
    print(f"state_proj frozen = {not model.suffix_embedder.state_proj.weight.requires_grad}")
    print(f"params            = {sum(p.numel() for p in policy.parameters())/1e6:.1f} M")
    streams = model.layers[0].self_attn._stream_attentions()
    print(f"streams           = {len(streams)} (action is proprio: {streams[1] is streams[2]})")

    config = policy.config
    print(
        f"state_layout      = {getattr(config, 'state_layout', 'raw')} "
        f"real_state_dim={config.real_state_dim} proprio_input_dim={config.proprio_input_dim}"
    )
    if args.leaf:
        check_preprocessor(args.ckpt, args.leaf, args.device)

    bsz = args.batch_size
    gen = torch.Generator().manual_seed(args.seed)
    views = len(config.image_features)
    images = [
        (torch.rand(bsz, 3, *config.image_resolution, generator=gen) * 2 - 1).to(args.device)
        for _ in range(views)
    ]
    img_masks = [torch.ones(bsz, dtype=torch.bool, device=args.device) for _ in range(views)]
    tokens = torch.randint(0, 256, (bsz, args.text_len), generator=gen).to(args.device)
    masks = torch.ones(bsz, args.text_len, dtype=torch.bool, device=args.device)
    # The proprio encoder only consumes the first `proprio_input_dim` columns: 7 on
    # bridge (euler layout), 8 on fractal, while `real_state_dim` stays at the
    # dataset's 8 in both cases.
    state_dims = (
        config.proprio_input_dim
        if getattr(config, "use_proprio_stack", False)
        else config.real_state_dim
    )
    states = torch.zeros(bsz, config.max_state_dim, device=args.device)
    states[:, :state_dims] = torch.randn(
        bsz, state_dims, generator=gen
    ).to(args.device) * 0.1

    buffer, step = None, 0
    for _ in range(model.N + 1):
        actions, buffer, step, _ = model.sample_actions(
            images, img_masks, tokens, masks, states, buffer, step,
        )
        if actions is None:
            print(f"  cold step {step-1}: v_t ok, buffer finite={bool(torch.isfinite(buffer).all())}")
        else:
            print(
                f"  steady step {step-1}: actions {tuple(actions.shape)} "
                f"mean={actions.mean().item():.4f} absmax={actions.abs().max().item():.4f} "
                f"finite={bool(torch.isfinite(actions).all())}"
            )
    torch.cuda.synchronize()
    print(f"peak GPU memory   = {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
