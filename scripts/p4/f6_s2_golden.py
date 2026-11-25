#!/usr/bin/env python
"""Pin the two-stream forward pass so the F6/S2 refactor can be proved inert.

`use_proprio_stack=false` must keep producing *the same numbers* after the
three-stream fusion lands (`f6.md` S2 conservation test). This script dumps a
fingerprint of `PI0FlashVLAModel.forward_shared_observation` on a synthetic
batch with fixed seeds, and can diff two dumps afterwards.

The fingerprint is computed for two cases:

* ``pad``   -- ``action_is_pad`` marks the not-yet-started slots of the
  cold-start schedule, so the loss masking path is exercised.
* ``nopad`` -- every slot is real, so nothing is dropped by ``losses[loss_mask]``
  (a bug confined to padded slots would otherwise be invisible).

Usage:
  # before the refactor (once per dtype/device)
  CUDA_VISIBLE_DEVICES=2 <env>/bin/python scripts/p4/f6_s2_golden.py \
      --ckpt ckpt/pi0star_fractal_flashvla_proprio --device cuda --dtype bfloat16 \
      --out logs/p4/f6_s2_golden_gpu_bf16.pt
  <env>/bin/python scripts/p4/f6_s2_golden.py \
      --ckpt ckpt/pi0star_fractal_flashvla_proprio --device cpu --dtype float32 \
      --out logs/p4/f6_s2_golden_cpu_fp32.pt

  # after the refactor
  ... --out logs/p4/f6_s2_after_gpu_bf16.pt --compare logs/p4/f6_s2_golden_gpu_bf16.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402

from flashvla.policies.pi0.modeling_pi0_flashvla import PI0FlashVLAPolicy  # noqa: E402


def build_policy(ckpt: str, *, device: str, dtype: str, num_layers: int | None):
    # `config.json` carries lerobot's `type` discriminator, which
    # `PI0FlashVLAConfig.from_pretrained` rejects; go through the generic parser
    # (this is what `PI0FlashVLAPolicy.from_pretrained` does internally).
    config = PreTrainedConfig.from_pretrained(pretrained_name_or_path=ckpt)
    if num_layers is not None:
        config.vlm_config.text_config.num_hidden_layers = num_layers
        config.action_expert_config.num_hidden_layers = num_layers
    policy = PI0FlashVLAPolicy.from_pretrained(ckpt, config=config)
    policy.eval()
    if dtype == "bfloat16":
        policy.model.to_bfloat16_for_selected_params("bfloat16")
    else:
        policy.model.to(dtype=torch.float32)
    return policy.to(device=device)


def make_batch(policy, *, batch_size: int, text_len: int, device: str, seed: int):
    """Synthetic batch with the exact shapes `forward_shared_observation` wants."""
    config = policy.config
    model = policy.model
    num_views = len(config.image_features)
    image_size = (3, *config.image_resolution)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    # `prepare_images` maps uint8 [0, 1] frames to [-1, 1]; mirror that range.
    images = [
        (torch.rand(batch_size, *image_size, generator=gen) * 2.0 - 1.0).to(device)
        for _ in range(num_views)
    ]
    img_masks = [torch.ones(batch_size, dtype=torch.bool, device=device) for _ in range(num_views)]
    tokens = torch.randint(0, 256, (batch_size, text_len), generator=gen).to(device)
    masks = torch.ones(batch_size, text_len, dtype=torch.bool, device=device)

    real_state_dim = config.real_state_dim
    states = torch.zeros(batch_size, config.max_state_dim, device=device)
    states[:, :real_state_dim] = torch.randn(batch_size, real_state_dim, generator=gen).to(device) * 0.1

    N, C = model.N, model.C
    total = N * N * C
    actions = torch.zeros(batch_size, total, config.max_action_dim, device=device)
    actions[..., : config.real_action_dim] = torch.randn(
        batch_size, total, config.real_action_dim, generator=gen
    ).to(device) * 0.05
    noise = torch.randn(batch_size, total, config.max_action_dim, generator=gen).to(device)

    # Cold-start schedule: slot k is only real once the buffer has k+1 entries.
    slot_pad = torch.zeros(N, N * C, dtype=torch.bool, device=device)
    slot_idx = torch.arange(N * C, device=device) // C
    for k in range(N):
        slot_pad[k] = slot_idx > k
    action_is_pad = (
        slot_pad.unsqueeze(0)
        .expand(batch_size, N, N * C)
        .reshape(batch_size, N * N * C)
        .contiguous()
    )
    return dict(
        images=images,
        img_masks=img_masks,
        tokens=tokens,
        masks=masks,
        states=states,
        actions=actions,
        noise=noise,
        action_is_pad=action_is_pad,
    )


@torch.no_grad()
def fingerprint(policy, batch, *, seed: int):
    model = policy.model
    out = {}
    cases = {
        "pad": batch["action_is_pad"],
        "nopad": None,
    }
    for name, action_is_pad in cases.items():
        torch.manual_seed(seed)
        losses = model.forward_shared_observation(
            batch["images"],
            batch["img_masks"],
            batch["tokens"],
            batch["masks"],
            batch["states"],
            batch["actions"],
            action_is_pad,
            noise=batch["noise"],
        )
        out[name] = {
            "shape": list(losses.shape),
            "sum": float(losses.float().sum().item()),
            "mean": float(losses.float().mean().item()),
            "values": losses.detach().float().cpu(),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--text-len", type=int, default=32)
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--atol", type=float, default=1e-6)
    args = ap.parse_args()

    policy = build_policy(
        args.ckpt, device=args.device, dtype=args.dtype, num_layers=args.num_layers
    )
    print(f"policy loaded: dtype={policy.model.backbone_dtype()} layers={len(policy.model.layers)}")
    batch = make_batch(
        policy, batch_size=args.batch_size, text_len=args.text_len,
        device=args.device, seed=args.seed,
    )
    fp = fingerprint(policy, batch, seed=args.seed)
    for name, entry in fp.items():
        print(f"  [{name}] shape={entry['shape']} sum={entry['sum']:.9g} mean={entry['mean']:.9g}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ckpt": args.ckpt,
            "device": args.device,
            "dtype": args.dtype,
            "seed": args.seed,
            "num_layers": args.num_layers,
            "cases": fp,
        },
        args.out,
    )
    print(f"saved {args.out}")

    if args.compare:
        ref = torch.load(args.compare, map_location="cpu")
        ok = True
        for name, entry in fp.items():
            ref_vals = ref["cases"][name]["values"]
            cur = entry["values"]
            if tuple(cur.shape) != tuple(ref_vals.shape):
                print(f"FAIL [{name}] shape {tuple(cur.shape)} != {tuple(ref_vals.shape)}")
                ok = False
                continue
            diff = (cur - ref_vals).abs().max().item()
            print(f"  [{name}] max|diff| = {diff:.3e} (atol {args.atol:g})")
            ok = ok and diff <= args.atol
        print("CONSERVED" if ok else "DRIFT")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
