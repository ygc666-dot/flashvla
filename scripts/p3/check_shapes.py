#!/usr/bin/env python
"""S4 of `3.4_todo.md`: what resolution actually reaches the vision tower?

The question decides the whole schedule (40 h vs impossible): SIMPLER frames are
512x640, and `PI05FlashVLAPolicy.forward` calls
`resize_with_pad(img, *self.config.image_resolution)` before embedding, so the
number of visual tokens depends on `policy.image_resolution`, not on the dataset.

This script builds the real dataset + the real preprocessor from a training config
(no model, no GPU) and prints:

    dataset image shape   -> what comes out of the dataset
    preprocessed shapes   -> what the policy receives
    policy.image_resolution (and the resulting per-frame token estimate)

Usage:
    python scripts/p3/check_shapes.py --config train/configs/pi05/simpler/_smoke.yaml
"""

import argparse
import json
import os

# The tokenizer (`google/paligemma-3b-pt-224`) must come from the local HF cache;
# a stray network probe would hang the whole unattended run.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import draccus
import torch

from flashvla.configs.train_config import FlashVLATrainConfig
from flashvla.datasets.flashvla_dataset import flashvla_collate_fn
from flashvla.policies.factory import make_pre_post_processors
from flashvla.train import make_flashvla_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = draccus.parse(FlashVLATrainConfig, args=[], config_path=args.config)
    print(f"[cfg] policy={cfg.policy.type} pretrained_path={cfg.policy.pretrained_path}")
    print(f"[cfg] empty_cameras={cfg.policy.empty_cameras} num_buffer_slots={cfg.policy.num_buffer_slots} "
          f"chunk_size={cfg.policy.chunk_size} state_cond={cfg.policy.state_cond}")

    dataset = make_flashvla_dataset(cfg)
    print(f"[data] frames={dataset.num_frames} episodes={dataset.num_episodes} leaves={len(dataset.subsets)}")

    raw = dataset[0]
    for key, value in raw.items():
        if isinstance(value, torch.Tensor):
            print(f"[dataset] {key}: {tuple(value.shape)} {value.dtype}")

    batch = flashvla_collate_fn([raw])
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=dataset.meta.stats,
    )
    processed = preprocessor(batch)

    image_resolution = tuple(getattr(cfg.policy, "image_resolution", (224, 224)))
    patches_per_side = image_resolution[0] // 14  # paligemma/SigLIP: patch 14
    tokens_per_image = patches_per_side * (image_resolution[1] // 14)

    report = {
        "image_resolution": image_resolution,
        "tokens_per_image": tokens_per_image,
        "preprocessed": {},
    }
    for key, value in processed.items():
        if isinstance(value, torch.Tensor):
            print(f"[preprocessed] {key}: {tuple(value.shape)} {value.dtype}")
            if "image" in key:
                report["preprocessed"][key] = list(value.shape)

    print(f"[policy] image_resolution={image_resolution} -> "
          f"{tokens_per_image} tokens/image, "
          f"{tokens_per_image * (1 + cfg.policy.empty_cameras)} visual tokens/frame")
    print("[json] " + json.dumps(report))


if __name__ == "__main__":
    main()
