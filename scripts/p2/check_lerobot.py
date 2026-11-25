#!/usr/bin/env python
"""G2 acceptance for the converted SIMPLER LeRobot datasets (`3_2_todo.md` §6).

Checks, in the order of the checklist:

    G2-0  root count / per-task frame counts vs the §5.5 table, fps, image shape
    G2-1  FlashVLADataset single sample: action [N*C, 7], action_is_pad [N*C],
          observation.state [8], image [3, H, W]
    G2-2  pad structure: block k inherits the first k*C rows of the base sample and
          pads exactly the tail
    G2-3  the action window never crosses an episode boundary
    G2-4  `meta.stats` == the real global statistics of the roots that train
          (normalise -> mean 0 / std 1, relative error < 1e-3)
    G2-5  action continuity: no isolated |a_{t+1} - a_t| spike inside an episode
    G2-7  `task` strings: the apple task must contain both sub-instructions
    G2-8  identical feature keys / fps across the leaves of one group
    T6    `len(dataset)` and steps/epoch for the P3 configs (batch 32)

and optionally (T5) exports a few episodes to gif + action curve and measures the
PSNR of the re-encoded video against the P1 source mp4 (risk 1 of §9).

Usage::

    export HF_HUB_OFFLINE=1
    python scripts/p2/check_lerobot.py --root /data/zjr/simpler_lerobot \\
        --json /data/zjr/simpler_lerobot/g2_report.json
    python scripts/p2/check_lerobot.py --root /data/zjr/simpler_lerobot \\
        --spotcheck 10 --spotcheck-dir /data/zjr/simpler_lerobot/spotcheck
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# §5.5 acceptance baseline (frames per leaf, after first_success + 1 + k, k = 2)
EXPECTED = {
    "fractal": {
        "google_robot_place_apple_in_closed_top_drawer": {"success": 20_273, "failure": 76_600},
        "google_robot_open_drawer": {"success": 14_793, "failure": 32_132},
        "google_robot_close_drawer": {"success": 8_010, "failure": 4_756},
        "google_robot_move_near": {"success": 3_984, "failure": 4_480},
        "google_robot_pick_coke_can": {"success": 3_783, "failure": 1_280},
    },
    "bridge": {
        "widowx_carrot_on_plate": {"success": 6_007, "failure": 7_740},
        "widowx_put_eggplant_in_basket": {"success": 4_560, "failure": 2_280},
        "widowx_spoon_on_towel": {"success": 3_564, "failure": 1_860},
        "widowx_stack_cube": {"success": 3_653, "failure": 2_520},
    },
}
EXPECTED_FPS = {"fractal": 3, "bridge": 5}
EXPECTED_SHAPE = {"fractal": (512, 640, 3), "bridge": (480, 640, 3)}
# §5.2①: these two tasks must carry more than one instruction string
MULTI_INSTRUCTION_TASKS = {"google_robot_place_apple_in_closed_top_drawer"}
STATS_RTOL = 1e-3
SPIKE_FACTOR = 10.0


# ------------------------------------------------------------------ utilities
def read_json(path: Path):
    return json.loads(path.read_text())


def leaves(root: Path, group: str, subdirs) -> list[tuple[str, str, Path]]:
    """[(task, subdir, leaf_root)] for every leaf that has meta/info.json."""
    out = []
    group_root = root / group
    if not group_root.is_dir():
        return out
    for task_dir in sorted(p for p in group_root.iterdir() if p.is_dir()):
        for subdir in subdirs:
            leaf = task_dir / subdir
            if (leaf / "meta" / "info.json").is_file():
                out.append((task_dir.name, subdir, leaf))
    return out


def read_parquet(leaf: Path, columns) -> pd.DataFrame:
    files = sorted((leaf / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet under {leaf/'data'}")
    frames = [pd.read_parquet(f, columns=columns) for f in files]
    return pd.concat(frames, ignore_index=True)


def as_matrix(series) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float64) for v in series])


# ------------------------------------------------------------------- checks
def check_roots(root: Path, groups, subdirs, report: dict) -> bool:
    print("\n## G2-0 root inventory and frame counts\n")
    print("| group | task | outcome | episodes | frames | expected | fps | image shape |")
    print("| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |")
    ok = True
    inv = {}
    for group in groups:
        found = {leaf[0]: leaf for leaf in leaves(root, group, subdirs)}
        missing_tasks = set(EXPECTED[group]) - set(found)
        if missing_tasks:
            ok = False
            print(f"- FAIL `{group}`: missing leaves for {sorted(missing_tasks)}")
        total = defaultdict(int)
        for task, subdir, leaf in leaves(root, group, subdirs):
            info = read_json(leaf / "meta" / "info.json")
            frames = int(info["total_frames"])
            eps = int(info["total_episodes"])
            shape = tuple(info["features"]["observation.images.top"]["shape"])
            exp = EXPECTED[group].get(task, {}).get(subdir)
            flag = "ok"
            if exp is None:
                flag = "extra leaf"
            elif exp != frames:
                flag = f"EXPECTED {exp}"
                ok = False
            if int(info["fps"]) != EXPECTED_FPS[group]:
                flag += f" fps={info['fps']} EXPECTED {EXPECTED_FPS[group]}"
                ok = False
            if shape != EXPECTED_SHAPE[group]:
                flag += f" shape={shape}"
                ok = False
            print(f"| {group} | {task} | {subdir} | {eps} | {frames} | "
                  f"{'' if exp is None else exp} | {info['fps']} | {shape} | {flag}")
            total[subdir] += frames
            inv.setdefault(group, {}).setdefault(task, {})[subdir] = {
                "root": str(leaf), "episodes": eps, "frames": frames,
                "fps": int(info["fps"]), "image_shape": list(shape),
                "expected": exp, "match": exp == frames,
            }
        for subdir, frames in sorted(total.items()):
            exp = sum(v.get(subdir, 0) for v in EXPECTED[group].values())
            good = exp == frames
            ok &= good
            print(f"- `{group}/{subdir}` total {frames} frames (expected {exp}) "
                  f"{'PASS' if good else 'FAIL'}")
        report.setdefault("roots", {})[group] = {"total_frames": dict(total), "leaves": inv[group]}
    return ok


def build_subsets(root: Path, group: str, subdirs, args):
    """FlashVLADataset leaves of one group, concatenated (metadata only, no video IO yet)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from flashvla.datasets.flashvla_dataset import FlashVLADataset, MultiFlashVLADataset

    task_leaves = leaves(root, group, subdirs)
    if not task_leaves:
        raise FileNotFoundError(f"No leaves for group {group} under {root/group}")
    delta = None
    subsets = []
    for task, subdir, leaf in task_leaves:
        meta = LeRobotDatasetMetadata(f"simpler/{group}/{task}/{subdir}", root=leaf, revision=None)
        if delta is None:
            # same dict `resolve_delta_timestamps` builds for a pi05 config:
            # observation_delta_indices is None, action_delta_indices = range(N*C)
            fps = meta.fps
            horizon = args.num_buffer_slots * args.chunk_size
            delta = {"action": [i / fps for i in range(horizon)]}
        subsets.append(FlashVLADataset(
            f"simpler/{group}/{task}/{subdir}", root=leaf, delta_timestamps=delta,
            num_buffer_slots=args.num_buffer_slots, chunk_size=args.chunk_size,
            revision=None, video_backend=args.video_backend,
        ))
    return MultiFlashVLADataset(subsets), task_leaves


def episode_table(meta) -> pd.DataFrame:
    eps = meta.episodes
    if eps is None:
        return pd.DataFrame()
    return eps if isinstance(eps, pd.DataFrame) else eps.to_pandas()


def episode_bounds(ds) -> list[tuple[int, int]]:
    """[(from_index, to_index)] per episode, in global index space."""
    tbl = episode_table(ds.meta).sort_values("episode_index")
    return [(int(r["dataset_from_index"]), int(r["dataset_to_index"])) for _, r in tbl.iterrows()]


def task_names(leaf: Path) -> list[str]:
    tbl = pd.read_parquet(leaf / "meta" / "tasks.parquet")
    if "task" in tbl.columns:
        return sorted(str(v) for v in tbl["task"].tolist())
    return sorted(str(v) for v in tbl.index.tolist())


def episode_video_window(leaf: Path, meta, episode_index: int):
    """(mp4 path, first frame index, last frame index) of one episode.

    LeRobot v3 packs many episodes into one `file-XXX.mp4`; the episode is the
    `[from_timestamp, to_timestamp)` slice of that file.
    """
    tbl = episode_table(meta)
    row = tbl[tbl["episode_index"] == episode_index].iloc[0]
    key = "videos/observation.images.top"
    chunk = int(row[f"{key}/chunk_index"])
    file_idx = int(row[f"{key}/file_index"])
    fps = float(meta.fps)
    f0 = int(round(float(row[f"{key}/from_timestamp"]) * fps))
    f1 = int(round(float(row[f"{key}/to_timestamp"]) * fps))
    path = leaf / "videos" / "observation.images.top" / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.mp4"
    return path, f0, f1


def decode_window(path: Path, f0: int, f1: int) -> np.ndarray:
    """Decode frames [f0, f1) of an mp4.

    `list(reader)` must not be used: these files report `nframes=inf`, so `__len__`
    returns 2**63-1 and `list()` preallocates until MemoryError.
    """
    import imageio.v2 as iio

    reader = iio.get_reader(str(path))
    try:
        return np.stack([reader.get_data(i) for i in range(f0, f1)])
    finally:
        reader.close()


def check_samples(root: Path, group: str, subdirs, args, report: dict) -> bool:
    """G2-1 (shapes) + G2-2 (pad structure) on every leaf of one group."""
    print(f"\n## G2-1/2 sample geometry — {group} {subdirs}\n")
    import torch

    ds, task_leaves = build_subsets(root, group, subdirs, args)
    N, C = args.num_buffer_slots, args.chunk_size
    H = N * C
    ok = True

    for subset, (task, subdir, _leaf) in zip(ds.subsets, task_leaves):
        sample = subset[0]
        shape_a = tuple(sample["action"].shape)
        shape_p = tuple(sample["action_is_pad"].shape)
        shape_s = tuple(sample["observation.state"].shape)
        keys = [k for k in sample if k.startswith("observation.images.")]
        shape_i = tuple(sample[keys[0]].shape) if keys else None
        # `__getitem__` concatenates N configs, each of length H_cfg = N*C (§6 warning);
        # the image comes back channel-first even though info.json stores (H, W, 3)
        want_img = (3, EXPECTED_SHAPE[group][0], EXPECTED_SHAPE[group][1])
        good = (shape_a == (N * H, 7) and shape_p == (N * H,) and shape_s == (8,)
                and shape_i == want_img)
        ok &= good
        print(f"- `{task}/{subdir}`: action {shape_a}, pad {shape_p}, state {shape_s}, "
              f"image {shape_i} {'PASS' if good else 'FAIL'}")

        # G2-2: block k carries the first k*C rows of the base sample and pads the tail.
        # The base sample is not returned directly, but block k=N is exactly it
        # (k*C == H, so nothing is zeroed and nothing is padded).
        base_a = sample["action"][(N - 1) * H:N * H]
        base_p = sample["action_is_pad"][(N - 1) * H:N * H]
        for k in range(1, N + 1):
            blk_a = sample["action"][(k - 1) * H:k * H]
            blk_p = sample["action_is_pad"][(k - 1) * H:k * H]
            want_a = base_a.clone()
            want_a[k * C:] = 0
            want_p = torch.ones(H, dtype=torch.bool)
            want_p[:k * C] = base_p[:k * C]
            same_a = bool(torch.equal(blk_a, want_a))
            same_p = bool(torch.equal(blk_p, want_p))
            if not (same_a and same_p):
                ok = False
                print(f"    FAIL `{task}/{subdir}` block k={k}: "
                      f"action match={same_a}, pad match={same_p}")

    print(f"- leaves checked: {len(ds.subsets)}, len(dataset) = {len(ds)} "
          f"(total_frames = {ds.meta.total_frames})")
    report.setdefault("geometry", {})[group] = {
        "num_leaves": len(ds.subsets), "len_dataset": len(ds),
        "feature_keys": sorted(ds.meta.features),
    }
    return ok


def check_pad_and_cross_episode(root: Path, group: str, subdirs, args, report: dict) -> bool:
    """G2-3: the action window never reaches past the episode end.

    For a sample at global index `idx` of an episode ending at `to_index`, exactly
    `min(to_index - idx, H_cfg)` of the `H_cfg` action rows may be real.  Checking the
    first, middle and last frames of every episode catches a window that borrows
    actions from the next episode.
    """
    print(f"\n## G2-3 episode-boundary padding — {group}\n")
    ds, task_leaves = build_subsets(root, group, subdirs, args)
    N, C = args.num_buffer_slots, args.chunk_size
    H = N * C
    ok = True
    checked = 0
    for subset, (task, subdir, _leaf) in zip(ds.subsets, task_leaves):
        bounds = episode_bounds(subset)
        picks = []
        for frm, to in bounds:
            picks.extend({frm, (frm + to) // 2, to - 1})
        bad = 0
        for idx in sorted(picks):
            sample = subset[idx]
            # block N (= last) is the base sample: its pad is the per-frame ground truth
            pad = sample["action_is_pad"][(N - 1) * H:N * H]
            real = int((~pad).sum())
            frm, to = next((f, t) for f, t in bounds if f <= idx < t)
            want = min(to - idx, H)
            if real != want:
                ok = False
                bad += 1
                print(f"- FAIL `{task}/{subdir}` idx {idx} (ep ends at {to}): "
                      f"{real} real rows, expected {want}")
            checked += 1
        if not bad:
            print(f"- `{task}/{subdir}`: {len(picks)} boundary samples OK "
                  f"(episodes={len(bounds)})")
    print(f"- boundary samples checked: {checked} {'PASS' if ok else 'FAIL'}")
    report.setdefault("episode_boundaries", {})[group] = {"checked": checked, "pass": ok}
    return ok


def check_stats(root: Path, group: str, subdir_sets, report: dict) -> bool:
    """G2-4: the pooled `meta.stats` must be the statistics of the roots that train."""
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.io_utils import cast_stats_to_numpy

    print(f"\n## G2-4 stats vs the actual training roots — {group}\n")
    ok = True
    for name, subdirs in subdir_sets:
        tbl = leaves(root, group, subdirs)
        if not tbl:
            print(f"- SKIP `{name}`: no leaves")
            continue
        chunks = {"action": [], "observation.state": []}
        pooled = None
        for _task, _sub, leaf in tbl:
            df = read_parquet(leaf, ["action", "observation.state"])
            chunks["action"].append(as_matrix(df["action"]))
            chunks["observation.state"].append(as_matrix(df["observation.state"]))
            leaf_stats = cast_stats_to_numpy(read_json(leaf / "meta" / "stats.json"))
            pooled = leaf_stats if pooled is None else aggregate_stats([pooled, leaf_stats])
        for feat in ("action", "observation.state"):
            x = np.concatenate(chunks[feat], axis=0)
            mean = np.asarray(pooled[feat]["mean"], dtype=np.float64).ravel()
            std = np.asarray(pooled[feat]["std"], dtype=np.float64).ravel()
            z = (x - mean) / std
            err_mean = float(np.max(np.abs(z.mean(0))))
            err_std = float(np.max(np.abs(z.std(0) - 1.0)))
            good = err_mean < STATS_RTOL and err_std < STATS_RTOL
            ok &= good
            print(f"- `{group}/{name}` {feat:18s} frames={len(x):7d} "
                  f"|mean|max={err_mean:.2e} |std-1|max={err_std:.2e} "
                  f"{'PASS' if good else 'FAIL'}")
            # per-root spread is what the augmentation arm will inherit
            report.setdefault("stats", {}).setdefault(f"{group}/{name}", {})[feat] = {
                "frames": int(len(x)), "max_abs_mean": err_mean, "max_abs_std_minus_1": err_std,
                "pass": good,
            }
    return ok


def check_action_continuity(root: Path, group: str, subdirs, report: dict) -> bool:
    print(f"\n## G2-5 action continuity — {group} {subdirs}\n")
    rows = []
    ok = True
    for task, sub, leaf in leaves(root, group, subdirs):
        df = read_parquet(leaf, ["action", "episode_index"])
        acts = as_matrix(df["action"])
        ep_idx = df["episode_index"].to_numpy()
        per_p99, spikes, worst = [], 0, 0.0
        for ep in np.unique(ep_idx):
            a = acts[ep_idx == ep]
            if len(a) < 3:
                continue
            d = np.linalg.norm(np.diff(a, axis=0), axis=1)
            p99 = float(np.percentile(d, 99))
            per_p99.append(p99)
            hi = float(d.max())
            worst = max(worst, hi / max(p99, 1e-9))
            spikes += int((d > SPIKE_FACTOR * max(p99, 1e-9)).sum())
        p99 = float(np.median(per_p99)) if per_p99 else float("nan")
        good = spikes == 0
        ok &= good
        print(f"- `{task}/{sub}`: episodes={len(per_p99)} median p99 |da|={p99:.4f} "
              f"max/p99={worst:.2f} spikes(>{SPIKE_FACTOR:g}x p99)={spikes} "
              f"{'PASS' if good else 'FAIL'}")
        rows.append({"task": task, "subdir": sub, "episodes": len(per_p99),
                     "median_p99": p99, "max_over_p99": worst, "spikes": spikes})
    report.setdefault("continuity", {})[group] = rows
    return ok


def check_tasks(root: Path, groups, subdirs, report: dict) -> bool:
    print("\n## G2-7 language labels\n")
    ok = True
    for group in groups:
        for task, sub, leaf in leaves(root, group, subdirs):
            names = task_names(leaf)
            counts = read_parquet(leaf, ["task_index"])["task_index"].value_counts().to_dict()
            print(f"- `{group}/{task}/{sub}`: {len(names)} instruction(s) {counts}")
            if task in MULTI_INSTRUCTION_TASKS and len(names) < 2:
                ok = False
                print(f"    FAIL: expected two sub-instructions, got {names}")
            report.setdefault("tasks", {}).setdefault(f"{group}/{task}/{sub}", names)
    return ok


def check_schema(root: Path, groups, subdirs, report: dict) -> bool:
    print("\n## G2-8 cross-leaf schema consistency\n")
    ok = True
    for group in groups:
        keys, fps_set = None, set()
        for _task, _sub, leaf in leaves(root, group, subdirs):
            info = read_json(leaf / "meta" / "info.json")
            k = frozenset(info["features"].keys())
            fps_set.add(int(info["fps"]))
            if keys is None:
                keys = k
            elif k != keys:
                ok = False
                print(f"- FAIL `{leaf}`: feature keys differ")
        good = len(fps_set) == 1
        ok &= good
        print(f"- `{group}`: fps={sorted(fps_set)} keys={len(keys or [])} "
              f"{'PASS' if good and keys else 'FAIL'}")
        report.setdefault("schema", {})[group] = {"fps": sorted(fps_set),
                                                  "feature_keys": sorted(keys or [])}
    return ok


def check_config_pipeline(configs, report: dict) -> bool:
    """Build the dataset through the real P3 entry point (`robotwin_multitask` + auto discovery).

    This validates the delivered yaml configs themselves, not just the raw leaves:
    a wrong `root` / `config_subdirs` / feature mismatch raises here, at start-up,
    exactly like it would in training.
    """
    print("\n## T6 config pipeline (`make_robotwin_multitask_flashvla_dataset`)\n")
    import draccus
    from flashvla.configs.train_config import FlashVLATrainConfig
    from flashvla.datasets.flashvla_dataset import make_robotwin_multitask_flashvla_dataset

    ok = True
    for path in configs:
        cfg = draccus.parse(FlashVLATrainConfig, config_path=str(path), args=[])
        ds = make_robotwin_multitask_flashvla_dataset(cfg)
        sample = ds[0]
        a_shape = tuple(sample["action"].shape)
        p_shape = tuple(sample["action_is_pad"].shape)
        s_shape = tuple(sample["observation.state"].shape)
        n_cfg = cfg.policy.num_buffer_slots
        horizon = n_cfg * cfg.policy.total_action_horizon if hasattr(cfg.policy, "total_action_horizon") \
            else n_cfg * n_cfg * cfg.policy.chunk_size
        good = a_shape == (horizon, 7) and p_shape == (horizon,) and s_shape == (8,)
        ok &= good
        spe = math.ceil(len(ds) / cfg.batch_size / cfg.grad_accum_steps)
        print(f"- `{path.name}`: leaves={len(ds.subsets)} len={len(ds)} fps={ds.meta.fps} "
              f"action={a_shape} state={s_shape} "
              f"step/epoch={spe} ({cfg.steps / spe:.1f} epoch at steps={cfg.steps}) "
              f"{'PASS' if good else 'FAIL'}")
        report.setdefault("configs", {})[path.name] = {
            "root": str(cfg.robotwin_multitask.root),
            "config_subdirs": list(cfg.robotwin_multitask.config_subdirs),
            "len_dataset": len(ds), "num_leaves": len(ds.subsets), "fps": ds.meta.fps,
            "action_shape": list(a_shape), "steps_per_epoch": spe, "steps": cfg.steps,
        }
    return ok


def len_table(root: Path, groups, report: dict, batch_size: int = 32) -> None:
    print("\n## T6 len(dataset) / steps per epoch (batch 32)\n")
    print("| training set | frames | len(dataset) | step/epoch | steps 12000 ≈ |")
    print("| :--- | ---: | ---: | ---: | ---: |")
    rows = {}
    for group in groups:
        for name, subdirs in (("success", ["success"]),
                              ("success+failure", ["success", "failure"])):
            frames = 0
            for _t, _s, leaf in leaves(root, group, subdirs):
                frames += int(read_json(leaf / "meta" / "info.json")["total_frames"])
            if not frames:
                continue
            spe = math.ceil(frames / batch_size)
            print(f"| {group} [{name}] | {frames} | {frames} | {spe} | {12000 / spe:.1f} epoch |")
            rows[f"{group}/{name}"] = {"frames": frames, "len_dataset": frames,
                                       "steps_per_epoch": spe}
    report["lengths"] = rows


# ------------------------------------------------------------------ spotcheck
def spotcheck(root: Path, groups, n: int, out_dir: Path, report: dict) -> bool:
    """T5: export gif + action curve and measure the re-encode PSNR (risk 1)."""
    import imageio.v2 as iio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n## T5 spotcheck ({n} episodes per outcome) -> {out_dir}\n")
    ok = True
    rng = np.random.default_rng(0)
    psnrs = []
    for group in groups:
        for task, sub, leaf in leaves(root, group, ["success", "failure"]):
            info = read_json(leaf / "meta" / "info.json")
            n_eps = int(info["total_episodes"])
            if n_eps == 0:
                continue
            pick = sorted(rng.choice(n_eps, size=min(n, n_eps), replace=False).tolist())
            df = read_parquet(leaf, ["action", "episode_index", "task_index"])
            names = task_names(leaf)
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
            meta = LeRobotDatasetMetadata(f"local/{task}/{sub}", root=leaf, revision=None)
            for ep in pick:
                video, f0, f1 = episode_video_window(leaf, meta, ep)
                if not video.exists():
                    print(f"- SKIP `{task}/{sub}` ep {ep}: no video {video.name}")
                    continue
                frames = decode_window(video, f0, f1)
                rows = df[df["episode_index"] == ep]
                acts = as_matrix(rows["action"])
                t_idx = int(rows["task_index"].iloc[0])
                tag = f"{group}_{task}_{sub}_ep{ep:06d}"
                gif = out_dir / f"{tag}.gif"
                iio.mimsave(gif, frames[::max(1, len(frames) // 40)], duration=0.2)
                fig, ax = plt.subplots(4, 1, figsize=(6, 7), sharex=True)
                for i, (name, a) in enumerate(zip(("dx", "dy", "dz", "gripper"), acts.T)):
                    ax[i].plot(a, lw=1.2)
                    ax[i].set_ylabel(name)
                ax[-1].set_xlabel(f"frame (task={names[t_idx]!r})")
                fig.suptitle(f"{task}/{sub} ep{ep}")
                fig.tight_layout()
                png = out_dir / f"{tag}_action.png"
                fig.savefig(png, dpi=90)
                plt.close(fig)

                # PSNR of the re-encoded (AV1) frames against the P1 source mp4
                psnr = _psnr_vs_source(leaf, ep, frames)
                if psnr is not None:
                    psnrs.append(psnr)
                    if psnr < 30.0:
                        ok = False
                        print(f"- WARN `{task}/{sub}` ep {ep}: PSNR {psnr:.1f} dB < 30")
                print(f"- `{task}/{sub}` ep {ep}: {len(frames)} frames, task={names[t_idx]!r}, "
                      f"psnr={psnr if psnr is None else round(psnr, 2)}")
    if psnrs:
        print(f"- PSNR over {len(psnrs)} episodes: min {min(psnrs):.2f} dB, "
              f"median {float(np.median(psnrs)):.2f} dB "
              f"{'PASS' if min(psnrs) >= 30 else 'FAIL (<30 dB: fall back to h264 / PNG)'}")
    report["spotcheck"] = {"episodes": len(psnrs), "psnr_min": min(psnrs) if psnrs else None,
                           "psnr_median": float(np.median(psnrs)) if psnrs else None,
                           "out_dir": str(out_dir)}
    return ok


def _psnr_vs_source(leaf: Path, ep: int, frames: np.ndarray):
    """PSNR between the LeRobot video and the P1 source mp4 named in p1_provenance.json.

    `ep` is the local episode index of this leaf, which is exactly the position in
    the provenance list (episodes are appended in conversion order); the provenance
    `episode` field itself is the P1 seed.
    """
    prov = leaf / "meta" / "p1_provenance.json"
    if not prov.is_file():
        return None
    entries = read_json(prov).get("episodes", [])
    if ep >= len(entries):
        return None
    src = entries[ep].get("source")
    if not src:
        return None
    video = Path(src).with_suffix(".mp4")
    if not video.is_file():
        return None
    try:
        import imageio.v2 as iio
        reader = iio.get_reader(str(video))
        try:
            ref = np.stack([f for f in reader])
        finally:
            reader.close()
    except Exception:  # noqa: BLE001
        return None
    m = min(len(ref), len(frames))
    a = frames[:m].astype(np.float64)
    b = ref[:m].astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    return 10 * math.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("/data/zjr/simpler_lerobot"))
    ap.add_argument("--groups", nargs="+", default=["fractal", "bridge"],
                    choices=["fractal", "bridge"])
    ap.add_argument("--subdirs", nargs="+", default=["success"],
                    help="Leaves used for the geometry / stats gates.")
    ap.add_argument("--stats-subsets", nargs="+", default=["success", "success,failure"],
                    help="Comma-separated subdir sets to pool for G2-4.")
    ap.add_argument("--num-buffer-slots", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=4)
    ap.add_argument("--video-backend", default="torchcodec")
    ap.add_argument("--spotcheck", type=int, default=0,
                    help="Export N episodes per outcome to gif + action curve (T5).")
    ap.add_argument("--spotcheck-dir", type=Path, default=None)
    ap.add_argument("--configs", nargs="*", default=[],
                    help="P3 yaml configs to build through the real dataset factory (T6).")
    ap.add_argument("--json", default="", help="Write the machine-readable report here.")
    ap.add_argument("--skip-samples", action="store_true",
                    help="Skip G2-1/2/3 (needs video decoding).")
    args = ap.parse_args()

    report = {"root": str(args.root), "groups": args.groups, "subdirs": args.subdirs}
    verdicts = {}

    verdicts["G2-0"] = check_roots(args.root, args.groups, ["success", "failure"], report)
    if not args.skip_samples:
        print("\n" + "#" * 70)
        geom_ok, boundary_ok = True, True
        for group in args.groups:
            geom_ok &= check_samples(args.root, group, args.subdirs, args, report)
            boundary_ok &= check_pad_and_cross_episode(args.root, group, args.subdirs, args, report)
        verdicts["G2-1/2"] = geom_ok
        verdicts["G2-3"] = boundary_ok
    subdir_sets = [(name, name.split(",")) for name in args.stats_subsets]
    stats_ok = True
    for group in args.groups:
        stats_ok &= check_stats(args.root, group, subdir_sets, report)
    verdicts["G2-4"] = stats_ok
    cont_ok = True
    for group in args.groups:
        cont_ok &= check_action_continuity(args.root, group, ["success", "failure"], report)
    verdicts["G2-5"] = cont_ok
    verdicts["G2-7"] = check_tasks(args.root, args.groups, ["success", "failure"], report)
    verdicts["G2-8"] = check_schema(args.root, args.groups, ["success", "failure"], report)
    len_table(args.root, args.groups, report)
    if args.configs:
        verdicts["T6"] = check_config_pipeline([Path(p) for p in args.configs], report)
    if args.spotcheck:
        out_dir = args.spotcheck_dir or (args.root / "spotcheck")
        verdicts["T5"] = spotcheck(args.root, args.groups, args.spotcheck, out_dir, report)

    print("\n## verdict\n")
    for k, v in verdicts.items():
        print(f"- {k}: {'PASS' if v else 'FAIL'}")
    all_ok = all(verdicts.values())
    report["verdicts"] = verdicts
    report["verdict"] = "PASS" if all_ok else "FAIL"
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nreport written to {args.json}")
    print(f"\n### overall: {report['verdict']}\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
