#!/usr/bin/env python
"""G1 health check for the P1 raw dumps (`3.flashvla_simpler.md` §3.3).

Scans a dump root produced by
`third_party/SDN/policy/pizero/evaluate.py --dump-episodes` and asserts the six
gates of §3.3, per task:

    success episodes >= 50           (>= 30 allowed for apple / open_drawer, flagged)
    total frames     >= 3000
    action           == the exact 7-dim vector handed to env.step  (100%)
    image size       identical inside one robot family
    first_success    median < 0.6 * horizon
    determinism      same seed -> identical first_success and per-frame action

The first five are checked here; determinism is checked with `--compare` against a
re-run of one episode into a second root (see the command printed at the end).

Usage (either the `sdn` or the `flashvla` env works; numpy + imageio only):

    python scripts/p1/check_raw.py --root third_party/SDN/eval_logs/p1_dump
    python scripts/p1/check_raw.py --root <rerun root> --compare <original root>
    python scripts/p1/check_raw.py --root <root> --json /tmp/g1.json --no-decode

Exit code is 0 only when every gate passes.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from collections import defaultdict

import numpy as np

REQUIRED_KEYS = (
    "proprio", "action", "world_vector", "rot_axangle", "gripper", "reward",
    "success", "truncated", "query_index", "query_step", "query_instruction_id",
    "raw_chunk", "instruction_id", "instruction_vocab", "meta",
)
MIN_SUCCESS = 50
MIN_FRAMES = 3000
# tasks whose pi0* success rate is low enough that 500 episodes may still yield < 50
LOW_YIELD_MIN_SUCCESS = {"google_robot_place_apple_in_closed_top_drawer": 30,
                         "google_robot_open_drawer": 30}
FIRST_SUCCESS_FRAC = 0.6


def iter_npz(root):
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.startswith("episode_") and name.endswith(".npz"):
                yield osp.join(dirpath, name)


def load_meta(z):
    return json.loads(str(z["meta"]))


def video_frame_count(path):
    """(n_frames, (width, height)) of an mp4; (None, None) when it cannot be decoded."""
    try:
        import imageio
    except ImportError:
        return None, None
    try:
        reader = imageio.get_reader(path)
        try:
            meta = reader.get_meta_data()
            size = tuple(meta["size"]) if meta.get("size") else None
            n = meta.get("nframes")
            # the container often reports a bogus nframes, so count when it is not a sane int
            if isinstance(n, (int, float)) and np.isfinite(n) and n > 0:
                return int(n), size
            return sum(1 for _ in reader), size
        finally:
            reader.close()
    except Exception:  # noqa: BLE001 - a broken video is a finding, not a crash
        return None, None


def check_episode(path, decode=True):
    """Returns a dict of per-episode facts plus a list of hard errors."""
    errors = []
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in REQUIRED_KEYS if k not in z.files]
        if missing:
            return {"path": path, "errors": [f"missing keys {missing}"]}, None
        meta = load_meta(z)
        proprio, action = z["proprio"], z["action"]
        steps = action.shape[0]
        facts = {
            "path": path,
            "task": meta["task"],
            "episode": int(meta["episode"]),
            "robot_type": meta["robot_type"],
            "fps": int(meta["fps"]),
            "steps": steps,
            "n_queries": int(z["query_step"].shape[0]),
            "first_success": int(meta["first_success"]),
            "success": bool(meta["success"]),
            "image_shape": tuple(meta["image_shape"]) if meta.get("image_shape") else None,
            "image_file": meta.get("image_file"),
            "image_mode": meta.get("image_mode"),
        }

        # --- structural checks
        if action.ndim != 2 or action.shape[1] != 7:
            errors.append(f"action shape {action.shape} != (T, 7)")
        if proprio.ndim != 2 or proprio.shape[1] != 8:
            errors.append(f"proprio shape {proprio.shape} != (T, 8)")
        for key in ("reward", "success", "truncated", "query_index", "instruction_id"):
            if z[key].shape != (steps,):
                errors.append(f"{key} shape {z[key].shape} != ({steps},)")
        if not np.all(np.isfinite(action)):
            errors.append("non-finite action")
        if steps != int(meta["steps"]):
            errors.append(f"meta steps {meta['steps']} != array length {steps}")

        # --- the §3.3 hard constraint: what is dumped is what env.step received
        rebuilt = np.concatenate([z["world_vector"], z["rot_axangle"], z["gripper"]], axis=1)
        if rebuilt.shape != action.shape or not np.array_equal(rebuilt, action):
            errors.append("action != concat(world_vector, rot_axangle, gripper)")

        # --- query bookkeeping: every step must point at a real query and a real chunk row
        query_step = z["query_step"].astype(np.int64)
        raw_chunk = z["raw_chunk"]
        horizon = raw_chunk.shape[1] if raw_chunk.ndim == 3 else 0
        if query_step.size == 0:
            errors.append("no query recorded")
        else:
            if query_step[0] != 0:
                errors.append(f"query_step[0] = {query_step[0]} != 0")
            if np.any(np.diff(query_step) <= 0):
                errors.append("query_step is not strictly increasing")
            if query_step[-1] >= steps:
                errors.append(f"query_step[-1] = {query_step[-1]} >= steps {steps}")
            offset = np.arange(steps) - query_step[z["query_index"].astype(np.int64)]
            if np.any(offset < 0):
                errors.append("a step precedes its own query")
            elif horizon and np.any(offset >= horizon):
                errors.append(f"a step uses chunk row {int(offset.max())} >= horizon {horizon}")
            if int((np.diff(query_step) > horizon).sum()) > 0:
                errors.append("gap between consecutive queries exceeds the chunk length")
            facts["row_offset_max"] = int(offset.max()) if offset.size else 0

        vocab = z["instruction_vocab"]
        if int(z["instruction_id"].max(initial=0)) >= vocab.size:
            errors.append("instruction_id out of range")
        facts["n_instructions"] = int(vocab.size)
        facts["first_instruction"] = str(vocab[0]) if vocab.size else ""

        # --- images
        if facts["image_mode"] == "npz":
            images = z["images"]
            if images.shape[0] != steps:
                errors.append(f"npz images count {images.shape[0]} != steps {steps}")
            facts["image_shape"] = tuple(images.shape[1:])
        elif facts["image_mode"] == "mp4" and decode and facts["image_file"]:
            video = osp.join(osp.dirname(path), facts["image_file"])
            if not osp.exists(video):
                errors.append(f"missing video {facts['image_file']}")
            else:
                n, size = video_frame_count(video)
                facts["video_frames"] = n
                if n is None:
                    errors.append("video could not be decoded")
                elif n != steps:
                    errors.append(f"video frames {n} != steps {steps}")
                if size and facts["image_shape"] and tuple(size) != (facts["image_shape"][1], facts["image_shape"][0]):
                    errors.append(f"video {size} != image_shape {facts['image_shape']}")

        facts["action_absmax"] = float(np.abs(action).max())

    return facts, errors


def compare_episodes(path_a, path_b):
    """Bit-exact determinism check between two dumps of the same (task, episode).

    Same seed must reproduce `success`, `first_success`, the query schedule and every
    per-step action; frames are compared by their mp4 bytes when both exist.
    """
    problems = []
    with np.load(path_a, allow_pickle=False) as a, np.load(path_b, allow_pickle=False) as b:
        meta_a, meta_b = load_meta(a), load_meta(b)
        tag = f"{meta_a['task']}#{meta_a['episode']}"
        if meta_a["success"] != meta_b["success"]:
            problems.append(f"success {meta_a['success']} != {meta_b['success']}")
        if int(meta_a["first_success"]) != int(meta_b["first_success"]):
            problems.append(f"first_success {meta_a['first_success']} != {meta_b['first_success']}")
        if a["action"].shape != b["action"].shape:
            problems.append(f"action shape {a['action'].shape} != {b['action'].shape}")
        elif not np.array_equal(a["action"], b["action"]):
            diff = np.abs(a["action"] - b["action"])
            problems.append(f"per-step action differs (max |d| = {diff.max():.3e} at row {int(np.argmax(diff.max(1)))}; "
                            f"{int((diff.max(1) > 0).sum())}/{a['action'].shape[0]} rows)")
        if not np.array_equal(a["query_step"], b["query_step"]):
            problems.append("query_step differs")
        if not np.array_equal(a["proprio"], b["proprio"]):
            problems.append("proprio differs")
        video_a, video_b = meta_a.get("image_file"), meta_b.get("image_file")
    if video_a and video_b:
        va, vb = osp.join(osp.dirname(path_a), video_a), osp.join(osp.dirname(path_b), video_b)
        if osp.exists(va) and osp.exists(vb):
            with open(va, "rb") as fa, open(vb, "rb") as fb:
                if fa.read() != fb.read():
                    problems.append("mp4 bytes differ")
    return (tag, problems) if problems else None


def stats_per_robot(arrays):
    out = {}
    for robot, acts in arrays.items():
        a = np.concatenate(acts, axis=0)
        out[robot] = {
            "n_frames": int(a.shape[0]),
            "action_mean": np.round(a.mean(0), 5).tolist(),
            "action_std": np.round(a.std(0), 5).tolist(),
            "action_p01": np.round(np.percentile(a, 1, axis=0), 5).tolist(),
            "action_p99": np.round(np.percentile(a, 99, axis=0), 5).tolist(),
            "action_absmax": np.round(np.abs(a).max(0), 5).tolist(),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dump root (searched recursively for episode_*.npz)")
    ap.add_argument("--compare", default="", help="second dump root; episodes present in both are compared bit-exactly")
    ap.add_argument("--json", default="", help="write the full report here")
    ap.add_argument("--no-decode", action="store_true", help="skip mp4 frame counting (faster)")
    ap.add_argument("--no-action-stats", action="store_true", help="skip the per-robot action statistics")
    args = ap.parse_args()

    paths = list(iter_npz(args.root))
    if not paths:
        print(f"no episode_*.npz under {args.root}", file=sys.stderr)
        return 2

    per_task = defaultdict(lambda: {"episodes": 0, "success": 0, "frames": 0, "success_frames": 0,
                                    "first_success": [], "steps": set(), "image_shapes": set(),
                                    "fps": set(), "errors": [], "errors_n": 0})
    action_arrays = defaultdict(list)
    grippers = set()
    facts_by_key = {}

    for path in paths:
        facts, errors = check_episode(path, decode=not args.no_decode)
        if facts is None:
            per_task["<unreadable>"]["errors"].append(f"{path}: {errors}")
            per_task["<unreadable>"]["errors_n"] += 1
            continue
        task = facts["task"]
        row = per_task[task]
        row["episodes"] += 1
        row["success"] += int(facts["success"])
        row["frames"] += facts["steps"]
        row["success_frames"] += facts["steps"] if facts["success"] else 0
        row["steps"].add(facts["steps"])
        row["fps"].add(facts["fps"])
        if facts["image_shape"]:
            row["image_shapes"].add(tuple(facts["image_shape"]))
        if facts["success"]:
            row["first_success"].append(facts["first_success"])
        if errors:
            row["errors_n"] += 1
            row["errors"].extend(f"{osp.basename(path)}: {e}" for e in errors)
        facts_by_key[(task, facts["episode"])] = facts
        if not args.no_action_stats:
            with np.load(path, allow_pickle=False) as z:
                action_arrays[facts["robot_type"]].append(z["action"])
                grippers.update(np.round(z["gripper"].ravel(), 3).tolist())
        row["first_instruction"] = facts["first_instruction"]

    # ------------------------------------------------------------------ report
    print(f"\n## G1 health check — {len(paths)} episodes under {args.root}\n")
    header = ("| task | ep | succ | rate | frames | succ frames | horizon | first_succ med "
              "| img | fps | bad eps |")
    print(header)
    print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: |")
    gates = {}
    for task in sorted(per_task):
        row = per_task[task]
        horizon = max(row["steps"]) if row["steps"] else 0
        med = float(np.median(row["first_success"])) if row["first_success"] else float("nan")
        rate = 100.0 * row["success"] / max(1, row["episodes"])
        img = ",".join(f"{h}x{w}" for h, w, _ in sorted(row["image_shapes"])) if row["image_shapes"] else "-"
        fps = ",".join(str(f) for f in sorted(row["fps"]))
        print(f"| {task} | {row['episodes']} | {row['success']} | {rate:.1f}% | {row['frames']} "
              f"| {row['success_frames']} | {horizon} | {med:.1f} | {img} | {fps} | {row['errors_n']} |")
        min_success = LOW_YIELD_MIN_SUCCESS.get(task, MIN_SUCCESS)
        gates[task] = {
            "success_count": row["success"] >= min_success,
            "success_count_relaxed": row["success"] < MIN_SUCCESS and row["success"] >= min_success,
            "total_frames": row["frames"] >= MIN_FRAMES,
            "action_roundtrip": row["errors_n"] == 0,
            "image_size_consistent": len(row["image_shapes"]) <= 1,
            "first_success_median": bool(row["first_success"]) and med < FIRST_SUCCESS_FRAC * horizon,
            "horizon_consistent": len(row["steps"]) <= 1,
        }

    hard = ("success_count", "total_frames", "action_roundtrip", "image_size_consistent",
            "first_success_median", "horizon_consistent")
    print("\n### gates\n")
    all_ok = True
    for task, g in gates.items():
        failed = [k for k in hard if not g[k]]
        relaxed = [k for k in failed if k == "success_count" and g["success_count_relaxed"]]
        real = [k for k in failed if k not in relaxed]
        status = "PASS" if not real else "FAIL"
        if real:
            all_ok = False
        note = f" (relaxed: {', '.join(relaxed)}, report it)" if relaxed else ""
        print(f"- {status} `{task}`{note}" + (f" — failed: {', '.join(real)}" if real else ""))

    if not args.no_action_stats and action_arrays:
        print("\n### per-robot action statistics (raw, denormalized; compare with the "
              "fractal/bridge tables when converting to LeRobot)\n")
        for robot, s in stats_per_robot(action_arrays).items():
            print(f"- `{robot}`: {s['n_frames']} frames")
            for k in ("action_mean", "action_std", "action_p01", "action_p99", "action_absmax"):
                print(f"    {k:13s} {s[k]}")
        values = sorted(grippers)
        print(f"\n- gripper values ({len(values)} distinct): {values[:12]}" + (" ..." if len(values) > 12 else ""))

    # ------------------------------------------------------- determinism (--compare)
    determinism = None
    if args.compare:
        other = {}
        for path in iter_npz(args.compare):
            facts, _errors = check_episode(path, decode=False)
            if facts is not None:
                other[(facts["task"], facts["episode"])] = (path, facts)
        common = sorted(set(facts_by_key) & set(other))
        mismatches = [compare_episodes(facts_by_key[key]["path"], other[key][0]) for key in common]
        mismatches = [m for m in mismatches if m]
        print(f"\n### determinism: {len(common)} episodes appear in both roots")
        if not common:
            print("- FAIL: no shared (task, episode); re-run the same seeds into --compare")
            all_ok = False
        elif mismatches:
            all_ok = False
            for m in mismatches:
                print(f"- FAIL {m[0]}: {', '.join(m[1])}")
        else:
            print(f"- PASS: identical success / first_success / per-step action for "
                  f"{len(common)} episodes (seeds: {sorted({k[1] for k in common})[:10]})")
        determinism = {"common": len(common), "mismatches": mismatches}
    else:
        sample = sorted(per_task)[0] if per_task else "<task>"
        seed = next((k[1] for k in sorted(facts_by_key) if k[0] == sample), 0)
        print("\n### determinism: not checked (pass --compare)")
        print("Re-run one episode into a second root and compare:")
        print(f"  cd third_party/SDN && python -m policy.pizero.evaluate --method vanilla --task {sample} \\")
        print(f"      --n-trajs 1 --start-seed {seed} --dump-episodes --dump-images mp4 --no-video \\")
        print("      --result-root ./eval_logs/p1_determinism")
        print(f"  python scripts/p1/check_raw.py --root ./eval_logs/p1_determinism --compare {args.root}")

    print(f"\n### verdict: {'PASS' if all_ok else 'FAIL'}\n")

    if args.json:
        report = {
            "root": args.root,
            "compare": args.compare or None,
            "n_episodes": len(paths),
            "per_task": {t: {k: (sorted(v) if isinstance(v, set) else v)
                             for k, v in row.items() if k != "errors"}
                         for t, row in per_task.items()},
            "errors": {t: row["errors"][:50] for t, row in per_task.items() if row["errors"]},
            "gates": gates,
            "determinism": determinism,
            "verdict": "PASS" if all_ok else "FAIL",
        }
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"report written to {args.json}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
