#!/usr/bin/env python
"""Convert the P1 SIMPLER dumps into LeRobot v3.0 datasets (P2 / T1-T3).

Input contract (`3_1.md` §1.2): one `episode_XXXXX.{npz,mp4}` pair per episode, where
the npz holds `proprio[T,8]`, `action[T,7]`, `instruction_id[T]`,
`instruction_vocab[V]` and a json `meta` (`steps` / `first_success` / `success` /
`fps` / `image_shape` / `camera`); the images live in the sibling mp4.

Output layout (decision 2 + `3_2_todo.md` §5.5)::

    <out-root>/<task>/success/    # success trajectories, truncated at first_success + 1 + k
    <out-root>/<task>/failure/    # failure trajectories, kept in full
    <out-root>/<task>/all/        # only with --no-outcome-split

Each leaf is a complete LeRobotDataset, so P3 can point
`robotwin_multitask.root` at `<out-root>` and pick `config_subdirs: [success]`
(or `[success, failure]` for the drift-augmentation arm) without hard-coding paths.

Three things this script deliberately does (`3_2_todo.md` §5.2):
  1. `task` is read **per frame** from `instruction_vocab[instruction_id[t]]`
     (apple has two sub-instruction spans inside a single episode);
  2. no `state = a[:-1] / action = a[1:]` shift — `action[t]` is already the action
     executed from the observation at row t (P1 asserts this bit-exactly);
  3. `finalize()` is always called, otherwise `meta/info.json` stays incomplete and
     every later `LeRobotDataset(root=...)` falls back to a Hub lookup.

Usage::

    export HF_HUB_OFFLINE=1
    python sim_eval/simpler/convert_simpler_to_lerobot.py \
        --dump-root third_party/SDN/eval_logs/p1_dump/vanilla \
        --out-root /data/zjr/simpler_lerobot/fractal \
        --group fractal --truncate-k 2

`--dump-root` may be the `vanilla/` directory (several `seeds=*` siblings, which is
how P1 laid the data out), a single `seeds=*` directory, one `<task>/` directory, or
one `<task>/dump/` directory; everything is discovered recursively.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import imageio
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

STATE_NAMES = ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]
ACTION_NAMES = ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]

# decision 3: one image key everywhere, both inside a robot family and across them
IMAGE_KEY = "observation.images.top"

GROUPS = {
    # the only differences left between the two families
    "fractal": dict(fps=3, shape=(512, 640, 3), robot_type="google_robot",
                    source_camera="overhead_camera"),
    "bridge": dict(fps=5, shape=(480, 640, 3), robot_type="widowx",
                   source_camera="3rd_view_camera"),
}

OUTCOMES = ("success", "failure")


# --------------------------------------------------------------------------- io
def episode_num(path: Path) -> int:
    return int(path.stem.replace("episode_", ""))


def read_meta(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as z:
        return json.loads(str(z["meta"]))


def iter_task_dirs(root: Path):
    """Yield `(task, dump_dir)` for every `<task>/dump/` under `root`."""
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"--dump-root does not exist: {root}")
    if root.name == "dump" and root.is_dir():
        yield root.parent.name, root
        return
    if (root / "dump").is_dir():
        yield root.name, root / "dump"
        return
    subs = sorted(p for p in root.iterdir() if p.is_dir())
    if not subs:
        raise FileNotFoundError(f"No sub-directory under {root}")
    found = False
    for sub in subs:
        found = True
        yield from iter_task_dirs(sub)
    if not found:
        raise FileNotFoundError(f"No <task>/dump/ under {root}")


def collect_tasks(dump_roots, wanted) -> dict[str, list[Path]]:
    """task -> sorted list of dump dirs (a task may span several `seeds=*` roots)."""
    wanted = set(wanted or [])
    tasks: dict[str, list[Path]] = defaultdict(list)
    for root in dump_roots:
        for task, dump_dir in iter_task_dirs(Path(root)):
            if wanted and task not in wanted:
                continue
            if dump_dir not in tasks[task]:
                tasks[task].append(dump_dir)
    missing = wanted - set(tasks)
    if missing:
        raise FileNotFoundError(f"requested tasks not found under {dump_roots}: {sorted(missing)}")
    if not tasks:
        raise FileNotFoundError(f"No tasks discovered under {dump_roots}")

    # Recursive discovery means one task can now be seen from several `seeds=*`
    # siblings. That is intended (P1 spread the 9 tasks over 4 directories, with
    # disjoint seed ranges), but the same episode reached through two roots would
    # be converted twice and silently inflate both `num_frames` and `stats`.
    dups: dict[str, list[str]] = {}
    for task, dump_dirs in tasks.items():
        seen: dict[int, Path] = {}
        for dump_dir in dump_dirs:
            for npz in sorted(dump_dir.glob("episode_*.npz"), key=episode_num):
                n = episode_num(npz)
                if n in seen and seen[n] != dump_dir:
                    dups.setdefault(task, []).append(f"episode {n} in both {seen[n]} and {dump_dir}")
                seen[n] = dump_dir
    if dups:
        detail = "; ".join(f"{t}: {rows[0]}" + (f" (+{len(rows) - 1} more)" if len(rows) > 1 else "")
                           for t, rows in sorted(dups.items()))
        raise ValueError(f"duplicate episodes reached through several --dump-root entries: {detail}. "
                         f"Pass the narrowest common root, or drop the overlapping one.")
    return dict(tasks)


def filter_by_group(tasks: dict[str, list[Path]], group: str, explicit: bool) -> dict[str, list[Path]]:
    """Drop tasks that belong to the other robot family (fractal vs bridge)."""
    robot_type = GROUPS[group]["robot_type"]
    kept, dropped = {}, {}
    for task, dump_dirs in tasks.items():
        npz = sorted(dump_dirs[0].glob("episode_*.npz"), key=episode_num)
        found = read_meta(npz[0])["robot_type"] if npz else "?"
        if found == robot_type:
            kept[task] = dump_dirs
        else:
            if explicit:
                raise ValueError(f"task {task} has robot_type={found}, but --group {group} "
                                 f"expects {robot_type}")
            dropped[task] = found
    if dropped:
        print(f"skipping {len(dropped)} task(s) of other robot types: "
              f"{', '.join(sorted(dropped))}")
    if not kept:
        raise FileNotFoundError(f"No task with robot_type={robot_type} found")
    return kept


def build_features(group: str) -> dict:
    g = GROUPS[group]
    return {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": {"motors": STATE_NAMES}},
        "action": {"dtype": "float32", "shape": (7,), "names": {"motors": ACTION_NAMES}},
        IMAGE_KEY: {"dtype": "video", "shape": g["shape"], "names": ["height", "width", "rgb"]},
    }


# ------------------------------------------------------------------ planning
def plan_episode(meta: dict, truncate_k: int, outcome_split: bool):
    """Return `(outcome, keep, success, first_success, steps)` for one episode."""
    success = bool(meta["success"])
    steps = int(meta["steps"])
    first_success = int(meta.get("first_success", -1))
    if success:
        # decision 1: k = 2 -> first_success + 1 + k frames (0-based index)
        keep = steps if first_success < 0 else min(steps, first_success + 1 + truncate_k)
    else:
        keep = steps  # decision 2: failure trajectories are kept in full
    outcome = "success" if success else "failure"
    if not outcome_split:
        outcome = "all"
    return outcome, keep, success, first_success, steps


def scan_task(npz_paths, truncate_k, outcome_split):
    """Dry-run the plan: per-outcome episode counts and kept frames."""
    stats = {o: {"episodes": 0, "frames": 0} for o in OUTCOMES + ("all",)}
    per_episode = []
    for path in npz_paths:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            n_instructions = int(z["instruction_vocab"].size)
        outcome, keep, *_ = plan_episode(meta, truncate_k, outcome_split)
        stats[outcome]["episodes"] += 1
        stats[outcome]["frames"] += keep
        per_episode.append({"path": path, "outcome": outcome, "keep": keep,
                            "meta": meta, "n_instructions": n_instructions})
    return {o: s for o, s in stats.items() if s["episodes"]}, per_episode


# ------------------------------------------------------------------- writing
def ensure_writer(writers, root: Path, group: str, task: str, outcome: str, args):
    ds = writers.get(outcome)
    if ds is not None:
        return ds
    g = GROUPS[group]
    # `LeRobotDataset.create` insists on a non-existing root
    if root.exists():
        if any(root.iterdir()):
            raise FileExistsError(f"{root} is not empty; pass --overwrite to re-convert")
        root.rmdir()
    root.parent.mkdir(parents=True, exist_ok=True)
    ds = LeRobotDataset.create(
        repo_id=f"simpler/{group}/{task}/{outcome}",
        fps=g["fps"],
        features=build_features(group),
        root=root,
        robot_type=g["robot_type"],
        use_videos=True,
        vcodec=args.vcodec,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=0,
    )
    writers[outcome] = ds
    print(f"  [create] {root}  (fps={g['fps']}, {g['shape']}, vcodec={args.vcodec})", flush=True)
    return ds


def write_episode(ds: LeRobotDataset, npz_path: Path, keep: int, group: str) -> int:
    """Append `keep` frames of one episode; returns the number of frames written."""
    expected_shape = GROUPS[group]["shape"]
    with np.load(npz_path, allow_pickle=False) as z:
        proprio = z["proprio"]
        action = z["action"]
        instruction_id = z["instruction_id"]
        vocab = z["instruction_vocab"]
    video_path = npz_path.with_suffix(".mp4")
    if keep > min(len(proprio), len(action), len(instruction_id)):
        raise ValueError(f"{npz_path.name}: keep={keep} exceeds array length")
    if not video_path.exists():
        raise FileNotFoundError(f"missing video for {npz_path.name}: {video_path}")

    reader = imageio.get_reader(str(video_path))
    written = 0
    try:
        for t, frame in enumerate(reader):
            if t >= keep:
                break
            if frame.shape != expected_shape:
                raise ValueError(
                    f"{video_path.name}: frame shape {frame.shape} != {expected_shape} "
                    f"for group {group}"
                )
            ds.add_frame({
                "observation.state": proprio[t].astype(np.float32),
                "action": action[t].astype(np.float32),
                # per-frame instruction: `task` must never be taken from row 0 only
                "task": str(vocab[instruction_id[t]]),
                IMAGE_KEY: frame,
            })
            written += 1
    finally:
        reader.close()
    if written != keep:
        raise ValueError(f"{video_path.name}: decoded {written} frames, expected {keep}")
    ds.save_episode()
    return written


def warn_if_safe_delete_shim() -> None:
    """Warn when a delete-hook shim is loaded, before it can abort a long conversion.

    The CodeBuddy IDE injects a `sitecustomize` that turns `os.remove/rmdir/
    shutil.rmtree` into "move to `<mount>/.Trash-$uid`".  lerobot deletes the
    staging PNGs at the end of every `save_episode`, so on a mount where that
    trash directory cannot be created (e.g. root-owned `/data`) the process dies
    *after* writing the episode.  The hook reads its switch at import time, so it
    cannot be disabled from here -- only reported.
    """
    if os.environ.get("CODEBUDDY_SAFE_DELETE_ENABLED") == "0":
        return
    shim = sys.modules.get("sitecustomize")
    hooked = getattr(shutil.rmtree, "__name__", "rmtree") != "rmtree" or (
        shim is not None and hasattr(shim, "_try_trash"))
    if not hooked:
        return
    print("[warn] a delete-hook shim is active (os.remove/rmdir/rmtree -> Trash).\n"
          "[warn] lerobot removes its staging PNGs on every save_episode, which will fail\n"
          "[warn] with 'Trash ... Permission denied' on mounts without a writable trash dir.\n"
          "[warn] re-run with CODEBUDDY_SAFE_DELETE_ENABLED=0 to turn the hook off.",
          file=sys.stderr, flush=True)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def convert_task(task: str, dump_dirs, out_root: Path, args, manifest: dict) -> dict:
    npz_paths = []
    for d in dump_dirs:
        npz_paths.extend(sorted(d.glob("episode_*.npz"), key=episode_num))
    if not npz_paths:
        raise FileNotFoundError(f"No episode_*.npz for task {task} in {dump_dirs}")

    stats, per_episode = scan_task(npz_paths, args.truncate_k, args.outcome_split)
    total_kept = sum(ep["keep"] for ep in per_episode)
    summary = ", ".join(f"{o}={s['episodes']}ep/{s['frames']}f" for o, s in stats.items())
    print(f"\n== {task}: {len(per_episode)} episodes, {total_kept} frames to encode ({summary})",
          flush=True)
    if args.dry_run:
        return {"task": task, "episodes": len(per_episode), "frames": total_kept, "stats": stats}

    task_root = out_root / task
    if task_root.exists() and args.overwrite:
        shutil.rmtree(task_root)

    writers: dict[str, LeRobotDataset] = {}
    provenance: dict[str, list] = defaultdict(list)
    counts = {o: 0 for o in OUTCOMES + ("all",)}
    kept_frames = {o: 0 for o in OUTCOMES + ("all",)}
    skipped = 0
    t0 = time.time()
    done = 0

    for ep in per_episode:
        npz_path, outcome, keep, meta = ep["path"], ep["outcome"], ep["keep"], ep["meta"]
        if args.limit and counts[outcome] >= args.limit:
            skipped += 1
            continue
        g = GROUPS[args.group]
        ds = ensure_writer(writers, task_root / outcome, args.group, task, outcome, args)
        written = write_episode(ds, npz_path, keep, args.group)
        counts[outcome] += 1
        kept_frames[outcome] += written
        provenance[outcome].append({
            "episode": int(meta["episode"]),
            "seed": int(meta["episode"]),  # P1 uses seed == episode index
            "outcome": outcome,
            "steps": int(meta["steps"]),
            "first_success": int(meta.get("first_success", -1)),
            "kept": written,
            "truncate_k": args.truncate_k,
            "n_instructions": ep["n_instructions"],
            "source": str(npz_path),
            "source_camera": meta.get("camera", g["source_camera"]),
            "source_fps": int(meta.get("fps", g["fps"])),
        })
        done += 1
        if done % 25 == 0 or done == len(per_episode) - skipped:
            frames_done = sum(kept_frames.values())
            print(f"   {done}/{len(per_episode) - skipped} eps, {frames_done} frames, "
                  f"{time.time() - t0:.1f}s", flush=True)

    for outcome, ds in writers.items():
        ds.finalize()  # §5.2③: without this meta/info.json is incomplete
        write_json(task_root / outcome / "meta" / "p1_provenance.json", {
            "task": task,
            "group": args.group,
            "outcome": outcome,
            "truncate_k": args.truncate_k,
            "outcome_split": args.outcome_split,
            "vcodec": args.vcodec,
            "fps": GROUPS[args.group]["fps"],
            "image_key": IMAGE_KEY,
            "source_camera": GROUPS[args.group]["source_camera"],
            "episodes": provenance[outcome],
        })
        print(f"  [finalize] {ds.root} -> {len(provenance[outcome])} episodes, "
              f"{kept_frames[outcome]} frames", flush=True)

    row = {
        "task": task,
        "episodes_seen": len(per_episode),
        "episodes_written": {o: counts[o] for o in counts if counts[o]},
        "frames_written": {o: kept_frames[o] for o in counts if kept_frames[o]},
        "skipped_by_limit": skipped,
        "roots": [str(w.root) for w in writers.values()],
        "elapsed_s": round(time.time() - t0, 1),
    }
    manifest["tasks"][task] = row
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-root", type=Path, nargs="+", required=True,
                    help="P1 dump root(s); `vanilla/`, a `seeds=*` dir, a <task>/ dir, "
                         "or a <task>/dump/ dir. Several are merged per task.")
    ap.add_argument("--out-root", type=Path, required=True,
                    help="Output root of the group tree (<out-root>/<task>/<outcome>/).")
    ap.add_argument("--group", required=True, choices=sorted(GROUPS),
                    help="Robot family: fixes fps, image shape and robot_type.")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Tasks to convert (default: every task discovered).")
    ap.add_argument("--truncate-k", type=int, default=2,
                    help="Success trajectories keep first_success + 1 + k frames (decision 1).")
    ap.add_argument("--outcome-split", dest="outcome_split", action="store_true", default=True,
                    help="Write <task>/success and <task>/failure roots (default).")
    ap.add_argument("--no-outcome-split", dest="outcome_split", action="store_false",
                    help="Write a single <task>/all root (fallback plan B, mixed 66%% failure).")
    ap.add_argument("--vcodec", default="libsvtav1", help="Video codec (AV1 by default, decision 4).")
    ap.add_argument("--limit", "--limit-per-outcome", dest="limit", type=int, default=0,
                    help="At most N episodes per (task, outcome); 0 = no limit. "
                         "`--limit-per-outcome 1` converts one success and one failure. "
                         "(`--limit` is kept as an alias.)")
    ap.add_argument("--image-writer-threads", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true",
                    help="Delete <out-root>/<task>/ first. Required to re-convert a task.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only print the episode/frame plan (fast, no encoding).")
    args = ap.parse_args()
    warn_if_safe_delete_shim()

    dump_roots = [Path(r) for r in args.dump_root]
    tasks = collect_tasks(dump_roots, args.tasks)
    tasks = filter_by_group(tasks, args.group, explicit=bool(args.tasks))
    args.out_root.mkdir(parents=True, exist_ok=True)

    print(f"group={args.group} fps={GROUPS[args.group]['fps']} "
          f"shape={GROUPS[args.group]['shape']} truncate_k={args.truncate_k} "
          f"outcome_split={args.outcome_split} vcodec={args.vcodec}")
    print(f"tasks: {', '.join(sorted(tasks))}")

    manifest = {
        "group": args.group,
        "out_root": str(args.out_root.resolve()),
        "dump_roots": [str(p.resolve()) for p in dump_roots],
        "truncate_k": args.truncate_k,
        "outcome_split": args.outcome_split,
        "vcodec": args.vcodec,
        "fps": GROUPS[args.group]["fps"],
        "image_key": IMAGE_KEY,
        "state_names": STATE_NAMES,
        "action_names": ACTION_NAMES,
        "tasks": {},
    }

    for task in sorted(tasks):
        if not args.dry_run:
            existing = args.out_root / task
            if existing.exists() and not args.overwrite:
                print(f"\n== {task}: {existing} already exists, skipping "
                      f"(pass --overwrite to re-convert)", flush=True)
                continue
        row = convert_task(task, tasks[task], args.out_root, args, manifest)
        if args.dry_run:
            manifest["tasks"][task] = row["stats"]
            print(f"== {task}: {row['episodes']} eps / {row['frames']} frames kept", flush=True)

    if args.dry_run:
        ep = sum(sum(s["episodes"] for s in r.values()) for r in manifest["tasks"].values())
        fr = sum(sum(s["frames"] for s in r.values()) for r in manifest["tasks"].values())
        manifest["totals"] = {"episodes": ep, "frames": fr}
        plan_path = args.out_root / "p2_plan_dry_run.json"
        write_json(plan_path, manifest)
        print(f"\n[dry-run] {ep} episodes / {fr} frames planned -> {plan_path}")
        return 0

    manifest["totals"] = {
        "episodes": sum(sum(r.get("episodes_written", {}).values()) for r in manifest["tasks"].values()),
        "frames": sum(sum(r.get("frames_written", {}).values()) for r in manifest["tasks"].values()),
    }
    # one task per process is the recommended way to parallelise, so keep the
    # per-process manifest name distinct to avoid last-writer-wins
    name = ("p2_conversion_manifest.json" if len(manifest["tasks"]) != 1
            else f"p2_manifest_{next(iter(manifest['tasks']))}.json")
    write_json(args.out_root / name, manifest)
    print(f"\n[done] group={args.group}: {manifest['totals']['episodes']} episodes, "
          f"{manifest['totals']['frames']} frames -> {args.out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
