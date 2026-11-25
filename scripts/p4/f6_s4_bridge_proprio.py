#!/usr/bin/env python
"""F6/S4: build and validate the 7-D bridge proprio that π0*'s encoder expects.

π0*'s bridge mixture consumes a 7-D ``POS_EULER`` proprio
(``[x, y, z, roll, pitch, yaw, grip]``), our datasets carry the 8-D raw
``eef_pos`` (``[x, y, z, qw, qx, qy, qz, grip]``). `utils.bridge_proprio_from_state`
does the conversion inside the processor; this script

1. re-derives the same numbers with π0*'s own ``quat2mat`` / ``mat2euler``
   (loaded straight out of ``third_party/.../open_pi_zero/src/utils/geometry.py``)
   and asserts the two agree element-wise, so the torch port is not "probably
   right";
2. measures the angle ranges, i.e. checks the assumption that makes a plain
   QUANTILES/percentile normalization legal (no +-pi wraparound in this data);
3. writes the two candidate 7-D normalization-stat sets:
   * ``--stats-out``: q01/q99/mean/std/min/max of *our* leaves (keeps the
     project's QUANTILES 口径, the default for the S4 arm);
   * ``--pi0-stats-out``: π0*'s own p01/p99 from
     ``open_pi_zero/config/bridge_statistics.json`` (the distribution the
     pretrained encoder actually saw; kept for the follow-up arm).

Usage::

    <env>/bin/python scripts/p4/f6_s4_bridge_proprio.py \
        --root /data/zjr/simpler_lerobot/bridge \
        --json logs/p4/f6_s4_bridge_proprio.json
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GEOMETRY = (
    REPO_ROOT
    / "third_party/SDN/simpler_env/policies/pizero/open_pi_zero/src/utils/geometry.py"
)
PI0_BRIDGE_STATS = (
    REPO_ROOT
    / "third_party/SDN/simpler_env/policies/pizero/open_pi_zero/config/bridge_statistics.json"
)

DIMS = ("x", "y", "z", "roll", "pitch", "yaw", "grip")


def load_geometry():
    """Import π0*'s geometry.py by path (its package has no __init__.py)."""
    spec = importlib.util.spec_from_file_location("pi0star_geometry", GEOMETRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_proprio(states: np.ndarray, geometry) -> np.ndarray:
    """π0* `BridgeSimplerAdapter.preprocess_proprio`, frame by frame, float64."""
    rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
    out = np.empty((states.shape[0], 7), dtype=np.float64)
    for i, s in enumerate(np.asarray(states, dtype=np.float64)):
        rpy = geometry.mat2euler(geometry.quat2mat(s[3:7]) @ rot.T)
        out[i, :3] = s[:3]
        out[i, 3:6] = rpy
        out[i, 6] = s[7]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/zjr/simpler_lerobot/bridge")
    ap.add_argument("--leaves", nargs="*", default=None, help="default: every */success")
    ap.add_argument("--json", default=None)
    ap.add_argument(
        "--stats-out", default="logs/p4/f6_s4_bridge_state7_quantiles.json"
    )
    ap.add_argument(
        "--pi0-stats-out", default="logs/p4/f6_s4_bridge_state7_pi0bound.json"
    )
    args = ap.parse_args()

    import pandas as pd

    from flashvla.policies.pi0.utils import bridge_proprio_from_state

    geometry = load_geometry()
    leaves = args.leaves or sorted(glob.glob(str(Path(args.root) / "*" / "success")))
    if not leaves:
        raise SystemExit(f"no leaves under {args.root}")

    per_leaf = {}
    states = []
    for leaf in leaves:
        files = sorted(glob.glob(str(Path(leaf) / "data" / "**" / "*.parquet"), recursive=True))
        if not files:
            raise SystemExit(f"no parquet under {leaf}")
        col = pd.concat([pd.read_parquet(f, columns=["observation.state"]) for f in files])
        arr = np.stack(col["observation.state"].values).astype(np.float64)
        per_leaf[Path(leaf).parent.name] = {"frames": int(arr.shape[0])}
        states.append(arr)
    states = np.concatenate(states)
    print(f"loaded {states.shape[0]} frames from {len(leaves)} leaves")

    # 1) torch port vs π0*'s reference, float64
    ref = reference_proprio(states, geometry)
    ours = bridge_proprio_from_state(torch.from_numpy(states)).numpy()
    delta = np.abs(ours - ref)
    per_dim_delta = {d: float(delta[:, i].max()) for i, d in enumerate(DIMS)}
    worst = float(delta.max())
    print(f"max |torch - pi0star| = {worst:.3e}")
    for d, v in per_dim_delta.items():
        print(f"  {d:>5}: {v:.3e}")
    tol = 1e-6
    ok = worst < tol

    # 2) ranges: a plain percentile normalization needs no +-pi wraparound
    ranges = {}
    for i, d in enumerate(DIMS):
        col = ref[:, i]
        ranges[d] = {
            "min": float(col.min()),
            "max": float(col.max()),
            "q01": float(np.percentile(col, 1)),
            "q99": float(np.percentile(col, 99)),
            "mean": float(col.mean()),
            "std": float(col.std()),
            "n_abs_gt_2": int((np.abs(col) > 2.0).sum()),
        }
    for d in ("roll", "pitch", "yaw"):
        if ranges[d]["n_abs_gt_2"]:
            raise SystemExit(
                f"{d} reaches |angle| > 2 rad ({ranges[d]['n_abs_gt_2']} frames): "
                "percentile normalization of a wrapping angle is not safe here"
            )
    print("ranges (no |angle| > 2 rad, so percentile normalization is safe):")
    for d in DIMS:
        r = ranges[d]
        print(f"  {d:>5}: [{r['min']:+.3f}, {r['max']:+.3f}] q01/q99 {r['q01']:+.4f}/{r['q99']:+.4f}")

    # 3a) our leaves' stats (QUANTILES: q01/q99 is all the normalizer needs)
    our_stats = {
        k: [round(float(x), 8) for x in vals]
        for k, vals in (
            ("q01", np.percentile(ref, 1, axis=0)),
            ("q99", np.percentile(ref, 99, axis=0)),
            ("mean", ref.mean(axis=0)),
            ("std", ref.std(axis=0)),
            ("min", ref.min(axis=0)),
            ("max", ref.max(axis=0)),
        )
    }
    our_stats["_dims"] = list(DIMS)
    our_stats["_source"] = (
        f"{states.shape[0]} frames from "
        + ", ".join(sorted(per_leaf))
    )
    Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_out).write_text(json.dumps(our_stats, indent=2))
    print(f"stats -> {args.stats_out}")

    # 3b) π0*'s own bound stats, expressed in the keys the normalizer reads
    pi0 = json.loads(PI0_BRIDGE_STATS.read_text())["proprio"]
    pi0_stats = {
        "q01": [round(float(x), 8) for x in pi0["p01"]],
        "q99": [round(float(x), 8) for x in pi0["p99"]],
        "mean": [round(float(x), 8) for x in pi0["mean"]],
        "std": [round(float(x), 8) for x in pi0["std"]],
        "min": [round(float(x), 8) for x in pi0["min"]],
        "max": [round(float(x), 8) for x in pi0["max"]],
        "_dims": list(DIMS),
        "_source": f"pi0* {PI0_BRIDGE_STATS.name} (2195527 bridge transitions)",
        "_note": "p01/p99 exposed as q01/q99: identical to openpi's `bound` mode",
    }
    Path(args.pi0_stats_out).write_text(json.dumps(pi0_stats, indent=2))
    print(f"pi0* bound stats -> {args.pi0_stats_out}")

    span_ratio = (np.array(our_stats["q99"]) - np.array(our_stats["q01"])) / (
        np.array(pi0_stats["q99"]) - np.array(pi0_stats["q01"])
    )
    print("our q01..q99 span / pi0* p01..p99 span: " + " ".join(
        f"{d}={r:.3f}" for d, r in zip(DIMS, span_ratio)
    ))

    report = {
        "leaves": per_leaf,
        "frames": int(states.shape[0]),
        "alignment": {
            "reference": str(GEOMETRY.relative_to(REPO_ROOT)),
            "tolerance": tol,
            "max_abs_delta": worst,
            "per_dim_max_abs_delta": per_dim_delta,
            "ok": bool(ok),
        },
        "ranges": ranges,
        "span_ratio_ours_over_pi0": {d: float(r) for d, r in zip(DIMS, span_ratio)},
        "stats_out": str(args.stats_out),
        "pi0_stats_out": str(args.pi0_stats_out),
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"report -> {args.json}")
    if not ok:
        print("\nALIGNMENT FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
