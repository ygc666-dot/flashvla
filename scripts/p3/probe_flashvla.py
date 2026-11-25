#!/usr/bin/env python
"""Offline gate for the FlashVLA G3 adapter (`3_5_p3_execution_report.md` §10.2).

Three checks, all runnable without the simulator (~40 s after load):

  P1 cold start    the first `num_buffer_slots - 1` queries return the hold pose
                   (all-zero action: cold_start_mode=zero_delta, unnormalized)
  P2 steady chunk  from query `num_buffer_slots` on, the action is finite, non-zero,
                   varies inside the chunk, and two consecutive queries differ
                   (i.e. the 4 actions are not one resampled vector)
  P3 state_cond    with an identical RNG stream, changing only `observation.state`
                   must change the action -- this is the open question left in
                   §3.2 (are the 6 `state_cond` tensors actually used?) and a
                   control run (same state twice) must reproduce bit-exactly

P3 is the decisive one: the `state_mlp_out` projections are zero-initialized when
`ckpt/pi05_base` lacks them, so "identical action for a different state" would
mean the FiLM never learned a non-zero contribution and G3 would be measuring a
state-blind policy.

Run (flashvla env, no GPU contention needed beyond one card):
  CUDA_VISIBLE_DEVICES=4 <flashvla>/bin/python scripts/p3/probe_flashvla.py \
      --ckpt /data/.../checkpoints/004000/pretrained_model
  # or against a running server:
  ... scripts/p3/probe_flashvla.py --port 12404
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def make_shim(ckpt: str | None, port: int | None):
    """Return something with reset()/step(); the same surface for both transports."""
    if port is not None:
        sys.path.insert(0, "/home/zjr/workspace_tj/StreamingVLA/flashvla/third_party/SDN")
        from policy.pizero.flashvla_client import FlashVLAInference

        return FlashVLAInference(port=port, exec_steps=4), "tcp"

    import os.path as osp

    sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
    from flashvla_server import FlashVLAPolicyServer

    class _Local:
        """Same surface as the TCP client, without the socket."""

        def __init__(self):
            self.srv = FlashVLAPolicyServer(ckpt=ckpt, warmup=False)
            self.info = self.srv.info()
            self.last_was_cold = False
            self.last_gripper_raw = None

        def reset(self, instruction, seed=None):
            return self.srv.reset(seed=int(seed or 0), instruction=instruction)

        def step(self, image, instruction, proprio):
            resp = self.srv.act(np.asarray(image), np.asarray(proprio, np.float32), instruction)
            self.last_was_cold = bool(resp["cold"])
            self.last_gripper_raw = resp.get("gripper_raw")
            return np.asarray(resp["actions"], np.float32), resp

    return _Local(), "in-process"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help=".../checkpoints/<step>/pretrained_model")
    ap.add_argument("--port", type=int, default=None, help="probe a running server instead")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="", help="write the JSON report here")
    args = ap.parse_args()
    if (args.ckpt is None) == (args.port is None):
        raise SystemExit("pass exactly one of --ckpt / --port")

    policy, transport = make_shim(args.ckpt, args.port)
    info = policy.info
    n_slots = int(info["num_buffer_slots"])
    n_steps = int(info["n_action_steps"])
    action_dim = int(info["action_dim"])
    print(f"[probe] {transport} {info['type']} ckpt={info['ckpt']}")
    print(f"[probe] slots={n_slots} chunk={info['chunk_size']} exec={n_steps} "
          f"state_cond={info['state_cond']} empty_cams={info['empty_cameras']} "
          f"cold={info['cold_start_mode']} binarize={info['gripper_binarize']}")

    rng = np.random.RandomState(0)
    image = rng.randint(60, 200, size=(480, 640, 3)).astype(np.uint8)  # textured, not OOD-black
    instruction = "put the cube on top of the other cube"
    state_a = np.array([-0.02, 0.11, 0.28, 1.0, 0.0, 0.0, 0.0, 0.85], dtype=np.float32)
    state_b = np.array([0.13, -0.07, 0.05, 0.5, 0.5, 0.5, 0.5, 0.10], dtype=np.float32)

    report: dict = {"transport": transport, "info": info, "checks": {}}

    def query(state, seed=args.seed, warm=None):
        """reset, run the cold-start phase, then return the first steady chunk."""
        policy.reset(instruction, seed=seed)
        cold = []
        for _ in range(max(0, n_slots - 1)):
            raw, _ = policy.step(image, instruction, warm if warm is not None else state)
            cold.append(np.asarray(raw, np.float32))
        raw, _ = policy.step(image, instruction, state)
        return np.asarray(raw, np.float32), cold

    # -- P1 cold start ----------------------------------------------------- #
    cold_actions = []
    policy.reset(instruction, seed=args.seed)
    for _ in range(n_slots - 1):
        raw, _ = policy.step(image, instruction, state_a)
        cold_actions.append(np.asarray(raw, np.float32))
    cold_arr = np.stack(cold_actions)  # [n_slots-1, n_steps, 7]
    # Not exactly 0: the hold pose is unnormalized from a zero *normalized*
    # action, which costs ~1e-10 in float32 round-off.
    p1_ok = bool(np.abs(cold_arr).max() < 1e-6)
    report["checks"]["P1_cold_start_hold_pose"] = {
        "ok": p1_ok,
        "max_abs": float(np.abs(cold_arr).max()),
        "shape": list(cold_arr.shape),
    }
    print(f"[P1] cold start ({n_slots - 1} queries) all-zero hold pose: "
          f"{p1_ok} (max|a|={np.abs(cold_arr).max():.3e})")

    # -- P2 steady chunk --------------------------------------------------- #
    # Two CONSECUTIVE queries of one episode (no reset in between): the rolling
    # buffer must shift, so the executed actions must change. Replaying the same
    # episode from the same seed reproduces it bit-exactly (that is P3's control).
    policy.reset(instruction, seed=args.seed)
    for _ in range(n_slots - 1):
        policy.step(image, instruction, state_a)
    a_first, _ = policy.step(image, instruction, state_a)
    a_first = np.asarray(a_first, np.float32)
    gripper_raw_1 = getattr(policy, "last_gripper_raw", None)
    a_second, _ = policy.step(image, instruction, state_a)
    a_second = np.asarray(a_second, np.float32)

    finite = bool(np.isfinite(a_first).all() and np.isfinite(a_second).all())
    # Non-zero translation/rotation, not just a stuck gripper.
    motion = float(np.abs(a_first[:, :6]).mean())
    within_chunk = float(a_first[:, :6].std(axis=0).mean())  # 0 => one vector repeated
    between = float(np.abs(a_first - a_second).max())         # 0 => buffer not shifting
    p2_ok = bool(finite and motion > 1e-4 and within_chunk > 1e-5 and between > 1e-5)
    report["checks"]["P2_steady_chunk"] = {
        "ok": p2_ok,
        "finite": finite,
        "absmean_xyzrpy": round(motion, 6),
        "within_chunk_std": round(within_chunk, 6),
        "consecutive_query_maxdiff": round(between, 6),
        "gripper_binarized": a_first[:, 6].round(3).tolist(),
        "gripper_raw": gripper_raw_1,
        "action_first_row": a_first[0].round(4).tolist(),
        "action_last_row": a_first[-1].round(4).tolist(),
        "absmean_per_dim": np.abs(a_first).mean(axis=0).round(4).tolist(),
        "absmax_per_dim": np.abs(a_first).max(axis=0).round(4).tolist(),
    }
    print(f"[P2] steady chunk finite={finite} |xyzrpy|={motion:.4f} "
          f"within-chunk std={within_chunk:.4f} consecutive-query diff={between:.4f} -> {p2_ok}")
    print(f"     chunk[0]={a_first[0].round(3).tolist()}  chunk[-1]={a_first[-1].round(3).tolist()}")
    print(f"     gripper raw={gripper_raw_1} -> binarized {a_first[:, 6].round(2).tolist()}")

    # -- P3 state_cond sensitivity ----------------------------------------- #
    # Same seed => identical noise stream, so any difference comes from the state.
    aA, _ = query(state_a, warm=state_a)
    aB, _ = query(state_b, warm=state_a)  # warm up on the SAME state as A
    aA2, _ = query(state_a, warm=state_a)
    control = float(np.abs(aA - aA2).max())  # determinism check, must be 0
    delta = float(np.abs(aA - aB).max())
    rel = float(np.abs(aA - aB).mean() / max(np.abs(aA).mean(), 1e-9))
    p3_ok = bool(control == 0.0 and delta > 1e-5)
    report["checks"]["P3_state_cond_sensitivity"] = {
        "ok": p3_ok,
        "determinism_control_maxdiff": control,
        "state_ab_maxdiff": round(delta, 6),
        "state_ab_relmean": round(rel, 4),
        "state_a_action0": aA[0].round(4).tolist(),
        "state_b_action0": aB[0].round(4).tolist(),
    }
    print(f"[P3] state_cond: control diff={control:.3e} (must be 0), "
          f"state A vs B maxdiff={delta:.4e} rel={rel:.3f} -> {p3_ok}")
    print(f"     A[0]={aA[0].round(3).tolist()}")
    print(f"     B[0]={aB[0].round(3).tolist()}")

    all_ok = all(v["ok"] for v in report["checks"].values())
    report["all_ok"] = all_ok
    print(f"[probe] {'PASS' if all_ok else 'FAIL'} "
          f"({sum(v['ok'] for v in report['checks'].values())}/3 checks)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
