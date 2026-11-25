#!/usr/bin/env python
"""P0 client: drive SIMPLER (py3.10 `sdn` env) with actions from `pi05_server.py`.

Mirrors `third_party/SDN/policy/pizero/evaluate.py:188-262` (same reset/step/success
bookkeeping) so the numbers are comparable with the R1 SDN runs in
`docs/sdn_flashvla/sdn/results.md`. Differences, all deliberate for P0:
  * the policy is a TCP client instead of an in-process pi0*;
  * every `--exec-steps` env steps the latest observation is re-sent (no chunk cache),
    matching pi0*'s act_steps=4 execution pattern;
  * episodes are NOT cut early on success (same as R1), so `first_success` is comparable.

Run (from <flashvla>/third_party/SDN):
  CUDA_VISIBLE_DEVICES=4 <sdn>/bin/python ../../scripts/p0/simpler_client_eval.py \
      --task google_robot_pick_coke_can --n-episodes 5 --start-seed 100 --port 12345
"""
from __future__ import annotations

import argparse
import json
import pickle
import socket
import struct
import time
from pathlib import Path

import numpy as np

SDN_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "SDN"
import sys
if str(SDN_ROOT) not in sys.path:
    sys.path.insert(0, str(SDN_ROOT))

import simpler_env  # noqa: E402


def get_camera_name(env) -> str:
    uid = env.unwrapped.robot_uid
    if "google_robot" in uid:
        return "overhead_camera"
    if "widowx" in uid:
        return "3rd_view_camera"
    raise NotImplementedError(uid)


def recv_msg(conn) -> dict:
    raw = b""
    while len(raw) < 4:
        c = conn.recv(4 - len(raw))
        if not c:
            raise ConnectionError("server closed")
        raw += c
    (n,) = struct.unpack(">I", raw)
    buf = b""
    while len(buf) < n:
        c = conn.recv(min(1 << 20, n - len(buf)))
        if not c:
            raise ConnectionError("server closed mid-message")
        buf += c
    return pickle.loads(buf)


def send_msg(conn, obj) -> None:
    payload = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack(">I", len(payload)) + payload)


def run_episode(env, conn, cam, episode, exec_steps: int, debug: bool = False) -> dict:
    obs, _ = env.reset(seed=episode, options=dict(reconfigure=True))
    instruction = env.unwrapped.get_language_instruction()

    timestep = 0
    first_success = -1
    any_success = False
    truncated = False
    latencies = []
    all_a = []
    states = []
    steps_since_query = exec_steps  # force a query on the first step

    while not truncated:
        if steps_since_query >= exec_steps:
            image = np.asarray(obs["image"][cam]["rgb"], dtype=np.uint8)
            state = np.asarray(obs["agent"]["eef_pos"], dtype=np.float32).reshape(-1)[:8]
            # Plain-Python payload only: numpy pickles are not portable between the
            # numpy 1.24 (`sdn`) and numpy 2.x (`flashvla`) environments.
            send_msg(conn, {
                "image": image.tobytes(),
                "shape": tuple(image.shape),
                "state": state.tolist(),
                "instruction": instruction,
                "seed": episode,
                "domain": "fractal" if "google_robot" in env.unwrapped.robot_uid else "bridge",
            })
            resp = recv_msg(conn)
            action_chunk = np.asarray(resp["actions"], dtype=np.float32)
            latencies.append(float(resp["latency_ms"]))
            if debug:
                all_a.append(action_chunk.copy())
                states.append(state.copy())
            steps_since_query = 0

        a = action_chunk[steps_since_query]
        command = np.concatenate([a[0:3], a[3:6], a[6:7]])
        obs, _reward, _term, truncated, info = env.step(command)
        steps_since_query += 1
        timestep += 1

        if bool(info.get("success", False)):
            any_success = True
            if first_success < 0:
                first_success = timestep

        new_instruction = env.unwrapped.get_language_instruction()
        if new_instruction != instruction:
            instruction = new_instruction

    out = {
        "episode": episode,
        "success": any_success,
        "first_success": first_success,
        "steps": timestep,
        "query_ms_mean": float(np.mean(latencies)) if latencies else 0.0,
        "n_queries": len(latencies),
    }
    if debug and all_a:
        A = np.concatenate(all_a, axis=0)  # [n_queries*exec, 7]
        S = np.stack(states, axis=0)       # [n_queries, 8]
        out["action_absmean_per_dim"] = np.abs(A).mean(0).round(4).tolist()
        out["action_std_per_dim"] = A.std(0).round(4).tolist()
        out["action_first_row"] = A[0].round(4).tolist()
        out["action_last_row"] = A[-1].round(4).tolist()
        out["gripper_unique"] = int(len(np.unique(np.round(A[:, 6], 2))))
        out["state_std_per_dim"] = S.std(0).round(4).tolist()
        out["eef_travel_l2"] = float(np.linalg.norm(S[-1][:3] - S[0][:3]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--start-seed", type=int, default=100)
    ap.add_argument("--port", type=int, default=12345)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--exec-steps", type=int, default=4)
    ap.add_argument("--debug-actions", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    env = simpler_env.make(args.task)
    cam = get_camera_name(env)
    print(f"[client] task={args.task} cam={cam} seeds={args.start_seed}"
          f"..{args.start_seed + args.n_episodes - 1}", flush=True)

    conn = socket.create_connection((args.host, args.port), timeout=600)
    results = []
    for i in range(args.n_episodes):
        ep = args.start_seed + i
        t0 = time.time()
        r = run_episode(env, conn, cam, ep, args.exec_steps, debug=args.debug_actions)
        r["walltime_s"] = round(time.time() - t0, 1)
        results.append(r)
        print(f"[client] ep {ep}: success={r['success']} first_success={r['first_success']} "
              f"steps={r['steps']} q={r['query_ms_mean']:.0f}ms wall={r['walltime_s']}s", flush=True)

    conn.close()
    rate = np.mean([r["success"] for r in results]) * 100
    summary = {
        "task": args.task,
        "n_episodes": args.n_episodes,
        "start_seed": args.start_seed,
        "success_rate": float(rate),
        "n_success": int(sum(r["success"] for r in results)),
        "episodes": results,
    }
    print(f"[client] {args.task}: {summary['n_success']}/{args.n_episodes} = {rate:.1f}%", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
