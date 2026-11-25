#!/usr/bin/env python
"""P0 policy server: run a FlashVLA-family pi05 checkpoint in the py3.12 `flashvla` env and
serve action chunks over TCP, so that SIMPLER (py3.10 `sdn` env) can drive it without
installing lerobot/torch2.9 into the simulator env.

Protocol (length-prefixed pickle, both directions):
  request  = {"image": uint8 HxWx3, "state": float32[8], "instruction": str, "seed": int}
  response = {"actions": float32[A,7], "latency_ms": float}

Run (from <flashvla> root):
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=4 \
    python scripts/p0/pi05_server.py --ckpt ckpt/pi05_libero --port 12345 --exec-steps 4
"""
from __future__ import annotations

import argparse
import pickle
import socket
import struct
import time

import json
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from flashvla.policies.factory import make_pre_post_processors
from flashvla.policies.pi05.modeling_pi05 import PI05Policy


def recv_msg(conn) -> dict:
    raw = b""
    while len(raw) < 4:
        chunk = conn.recv(4 - len(raw))
        if not chunk:
            raise ConnectionError("peer closed")
        raw += chunk
    (n,) = struct.unpack(">I", raw)
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError("peer closed mid-message")
        buf += chunk
    return pickle.loads(buf)


def send_msg(conn, obj) -> None:
    payload = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack(">I", len(payload)) + payload)


def build_batch(req: dict, duplicate_view: bool):
    # NOTE: numpy objects are NOT pickled across processes -- the `sdn` env has numpy 1.24
    # while `flashvla` has numpy 2.x, and numpy's pickle module path changed (`_core` vs
    # `core`). Only plain Python objects cross the wire.
    img = np.frombuffer(req["image"], dtype=np.uint8).reshape(req["shape"]).astype(np.float32) / 255.0
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    batch = {
        "observation.images.image": x,
        "observation.state": torch.tensor(req["state"], dtype=torch.float32).unsqueeze(0),
        "task": [req["instruction"]],
    }
    # SIMPLER exposes a single camera; the checkpoint expects a second (wrist) view.
    # Feeding a duplicate is the *optimistic* choice: if even this fails, the honest
    # (blank second view) variant cannot do better.
    if duplicate_view:
        batch["observation.images.image2"] = x.clone()
    return batch


def load_rescale_tables(ckpt: str, stats_dir: str) -> dict:
    """Build per-domain affine maps from LIBERO-normalized space to SIMPLER deltas.

    pi05_libero was trained with MEAN_STD on LIBERO stats, so its postprocessed output is a
    physical delta whose *scale* is LIBERO's (xyz std ~0.34-0.44 m) while SIMPLER deltas are
    5-30x smaller. pi0* on SIMPLER unnormalizes with SIMPLER stats instead, so mapping
    ``u = (a_libero - mean_L) / std_L; a_simpler = u * std_S + mean_S`` is the fair variant.
    """
    from safetensors.torch import load_file

    sd = load_file(str(Path(ckpt) / "policy_preprocessor_step_2_normalizer_processor.safetensors"))
    mean_L = np.asarray(sd["action.mean"].flatten().tolist(), dtype=np.float32)[:7]
    std_L = np.maximum(np.asarray(sd["action.std"].flatten().tolist(), dtype=np.float32)[:7], 1e-6)

    tables = {}
    for domain in ("fractal", "bridge"):
        with open(Path(stats_dir) / f"{domain}_statistics.json") as f:
            st = json.load(f)["action"]
        tables[domain] = {
            "mean_L": mean_L,
            "std_L": std_L,
            "mean_S": np.asarray(st["mean"], dtype=np.float32)[:7],
            "std_S": np.maximum(np.asarray(st["std"], dtype=np.float32)[:7], 1e-6),
        }
    return tables


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/pi05_libero")
    ap.add_argument("--port", type=int, default=12345)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--exec-steps", type=int, default=4, help="actions returned per request")
    ap.add_argument("--no-dup-view", action="store_true", help="do not duplicate the single camera")
    ap.add_argument("--clip-state", type=float, default=1.0,
                    help="clip normalized state into [-v,v] before prompt discretization; <=0 disables")
    ap.add_argument("--clip-action", type=float, default=0.0,
                    help="clip postprocessed delta actions into [-v,v]; <=0 disables")
    ap.add_argument("--rescale-stats", action="store_true",
                    help="re-express output deltas with SIMPLER's per-dim action mean/std "
                         "(fair variant: pi0* itself unnormalizes with SIMPLER stats)")
    ap.add_argument("--simpler-stats-dir",
                    default=str((Path(__file__).resolve().parents[2] / "third_party" / "SDN" /
                                 "simpler_env" / "policies" / "pizero" / "open_pi_zero" / "config")))
    args = ap.parse_args()

    cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    cfg.compile_model = False
    cfg.device = "cuda"
    t0 = time.time()
    policy = PI05Policy.from_pretrained(args.ckpt, config=cfg)
    pre, post = make_pre_post_processors(cfg, pretrained_path=args.ckpt)
    n_param = sum(p.numel() for p in policy.parameters()) / 1e9
    rescale = load_rescale_tables(args.ckpt, args.simpler_stats_dir) if args.rescale_stats else None
    print(f"[server] loaded {type(cfg).__name__} {n_param:.2f}B in {time.time() - t0:.1f}s "
          f"(chunk={cfg.chunk_size} exec_steps={args.exec_steps} "
          f"dup_view={not args.no_dup_view} clip_state={args.clip_state} "
          f"rescale={args.rescale_stats})", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[server] listening on {args.host}:{args.port}", flush=True)

    n_calls = 0
    while True:
        conn, addr = srv.accept()
        print(f"[server] client {addr}", flush=True)
        try:
            while True:
                req = recv_msg(conn)
                seed = int(req.get("seed", 0))
                torch.manual_seed(seed * 100000 + n_calls * 7919)
                n_calls += 1

                batch = pre(build_batch(req, duplicate_view=not args.no_dup_view))
                if args.clip_state > 0:
                    batch["observation.state"] = batch["observation.state"].clamp(
                        -args.clip_state, args.clip_state)

                t1 = time.time()
                with torch.no_grad():
                    chunk = policy.predict_action_chunk(batch)
                torch.cuda.synchronize()
                dt_ms = (time.time() - t1) * 1000.0

                act = post(chunk)[0, : args.exec_steps, :7].float().cpu().numpy()
                if rescale is not None:
                    tb = rescale[req.get("domain", "fractal")]
                    u = (act - tb["mean_L"]) / tb["std_L"]
                    act = (u * tb["std_S"] + tb["mean_S"]).astype(np.float32)
                if args.clip_action > 0:
                    act = np.clip(act, -args.clip_action, args.clip_action)
                send_msg(conn, {"actions": act.tolist(), "latency_ms": dt_ms})
        except (ConnectionError, EOFError, BrokenPipeError):
            print("[server] client disconnected", flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
