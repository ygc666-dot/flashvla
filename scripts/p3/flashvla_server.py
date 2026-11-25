#!/usr/bin/env python
"""G3/p4 policy server: serve a `pi0-flashvla` or `pi05-flashvla` checkpoint to SIMPLER over TCP.

Why a separate process (`3.4_todo.md` §6, `3.5_p3_execution_report.md` §10):

* SIMPLER lives in the py3.10 `sdn` env (torch 2.3.1 / numpy 1.24 / sapien);
  the checkpoint needs the py3.12 `flashvla` env (torch 2.9 / lerobot 0.5.1).
  Installing lerobot into `sdn` would break sapien, so the policy runs here and
  the simulator process talks to it. Same split as `scripts/p0/pi05_server.py`.

Protocol (length-prefixed pickle, both directions, plain Python only -- numpy
pickles are NOT portable between numpy 1.24 and 2.x):

  request  {"cmd": "ping"}
           {"cmd": "reset", "seed": int, "instruction": str}
           {"cmd": "act", "image": uint8 bytes, "shape": [H, W, 3],
            "state": [8 floats], "instruction": str}
  response {"ok": bool, "error": str|None, ...}

Inference semantics replicated from the training-time/streaming reference
(`flashvla/async_manager.py:176-217`, `AsyncStreamingActionManager`):

* one `act` == one `predict_action_chunk` == `n_action_steps` env steps;
  the checkpoint is a rolling 4-slot buffer, so a query must be issued once
  per chunk, not once per env step;
* the first `num_buffer_slots - 1` queries are cold start: the policy returns
  `None` and the framework substitutes `compute_cold_start_action` (ckpt's
  `cold_start_mode`, here `zero_delta` => hold pose, gripper 0), unnormalized
  with the checkpoint's own postprocessor;
* with `cold_start_burst` (default on) those cold queries are drained against
  the first observation before any action is emitted, so the episode does not
  spend `(slots - 1) * n_action_steps` = 12 of its 60 widowx steps on the hold
  pose. This is also closer to training, which pairs one observation with
  prefix-length buffers (`_build_per_chunk_time`). `--no-cold-burst` restores
  the step-spread behaviour for A/B;
* every request seeds torch/numpy from the episode seed at `reset`, so an
  episode outcome does not depend on how many workers share the server.

Run (from the <flashvla> root):
  CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 \
    <flashvla>/bin/python scripts/p3/flashvla_server.py \
      --ckpt /data/.../checkpoints/004000/pretrained_model --port 12404
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import socket
import struct
import time

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# wire format
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# policy
# --------------------------------------------------------------------------- #
class FlashVLAPolicyServer:
    """Wraps a FlashVLA policy (`pi0-flashvla` / `pi05-flashvla`) with chunk/cold-start bookkeeping."""

    def __init__(
        self,
        ckpt: str,
        device: str = "cuda",
        exec_steps: int | None = None,
        gripper_binarize: bool = True,
        cold_start_burst: bool = True,
        warmup: bool = True,
    ) -> None:
        # Registers the "pi0-flashvla" / "pi05-flashvla" PreTrainedConfig subclasses;
        # without it `PreTrainedConfig.from_pretrained` cannot decode config.json's `type`.
        import flashvla.configs  # noqa: F401
        from lerobot.configs.policies import PreTrainedConfig
        from flashvla.policies.factory import get_policy_class, make_pre_post_processors

        self.ckpt = ckpt
        cfg = PreTrainedConfig.from_pretrained(ckpt)
        cfg.compile_model = False  # one denoise step per query; compile buys nothing
        cfg.device = device

        # Dispatch on the checkpoint's own `type` (`pi0-flashvla` = π0*/P4 route,
        # `pi05-flashvla` = P3 route). Hard-coding the pi05 class made a pi0
        # checkpoint die in `PI05FlashVLALayer.__init__` (`config.state_cond`).
        policy_cls = get_policy_class(cfg.type)

        t0 = time.time()
        self.policy = policy_cls.from_pretrained(ckpt, config=cfg)
        self.policy.to(device)
        self.policy.eval()
        self.load_s = time.time() - t0

        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=ckpt,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.cfg = cfg
        self.device = device
        self.action_dim = int(self.policy.config.action_feature.shape[0])
        self.state_dim = int(self.policy.config.input_features["observation.state"].shape[0])
        self.n_action_steps = int(cfg.n_action_steps)
        self.exec_steps = int(exec_steps or self.n_action_steps)
        self.n_slots = int(getattr(cfg, "num_buffer_slots", 4))
        self.gripper_binarize = bool(gripper_binarize)
        self.cold_start_burst = bool(cold_start_burst)

        # Cold-start fallback, exactly as `AsyncStreamingActionManager` does it.
        from flashvla.policies.pi05.utils import compute_cold_start_action

        self._compute_cold_start_action = compute_cold_start_action
        self.cold_start_mode = getattr(cfg, "cold_start_mode", "zero_delta")
        self.cold_start_stats = getattr(self.policy, "_cold_start_stats", None)
        self._cold_normalized = None  # [1, action_dim] in normalized space, cached

        self._n_queries = 0
        self._n_cold = 0
        self.last_instruction = None

        # NOTE: n_action_steps must equal chunk_size for the buffer to advance by
        # exactly one chunk per query; a mismatch silently drops or repeats actions.
        if self.exec_steps > self.n_action_steps:
            raise ValueError(
                f"exec_steps={self.exec_steps} > n_action_steps={self.n_action_steps}"
            )

        if warmup:
            self._warmup()

    # -- lifecycle ---------------------------------------------------------- #
    def _warmup(self) -> None:
        """Walk the cold-start phase once so the first real query is not 3x slower.

        The streaming buffer only reaches `_steady_streaming` after
        `num_buffer_slots - 1` queries, so a single call would warm up the cold
        path only. State is reset afterwards, as `AsyncStreamingActionManager.
        warmup` does.
        """
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        state = np.zeros(self.state_dim, dtype=np.float32)
        self.reset(seed=0)
        t0 = time.time()
        for _ in range(int(getattr(self.cfg, "num_buffer_slots", 4)) + 1):
            self.act(img, state, "warmup")
        torch.cuda.synchronize()
        self.warmup_s = time.time() - t0
        self.reset(seed=0)

    def reset(self, seed: int = 0, instruction: str | None = None) -> dict:
        """Seed the RNG and clear the streaming buffer (mirrors setup_torch_seed)."""
        seed = int(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        self.policy.reset()
        self._n_queries = 0
        self._n_cold = 0
        self.last_instruction = instruction
        return {"ok": True, "seed": seed, "step_counter": int(self.policy._step_counter)}

    # -- inference ---------------------------------------------------------- #
    def _build_batch(self, image_u8: np.ndarray, state: np.ndarray, instruction: str) -> dict:
        # The checkpoint was trained on `observation.images.top` (the bridge
        # single camera, 480x640 uint8), so no camera remap is needed. Values are
        # fed in [0, 1]; `prepare_images` does resize_with_pad(224) + *2-1.
        img = torch.from_numpy(
            np.ascontiguousarray(image_u8, dtype=np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0)
        return {
            "observation.images.top": img,
            "observation.state": torch.as_tensor(state, dtype=torch.float32).reshape(1, -1),
            "task": [instruction],
        }

    def _cold_chunk(self) -> np.ndarray:
        """Hold-pose chunk, `exec_steps` copies, in physical (unnormalized) space."""
        if self._cold_normalized is None:
            self._cold_normalized = self._compute_cold_start_action(
                mode=self.cold_start_mode,
                stats=self.cold_start_stats,
                action_dim=self.action_dim,
                device=torch.device(self.device),
            )
        cold = self._cold_normalized.expand(1, self.exec_steps, self.action_dim)
        return self.post(cold)[0].float().cpu().numpy()

    def act(self, image_u8: np.ndarray, state: np.ndarray, instruction: str) -> dict:
        t0 = time.time()
        batch = self.pre(self._build_batch(image_u8, state, instruction))

        with torch.inference_mode():
            if self.cold_start_burst:
                # Drain the cold-start phase on this single observation before
                # emitting anything. Training gives the model ONE observation plus
                # a prefix-length buffer (`_build_per_chunk_time`), so replaying the
                # cold queries against a static scene is the in-distribution form of
                # it. Spreading them over 12 env steps instead costs 20% of a
                # 60-step widowx episode on a hold pose, and refreshes the
                # observation mid-buffer, which training never does.
                while self.policy._step_counter < self.n_slots - 1:
                    self.policy.predict_action_chunk(batch)  # cold: returns None
            chunk = self.policy.predict_action_chunk(batch)

        cold = chunk is None
        raw_gripper = None
        if cold:
            actions = self._cold_chunk()
            self._n_cold += 1
        else:
            # Unnormalize with the checkpoint's own statistics (bridge/fractal
            # mixed stats), then keep only the true action dimensionality.
            actions = self.post(chunk)[0, : self.exec_steps, : self.action_dim]
            actions = actions.float().cpu().numpy()
            raw_gripper = actions[:, -1].copy()
            if self.gripper_binarize:
                # The training action is strictly {-1, +1} on the gripper dim
                # (`simpler_adapter.py:194-199` binary threshold in the dumper),
                # so reproduce that instead of sending a continuous ~0.9.
                actions = actions.copy()
                actions[:, -1] = np.where(actions[:, -1] > 0.0, 1.0, -1.0)

        self._n_queries += 1
        self.last_instruction = instruction
        return {
            "ok": True,
            "actions": actions.tolist(),
            "gripper_raw": None if raw_gripper is None else raw_gripper.round(4).tolist(),
            "cold": bool(cold),
            "step_counter": int(self.policy._step_counter),
            "n_queries": self._n_queries,
            "latency_ms": (time.time() - t0) * 1000.0,
        }

    def info(self) -> dict:
        return {
            "ok": True,
            "ckpt": self.ckpt,
            "device": self.device,
            "type": self.cfg.type,
            "chunk_size": int(self.cfg.chunk_size),
            "n_action_steps": self.n_action_steps,
            "num_buffer_slots": int(getattr(self.cfg, "num_buffer_slots", -1)),
            "state_cond": bool(getattr(self.cfg, "state_cond", False)),
            "empty_cameras": int(getattr(self.cfg, "empty_cameras", 0)),
            "cold_start_mode": self.cold_start_mode,
            "gripper_binarize": self.gripper_binarize,
            "cold_start_burst": self.cold_start_burst,
            "exec_steps": self.exec_steps,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "n_params_B": round(sum(p.numel() for p in self.policy.parameters()) / 1e9, 3),
            "load_s": round(self.load_s, 1),
            "warmup_s": round(getattr(self, "warmup_s", 0.0), 2),
        }


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True,
                    help=".../checkpoints/<step>/pretrained_model")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--exec-steps", type=int, default=0,
                    help="actions per query (0 = the ckpt's n_action_steps)")
    ap.add_argument("--no-gripper-binarize", action="store_true",
                    help="send the continuous unnormalized gripper instead of +-1")
    ap.add_argument("--no-cold-burst", action="store_true",
                    help="spread the cold-start queries over the episode (12 hold steps "
                         "on widowx) instead of draining them on the first observation")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    server = FlashVLAPolicyServer(
        ckpt=args.ckpt,
        device=args.device,
        exec_steps=args.exec_steps or None,
        gripper_binarize=not args.no_gripper_binarize,
        cold_start_burst=not args.no_cold_burst,
        warmup=not args.no_warmup,
    )

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[server] {json.dumps(server.info())}", flush=True)
    print(f"[server] listening on {args.host}:{args.port}", flush=True)

    while True:
        conn, addr = srv.accept()
        print(f"[server] client {addr[0]}:{addr[1]}", flush=True)
        try:
            while True:
                req = recv_msg(conn)
                cmd = req.get("cmd")
                if cmd == "ping":
                    send_msg(conn, server.info())
                elif cmd == "reset":
                    send_msg(conn, server.reset(seed=req.get("seed", 0),
                                                instruction=req.get("instruction")))
                elif cmd == "act":
                    img = np.frombuffer(req["image"], dtype=np.uint8).reshape(req["shape"])
                    state = np.asarray(req["state"], dtype=np.float32).reshape(-1)
                    if state.shape[0] != server.state_dim:
                        raise ValueError(
                            f"state dim {state.shape[0]} != ckpt state_dim {server.state_dim}"
                        )
                    resp = server.act(img, state, req["instruction"])
                    send_msg(conn, resp)
                    if resp["cold"]:
                        print(f"[server] cold-start query {resp['n_queries']} "
                              f"({resp['step_counter']}/{server.cfg.num_buffer_slots})",
                              flush=True)
                else:
                    raise ValueError(f"unknown cmd {cmd!r}")
        except (ConnectionError, EOFError, BrokenPipeError):
            print("[server] client disconnected", flush=True)
        except Exception as exc:  # keep serving: one bad episode must not kill the run
            import traceback
            traceback.print_exc()
            try:
                send_msg(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            except OSError:
                pass
        finally:
            conn.close()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
