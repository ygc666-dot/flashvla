"""M0 离线自检：G1 逐位一致性 + N>1 路径可用性 + N/M 成本剖面。不属于仓库代码。

两个必须知道的点（2026-11-19 实测踩到）：

1. ``ckpt/pi05_libero/config.json`` 里 ``compile_model=True`` + ``compile_mode=max-autotune``。
   直接 ``PI05Policy.from_pretrained`` 会拿到编译版：**每换一个 batch size 就要
   autotune 约 5–10 分钟**（N=4 就把脚本拖过了 20 分钟超时）。而 ``eval.sh`` 的
   ``OVERLAP_STEPS=0`` 会显式传 ``--policy.compile_model=false``，所以 R3 真实
   rollout 走的是**未编译**路径。本脚本用 ``_load_policy(compile_model=False)``
   对齐真实路径，跑完约 2 分钟。

2. CUDA Graph 会**覆盖上一次输出张量**：若开 compile，比较前必须 ``.clone()``，
   否则报 ``accessing tensor output of CUDAGraphs that has been overwritten``。
"""

import json
import tempfile
import time

import draccus
import torch

from flashvla.policies.pi05 import PI05Config, PI05Policy
from flashvla.policies.pi05.sdn_selector import SDNChunkSelector, attach_sdn

CKPT = "ckpt/pi05_libero"
DEV = "cuda"
REPEAT = 5


def _load_policy(**overrides):
    """按 lerobot 的解析路径构造 config（先摘掉 ``type`` 键），再按需覆盖字段。

    直接 ``PI05Config.from_pretrained`` 会因 config.json 里的 ``type: pi05``
    报 ``DecodingError: The fields `type` are not valid``。
    """
    with open(f"{CKPT}/config.json") as f:
        raw = json.load(f)
    raw.pop("type", None)
    raw.update(overrides)
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
        json.dump(raw, f)
        tmp = f.name
    with draccus.config_type("json"):
        cfg = draccus.parse(PI05Config, tmp, args=[])
    policy = PI05Policy(cfg)
    policy = PI05Policy._load_as_safetensor(policy, f"{CKPT}/model.safetensors", cfg.device, False)
    return policy.to(cfg.device).eval()


def _time(fn, repeat=REPEAT):
    for _ in range(2):  # warmup（含首块偶发的一次性分配）
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat


policy = _load_policy(n_action_steps=5, compile_model=False)  # 对齐 eval.sh ov0
cfg = policy.config
print(
    f"loaded type={cfg.type} chunk_size={cfg.chunk_size} n_action_steps={cfg.n_action_steps} "
    f"max_action_dim={cfg.max_action_dim} num_inference_steps={cfg.num_inference_steps} "
    f"compile_model={cfg.compile_model}"
)

bsz = 1
batch = {
    "observation.images.image": torch.rand(bsz, 3, 256, 256, device=DEV),
    "observation.images.image2": torch.rand(bsz, 3, 256, 256, device=DEV),
    "observation.images.empty_camera_0": torch.zeros(bsz, 3, 224, 224, device=DEV),
    "observation.state": torch.rand(bsz, 8, device=DEV),
    "observation.language.tokens": torch.tensor([[2, 100, 200, 300]], device=DEV),
    "observation.language.attention_mask": torch.ones(bsz, 4, dtype=torch.bool, device=DEV),
}

# ---------- G1: N=1 必须与基线逐位一致 ----------
noise = torch.randn(bsz, cfg.chunk_size, cfg.max_action_dim, device=DEV)
base = policy.predict_action_chunk(dict(batch), noise=noise.clone()).clone()
sel1 = SDNChunkSelector(policy, num_samples=1)
wrapped = sel1.predict_action_chunk(dict(batch), noise=noise.clone()).clone()
print(
    f"G1 base{tuple(wrapped.shape)} bitwise_equal={torch.equal(base, wrapped)} "
    f"max|diff|={(base - wrapped).abs().max().item():.3e}"
)
assert torch.equal(base, wrapped), "G1 失败：N=1 未逐位一致"

# ---------- attach_sdn: 必须保持 isinstance(policy, PreTrainedPolicy) 且 N=1 不自递归 ----------
# 复用同一个 policy（不加载第二份 7 GB 权重）：patch 后 N=1 走 _base_predict，
# N>1 分支直接调 model.sample_actions，两者都不受这次 patch 影响。
from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402

attach_sdn(policy, num_samples=1)
print(f"attach_sdn isinstance(policy, PreTrainedPolicy)={isinstance(policy, PreTrainedPolicy)}")
assert isinstance(policy, PreTrainedPolicy), "attach_sdn 破坏了 PreTrainedPolicy 类型"
out_p = policy.predict_action_chunk(dict(batch), noise=noise.clone()).clone()
print(
    f"attach_sdn N=1 透传 out{tuple(out_p.shape)} finite={torch.isfinite(out_p).all().item()} "
    f"same_as_base={torch.equal(out_p, base)}"
)
assert torch.equal(out_p, base), "attach_sdn N=1 未透传（可能自递归或改了路径）"

# ---------- N>1: 形状 / 有效延迟 / 峰值显存 ----------
t_base = _time(lambda: policy.predict_action_chunk(dict(batch)))
print(f"N=1 baseline latency={1000 * t_base:.1f} ms/chunk")
rows = []
for n in (4, 8, 12):
    torch.cuda.reset_peak_memory_stats()
    sel = SDNChunkSelector(policy, num_samples=n)
    out = sel.predict_action_chunk(dict(batch))
    dt = _time(lambda: sel.predict_action_chunk(dict(batch)))
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"N={n:2d} out{tuple(out.shape)} dtype={out.dtype} finite={torch.isfinite(out).all().item()} "
        f"latency={1000 * dt:7.1f} ms  x{dt / t_base:4.2f}  peak_alloc={peak:.2f} GiB"
    )
    assert out.shape == (bsz, cfg.n_action_steps, cfg.action_feature.shape[0])
    rows.append((n, dt / t_base))
    del sel, out
    torch.cuda.empty_cache()

# ---------- 复算候选，确认 argmin 真的选了最平滑者（而非恒选 0）----------
n = 4  # 复用上面已跑过的 batch size，避免重复分配
sel = SDNChunkSelector(policy, num_samples=n)
tiled = {k: (torch.cat([v] * n, dim=0) if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == bsz else v) for k, v in batch.items()}
images, img_masks = policy.prepare_images(tiled)
state = policy.prepare_state(tiled)
noise_all = policy.model.sample_noise((n * bsz, cfg.chunk_size, cfg.max_action_dim), tiled["observation.language.tokens"].device)
acts, _ = policy.model.sample_actions(
    images, img_masks, tiled["observation.language.tokens"],
    tiled["observation.language.attention_mask"], state, noise=noise_all,
)
dim = cfg.action_feature.shape[0]
cand = acts[..., :dim].view(n, bsz, acts.shape[-2], dim)[:, 0].clone()
from sdn.selection import smoothness_scores  # noqa: E402

sc = smoothness_scores(cand, lambda_energy=0.05, ignore_gripper=True)
print(f"candidate scores={[round(float(s), 4) for s in sc]}")
print(f"argmin={int(sc.argmin())} candidate_std={float(cand.std(dim=0).mean()):.5f}")
# 注入同一份 noise_all：否则 sel 内部会重新采噪声，与 cand 不是同一组候选（G2 校验无效）
out_multi = sel.predict_action_chunk(dict(batch), noise=noise_all)
ok = torch.allclose(out_multi[0], cand[int(sc.argmin())][: cfg.n_action_steps].float())
print(f"selected chunk == cand[argmin]: {ok}")
assert ok, "选择路径与手工复算不一致"
print(f"peak_alloc_total={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
print("ALL CHECKS PASSED")
