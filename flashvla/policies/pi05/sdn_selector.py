"""R3: 读出侧 SDN 选择（非流式 pi0.5 + LIBERO）。

不改模型、不改调度器：包装 policy，只替换 ``predict_action_chunk``。
同一观测采 N 个独立噪声得到 N 个候选 chunk，按 Stage 2（平滑度）择优执行。

``num_samples <= 1`` 时严格透传基线路径，保证与基线逐位一致（闸门 G1）。

参考实现：``third_party/SDN``（submodule），只依赖其中的
``sdn/selection.py`` —— 纯 torch、backbone-agnostic。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

_REPO_ROOT = Path(__file__).resolve().parents[3]  # flashvla/ 仓库根
_SDN_REF = _REPO_ROOT / "third_party" / "SDN"  # submodule


def _sdn_selection():
    """从 third_party/SDN 导入打分函数，保持单一事实源（不复制代码）。"""
    if not (_SDN_REF / "sdn" / "selection.py").is_file():
        raise RuntimeError(
            f"SDN 参考实现缺失：{_SDN_REF}\n"
            "修复：cd <flashvla> && git submodule update --init --recursive"
        )
    if str(_SDN_REF) not in sys.path:
        sys.path.insert(0, str(_SDN_REF))
    from sdn.selection import select_by_smoothness, sdn_select

    return sdn_select, select_by_smoothness


class SDNChunkSelector(nn.Module):
    """N 个独立噪声 -> N 个候选 chunk -> 择优执行。

    Args:
        policy: ``PI05Policy`` 实例（需具备 ``.model.sample_actions`` 与 ``.config``）。
        num_samples: N，每次推理的候选数；``<=1`` 时透传基线。
        smoothness: ``'acceleration'`` 或 ``'jerk'``，见 ``sdn.selection.SMOOTHNESS_SCORES``。
        long_horizon: 打分用的 chunk 长度（None = 用满 ``chunk_size``）。
        lambda_energy: 能量项权重，防止选中近零的"怠工" chunk。
        ignore_gripper: 打分时丢弃最后一维（二值夹爪）。
        top_m / knn_k: Stage 1 参数，M0 未用，留给 M2。
    """

    def __init__(
        self,
        policy,
        num_samples: int = 12,
        smoothness: str = "acceleration",
        long_horizon: int | None = None,
        lambda_energy: float = 0.05,
        ignore_gripper: bool = True,
        top_m: int | None = None,
        knn_k: int | None = None,
    ):
        super().__init__()
        self.policy = policy
        # 先存基线入口：attach_sdn 会把 policy.predict_action_chunk 换成实例属性，
        # 若这里再查 policy.predict_action_chunk 会拿到被替换后的版本（N=1 自递归）。
        self._base_predict = policy.predict_action_chunk
        self.config = policy.config  # 调度器读 config.type / num_buffer_slots
        self.num_samples = int(num_samples)
        self.smoothness = smoothness
        self.long_horizon = long_horizon
        self.lambda_energy = lambda_energy
        self.ignore_gripper = ignore_gripper
        self.top_m = top_m
        self.knn_k = knn_k
        self._select_sdn, self._select_smooth = _sdn_selection()

    # ---- 生命周期透传（nn.Module 的 eval/train 不会传播到被包装的 policy）----
    def eval(self):
        super().eval()
        self.policy.eval()
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.policy.train(mode)
        return self

    def reset(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    # ---- 主入口 ----
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, noise: torch.Tensor | None = None, profile: bool = False):
        if self.num_samples <= 1:
            # G1：与基线走同一条代码路径（含内部噪声采样与 RNG 消耗顺序）
            return self._base_predict(batch, noise=noise, profile=profile)

        policy = self.policy
        cfg = policy.config
        bsz = int(batch[OBS_LANGUAGE_TOKENS].shape[0])
        n = self.num_samples

        # 同一观测复制 N 份：prefix 相同、噪声不同 —— 这才是 SDN 的候选定义
        tiled = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == bsz:
                tiled[key] = torch.cat([value] * n, dim=0)
            else:
                tiled[key] = value

        images, img_masks = policy.prepare_images(tiled)
        state = policy.prepare_state(tiled)
        tokens = tiled[OBS_LANGUAGE_TOKENS]
        masks = tiled[OBS_LANGUAGE_ATTENTION_MASK]

        # 必须一次前向跑完 N*B：KV cache 是模块属性，循环调用会互相串味
        noise_shape = (tokens.shape[0], cfg.chunk_size, cfg.max_action_dim)
        if noise is None:
            noise_all = policy.model.sample_noise(noise_shape, tokens.device)
        elif noise.shape[0] == noise_shape[0]:
            noise_all = noise  # [N*B, C, D]：可复现（闸门 G2 需要注入同一噪声序列）
        else:
            noise_all = torch.cat([noise] * n, dim=0)  # [B, C, D] → 复制 N 份
        actions, _ = policy.model.sample_actions(
            images, img_masks, tokens, masks, state, noise=noise_all
        )

        dim = cfg.action_feature.shape[0]
        actions = actions[..., :dim].view(n, bsz, actions.shape[-2], dim)  # [N, B, C, D]

        horizon = self.long_horizon or actions.shape[2]
        picked = []
        for b in range(bsz):
            cand = actions[:, b]  # [N, C, D]
            idx = self._select_smooth(
                cand[:, :horizon],
                self.smoothness,
                lambda_energy=self.lambda_energy,
                ignore_gripper=self.ignore_gripper,
            )
            picked.append(cand[idx, : cfg.n_action_steps])

        return torch.stack(picked, dim=0).float()


def attach_sdn(policy, **kwargs):
    """在既有 policy 实例上挂载 N 采样择优，返回**同一个** policy 对象。

    为什么不返回包装对象：``sim_eval/libero/eval.py`` 的 ``eval_policy`` 有
    ``isinstance(policy, PreTrainedPolicy)`` 硬校验（``eval.py:287``），返回
    ``nn.Module`` 包装层会直接 ``ValueError``；而 ``PreTrainedPolicy`` 带 5 个
    ``abstractmethod`` 与 ``config_class``/``name`` 元类约束，为包装去实现它只会
    换来更多耦合。直接用实例属性覆盖 ``predict_action_chunk`` 最小侵入：
    类型不变、``config`` 不变、``parameters()`` 不变、调度器路径不变。

    ``num_samples <= 1`` 时仍挂载，但内部走 ``_base_predict`` 透传（闸门 G1）。
    """
    selector = SDNChunkSelector(policy, **kwargs)

    def _predict_action_chunk(batch, noise=None, profile=False):
        return selector.predict_action_chunk(batch, noise=noise, profile=profile)

    policy.predict_action_chunk = _predict_action_chunk  # 实例属性优先于类方法
    return policy
