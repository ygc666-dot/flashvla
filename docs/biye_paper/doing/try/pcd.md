# PCD × FlashVLA 第一步：输出侧单步对比（可开关，基线零影响）

> 本文是「把 PCD 思想接入 FlashVLA 流式推理」的第一阶段工程方案。
> 目标不是刷点，而是**在不动基线任何一行执行路径的前提下**，拿到一个可以随时开关、可以逐项归因的对比分支。
>
> 关联文档：`flashvla/docs/analysis/code_analysis.md`（FlashVLA 算法实现分析）、`flashvla/docs/papers/PCD - Policy Contrastive Decoding for Robotic Foundation Models.md`。
> 代码引用格式 `起始行:结束行:文件路径`，路径相对 `flashvla/` 仓库根。

---

## 0. 结论摘要

1. **PCD 的 Eq.(2) 在 flow matching 里必然退化成速度场的线性复合**：
   $v^*(x_t,t)=(1+\alpha)v_o(x_t,t)-\alpha v_{\hat o}(x_t,t)$。
   推导见 §1.1。这不是近似，而是 score 线性复合的严格等价。
2. **FlashVLA 拿不到分布**。它一次调用只产出一个 chunk（slot 0 被推到 $t=0$），要复现 PCD 的 KDE-PM（$N'=24$）就得付 24 倍代价，FlashVLA 的意义直接消失。在 $N'=1$ 下取 KDE 的众数，恰好得到同一形式的解（§1.2），所以**速度场对比就是 PCD 在流式下的唯一可行形态**。
3. **第一步只做「输出侧单步对比」**：只修正即将被读出的 slot 0，buffer 演化仍用原始速度。因为 slot 0 在左移时本来就被丢弃（`modeling_pi05_flashvla.py:1021`），所以**流式动力学逐位不变**。这是唯一能同时保证"有效修改"与"不惊动训练分布"的落点。
4. **开关设计**：`pcd_alpha=0` 时走**原始 `_steady_streaming`，一行不改、零额外开销、图不变**；`pcd_alpha>0` 时走新增的 `_steady_streaming_pcd`，单独编译、单独捕获。
5. **验证分五阶段**：零算力正确性闸门（5 个 test）→ 离线扰动数据定方向 → 成本剖面 → 分层成功率 → 归因消融。**在 clean LIBERO 上测不出差异是正常结果**，必须先在扰动设定上锚定效应量（§5.4 L2）。
6. **但"扰动设定"本身需要先过闸**：LIBERO 微调模型的脆弱点集中在相机视角与初始位姿（PCD 数学上够不到），而 PCD 能修的背景通道它本来就不依赖。**§5.4 的 L2 以 §9.3 的 30 分钟遮挡敏感度诊断通过为前置**，未通过就直接走 §9.5 的三条出路，避免白跑几周算力。

---

## 1. 方法：为什么是"输出侧单步"

### 1.1 PCD → 速度场的严格等价

FlashVLA 沿用 π0.5 的路径 $x_t=t\varepsilon+(1-t)a$（$t=1$ 纯噪声，$t=0$ 纯动作），回归目标 $v=\varepsilon-a$，欧拉更新 $x\leftarrow x-\Delta t\,v$（`code_analysis.md` §1.1）。

令 $m=\mathbb{E}[\varepsilon\mid x_t]$，则 score $=-m/t$，而由 $v=(m-x_t)/(1-t)$ 得

$$\text{score}(x_t,t)=-\frac{x_t+(1-t)\,v(x_t,t)}{t}$$

在固定 $(x_t,t)$ 下 score 是 $v$ 的**仿射**函数，因此 $\log\pi^*=\log\pi(a|o)+\alpha[\log\pi(a|o)-\log\pi(a|\hat o)]$ 线性复合到速度场时权重原样保留：

$$\boxed{\;v^*=(1+\alpha)v_o-\alpha v_{\hat o}\;\Longleftrightarrow\;x_0^{\text{guided}}=x_0^{\text{base}}-\Delta t\cdot\alpha\,(v_o-v_{\hat o})\;}$$

### 1.2 KDE-PM 在 $N'=1$ 下退化到同一形式

把 $N'=1$ 代入 PCD 的 Eq.(3) 再代入 Eq.(2)：

$$\log\pi^*=-\frac{(1+\alpha)(a-\mu_o)^2}{2b^2}+\frac{\alpha(a-\mu_{\hat o})^2}{2b^2}+\text{const}\;\Rightarrow\;\operatorname{mode}(\pi^*)=(1+\alpha)\mu_o-\alpha\mu_{\hat o}$$

与 §1.1 同形。**丢掉的只是 KDE 的方差异质性信息，保住了方向性矫正。** 这解释了为什么"速度场对比"不是权宜之计，而是流式设定下的正确降级。

### 1.3 为什么只在最干净的一步做

同一 $\Delta v$ 对应的分布偏移量是 $\big|\Delta\,\text{score}\big|=\frac{1-t}{t}|\Delta v|$。$t=0.25$ 时的系数是 $t=0.75$ 时的 **9 倍**。也就是说：

- **越靠近 $t\to0$，每单位 $\Delta v$ 的分布级矫正越强**，因此单步即可产生可观测效果；
- 在大 $t$ 段做引导主要破坏 ODE 的粗结构而收益有限（与 PCD Appendix Table 7 的"CFG 越早介入越差"一致：step0 62.7 / step2 63.8 / step4 64.8 / step8 64.5，base 63.9）。

FlashVLA 的 slot 0 状态正好是 $t=1/N$，是流水线里最干净的一步，天然就是最佳落点。**并且它被读出后即被丢弃，改它不影响任何后续计算。**

### 1.4 与训练分布的关系

$a$ 的横向修正量是 $\Delta t\cdot\alpha\|v_o-v_{\hat o}\|=\frac{\alpha}{N}\|v_o-v_{\hat o}\|$。以 $\alpha=1$、$N=4$ 计，是速度量级的 25%。而 §2.3（`code_analysis.md`）已证明训练在每个槽的整个 $1/N$ 区间内随机采样，对栅格外的 $x_t$ 有覆盖。所以**默认强度下不会把状态推出训练分布**；但如果 α 拉到 4 以上，等效于把 slot 0 推到 $t=0$ 之外，需要重新检查。

### 1.5 α 的量级先验

由于修正被 $\Delta t=1/N$ 稀释，**方案 A 的 α 约等于 PCD 论文 α 的 $N$ 倍**。PCD 在 π0 上最优 $\alpha=0.2$，$N=4$ ⟹ 先验搜索区间 **$\{0.5,1.0,2.0,4.0\}$**。不要照抄论文的 0.2。

---

## 2. 开关设计：三条"不影响基线"的保证

| # | 保证 | 实现手段 |
| :-- | :--- | :--- |
| 1 | **执行路径不变** | `pcd_alpha <= 0` 时 `sample_actions` 仍调用原 `_steady_streaming`；不折 batch、不采第二张图、不建第二份 cache |
| 2 | **图不变** | `_pcd_active` 是 `__init__` 里一次性判定的 Python `bool`，不进入 torch 计算图；α 是 `register_buffer` 标量张量，可 `fill_` 更新 |
| 3 | **配置向后兼容** | 新字段全部带默认值（默认关闭）。旧 checkpoint 的 `config.json` 缺这些键时回落到 dataclass 默认值 ⟹ 加载即基线行为 |

补充：**当 `pcd_alpha>0` 但缺少反事实图像时，默认 `pcd_missing_cf="error"` 直接报错**，不静默降级。理由见 §4 第 5 条——静默降级会让 T1 自检"假通过"，最终表现为"方法无效"的误判。

### 开关矩阵

| 模式 | 配置 | 走哪条路径 | 用途 |
| :--- | :--- | :--- | :--- |
| `baseline` | `pcd_alpha=0` | 原 `_steady_streaming`（**未改动**） | 生产、回归基线 |
| `matched_compute` | `pcd_alpha=0, pcd_ctrl="fold2"` | `_steady_streaming_pcd`，但丢弃 $v_{\hat o}$ | 归因对照：排除"多算一次"的效应 |
| `pcd` | `pcd_alpha>0` | `_steady_streaming_pcd` | 实验组 |

---

## 3. 代码改动清单

### 3.1 `flashvla/policies/pi05/configuration_pi05.py`

在 `PI05FlashVLAConfig`（`configuration_pi05.py:218-249`）追加字段，全部带默认值：

```python
# ---- PCD (counterfactual decoding) — 默认全关，零侵入 ----
pcd_alpha: float = 0.0            # 主开关：<=0 完全关闭
pcd_step_scaled: bool = True      # True: 修正量按 dt=1/N 缩放（式(1.1)）
pcd_ctrl: str = "off"             # "off" | "fold2"（matched-compute 对照）
pcd_missing_cf: str = "error"     # "error" | "degrade"
pcd_image_suffix: str = "__pcd"   # 反事实图像在 batch 里的 key 后缀
pcd_log_stats: bool = True        # 记录 ||v_o - v_cf|| / ||v_o||
```

### 3.2 `flashvla/policies/pi05/modeling_pi05_flashvla.py`

**(a) 新增 `_steady_streaming_pcd`**（与 `_steady_streaming` 平级，`modeling_pi05_flashvla.py:967-1036` 之后）：

```python
@torch.no_grad()
def _steady_streaming_pcd(self, images, img_masks, images_cf, tokens, masks,
                          state, buffer, profile=False):
    """稳态流式 + 输出侧单步对比。buffer 走原始速度演化，动力学与基线逐位一致。"""
    N, C = self.N, self.C
    buf_len = N * C
    bsz = tokens.shape[0]
    dev = tokens.device

    # --- 1) 折 2：[original ; counterfactual]，顺序固定 ---
    images_f  = [torch.cat([im, im_cf], 0) for im, im_cf in zip(images, images_cf, strict=True)]
    img_masks_f = [torch.cat([m, m], 0) for m in img_masks]   # 每相机一个 [2B]
    tokens_f  = torch.cat([tokens, tokens], 0)
    masks_f   = torch.cat([masks, masks], 0)
    state_f   = torch.cat([state, state], 0)
    buffer_f  = buffer.repeat(2, 1, 1).contiguous()   # 共享 buffer，禁止重采样

    # --- 2) prefix 只编码一次（2B），prefill 一次 ---
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.prefix_embedder(
        images_f, img_masks_f, tokens_f, masks_f
    )
    for layer in self.layers:
        layer.self_attn.attn.reset_cache()
    prefix_attention_mask, prefix_position_ids = build_attention_mask_and_position_ids(
        prefix_pad_masks, prefix_att_masks, prefix_embs.dtype
    )
    hs = [prefix_embs, None]
    for layer in self.layers:
        hs = layer(hs, prefix_attention_mask, prefix_position_ids, [None, None], use_cache=True)

    # --- 3) 一次 denoise_step，两路速度 ---
    time_per_slot = torch.arange(1, N + 1, device=dev, dtype=torch.float32) / N
    time_f = time_per_slot.unsqueeze(0).unsqueeze(2).expand(2 * bsz, -1, C).reshape(2 * bsz, buf_len)
    v_all = self.denoise_step(prefix_pad_masks, prefix_att_masks, state_f,
                              buffer_f, time_f, padding_mask=None)
    v_o, v_cf = v_all.chunk(2, dim=0)             # 各 [B, buf_len, D]

    # --- 4) buffer 只用 v_o 演化（与基线逐位相同）---
    dt = 1.0 / N
    buffer_upd = buffer - dt * v_o
    actions = buffer_upd[:, :C, :]

    # --- 5) 只修正读出槽 ---
    if self._pcd_active:                          # alpha>0 且非纯对照
        delta = v_o[:, :C] - v_cf[:, :C]
        scale = dt if getattr(self.config, "pcd_step_scaled", True) else 1.0
        actions = actions - scale * self._pcd_alpha * delta
        if getattr(self.config, "pcd_log_stats", False):
            self._pcd_ratio_acc += (delta.norm() / v_o[:, :C].norm().clamp_min(1e-6)).detach()

    # --- 6) 左移 + 尾部灌噪声（半 batch 完成即可）---
    new_buffer = self.sample_noise((bsz, buf_len, self.config.max_action_dim), dev)
    new_buffer[:, :-C, :] = buffer_upd[:, C:, :]
    return actions, new_buffer, None
```

> 注意：`buffer_upd` / `new_buffer` 都在**半 batch** 上完成。两路分支的 buffer 演化完全相同（都用 $v_o$），没必要 fold；只有 `denoise_step` 的**输入**需要 fold。
> 另两点：`buffer` 本身宽度就是 `max_action_dim`，所以 `buffer_upd[:, :C, :]` 与基线 `buffer[:, :C, :self.config.max_action_dim]`（`modeling_pi05_flashvla.py:1019`）等价；`profile` 分段计时在本路径下直接返回 `None`（与 `_steady_streaming` 的 docstring 一致：CUDA Graph 回放下图内 event 无意义），调用方已能容忍 `None`。

**(b) 修改 `sample_actions` 调度**（`modeling_pi05_flashvla.py:1065-1079`）：

```python
if step_counter < N - 1:
    ...原有 _cold_start（完全不动，冷启动不产出动作，不需要对比）...
else:
    if self._pcd_path_on:                     # Python bool，编译期固定
        actions_to_execute, buffer, breakdown = self._steady_streaming_pcd(
            images, img_masks, images_cf, tokens, masks, state, buffer, profile=profile)
    else:
        actions_to_execute, buffer, breakdown = self._steady_streaming(
            images, img_masks, tokens, masks, state, buffer, profile=profile)
```

`sample_actions` 签名追加可选参数 `images_cf=None`。

**(c) 修改 `predict_action_chunk`**（`modeling_pi05_flashvla.py:1202`）：

```python
images, img_masks = self.prepare_images(batch)
images_cf = self.prepare_images_cf(batch) if self._pcd_path_on else None
```

**(d) 新增 `prepare_images_cf`**：镜像 `prepare_images`（`modeling_pi05_flashvla.py:1282-1313`）的预处理（同样的 `resize_with_pad` + `*2-1`），但按 `f"{key}{suffix}"` 取图，**与 `images` 一一对齐、顺序一致、相机数一致**；缺失时按 `pcd_missing_cf` 报错或回落到原图。空相机槽位同样用 `* -1` 占位以保持 prefix 长度不变。

**(e) `__init__` 追加**（`PI05FlashVLAPolicy.__init__`，`modeling_pi05_flashvla.py:1120-1142`）：

```python
alpha = float(getattr(config, "pcd_alpha", 0.0) or 0.0)
self._pcd_path_on = alpha > 0.0 or getattr(config, "pcd_ctrl", "off") == "fold2"
self._pcd_active  = alpha > 0.0                       # 纯对照模式下为 False
self.model.register_buffer("_pcd_alpha", torch.tensor(alpha), persistent=False)
self.model._pcd_active = self._pcd_active
self.model._pcd_path_on = self._pcd_path_on
self.model.register_buffer("_pcd_ratio_acc", torch.zeros((), dtype=torch.float32), persistent=False)
self.model._pcd_ratio_n = 0    # 纯 Python 计数，仅用于日志
```

另加一个便于扫参的 setter（避免每次重建 policy）：

```python
def set_pcd_alpha(self, alpha: float) -> None:
    self.model._pcd_alpha.fill_(float(alpha))
    self.model._pcd_active = float(alpha) > 0.0
```

> 变更 α 会改变 `_pcd_active`，而它参与 Python 分支判断，所以**同一进程内从 0 切到非 0 会重新捕获图**。扫参脚本应保证一个进程一个 α（例外是 T2 用同一进程比较 buffer，那里本来就不在意图）。

### 3.3 新增 `flashvla/policies/pi05/pcd.py`（反事实观测生成，模型外）

职责：把 `batch` 里的图像替换成"物体被移除"的版本。**必须与 policy 解耦**，只负责产出 `{key}__pcd`。

```python
class BasePCDMasker:
    def reset(self, obs=None): ...
    def __call__(self, batch): -> dict      # 原地或返回新 dict，写入 {key}__pcd
```

实现四个可切换版本，按 §5 的验证需要：

| 版本 | 说明 | 用途 |
| :--- | :--- | :--- |
| `identity` | 直接 `copy` 原图 | **T1 自检**、matched-compute 对照 |
| `gt` | 用仿真器 instance mask（LIBERO 可从 segmentation 渲染拿 obj id） | 仿真首选，准且免费 |
| `gdino_sam2` | GDINO 检首帧 + SAM2 逐帧跟踪 | 真机 / 无 GT 时 |
| `random_box` | 同面积随机矩形 | **归因消融**：区分"物体聚焦"与"任意扰动" |

三个必须实现的细节：

1. **填充方式**：默认**灰色**（`pcd_fill="gray"`，值为 `batch` 张量的均值）。不用 LaMa。理由：流式下一个 chunk 的 4 步看到 4 帧不同修复图，生成式修复的纹理差异会作为噪声进入 $\Delta v$；确定性填充保证 $\Delta v$ 只反映"物体有无"。LaMa 作为消融臂保留。
2. **时序一致性**：首帧确定目标后**冻结掩码集**，后续帧只做跟踪（或直接复用首帧掩码 + 小范围膨胀），避免逐帧分割抖动直接注入速度场。
3. **仿真里用 GT segmentation 而不是 GDINO+SAM2**（见 §9.4）：robosuite 可直接取 geom id 渲染 mask，既准又免费；同时**扰动区域与 PCD 掩码不得完全重合**，否则"PCD 掩掉的正好是你破坏的区域"会让扰动实验的结论退化为循环论证。

### 3.4 `sim_eval/libero/eval.py`

- `rollout()` 里 `env.reset()`（`eval.py:140`）之后调用 `masker.reset()`；
- 在 `env_preprocessor` 之后、`preprocessor` 之前（`eval.py:170-172`）插一个 `masker(observation)` step；
- `eval_info.json` 的 `overall` 里写入 `pcd: {alpha, ctrl, fill, masker, camera, ratio_mean}`，否则一堆 json 分不清；
- 已有的 `transition/internal` 延迟打印（`eval.py:258-264`）保留，它是 T4 的判据来源。

### 3.5 `sim_eval/libero/eval.sh`

`eval.sh:210-226` 的参数是硬编码的，加一个透传：

```bash
POLICY_EXTRA_ARGS="${POLICY_EXTRA_ARGS:-}"
... python sim_eval/libero/eval.py ... $POLICY_EXTRA_ARGS
```

---

## 4. 实现约束清单（不遵守必然踩坑）

1. **两路必须共享 buffer 与噪声**。$\Delta v=v_o-v_{\hat o}$ 的前提是同一个 $x_t$、同一个 $\epsilon$。fold 时 `buffer.repeat(2,1,1)`；尾部新噪声在**半 batch** 上采一次（本方案天然满足，因为后续处理都在半 batch）。若两路噪声不同，$\Delta v$ 会混入独立噪声项、方向随机，表现为"任何 α 都没效果"。
2. **α 必须是张量，不能是 Python float**。`_pcd_alpha` 用 `register_buffer(..., persistent=False)`，`fill_` 更新。否则 α 进 Python 控制流会导致 dynamo 重编译，破坏 `code_analysis.md` §3.5 的"单图捕获"。
   注意 `torch.compile` 包的是整个 `sample_actions`（`modeling_pi05_flashvla.py:501-502`），所以 §3.2(b) 里的 `if self._pcd_path_on:` 是**在编译图内**的 Python 常量分支——dynamo 会按该属性特化，一张图一个分支，安全；但不要在它上面再加任何依赖张量值的分支。
3. **冷启动完全不改**。`_cold_start`（`modeling_pi05_flashvla.py:872-965`）保持原样：不折 batch、不算 $v_{\hat o}$。冷启动 `N-1` 次调用不产出可执行动作，对比没有意义，且省一半开销。
4. **`bsz` 的账要改干净**：`_steady_streaming_pcd` 内部凡是构造 buffer/noise/time 的地方都已在半 batch 上；只有 `time_f` 和 `denoise_step` 的输入用 `2*bsz`。`denoise_step`（`modeling_pi05_flashvla.py:819-869`）本身**不需要改动**——它从 `suffix_embs.shape[1]` 推 `L_suf`，与 batch 无关。
5. **缺失反事实图不得静默降级**。默认 `pcd_missing_cf="error"`。若静默退化为"$\hat o=o$"，T1 会假通过，最终表现为"方法无效"，而这个误判要花几十 GPU·h 才能发现。
6. **`n_action_steps < chunk_size` 的影响要记录**。LIBERO 默认 `C=10, n_action_steps=5`，校正作用在整段 C 上但只有前 5 条被执行，后 5 条丢弃——不是 bug，但写进结论，避免误读。
7. **warmup 走同一条路径**。`eval.py:557-561` 的 `warmup_manager.warmup()` 必须在启用 PCD 的配置下运行，否则图在第一次真实 transition 才捕获，前 3 个 episode 的延迟数据作废。

---

## 5. 验证方案

五个阶段，按"越早、越便宜、排除假设越多"排序。

### 5.1 阶段 1：零算力正确性闸门

- T1–T3：新文件 `tests/test_pcd_streaming.py`（pytest，进 CI）；
- T4–T5：新文件 `scripts/pcd_cost_probe.py`（需要 CUDA + 一次编译，由 CI 的 nightly 或手动跑）。

| Test | 内容 | 断言 | 抓住什么 |
| :--- | :--- | :--- | :--- |
| **T1 identity 自检** | 令 `ô ≡ o`（batch 里 `{key}__pcd = batch[key]`），α 取 0.5/1/2 | `actions` 与基线 diff < 1e-5 | 噪声未共享、buffer 未共享、fold 顺序错位、掩码没接上、`L_suf` 切片错、KV cache 串味 |
| **T2 buffer 冻结** | 同一 seed 跑 20 次 `predict_action_chunk`，比较 α=0 与 α=3 的 `policy.action_buffer` 快照序列 | 逐位 `.equal()` | 是否悄悄把方案 A 变成了方案 B（改了动力学） |
| **T3 $\Delta v$ 单调性** | 真实帧 + GT mask，β ∈ {0, 0.25, 0.5, 0.75, 1.0} 掩码面积 | `ratio = ‖Δv‖/‖v_o‖` 随 β 单调上升；identity 时 ratio < 1e-6 | 掩码是否真的送到模型；**是否掩错了相机**（看得到目标的那一路 ratio 才大） |
| **T4 internal 延迟不变** | 1 条 episode，`overlap_steps=1` | `eval.py` 打印的 `internal mean` 与基线差 < 10%，`transition mean` 上升 | cf 分支是否被算到了每步；异步是否真的生效 |
| **T5 单图捕获** | 启用 PCD 后 warmup，读 `torch._dynamo.utils.counters["stats"]` | `unique_graphs == 1` | α 是否进了 Python 控制流导致反复重编译 |

T1 是全套里性价比最高的一条：**写这一条比写 T2–T5 加起来更能抓住实现错误**。

```bash
pytest tests/test_pcd_streaming.py -v
```

### 5.2 阶段 2：离线扰动数据——不用仿真就能定方向

用 LeRobot 格式的 LIBERO 数据（训练集即可），取 50 条 episode，做教师强迫式误差比较：

1. 用真实图像序列驱动稳态路径（`_steady_streaming` / `_steady_streaming_pcd`，保持 buffer/step_counter 正常推进），拿预测 chunk 与 GT 算 L1；
2. 对图像施加三类变换后重跑：

| 变换 | 期望 | 含义 |
| :--- | :--- | :--- |
| (a) 背景/纹理/亮度扰动 | α>0 **降低**误差 | PCD 的目标场景 |
| (b) 目标物体被遮挡 | α>0 **略微变差** | 此时对比会放大"无物体"这个本来正确的判断 |
| (c) clean | 基本不变 | 物体可见时 $\Delta v$ 本就小 |

**符号模式 (a)降 / (b)微升 / (c)平，才算"实现 + 方向都对"。** 若 (a) 不降，不要烧仿真算力，回去查实现。

同一步顺手筛 α：扫 `{0, 0.5, 1.0, 2.0, 4.0}`，挑平台区。注意 flow 模型的动作 L1 与成功率只是弱相关（多模态、chunk 内相关），**只看相对变化，不看绝对值**。

### 5.3 阶段 3：成本剖面（可行性的唯一硬闸门）

先用现成的 `profile=True`（`modeling_pi05_flashvla.py:1024-1032` 的 `encode/prefill/action` 三段）量 B 与 2B：

| 判据 | 结论 |
| :--- | :--- |
| `action` 段 2B/B < 1.3× | 方案 A 可行（batch=1 时 GPU 未饱和，翻倍 batch 常近乎免费） |
| > 1.8× | 先做 v1 优化（只算 slot 0 的 C 个 token + 掩码 prefix KV 跨调用复用）再谈效果 |

再跑 1 条完整 episode 看 `eval.sh` 的延迟输出（`eval.py:258-264`）：`internal mean` 应不变，`transition/internal mean ratio` 不应显著上升，`avg_time_per_action_step_ms` 上升 < 20% 可接受。**最终指标是 `avg_episode_walltime_s`**（PCD 原文报的就是执行时间 +24%，因为机器人物理执行占大头）。

### 5.4 阶段 4：仿真成功率——分层 + 统计功效

先算清楚功效，否则浪费算力：

| 设定 | episode | 成功率 95% CI（p≈0.7） |
| :--- | :--- | :--- |
| 1 suite × 1 seed × 50 ep | 500 | ±4.0% |
| 1 suite × 5 seeds × 50 ep | 2500 | ±1.8% |
| 4 suites × 5 seeds × 50 ep | 10000 | ±0.9% |

**单 seed 单 suite 只能分辨 >8% 的效应**，而 PCD 在 π0 上平均仅 +8.9%。所以：

- **L1 筛 α（便宜）**：`libero_object` × 1 seed × 50 ep，α ∈ {0, 1.0, 2.0}，加 matched-compute。目的是淘汰有害 α，不是证明有效。
  ```bash
  for A in 0 1.0 2.0; do
    SUITES="libero_object" N_EPISODES=50 OUTPUT_ROOT=/tmp/pcd/a$A \
      POLICY_EXTRA_ARGS="--policy.pcd_alpha=$A" \
      bash sim_eval/libero/eval.sh 1 1000
  done
  # matched-compute 对照臂
  SUITES="libero_object" N_EPISODES=50 OUTPUT_ROOT=/tmp/pcd/ctrl \
    POLICY_EXTRA_ARGS="--policy.pcd_ctrl=fold2" \
    bash sim_eval/libero/eval.sh 1 1000
  ```
- **L2 定方向（关键，别省，但有前置）**：在 **clean** 与 **perturbed** 两个设定下各跑基线 + α*。扰动插在 `env_preprocessor` 之后（`eval.py:170-172`，此时已是 tensor）：改亮度 / 叠 distractor / 换桌面纹理。
  > **前置**：先过 §9.3 的遮挡敏感度诊断。若 `ratio_背景 ≪ ratio_物体`，说明模型本来就不用背景，本步大概率无信号，直接转 §9.5。
  > 这是整个验证的锚点：要复现的是"baseline 掉 30%、PCD 拉回 10%"这种大效应。**如果 clean 无差异、perturbed 有恢复，那是符合预期的正常结果；只有当 perturbed 下也测不到恢复，才是方法或实现有问题。**
- **L3 显著性（贵）**：α* 与 matched-compute 各跑 5 seeds × 4 suites。用 `eval_info.json` 的 `per_episode[].seed / .success`（`eval.py:420-442`）做**同 seed 准配对 McNemar**，而不是比较两个边际 CI。注意这是准配对（动作不同会让轨迹发散），但对功效仍有明显提升。

> **最大的风险点**：LIBERO 四个 suite 渲染几乎无伪相关可破（背景固定、物体单一）。**在 clean LIBERO 上得到"无显著差异"几乎是必然的，不要把它当成实现错误。**

### 5.5 阶段 5：归因消融（reviewer 必问）

按诊断价值排序：

| # | 消融 | 判据 |
| :-- | :--- | :--- |
| 1 | **反事实类型**：object mask / 同面积随机矩形 / 固定噪声填充 / **陈旧帧**（$o_{t-k}$） | 若"随机矩形"效果与对象掩码相当，增益来自"任意输入扰动"的平滑效应，**不是 PCD 机制** |
| 2 | **matched-compute**：α=0 折 2；以及"两次独立噪声取平均" | 后者是 FlashVLA 本来就能做的集成基线，必须被超过 |
| 3 | **对比深度**：depth=1（本方案）vs depth=4（全槽引导） | 验证 §1.3 的 $(1-t)/t$ 预测；同时给出 A/B 方案的相对结论 |
| 4 | **填充方式**：灰色 vs 固定种子噪声 vs LaMa | 预测流式下确定性填充更好（与 PCD 原文相反，因为帧间一致性） |
| 5 | 指令侧反事实（只换物体名，复用视觉 token） | 区分"物体聚焦"与"语言先验"，并测出省掉 SigLIP 塔能省多少 |

### 5.6 结果判读表

| 现象 | 结论 |
| :--- | :--- |
| T1 不过 | 接线 / 噪声共享 / fold 顺序错，停止后续验证 |
| T2 不过 | 不小心改了 buffer 演化，方案 A 退化为方案 B |
| T3 的 full-mask ratio ≈ 0 | 掩码没送到模型，或掩错了相机 |
| T4 internal 延迟上涨 | 额外前向跑到了每步，或异步没生效 |
| 2B 的 action 段 > 1.8× | 成本闸门不过，先做 v1 |
| 离线 (a) 不降、(b) 不升 | 实现或方向有问题（此时 $\Delta v$ 可能只是噪声） |
| 离线对，clean LIBERO 无差异 | **正常**，去 perturbed 设定评 |
| perturbed 下 baseline 崩、PCD 不恢复 | 方法在流式下确实无效——这本身是可发表的结论 |
| 增益与"随机矩形掩码"相当 | 机制不是 PCD，需要重新表述 |

---

## 6. 里程碑

| 阶段 | 交付物 | 估算 |
| :--- | :--- | :--- |
| M1 | config 字段 + `_steady_streaming_pcd` + `predict_action_chunk` 接线 + T1 | 0.5 天 |
| M2 | `pcd.py` 的 identity/gt 两个 masker + T2/T3 | 0.5 天 |
| M3 | 阶段 2 离线扰动脚本 + α 筛选 | 0.5 天 |
| M4 | 成本剖面（T4/T5 + 2B/B 三段计时） | 0.5 天 |
| M5 | L1→L2 仿真；`random_box` 与 matched-compute 对照臂 | 2–3 天算力 |
| M6 | L3 全量（5 seeds × 4 suites × 2 条件） | 5–8 天算力 |

---

## 7. 风险与回退

| 风险 | 影响 | 回退 |
| :--- | :--- | :--- |
| 2B 墙钟远超预期 | 收益被成本吃掉 | 保留代码、默认关闭；转 v1（slot 0 单 block + prefix KV 复用） |
| clean LIBERO 无差异 | 无法直接发表 | 根源已在 §9.2 定位（LIBERO 无可破伪相关、主导通道是位姿），非实现问题。先过 §9.3 诊断；再走 §9.5 的换 benchmark / 人造伪相关 / 换叙事三条出路 |
| α 安全区间极窄 | 难调 | 检查 $\Delta t$ 缩放是否正确（`pcd_step_scaled`）；用 EMA 平滑 $\Delta v$ |
| 掩码抖动导致方向噪声 | 效果不稳 | 冻结掩码集 + 灰色填充 + $\Delta v$ EMA（$\bar{\Delta v}\leftarrow\beta\bar{\Delta v}+(1-\beta)\Delta v$） |

**任何情况下 `pcd_alpha=0` 都必须能原样复现基线**（T1 + T2 已覆盖），因此本改动不存在"污染基线"的风险——最坏情况是这段代码永远不启用。

---

## 8. 记录：本阶段明确不做的事

1. **不做全槽引导（depth=N）**：会改变 buffer 演化，属于方案 B，留作 §5.5 的对照臂 3。
2. **不做真正的 KDE-PM（多采样）**：$N'=24$ 需要 24× 代价；只在分析里作为"上界参照"。
3. **不缓存掩码分支的 prefix KV**：属于 v1 优化（可把 2× 压到约 1.1×），但需要第二套 `k_cache/v_cache`，与"最小可验证"目标不匹配。
4. **不改训练侧**：本阶段纯推理期。若后续发现需要微调（例如速度幅值漂移），再单独立项。
5. **不引入指令侧反事实**：作为 §5.5 的第 5 组消融保留，不进入主路径。

---

## 9. 前置调研：LIBERO 上做扰动评测是否可行（go/no-go）

> 本节回答两个问题：**(1) 在 LIBERO 上构造 perturbed 评测可行吗？(2) FlashVLA 已在 LIBERO 上微调，它本身对 LIBERO 的扰动是否已经鲁棒？**
>
> 结论：**工程上可行且不难（还有现成 benchmark 可复用），但"已经鲁棒"这个问法会让你走到岔路上——正确的问题是"模型在哪个通道上过拟合"。而从公开结果看，答案对本实验不利：LIBERO 微调模型的脆弱点集中在相机视角与初始位姿，这恰好是 PCD 数学上够不到的通道；而 PCD 能修的背景/纹理通道，模型本来就不怎么依赖。**
>
> **因此 §5.4 的 L2（扰动评测）必须先通过 §9.3 的 30 分钟诊断，否则大概率白跑几周算力。**

### 9.0 结论摘要

| 问题 | 回答 |
| :--- | :--- |
| 在 LIBERO 上构造 perturbed 可行吗？ | **可行**，比 §3.3/§5.4 里写的更省：robosuite 能直接出 GT segmentation（geom id），走图像空间合成即可，不需要 GDINO + SAM2。更省事的是直接复用 **LIBERO-Plus**（arXiv 2510.13626，7 个扰动维度 × 5 个难度等级，仓库 `sylvestf/LIBERO-plus`，支持按维度单独评测） |
| FlashVLA 对 LIBERO 的扰动本身鲁棒吗？ | **不鲁棒，但脆弱点不在 PCD 能修的通道上**——这才是问题 |

### 9.1 为什么"鲁棒性"和"PCD 能否帮上忙"是两个问题

PCD 做的是**通道权重再分配**：把"物体证据 $u$"的相对权重拉到 $1+\alpha$，把其余一切压低。它**不能创造模型内部不存在的表征**。所以扰动导致的失败必须分类：

| 失败模式 | 表现 | PCD 能修吗 |
| :--- | :--- | :--- |
| ① 通道权重错配 | 物体证据很强，但背景/纹理/臂姿的意见压过了它 | **能**，这正是 PCD 的机制 |
| ② 表征被破坏 | 物体像素本身变了（视角、亮度、遮挡），模型对 $u$ 的编码失效 | **不能**。此时 $\Delta v$ 本身也是错的，对比只会放大噪声 |
| ③ 分布外动力学 | 任务本身需要不同动作分布（初始位姿大变） | **不能**，这是策略分布问题 |

这是本节的核心判断依据：**LIBERO 上最常见的扰动（相机视角、机器人初始状态）属于 ② 和 ③，不是 ①。** PCD 原文在真机上拿到 +108%，正是因为真实场景里"物体清晰可见（①）+ 背景复杂且与训练分布不符"同时成立。

### 9.2 LIBERO 微调模型的依赖结构（有公开证据）

LIBERO-Plus 在 7 个维度做受控扰动：物体布局、相机视角、机器人初始状态、语言指令、光照、背景纹理、传感器噪声。与本实验直接相关的三条结论：

1. **对相机视角和机器人初始状态极其敏感**，中等扰动即把成功率从 95% 打到 **30% 以下**；
2. **对语言变化基本不敏感**，进一步分析显示模型常常**完全忽略语言指令**；
3. 即：LIBERO 微调模型的决策由**视觉几何 + 位姿**主导。

> 注：95%→<30% 是原文摘要给出的量级，主要指 camera / robot-init-state 两个维度。**background_texture 与 lighting 的单维度跌幅需要查它的 per-dimension 表确认**（仓库有 per-dimension 评测入口），但方向明确：显著小于 camera / init-state。

由此三条推论全部对本次实验不利：

- **推论 1**：背景/纹理这类"PCD 能修的通道"，恰恰是 baseline 掉得最少的通道。模型本来就没怎么用它 ⟹ PCD 的矫正空间很小。
- **推论 2**：主导伪相关是"**机械臂位姿 → 动作序列**"的记忆（LIBERO 每任务 50 个固定初始状态，评测用的还是同一批）。掩码物体对这个通道**完全无效**——你掩的是物体，不是臂姿。
- **推论 3**：物体布局扰动在 `libero_spatial` 里是**任务相关**的（"把碗放在盘子左边"这类指令中位置就是语义），掩掉物体等于掩掉任务定义。

**FlashVLA 特有的一处混淆**：`N=4` 的 buffer 让同一 chunk 的 4 步看到 4 个不同观测，扰动的污染会在 buffer 里**跨 4 次调用传播**（π0 的 chunk 只条件在一个观测上，不会如此）。所以扰动实验必须**整段 episode 保持同一个扰动**，不能逐帧抖动，否则测到的是"时序不一致"而非"背景偏移"。

### 9.3 先做 30 分钟的 go/no-go 诊断

**不要先建扰动评测。** 直接用现有 checkpoint，在**同一个 $(x_t,t)$** 上做**通道遮挡敏感度**：

$$\text{ratio}_c=\frac{\lVert v_o-v_{o\setminus c}\rVert}{\lVert v_o\rVert},\qquad c\in\{\text{目标物体},\ \text{背景},\ \text{机械臂},\ \text{夹爪},\ \text{桌面}\}$$

这就是 §5.1 的 T3 的扩展，代码几乎不用改（把 masker 换成"逐通道遮挡"）。

| 观测到的模式 | 结论 |
| :--- | :--- |
| `ratio_背景` ≪ `ratio_物体`（如 0.02 vs 0.30） | **模型几乎不用背景** ⟹ PCD 在 LIBERO 上没有可矫正空间，扰动评测大概率无信号，**不要做** |
| `ratio_背景` 与 `ratio_物体` 同量级（都 > 0.1） | 通道权重错配存在 ⟹ PCD 有机制上的机会，**值得做扰动评测** |
| `ratio_物体` 本身就很小 | 掩错相机（双路里只有看得到物体的那路敏感），先修实现 |
| `ratio_机械臂` 最大 | 印证位姿记忆假说 ⟹ PCD 帮不上，转 §9.5 |

额外价值：顺带把 T3 与"掩错相机"两个实现问题一起验证，且**不花仿真 rollout**。

### 9.4 若诊断通过：各扰动维度的可行性排序

| 扰动 | 工程难度 | baseline 预期跌幅 | PCD 机制上是否有机会 | LIBERO-Plus 现成 |
| :--- | :--- | :--- | :--- | :--- |
| **背景纹理替换**（只改非物体区域） | 低（GT seg + 图像合成） | 小（个位数 %） | 有，但小 | 有 |
| **外围 distractor**（物体 bbox 外贴干扰物） | 低 | 小～中 | 有（PCD Appendix Table 10 就是这项） | 有 |
| **光照/色温**（全局） | 低 | 中 | 弱：物体像素也变了（②） | 有 |
| **光照/色温**（只对非物体区域） | 低 | 小 | 有 | 需自建 |
| **相机视角** | 中 | **大（95→30）** | 弱：物体投影也变（②） | 有 |
| **机器人初始状态** | 低 | **大** | 无（③，掩码够不到） | 有 |
| **语言改写** | 低 | 很小 | 无（模型本就忽略指令） | 有 |

**结构性矛盾：效应量所在之处与可矫正之处不重叠。** 只有前两行是"PCD 有机会"的，而后两行恰好是 baseline 掉得最少的。

一个值得单独试、且便宜的假设：**相机视角扰动 + PCD**。视角变化同时改变物体与背景外观，但"物体 vs 无物体"的对比在视角变化下仍能分离出物体证据，也许能抵掉部分"背景随视角漂移"的污染。风险是 $\Delta v$ 在 ② 状态下本身失真。测法：视角扰动 × α ∈ {0,1,2}，看是否单调。

### 9.5 若诊断不通过（`ratio_背景` 很小），三条出路

1. **换 benchmark**。与 PCD 原文同构的设定是 SIMPLER（Google Robot / WidowX 的 real-to-sim，背景复杂、有 distractors），PCD 在它对 π0 拿到 +8.9%。RoboTwin + 复杂背景亦可。代价是重搭评测，但省掉"造扰动 + 解释为什么没效果"的大量时间。
2. **人造伪相关（唯一能在 LIBERO 上制造出"可被 PCD 修"的条件）**。在 LIBERO 微调数据里**人为植入**一条 train 时成立、test 时被打破的关联（例如固定把某个 distractor 放在某个位置，或让某块桌面纹理与"抓取"动作强相关），评测时打乱它。这样模型**真的**会依赖 $v$，PCD 才有东西可去。代价是多一次微调，但**对 PCD 本身仍然 training-free**，论文表述站得住；而且这是唯一能拿到干净因果证据的路线——伪相关是你控制的变量，不是猜的。
3. **换叙事**。把对比幅度 $\lVert\Delta v\rVert$ 当作**在线伪相关/冲突度量**，喂给 `async_manager` 做动态 `overlap_steps`（冲突大 → 同步重规划）。这条不依赖 LIBERO 有伪相关，而是把 PCD 的副产物变成系统级收益。

### 9.6 执行顺序（对本文档其他章节的修订）

1. **先跑 §9.3 的 30 分钟遮挡敏感度诊断**（真实 checkpoint，不加扰动，只在离线帧上做）。
2. `ratio_背景` 与 `ratio_物体` 同量级 → 走 §9.4 前两行：LIBERO-Plus 的 `background_texture` 维度 + 自建的"只改非物体区域"版本，1 suite × 1 seed 先看有没有信号（对应 §5.4 的 L2）。
3. `ratio_背景` 很小 → **不要在 LIBERO 上做扰动评测**，直接选 §9.5 路径 1 或 2。
4. §9.5 路径 3 与 1/2 不冲突，可并行，且不受 LIBERO 场景限制。

**对本文档的修订**：

- §5.4 的 **L2（扰动设定评测）升级为条件项**：以 §9.3 诊断通过为前置；未通过则跳过，直接进 §9.5。
- §7 风险表的"clean LIBERO 无差异"一行补充：该风险的**根源**已在 §9.2 定位（LIBERO 无可破的伪相关 + 主导通道是位姿），不是实现问题。
- §3.3 的 masker：仿真首选 `gt`（robosuite segmentation），且**扰动区域与 PCD 掩码不得完全重合**，否则"PCD 掩掉的正是你破坏的区域"会让结论变成循环论证。
