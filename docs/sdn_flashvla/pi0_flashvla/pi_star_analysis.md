# π0\* → FlashVLA 为什么这么低：归因、证据与下一步

> 2026-11-24。上游方案：`4_pi0star_flashvla.md`（π0\* 路线）、`../flashvla_simpler/3.flashvla_simpler.md`（pi05 路线）。
> 本文只回答两件事：**bridge 的 12.5% 是怎么来的**，**fractal 是不是同一个病**，然后给出可执行的修法。

---

## 0. TL;DR

1. **不是评测/协议问题**：pi05 的 005000 用同一条 `run_g3.sh`、同两张卡、同 seeds 复跑，逐任务**完全复现**
   42/78/62/54 = 59.00%（`logs/p4_eval/gates/G3_full_pi05_control.json`）。π0\* 的 12.5% 是模型自己的问题。
2. **主因是「动作归一化口径被换掉」**：π0\* 的头部是按 `2(x−q01)/(q99−q01)−1`（openpi 的 `bound`）标定的，
   我们换成 `MEAN_STD`，等于要求 6 个非夹爪维把输出**整体放大 ~1.95×**（逐维 1.84–2.35，夹爪 1.01 恰好对齐）。
   π0\* 初始的真实动作维 loss 是 **4.32**，pi05 是 **1.46**——前几百步全在补这个增益。
3. **帮凶是「32 维零填充吃掉 78% 的梯度」**：转换器把 `action_in_proj` 第 7–31 列、`action_out_proj` 第 7–31 行
   置零，而 noise 在 32 维采样 → 那 25 维**既看不到输入也发不出输出**，恒在 `E[noise²]=1.0`，
   占总 loss 的 25/32 = **0.781**（实测权重仍为零，见 §3.2）。pi05 的同位置是预训练过的满宽矩阵（P3 实测
   这 25 维 MSE 只有 0.0014），所以**它的 loss 可比，我们的不可比**。
4. **fractal 同病，但只掉了 22 pp 而不是 67 pp**：实测 `google_robot_close_drawer` = **46%**（π0\* 原生 68%）。
   同一套转换器/归一化/填充，机制 A/B/C 全在（增益错位甚至更大：ratio 1.94–2.45 vs bridge 1.84–2.35），
   差别在**三个放大器**：fractal 有 3× 的优化步数（12,000 vs 4,000）、预训练见过该域（L7 起点 2.12 vs 4.01）、
   以及 close_drawer 这个任务**对策略精度几乎不敏感**。bridge 的四个任务（尤其 stack_cube 需要精确顶抓）
   把同样的弱点放大成 12.5%。
5. **修法按性价比**：先 `ACTION/STATE → QUANTILES`（纯 config，10 分钟）+ 让那 25 维不再吃梯度，再考虑
   state 通路和更长配方。注意区分两种「多训」：**续跑**没用（lr 到 4000 步已在地板 2.5e-7，002750 与 004000 同分），
   但**新开 run 给更多步数**有用——fractal 用 12,000 步把 L7 从 4.01 压到 0.94。

---

## 1. 事实：四组数字放在一起

| 检查 | carrot | eggplant | spoon | stack_cube | macro |
| :--- | ---: | ---: | ---: | ---: | ---: |
| π0\* 原生（T0 vanilla，openpi 权重 + SDN 自己的路径） | 68% | 88% | 88% | 76% | **80.0%** |
| **π0\*→FlashVLA 004000** | 8% | 36% | 6% | 0% | **12.5%** |
| **π0\*→FlashVLA 002750** | 4% | 34% | 10% | 0% | **12.0%** |
| pi05→FlashVLA 005000（P3 原记录） | 42% | 78% | 62% | 54% | **59.0%** |
| pi05→FlashVLA 005000（**本轮对照复跑**） | 42% | 78% | 62% | 54% | **59.0%** |

fractal 族（`google_robot_close_drawer`，seeds 2000–2049，50 trajs，单卡）：

| 检查 | close_drawer | 相对原生 |
| :--- | ---: | ---: |
| π0\* 原生（T0 vanilla） | 68% | — |
| **π0\*→FlashVLA 012000** | **46%**（`eval_logs/p4_pi0star/fractal/012000/.../000_success_0.46.log`） | **−22 pp** |

对比同族的 bridge：**−67.5 pp**。同一套转换与配方，两族结局差 3 倍 → 见 §4。

训练侧（同样的 bs4×accum8、lr 5e-5 cosine、warmup 300、adafactor、7.2/7.55 epoch）：

| 族 | 步数 | 总 loss（起始→末） | 折出真实 7 维 loss | 备注 |
| :--- | ---: | :--- | :--- | :--- |
| π0\* bridge | 4,000 | 1.727 → **1.033** | 4.32 → **1.15** | 17,784 帧 / 4 任务，5 Hz |
| π0\* fractal | 12,000 | ~1.24(step50) → **0.986** | 2.12 → **0.94** | 50,843 帧 / 5 任务，3 Hz |
| pi05 bridge | 4,000~5,000 | 0.343(step50) → **0.18** | 1.56 → **0.82** | 同数据同配方 |

折算口径见 §3.2：`L7 = (total − 25/32 × L_pad) × 32/7`，π0\* 的 `L_pad ≡ 1.0`、pi05 的 `L_pad ≈ 0.0011`（P3 §11.3 实测）。

---

## 2. 先关掉三个“看起来像”的嫌疑

| 嫌疑 | 结论 | 证据 |
| :--- | :--- | :--- |
| 评测脚本/协议回归 | **排除** | pi05 005000 在同一条 `run_g3.sh`、同 GPU、同 seeds 下逐任务复现 42/78/62/54，误差 0 |
| 服务端推理路径错（符号/尺度/夹爪/冷启动） | **排除** | 离线探针 `scripts/p4/probe_actions.py` 走完全相同的 `FlashVLAPolicyServer.act`：π0\* 整体 MAE 0.0754，而“永远输出数据集均值”是 0.1515；幅值对得上（dx 预测 0.00485 vs 真值 0.00531），夹爪正确饱和到 ±1（pred_abs=1.0） |
| `empty_cameras=0` 与 π0\* 预训练不一致 | **排除（本来就是 1 张图）** | π0\* 参考路径 `simpler_adapter.py:73-81` 只喂 `images[None]` 一张；pi05 才需要 `empty_cameras=2` 去凑它自己的 3 槽前缀 |
| 夹爪二值化阈值 | **排除（噪声内）** | P3 §11.4 实测 54.0% vs 53.5% |
| “多训一点就好了” | **排除** | 002750 与 004000 同为 12%；且 lr 在 4000 步已到地板 2.5e-7（P3 §11.6 同款教训：续跑不买趋势） |

---

## 3. 归因：三个机制，A 最重、B 放大、C 结构性

### 3.1 A（主因）动作归一化口径被换掉 → 头部输出系统性偏小 ~1.95×

π0\* 的部署路径对动作做的是 **quantile 归一化**，而且**夹爪不归一化**：

```96:120:third_party/SDN/simpler_env/policies/pizero/simpler_adapter.py
        if self.action_normalization_type == "bound":
            raw_actions_except_gripper = self.denormalize_bound(
                actions[:, :-1],
                np.array(self.dataset_statistics["action"]["p01"])[:-1],
                np.array(self.dataset_statistics["action"]["p99"])[:-1],
```

```194:199:third_party/SDN/simpler_env/policies/pizero/simpler_adapter.py
    def postprocess_gripper(self, action: float) -> float:
        # trained with [0, 1], 0 for close, 1 for open
        action_gripper = 2.0 * (action > 0.5) - 1.0
```

而我们的两个 yaml（`train/configs/pi0/simpler/pi0_flashvla_{bridge,fractal}.yaml`）沿用了 P3 的口径：

```yaml
  normalization_mapping:
    VISUAL: IDENTITY
    STATE: MEAN_STD
    ACTION: MEAN_STD
```

两种口径的目标分布标准差（=`bound`  vs  `MEAN_STD`）在 bridge/stack_cube 上逐维算：

| 维 | 原始 std | q01 | q99 | `bound` 后 std = 2·std/(q99−q01) | `MEAN_STD` 后 std | **放大倍数** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| dx | 0.00799 | −0.0140 | 0.0154 | 0.543 | 1.0 | **1.84×** |
| dy | 0.00998 | −0.0176 | 0.0212 | 0.515 | 1.0 | **1.94×** |
| dz | 0.01089 | −0.0156 | 0.0314 | 0.464 | 1.0 | **2.16×** |
| rx | 0.01231 | −0.0288 | 0.0290 | 0.426 | 1.0 | **2.35×** |
| ry | 0.02367 | −0.0687 | 0.0374 | 0.446 | 1.0 | **2.24×** |
| rz | 0.02635 | −0.0592 | 0.0535 | 0.468 | 1.0 | **2.14×** |
| grip | 0.98905 | −1.0 | 1.0 | 0.989 | 1.0 | 1.01× |

**这是整条链上最“干净”的解释**：π0\* 的 `action_out_proj` 学到的输出幅度是按 ~0.5 的目标 std 来的，
我们要求它输出 std ≈1.0 的东西，于是**整个动作头一开始就小了近一半**；6 个运动维全中，唯独夹爪（±1 本来就是
单位尺度）不受影响——这与实测完全一致：夹爪“能用”，运动“发糊”。

旁证：**pi05 没有这一层错位**。`lerobot/pi05_base` 本身就是 MEAN_STD 预训练的，所以它的真实维 loss 起点
1.46 是“直接能用的量级”，而 π0\* 是 4.32（≈ 一个 2× 增益误差的平方量级）。这就是“明明 π0\* 已经适配 simpler”
却更差的直接原因：**适配 simpler 的是「openpi 的 π0\* 推理栈」，不是「FlashVLA 的动作空间」**。

> 注：4000 步后头部确实把增益补回来了（探针里 `pred_abs_mean` 已接近 `gt_abs_mean`），但代价是前 ~1/3 的步数
> 在学一个**线性重标定**，不是在学策略；剩下 2/3 又只有 22% 的梯度（见 3.2），于是只能到“能预测但不够准”。

### 3.2 B（放大）32 维零填充吃掉 78% 的 loss 与梯度

`max_action_dim=32`，真实动作 7 维。noise 在**全部 32 维**采样（`modeling_pi0_flashvla.py:383-384`），
损失对 32 维取均值（`:1062-1063`）。转换器把填充维**两边都置零**：

```76:100:scripts/p4/convert_pi0star.py
    """Right-pad a 2-D linear weight from ``[out, in]`` to ``[out, width]`` with zeros."""
    ...
    """Right-pad the leading dimension (action output projection) with zeros."""
```

实测权重（直接读 `model.safetensors` 的切片 rms，30 秒可复现，命令见 §6②）：

| 张量切片 | π0\* 转换后（init） | π0\* 004000（训完） | pi05 005000 |
| :--- | ---: | ---: | ---: |
| `action_out_proj` 行 0–6（真维） | 0.0180 | 0.0180 | 0.0201 |
| `action_out_proj` 行 7–31（填充） | **0** | **8.3e-06** | **1.92e-02** |
| `action_in_proj` 列 0–6 | 0.2180 | 0.2194 | 0.1950 |
| `action_in_proj` 列 7–31 | **0** | **1.3e-04** | **2.09e-01** |
| `state_proj`（全） | **0** | **1.7e-05** | 1.03e-01 |

填充维在 π0\* 里 4000 步后仍然是 `1e-5` 量级（相对真维小 3 个数量级）→ 它**看不见** `x_t` 的噪声、也**发不出**
`v_t`，于是那 25 维的 loss 恒等于 `E[(noise − 0)²] = 1.0`。因此：

```
total = (7/32)·L7 + (25/32)·1.0 = 0.781 + 0.219·L7
```

- π0\* bridge：total 1.033 → **L7 = 1.15**（起点 1.727 → L7 = 4.32）
- π0\* fractal：total 0.986 → **L7 = 0.94**
- pi05：填充维已被预训练成“可预测噪声”（P3 §11.3 实测 MSE 0.0014，仅贡献 0.0011）→ total 0.18 ≈ 0.219·L7 →
  **L7 = 0.82**

两个后果，都致命：
1. **78% 的梯度花在 25 个无意义的维上**（而且是“永远学不会”的），真实动作维只分到 22%；
2. **loss 曲线不可读**：0.781 是硬地板，§4.1 的「500 步 > 0.35 转 R2」判据对本方案恒成立，不能用。

### 3.3 C（结构性）proprio 通路被整体摘掉，且口径不能照抄

转换把 π0\* 的整条 proprio 栈（165 个张量）丢掉，`state_proj` 置零：

```231:234:scripts/p4/convert_pi0star.py
        out[f"{STATE_PROJ}.weight"] = torch.zeros(width, max_state_dim, dtype=dtype)
        out[f"{STATE_PROJ}.bias"] = torch.zeros(width, dtype=dtype)
        report["synthesized"][f"{STATE_PROJ}.weight"] = "zeros (state branch trained from scratch)"
```

实测训完 `state_proj` rms 只有 1.7e-5 → **模型是彻底 state-blind 的**（输入乘 1e-5 等于 0）。

要点是：π0\* 的 proprio **不是**我们数据集里的 8 维四元数状态，而是 7 维、且旋转被转到 top-down 系：

```179:192:third_party/SDN/simpler_env/policies/pizero/simpler_adapter.py
    def preprocess_proprio(self, proprio) -> np.array:
        # convert ee rotation to the frame of top-down
        rm_bridge = quat2mat(proprio[3:7])
        rpy_bridge_converted = mat2euler(rm_bridge @ self.default_rot.T)
        gripper_openness = proprio[7]
        raw_proprio = np.concatenate([proprio[:3], rpy_bridge_converted, [gripper_openness]])
```

所以 `scripts/p4/convert_pi0star.py:344` 的 `--map-proprio-to-state`（把 `proprio_encoder` 拷到 `state_proj`）
**只能算形状兼容**：7 维 euler 与 8 维 quat 语义不同，直接拷是把一个“读过 proprio 的线性层”接到另一套坐标上。

严重度评估：pi05 在 P3 里也近乎 state-blind（`state_cond` 换 state 只让动作变 6.6e-5，§11.3），却仍有 59%，
所以 **C 不是 12% vs 59% 的主因**；但对 π0\* 更致命——π0\* 原本就是靠 proprio 把“delta 动作”落到工作空间的，
摘掉之后只能靠视觉猜位姿，闭环漂移更大。

### 3.4 归因权重（可复核的判断）

| 机制 | 对 12.5% 的贡献 | 可复核证据 |
| :--- | :--- | :--- |
| A 归一化口径错位（~1.95× 增益） | **主因**，浪费前 1/3 步数 + 留下残余偏移 | §3.1 逐维表 + L7 起点 4.32 vs 1.46 |
| B 填充维吃掉 78% 梯度 | **放大**，把“能学会”压成“学不动” | §3.2 权重实测量级 |
| C 无 proprio | **次级**（pi05 也盲，但 π0\* 更依赖） | `state_proj` rms 1.7e-5 |
| D 配方（4000 步 / lr 地板 / 7.2 epoch） | **放大器**：fractal 用 12,000 步到 L7 0.94 / 46%，bridge 只有 4,000 步 | §4 |
| E 任务对精度的容忍度 | **放大器（决定性）**：同样的弱点在 close_drawer 只掉 22 pp，在 widowx 抓取类掉 67 pp | §4 逐任务表 |

---

## 4. fractal 是不是同一个病？——机制相同，结局差 3 倍

**是同一个病**：同一个转换器、同一份 yaml 模板、同一个 `MEAN_STD`、同样被置零的填充列/行、同样死掉的 state 通路。
增益错位 fractal 甚至**更大**（`bound`/`MEAN_STD` 的 std 比：fractal 1.94–2.45，bridge 1.84–2.35）：

| 指标 | bridge | fractal | 说明 |
| :--- | ---: | ---: | :--- |
| 数据 | 17,784 帧 / 4 任务 / 5 Hz | 50,843 帧 / 5 任务 / 3 Hz | fractal 里 close_drawer 8,010 帧 |
| 优化步数 / epoch | 4,000 / 7.20 | **12,000** / 7.55 | epoch 相同，**梯度更新次数 3×** |
| total loss | 1.033 | 0.986 | 都含 0.781 硬地板 |
| **真实 7 维 loss** | **1.15** | **0.94** | fractal 好 18% |
| step-50 真实维 loss | 4.01 | **2.12** | fractal 起点低一半 → `fractal_beta` 预训练见过该域 |
| 填充维 / state 通路 | 零 / 死 | 零 / 死 | 同病 |
| **M=1 结果** | **12.5%**（原生 80%） | **46%**（原生 68%） | **−67.5 pp vs −22 pp** |

**为什么同病却差这么多——三个放大器，缺一不可**：

1. **优化预算**：head 要先补掉 ~2× 的增益再学策略。fractal 有 12,000 次更新，bridge 只有 4,000 次
   （epoch 数一样，但“补增益 + 拟合”是按更新次数算的）。bridge 相当于在“刚补完增益”就结束了。
2. **预训练域吻合度**：fractal 用的 `fractal_beta` 就是在该域训的（起点 L7 2.12），bridge 的 `bridge_beta` 与本
   simpler widowx 的 delta 尺度不同（起点 L7 4.01）。
3. **任务对误差的容忍度（决定性）**：`close_drawer` 是单相位“推上去关掉”，策略发糊也能关；bridge 的四个任务
   全要求抓取时序，实测 π0\* 的夹爪通道有 ~23% 步数判错（grip MAE 0.458 vs pi05 0.333）、平移维度只比
   “预测均值”好 19%——这种精度在 `stack_cube`（精确顶抓）上就是 0%。

逐任务看 bridge 内部的分化，完全支持第 3 点：

| bridge 任务 | π0\* 原生 | π0\*→FlashVLA | Δ | 任务对精度的要求 |
| :--- | ---: | ---: | ---: | :--- |
| `widowx_put_eggplant_in_basket` | 88% | 36% | −52 | 走到篮子上方松手（最宽松） |
| `widowx_carrot_on_plate` | 68% | 8% | −60 | 需对准盘子 |
| `widowx_spoon_on_towel` | 88% | 6% | −82 | 需精确抓勺柄 |
| `widowx_stack_cube` | 76% | 0% | −76 | 需精确顶抓 + 放置 |

**结论**：
- fractal **不是另一个 bug**，它就是同一套 A/B/C，只是拿到 3× 更新 + 域内预训练 + 宽容任务，所以还剩 46%；
- **不能**因为 fractal 有 46% 就认为 bridge 的 12.5% 是评测噪声——bridge 的四个任务把同一个弱点放大了 3 倍；
- 反过来说，**fractal 的 46% 也不该被当成“通过”**：它比原生低 22 pp，A/B/C 修完仍应重跑一次。

---

## 5. 下一步：按性价比排的改法

### F1（必做，纯 config，~10 分钟）把归一化换回 π0\* 自己的口径

lerobot 的 `QUANTILES` 就是 `2(x−q01)/(q99−q01)−1`，与 openpi 的 `bound` **逐字一致**
（`lerobot/processor/normalize_processor.py:362-377`），且我们数据集 `meta/stats.json` 里 `q01/q99` 齐全：

```yaml
  normalization_mapping:
    VISUAL: IDENTITY
    STATE: QUANTILES     # π0* 的 proprio 也是 bound
    ACTION: QUANTILES    # 关键：让头部从第 0 步就落在原生输出尺度
```

预期：**初始真实维 loss 从 4.32 掉到 ~1.4**（即 pi05 的量级），前 1/3 步数不再浪费在学增益。
> 夹爪维在 π0\* 里本来就没归一化（±1 单位尺度），`QUANTILES` 对它是恒等映射（1.01×），不会引入新偏差。
>
> **⚠️ 2026-11-24 实测修正（见 §5.7）**：这个预期**没有兑现**。同一 batch、同一 noise、同一权重下，
> `QUANTILES` 初值 L7 = **4.14**，`MEAN_STD` 初值 L7 = **4.59**——只有 −10%，不是 −70%。
> 原因：初值 loss 的 77% 来自**夹爪通道**（grip MSE 19.1，其它 6 维 1.2–2.8），
> 而且 `MEAN_STD` 下头部输出是**偏大**的（v/u 比 1.1–1.4），不是本文猜的“偏小 1.95×”。
> F1 仍然值得做（口径对齐 pretrain、纯 config、无副作用），但**不要**用它当提升幅度的依据。

### F2（必做，二选一）让那 25 个填充维不再吃梯度

- **推荐做法**：训练时把 noise 掩到真实动作维（`sample_noise` 后 `noise[..., action_dim:] = 0`）。
  因为 `v_t` 在那 25 维恒为 0、`u_t` 也变 0 → loss 贡献恰为 0，loss 读数直接等于 `L7`（可读、可用于早停）。
  代价：与 pi05 的 loss 口径脱钩（pi05 的 0.18 里也含它的填充项），比较时必须换回同一口径。
- **保守做法**：不动 policy，只把「填充维的 loss」从监控里剔除（评测脚本侧重算），梯度稀释照旧。
- **不要做**：把 `action_in_proj` 的填充列随机初始化——那只会让模型把容量花在“预测纯噪声”上（pi05 就先学了
  这件无用功，P3 实测它在填 0.0014 的收益上花了预训练算力）。

### F3（必做，config + 1 个开关）给 state 一条真的通路

- 用 `ACTION/STATE: QUANTILES` 让 state 落在 π0\* 的 proprio 尺度上；
- 初始化不要全零：`state_proj` 用 `std=1e-3` 的随机初始化（零初始化 + 共享 lr 在 4000 步里只长到 1.7e-5，P3
  §11.7 的两个“便宜修法”同样失败过）；
- 该路径 lr ×3（`state_path_lr_mult: 3.0`，本仓库已支持）；
- 上线前先用 `scripts/p3/state_sensitivity.py` 验“换 state 动作要变”，不要再靠猜。
- 若要拷 π0\* 的 `proprio_encoder`，必须先把数据集 state 转成 π0\* 的 7 维 euler（top-down 系）口径，
  否则 `--map-proprio-to-state` 只是形状兼容（见 §3.3）。

### F4（必做）新开 run，不要续跑

P3 的教训同样适用：`num_decay_steps` 到点后 lr 恒在地板，续跑是在加噪声。
建议 `steps: 8000`（bridge 至少 8,000；参照 fractal 用了 12,000 才到 L7 0.94）、`warmup 300`、
`num_decay_steps: 8000`、`peak_lr 5e-5`、`decay_lr 2.5e-7`，
`pretrained_path` 指到**刚转换出的 ckpt**（`ckpt/pi0star_bridge_flashvla`）而不是旧 run 的 004000。
按实测 2.17 s/step，8,000 步 ≈ 4.8 h/卡。

### F5（方法论，省钱）先 500 步诊断，再决定要不要 2.4 小时

判据（**2026-11-24 修正版**，原判据见 §5.7 的对照；关键改动是把「绝对 loss」换成 scale-free 的探针量，
因为不同归一化口径的 loss 地板不同：`QUANTILES` ≈ 1.2、`MEAN_STD` ≈ 2.0，loss 不可跨口径比较）：

| 检查点 | 期望 | 不达标说明 |
| :--- | :--- | :--- |
| step 1 | `L7 ≈ 3.5–5.0`（QUANTILES 口径，grip 主导） | 明显 > 5 → 归一化/掩码没生效；**不要**再指望 1.3–1.6 |
| step 100 | `L7 ≤ 2.2` 且单调下降、无 NaN | 头部没在学（查 state/图像/tokenizer 预处理） |
| step 500 | `L7 ≤ 1.1`（已越过“预测零”地板 1.2） | 头部仍在学边际分布 |
| step 500（真正的判据） | `scripts/p4/probe_actions.py`：`corr(dx) ≥ 0.3`，`mae_overall` 明显低于 0.1515（mean baseline） | 走 `probe` 的部署路径，与归一化口径无关，可跨配方比较 |
| 评测 | 50 seeds 的 SE≈3.5 pp，**不要**用它调参 | 只在配方定稿后跑一次闸门 |

参照系（同一 leaf `widowx_stack_cube`、同 seeds、部署路径探针）：π0\* 旧配方 002500 → `mae 0.0787 / corr(dx) 0.674`；
004000 → `0.0754 / 0.706`；pi05 005000 → `0.0542`；mean baseline `0.1515`。

### F6（如果修完仍不过闸）

那就把结论写成“**π0\* 的动作口径（quantile/无夹爪归一化 + 独立 proprio 栈）无法低成本迁到 FlashVLA 的
`MEAN_STD` + 单 state token 结构**”，与 pi05 的 59% 并列作为 P4 的负结果；此时正确的下一步是
**换路线**（例如把 π0\* 的 proprio 栈与 quantile 归一化在 FlashVLA 侧做成原生 feature，而不是靠转换器补零），
而不是继续在评测上刷 seeds。

---

### F5 实测记录（2026-11-24，GPU4，bridge）

**① 落地清单（F1/F2/F3/F4）**

| 项 | 改动 |
| :--- | :--- |
| F1 | `train/configs/pi0/simpler/pi0_flashvla_{bridge,fractal}.yaml` + `_smoke_pi0star.yaml`：`STATE/ACTION → QUANTILES` |
| F2 | `PI0FlashVLAConfig.mask_padded_action_dims`（`configuration_pi0.py`）+ `modeling_pi0_flashvla.py:444`（掩 noise/action 尾部）与 `:1062`（loss 只在前 `real_action_dim` 维取均值）；三份 yaml 置 `true` |
| F3 | `scripts/p4/convert_pi0star.py --state-proj-init-std 1e-3`（种子 0 的 `N(0,1e-3)`），两族起点已重转（`verify.ok=true`、`unfilled=0`）；yaml `state_path_lr_mult: 3.0`（日志确认命中 `state_proj.weight/bias` 两个张量） |
| F4 | 新 run `runs/pi0_flashvla_bridge_q8000`（`steps/num_decay_steps 8000`、`warmup 300`、`save_freq 500`），实测 **2.17–2.22 s/step**、峰值沿用 25.4 GB |

**② 初值 loss 归因：同一 batch、同一 noise、同一权重（新转换的 ckpt）**

| 口径 | 初值 L7 | dx | dy | dz | rx | ry | rz | **grip** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `QUANTILES` | **4.14** | 2.79 | 1.42 | 1.67 | 1.35 | 1.52 | 1.16 | **19.08** |
| `MEAN_STD` | **4.59** | 3.82 | 2.00 | 2.62 | 1.87 | 2.62 | 1.69 | **17.47** |
| “预测零”基线 `E[u_t²]`（QUANTILES） | 1.22 | 1.35 | 1.06 | 1.24 | 1.09 | 1.20 | 0.95 | 1.72 |

判读（三条，均可复核）：

1. **F1 只值 −10%**，不是 §3.1 预测的 −70%；`L7` 的 77% 是**夹爪通道**（`v_t` rms 2.25 vs 目标 1.31）。
   夹爪是初值瞬态：同探针下旧 004000 的 grip MSE 降到 **0.44**。
2. **“`MEAN_STD` 要求整体放大 ~1.95×”方向相反**：`MEAN_STD` 下 `v/u` ≈ 1.1–1.4（头输出偏大），
   `QUANTILES` 下 `v/u` ≈ 0.13–0.55（偏小）。初值头与目标基本不相关，两种口径都只能靠训练。
   所以 §3.1 的“前 1/3 步数在学线性重标定”这个机制**没有被证据支持**。
3. **loss 地板随口径变**：`QUANTILES` ≈ 1.2、`MEAN_STD` ≈ 2.0 → §1 表中“总 loss / 折算 L7”的数字
   **不能跨口径比较**，§5 F5 的绝对判据据此改写（见上）。

**③ 500 步（新配方）与旧配方的部署路径探针**（`scripts/p4/probe_actions.py`，`widowx_stack_cube`，24 帧，seed 0）

| ckpt | 步数 | 配方 | `mae_overall` | dx | dy | dz | rx | ry | rz | grip | corr(dx) |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 新 `q8000/000500` | 500 | F1+F2+F3 | **0.1141** | .0108 | .0189 | .0187 | .0265 | .0349 | .0641 | **.625** | 0.277 |
| 旧 `ctrl_meanstd_500/000500` | 500 | 旧口径 | 0.1257 | **.0069** | **.0111** | **.0124** | **.0149** | **.0240** | **.0397** | .771 | **0.449** |
| 旧 `002500` | 2,500 | MEAN_STD | 0.0787 | .0045 | .0074 | .0082 | .0109 | .0176 | .0233 | .479 | 0.674 |
| 旧 `004000` | 4,000 | MEAN_STD | 0.0754 | .0043 | .0070 | .0080 | .0108 | .0173 | .0221 | .458 | 0.706 |
| （参照）pi05 005000 | 5,000 | MEAN_STD | 0.0542 | — | — | — | — | — | — | .333 | 0.872 |
| （参照）mean baseline | — | — | 0.1515 | — | — | — | — | — | — | — | — |

**步数对齐的结论（这是本轮最重要的一条）**：500 vs 500 相比，

- **`mae_overall` 看起来更好（−9%），但完全来自夹爪通道**（0.625 vs 0.771，corr 0.399 vs 0.243）；
- **6 个运动维全部变差**（dx −36%、dy −41%、dz −34%、rx −44%、ry −31%、rz −38% 的 MAE 恶化），
  `corr(dx)` 0.277 vs 0.449。也就是说 **F1 让运动维在同等步数下学得更慢**——与 §3.1 的预期相反：
  `QUANTILES` 把目标尺度压到 ~0.45σ，而转换来的头部在初值时**输出偏小**（v/u 0.13–0.55），
  要更多步数才长到位；`MEAN_STD` 的 1.0σ 反而让运动维更快“够到”。
- 因此**不能靠 `mae_overall` 判优劣**（它被夹爪一项主导）。是否在 2k–4k 步交叉，只能靠桥训练跑完看；
  下一批判据点：本 run 的 002000 / 004000 / 008000，对上表旧配方的 002500 / 004000 逐维比。
- 旧配方 004000 的数字与 §7 的表逐维一致（0.0754 / dx corr 0.706），说明探针本身可复现。

**④ 顺带修掉的一个真 bug：`state_path_lr_mult` 会让续跑直接崩**

- 现象：从 `q8000/000500` 续跑抛 `ValueError: List length mismatch: expected 1, got 2`
  （`lerobot/optim/schedulers.py:141` → `load_scheduler_state`）。
- 根因：`scale_state_path_lr` 在 scheduler 建好**之后**给 optimizer 加了一个 param group，而 `LambdaLR`
  在构造时就把 `base_lrs` / `lr_lambdas` 快照成“每组一项”，`get_lr()` 用 `zip` 因此**只更新第 0 组**：
  ① 第 1 组（`state_proj`，×3）的 lr 永远停在 warmup 初值（实测 `4.98e-07`，等于没放大）；
  ② 存档里 `base_lrs/lr_lambdas` 长度为 1、`_last_lr` 长度为 2，续跑时长度校验必失败。
- 修法（`flashvla/train.py:890` 一带）：改成「先建 optimizer → `scale_state_path_lr` → 再 `cfg.scheduler.build(optimizer, cfg.steps)`」，
  并打印 `LR scheduler built for 2 param group(s): base_lrs=['5e-05', '0.00015']` 以便复核。
  已存在的 `q8000/000500` 的 `training_state/scheduler_state.json` 手工补齐成两组
  （`base_lrs=[5e-5,1.5e-4]`、`lr_lambdas=[null,null]`），备份在 `/tmp/scheduler_state_000500.bak.json`；
  补齐后 `Auto-resume ... resume=True` 从 500 步正常继续（7500 步，2.17 s/step）。
- 推论：**之前 π0\*/pi05 的 `state_path_lr_mult` 实验（P3 §11.7 的 10× 那组）实际上也没生效**，
  第 1 组一直被冻在 warmup lr 上；`state_path_init_std` 的实验不受影响（它改的是权重不是 lr）。

**⑤ 结论**

- F2（掩码）没有理由回退：它把 25 个学不动的维从 loss/梯度里摘掉，读数变成可用的 `L7`。
- F1 从“提升来源”降级为“口径对齐”：500 vs 500 的对齐实验显示它**只改善夹爪、并拖慢 6 个运动维**。
  保留（与 pretrain 一致、纯 config），但不再指望它带来幅度。
- F3 现在才有意义：修掉 scheduler 分组 bug 之前，`state_path_lr_mult` 的那一组 lr 其实一直是冻结的；
  也就是说 P3/P4 过去关于“调大 state lr 没用”的结论要重新做（见 ④）。
- **12.5% 的主因需要重新归因**：现有证据只支持“π0\* 的头部在 FlashVLA 结构里初始化后与目标不相关、要靠训练学，
  且夹爪通道在初值阶段吃掉大部分梯度”，不再支持“归一化增益错位”。
- 训练进度：① 第一段 `logs/p4_bridge_q8000.log` step 50→500 的 `L7`：3.37 → 2.20 → 1.43 → 1.29 → 1.25 → 1.23
  → 1.20 → 1.17 → 1.14 → 1.08，按计划在 500 步停；② 对照组 `pi0_ctrl_meanstd_500`（旧口径，500 步，19 min 跑完）
  的 32 维 loss 1.145 ⇒ `L7 ≈ 1.66`（注意这是 MEAN_STD 口径，地板不同，**不可与新 run 的 1.08 直接比**）；
  ③ 新 run 已从 000500 续跑到 8000 步（2.17 s/step，≈4.3 h），判据点改为 002000 / 004000 / 008000 的探针 + 最终 M=1。

---

## 6. 复现命令

```bash
R=/home/zjr/workspace_tj/StreamingVLA/flashvla
PY=/home/zjr/workspace_tj/StreamingVLA/miniconda_ygc/envs/flashvla/bin/python
SPY=/home/zjr/workspace_tj/StreamingVLA/miniconda_ygc/envs/sdn/bin/python

# ① 离线动作探针（同一部署路径，比较两族“能预测到多准”）
$PY scripts/p4/probe_actions.py --ckpt <ckpt>/pretrained_model \
  --leaf /data/zjr/simpler_lerobot/bridge/widowx_stack_cube/success --samples 24 \
  --json $R/logs/p4/probe_<tag>.json

# ② 填充维权重体检（30 秒，CPU）
$PY -c "from safetensors.torch import load_file as L; sd=L('<ckpt>/model.safetensors'); \
o=sd['model.action_out_proj.weight']; i=sd['model.suffix_embedder.action_in_proj.weight']; \
print('out7+', o[7:].abs().max().item(), 'in7+', i[:,7:].abs().max().item(), \
'state', sd['model.suffix_embedder.state_proj.weight'].abs().max().item())"

# ③ M=1 评测（一次起两个 ckpt / 两卡）
env -u PYTHONPATH -u BASH_ENV \
  P3_LOG=$R/logs/p4_eval \
  P3_CKPT_ROOT=/data/zjr/simpler_lerobot/runs/pi0_flashvla_bridge_success/checkpoints \
  P3_G3_CKPTS="004000" P3_G3_GPUS="4 5" \
  P3_G3_ROOT=$R/third_party/SDN/eval_logs/p4_pi0star/bridge \
  bash $R/scripts/p3/run_g3.sh full

# ④ 单任务（非 widowx）手动评测：server 的端口必须是 12400+<该卡号>
CUDA_VISIBLE_DEVICES=4 $PY scripts/p3/flashvla_server.py --ckpt <ckpt>/pretrained_model --port 12404 &
CUDA_VISIBLE_DEVICES=4 $SPY -m policy.pizero.evaluate --method flashvla \
  --task google_robot_close_drawer --checkpoint <ckpt>/pretrained_model \
  --num-gpus 1 --n-trajs 50 --start-seed 2000 --reload-every 0 --no-video --result-root <out>
```

---

## 7. 附：离线探针逐维对照（stack_cube，24 帧，同一部署路径）

| 维 | π0\* 004000 MAE | pi05 005000 MAE | π0\* corr | pi05 corr | 真值 abs 均值 | 判读 |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| dx | 0.00426 | 0.00256 | 0.706 | 0.872 | 0.00531 | π0\* 只比"预测均值"好 19% |
| dy | 0.00702 | 0.00366 | 0.493 | 0.817 | 0.00608 | 同上，方向几乎没抓住 |
| dz | 0.00797 | 0.00491 | 0.562 | 0.750 | 0.00936 | 最弱的平移维 |
| rx | 0.01078 | 0.00845 | 0.298 | 0.350 | 0.00849 | 两族都差（旋转难学） |
| ry | 0.01734 | 0.01349 | 0.444 | 0.621 | 0.01677 | |
| rz | 0.02213 | 0.01291 | 0.289 | 0.689 | 0.01645 | **π0\* 劣化最明显**（corr 0.29 vs 0.69） |
| grip | 0.45833 | 0.33333 | 0.563 | 0.671 | 1.00000 | 都正确饱和到 ±1；π0\* 有 ~22% 步数判错 → 抓取类任务（stack_cube 0%）的致命项 |
| **整体** | **0.07541** | **0.05419** | | | 均值基线 0.15153 | π0\* 1.39× 差于 pi05，2.0× 好于均值基线 |

这张表说明：π0\* 不是“乱输出”，而是**全方位弱一档**（平移 1.7–1.9×、旋转最差、夹爪时序错 ~1/5），
闭环里这种弱会被放大成 12.5%——这也反过来印证了 §3 的判断：**问题在“学得不够准”，而不是“接错了”**。
