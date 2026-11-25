# F6 交接单：已完成 / 未完成 / **下一步做什么**

> 用途：给下一个对话直接开工用的单文件交接。细节数字与施工过程在
> `f6_s0_s1.md`、`f6_s2_s3.md`、`f6_s4.md`、`f6_s5.md`，本文只保证「状态正确 + 下一步可复制执行」。
> **最后更新：2026-11-25（S6 完成）。S6 之后的施工单已移到 `f7_next_steps.md`。**

---

## 0. 一页状态

| 阶段 | 内容 | 文档 | 状态 |
| :--- | :--- | :--- | :--- |
| S0 | `proprio_encoder` 装进 `state_proj`，判先验可迁移性 | `f6_s0_s1.md` §7 | ✅ 弱 GO（6/6 运动维更优，`mae 0.1361→0.1225`） |
| S1 | `use_proprio_stack` / `proprio_dim` / 转换器开关（原「独立第三路」已作废） | `f6_s0_s1.md` §7.5 | ✅（部分被 S2 取代） |
| S2 | 三路前向（**共享 action 权重**）+ 守恒测试 | `f6_s2_s3.md` | ✅ 两路逐位守恒 |
| S3 | `denoise_step` / `_cold_start` / `_steady_streaming` 三路化 | `f6_s2_s3.md` | ✅ |
| S4 | bridge 的 7 维 `POS_EULER` 口径 | `f6_s4.md` | ✅ 对齐 `4.44e-16`、ckpt/smoke/单测全通 |
| S5 | 三臂 A/B + GO/NO-GO | `f6_s5.md` | ✅ **GO，归因反转** |
| S6 | 8000 步长跑（N1 臂③ + N2 lr×10） | `f6_s5.md` §4 | ✅ **N1 有效、N2 判否** |
| S7 | M=1 SIMPLER 闸门 | **`f7_next_steps.md`** | ⚠️ **部分完成**：close_drawer 50 seeds 已评 4 个 ckpt（34/28/24% + 老锚点 46%），五任务未跑 |
| F7-P0 | **MEAN_STD 单变量臂**（头号嫌疑是 QUANTILES） | `f7_next_steps.md` §3.3 | ⬜ **下一步主任务** |

GPU4 / GPU2 现在都空闲（两条长跑已结束）。

---

## 1. 现在站得住的三条结论

1. **三路（proprio block）在架构上是通的**：与 π0\* 同构，suffix 的 state token 直接复用 action
   栈的 attention/MLP/norm 权重，只多 `proprio_in_proj`（1024×8 + bias ≈ 8.2 K 参数）。
   两路路径**逐位守恒**（CPU fp32 / GPU bf16 的 `max|diff| = 0`），老 ckpt 探针数字**逐位不变**。
2. **价值在权重，不在结构**（S5 的核心结论，也是与规划相反的地方）：

   | 臂 | 模型 | `mae_overall` | 6 运动维均值 | `corr(dx)` | 夹爪二分类 |
   | :--- | :--- | ---: | ---: | ---: | ---: |
   | ① | 两路 + 编码器装 `state_proj` | 0.12254 | 0.06533 | −0.021 | 0.7708 |
   | ② | 三路 + **随机** `N(0,0.1)` 编码器 | 0.12370 | 0.07307 | −0.007 | 0.7917 |
   | ③ | 三路 + **π0\* 编码器** | **0.10343** | **0.05984** | **+0.110** | **0.8229** |

   ③ vs ①：6/6 运动维 + 夹爪全胜、`corr(dx)` **+0.131**（阈值 +0.05）、`mae_overall` −15.6%。
   ② vs ①：运动维 **+11.8%（更差）**，6 维只赢 1 维 → **给 proprio 通路接随机编码器是有害偏置**。
3. **state 输入投影在这套 Adafactor 配方下几乎不训练**：三臂在 2000 步里
   `cos ≈ 0.9995`、rms 变化 <0.05%。且 `f6_s0_s1.md` §7.4 的「把 init 提到 O(0.1) 就能解锁」
   **机制上不成立**（`alpha = RMS(p)·rho_t` → 相对步长与参数 RMS 无关）。

### 1.1 S6（2026-11-25 完成）新增的三条，都推翻了原交接单里的预期

| # | 结论 | 证据 |
| :--- | :--- | :--- |
| S6-1 | **S5 的三臂是「短配方」**：`steps=2000 < num_decay_steps=12000` 触发调度器自动压缩（`lerobot/optim/schedulers.py:99-112`），实际是 **83 步 warmup + 2000 步退火**，2000 步时 lr 已到 2.5e-7。任何跨 `steps` 的对照都要先对齐这条 | `f6_s5.md` §4.0 |
| S6-2 | **长跑有真实收益，6000 步饱和**：N1（臂③ 8000 步）运动维均值 `0.05984 → 0.04697`（vs ③@2000 −21.5%，vs ①@2000 −28.1%），`corr(dx) +0.110 → +0.369`；6000→8000 只再降 0.3% | `f6_s5.md` §4.1 |
| S6-3 | **给 state 通路放大 lr 无效（N2 判否）**：`state_path_lr_mult 30` 让 `proprio_in_proj` 位移放大 **5.7–8.9×**（`cos 0.926` vs N1 `0.997`），但 4/4 个探针点上 6 个运动维全部打平（≤1%）。→ 那张 8.2 K 的表不是精度瓶颈 | `f6_s5.md` §4.2 / §4.3 |

**因此「下一步最值得动手的地方」不再是优化这条 proprio 通路，而是部署口径。已经跑了第一步
（`f7_next_steps.md` §3）：close_drawer × 50 seeds 上 π0\* 68% > 老锚点（MEAN_STD/12k/两路）46% >
N1（QUANTILES/8k/三路）34% > 6k 28% > 2k 24%。链路没有问题，**头号嫌疑变成归一化口径 QUANTILES**。**

---

## 2. 关键实现（下一个对话不要重写，直接用）

| 文件 | 已有什么 |
| :--- | :--- |
| `flashvla/policies/pi0/configuration_pi0.py` | `use_proprio_stack`、`proprio_dim`、`real_state_dim`、`proprio_input_dim`；S4：`state_layout`（`raw`/`bridge_euler7`）、`state_stats_path` |
| `flashvla/policies/pi0/modeling_pi0.py` | `PI0Attention` / `PI0MLP` / `PI0ModelLayer` 的 `share_proprio`、`_pack_streams` / `_unpack_streams`、长度断言 |
| `flashvla/policies/pi0/modeling_pi0_flashvla.py` | `project_state`、`_split_suffix_streams`、`_suffix_hidden_states`、`_empty_prefix_streams`、三条推理路径 |
| `flashvla/policies/pi0/utils.py` | `bridge_proprio_from_state`、`quat_wxyz_to_mat`、`mat_to_euler_sxyz`、`BRIDGE_DEFAULT_ROT` |
| `flashvla/policies/pi0/processor.py` | `ProprioLayoutProcessorStep`（注册名 `flashvla_pi0_proprio_layout_processor_step`）、`state_stats_from_json`；插在 **Normalizer 之前** |
| `flashvla/train.py` | `scale_state_path_lr`（覆盖 `suffix_embedder.proprio_in_proj`）、`reinit_proprio_in_proj`（`proprio_in_proj_init_std`） |
| `flashvla/configs/train_config.py` | `state_path_lr_mult`、`state_path_init_std`、`proprio_in_proj_init_std` |
| `scripts/p4/convert_pi0star.py` | `--emit-proprio-stack`、`--proprio-layout`、`--proprio-dim`、`--state-layout`、`--state-stats-path`、`--state-proj-init-std`、`--map-proprio-to-state` |
| `scripts/p4/f6_s4_bridge_proprio.py` | 7 维变换对齐 + 角度范围体检 + 统计量产出 |
| `scripts/p4/f6_s2_smoke_check.py` | ckpt 重载 + 三条推理路径；`--leaf` 走真实帧的 preprocessor |
| `scripts/p4/probe_actions.py` | 离线动作探针（逐维 MAE / corr / 夹爪准确率） |
| `scripts/p4/f6_s2_golden.py` | 两路守恒 golden |
| `tests/test_pi0_proprio_stack.py` | 6 条：三路等价、梯度覆盖、共享权重、长度断言（248 s CPU） |
| `tests/test_pi0_bridge_proprio.py` | 16 条：7 维变换与 processor step（0.02 s CPU） |

### 2.1 现成产物（都在本机，可直接用）

| 类型 | 路径 |
| :--- | :--- |
| fractal 三路起点 ckpt | `ckpt/pi0star_fractal_flashvla_proprio_stack`（`proprio_in_proj=(1024,8)`，rms 0.2044） |
| fractal 两路起点 ckpt（S0-B） | `ckpt/pi0star_fractal_flashvla_proprio`（编码器在 `state_proj`） |
| bridge 三路起点 ckpt | `ckpt/pi0star_bridge_flashvla_proprio_stack`（`(1024,7)`，`verify.ok=true`） |
| 7 维统计量 A（本次用） | `logs/p4/f6_s4_bridge_state7_quantiles.json` |
| 7 维统计量 B（备查） | `logs/p4/f6_s4_bridge_state7_pi0bound.json` |
| 臂① 2000 步 ckpt | `/data/zjr/simpler_lerobot/runs/f6_s0_b_fractal2000/checkpoints/002000/pretrained_model` |
| 臂② 2000 步 run | ~~`runs/f6_s5_2_fractal2000`~~（17 G）— **2026-11-25 已删**，探针 json 仍在 |
| 臂③ 2000 步 run | ~~`runs/f6_s5_3_fractal2000`~~（17 G）— **2026-11-25 已删**，探针 json 与 SIMPLER 24% 仍在 |
| **S6 N1 8000 步 run**（当前最好 ckpt） | `runs/f6_s5_3_fractal8000`（现 17 G，仅 006000/008000；2000/4000/7500 已删） |
| **S6 N2 lr30 8000 步 run**（判否） | ~~`runs/f6_s5_3_fractal8000_lr30`~~（41 G）— **2026-11-25 已删** |
| 探针结果 | `logs/p4/probe_f6s5_{1,2,3}.json`、`logs/p4/probe_f6s5_3_{2000,4000,6000,8000}.json`、`logs/p4/probe_f6s5_3lr30_{2000,4000,6000,8000}.json` |
| yaml | `train/configs/pi0/simpler/_f6s0_{a,b}{,2000}.yaml`、`_f6s2_{smoke,ref2s}.yaml`、`_f6s4_bridge_smoke.yaml`、`_f6s5_{2,3}.yaml`、`_f6s5_3_8000.yaml`、`_f6s5_3_8000_lr30.yaml` |

---

## 3. 未完成清单

### 3.1 必做（路径已明确）

| # | 任务 | 为什么 | 状态 |
| :--- | :--- | :--- | :--- |
| N1 | ~~臂③ 8000 步长跑 + 2000/4000/6000/8000 探针~~ | S5 已判 GO，只有 2000 步（1.26 epoch）的证据 | ✅ 完成，`f6_s5.md` §4.1 |
| N2 | ~~state 通路 lr×10（`state_path_lr_mult: 30`）的 8000 步臂~~ | §1.3：该通路 2000 步几乎不训练 | ✅ 完成并**判否**，`f6_s5.md` §4.2 |
| N3 | M=1 SIMPLER 闸门（fractal 各自任务，50 seeds） | 只有它能给出部署口径的结论 | ⚠️ close_drawer 已评（见 `f7_next_steps.md` §3），五任务未跑 |

### 3.2 可选（有明确条件才做）

| # | 任务 | 触发条件 |
| :--- | :--- | :--- |
| O1 | bridge 上的 proprio A/B（两路 baseline vs 三路 + 预训练编码器） | 要在 bridge 上部署这条通路时。**必须两边都用预训练编码器**（随机编码器已被证有害） |
| O2 | S4 的统计量口径 B（π0\* `bound` p01/p99）单独成臂 | 想把编码器输入尺度对齐 π0\* 训练分布时；换 `state_stats_path` 即可，但要独立成臂 |
| O3 | 三路 + 独立 `proprio_expert`（S1 原方案，+311 M 参数） | 只有当 N1/N2 出现「结构开始起作用」的信号时才值得 |
| O4 | `states_flat` 不再 expand 成 `B*N` 份 | N=4、1 token，FLOPs 可忽略，纯优化项 |
| O5 | FSDP2 多卡跑通三路 | 要上多卡时；目前单卡已验证 |

### 3.3 历史遗留（与本阶段无关，但别忘）

- `pi0_flashvla_bridge_q8000` 的 2000/4000/8000 探针仍**未补跑**（`f6.md` §3）。
- **fractal 的 SIMPLER 闸门工具链没打通**：`logs/p4_eval_fractal.log` 是 0 字节。
  `scripts/p3/run_g3.sh` 的任务表写死 4 个 `widowx_*`；fractal 需要 `google_robot_*` 任务，
  而 `scripts/p0/simpler_client_eval.py --task google_robot_pick_coke_can` 本身是支持 google_robot
  摄像头的（`get_camera_name` 有分支）。→ **N3 需要先补一个 fractal 版驱动脚本**（施工单见 `f7_next_steps.md` §2）。

---

## 4. 下一步施工单 → 已移交给 `f7_next_steps.md`

S6 的施工单（原 §4.1 N1 / §4.2 N2）已执行完毕，结果写在 `f6_s5.md` §4，**不要照原单重跑**：

| 原任务 | 结果 |
| :--- | :--- |
| §4.1 N1（臂③ 8000 步） | ✅ `runs/f6_s5_3_fractal8000`，运动维均值 `0.05984 → 0.04697`，6000 步饱和 |
| §4.2 N2（state 通路 lr×10） | ✅ 跑完但**判否**（权重位移 5.7–8.9× 而精度打平）；run 已于 2026-11-25 删除 |
| §4.3 N3（fractal SIMPLER 闸门） | ⬜ **未开工**，完整施工单见 `f7_next_steps.md` |

⚠️ 原 §4.1 的两条判断需要修正（实测）：
1. 「参考量级（2000 步窗口）`max|d| ≈ 5e-3`，若 6000 步窗口仍是这个量级 → 加步数没用」——
   **这个量级判据本身有偏**：位移严格正比于当步 lr（`f6_s5.md` §4.3），
   `5e-3` 是「lr 已退到 1e-6」的结果，不是「步数没用」。N1 的 2000→4000 窗口是 `2.53e-2`。
2. 「`steps` 改了但 `num_decay_steps` 也改成 8000，所以配方不变」——
   **只要 `steps` 变，lr 轨迹就变**（调度器自动压缩，`f6_s5.md` §4.0），跨 `steps` 不可直接比绝对指标。


---

## 5. 环境与操作陷阱（都踩过）

1. **`train/train.sh` 用裸 `python`**：必须 `PATH=<env>/bin:$PATH`，否则落到系统的
   `/home/zjr/miniconda3/lib/python3.4`（一个 `SyntaxError` 现场）。
2. **`env -u PYTHONPATH -u BASH_ENV`**：仓库根有 `PYTHONPATH` 污染时会串到别的包。
3. **`PI0FlashVLAConfig.from_pretrained()` 不接受 `config.json` 的 `type` 字段**：
   读配置走 `PreTrainedConfig.from_pretrained(path)`，并先 `import flashvla.configs` 注册。
4. **`output_dir` 必须是全新目录**，否则 `auto_resume` 接管并续跑到新 `steps`。
5. **`runs/` 与 `ckpt/` 都在 `.gitignore` 里**：ckpt 不会进 git，交接靠路径不要靠提交。
6. **探针必须沿用 `--samples 24 --seed 0 --leaf google_robot_close_drawer/success`**，
   否则与 `probe_f6s5_{1,2,3}.json` 不可比。
7. **`mae_overall` 不能单独做判据**（被 grip 主导）；**绝对 loss 不能跨归一化口径比**
   （QUANTILES 地板 ≈1.2、MEAN_STD ≈2.0）。本阶段三臂 loss 同口径，可以互比，仅作参考。
8. **bridge 的 smoke loss 天生大**（转换后初始瞬态）：同配方两路 run 在 step 50 是 3.369，
   step 150 才降到 1.428。看到 bridge 的 3.8 不要当成 bug。
9. **（S6 新增）改 `steps` 会改 lr 轨迹**：`CosineDecayWithWarmup.build(optimizer, cfg.steps)`，
   当 `cfg.steps < num_decay_steps` 时 warmup 与退火区间都被压缩（`lerobot/optim/schedulers.py:99-112`）。
   跨 `steps` 的 run 只能在「同一 step 的 lr 相同」时比绝对指标，否则先算 lr。
10. **（S6 新增）`checkpoint_keep_last: 2` + `save_freq: 500` 会删中间 ckpt**：探针需要
    2000/4000/6000 时，必须加 `checkpoint_keep_every_n_steps: 2000`（只改保留策略，不改配方），
    否则每个 2000 步 ckpt 只活 500 步（≈19 min）。
11. **（S6 新增）探针的夹爪分辨率是 1.04 pp**（24 帧 × chunk 4 = 96 个二值）；≤3 pp 的差异是噪声，
    不要把 `grip_accuracy` 0.8125 vs 0.8229 解释成「变差」。

---

## 6. 不要再做的事

1. **不要**给 proprio 通路接随机编码器再和「不用 proprio」对比 —— S5 臂② 已证明它**有害**
   （运动维 +11.8%，6 维只赢 1 维）。任何三路对照两边都要用预训练编码器。
2. **不要**再试图用 init 尺度去「解锁」Adafactor 下的小参数（§1.3 的机制推导）；要改就改 `lr`
   或换优化器，并单独成臂。
3. **不要**在归一化之后做布局转换：quat→euler 是非线性的，必须排在 `NormalizerProcessorStep` 之前。
4. **不要**在 `PI0Attention` 里直接 `cat([vlm, pro, act])`：suffix 是 `[pro_n, act_n…]` 分块交错，
   必须按 `_pack_streams` 重交错。
5. **不要**给 proprio block 单独一套 decoder 权重（π0\* 是共享的）。
6. **不要**删 `state_proj`：它是 `use_proprio_stack=false` 的回退路径，老 ckpt 都依赖它。
7. **不要**用绝对 loss 跨臂比较；不要在没有离线判据的情况下先跑 M=1。

---

## 7. 下一个对话的推荐开场

> 「读 `docs/sdn_flashvla/pi0_flashvla/f7_next_steps.md`，总目标是让 FlashVLA 在 SIMPLER 上跑通。
> 先做 §2（eval 侧 state 布局修复 + fractal 驱动脚本），再按 §3 做单任务 smoke，最后按 §4 跑五任务闸门。」

S6 已经把「加步数 / 加 lr」这条线走完（`f6_s5.md` §4）：**加步数有效但 6000 步饱和，加 lr 无效**。
再往这两个方向投算力都不划算；现在的瓶颈在**部署口径**，不在训练配方。
