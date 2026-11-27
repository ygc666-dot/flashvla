# F7 交接单：FlashVLA 在 SIMPLER 上的部署分数与瓶颈定位

> 总目标（用户口径）：**FlashVLA 在 SIMPLER 上能跑通并给出可引用的部署分数。**
> 上游：`f6_next_steps.md`（S0–S6 已完成，训练侧到顶）；S6 的实测结果与三条修正见 `f6_s5.md` §4。
> **最后更新：2026-11-26（F1 / F2 / 实验 A / 轨迹对比 四组结果全部落盘，诊断闭环）。**

---

## 0. 一页状态

| 项 | 状态 | 证据 |
| :--- | :--- | :--- |
| 训练 S0–S6（fractal，三路 + π0\* 编码器） | ✅ 完成 | `f6_s5.md`（S5 A/B + S6 长跑） |
| **F1：归一化单变量 A/B** | ✅ 完成，**假设成立** | QUANTILES 34% → **MEAN_STD 46%**（同任务同 seeds） |
| **F1 主 ckpt** | ✅ `runs/f7_s1_fractal8000_meanstd/checkpoints/008000` | MEAN_STD + 三路 + 8000 步；离线运动维 MAE **0.04435**（全历史最好） |
| **F2：五任务闸门** | ✅ 完成，**未过关** | macro **19.6%** vs T0 56.4%（−36.8 pp），250 集全跑完 | 
| 逐任务离线探针 | ✅ 五任务全部补测 | 失败任务的 MAE **优于**唯一能跑的任务 → 离线筛选出局（§4.3） |
| 实验 A：同种子对照 | ✅ 完成，**「覆盖不足」被否** | in-dist 26%/28% vs OOD 22%/14%（§4.4） |
| 同种子轨迹对比 | ✅ 完成，**误差是偏置主导** | 第 10 步就偏 4–5 cm，去趋势残差只有 0.3–2.0 cm（§4.5） |
| 瓶颈定位 | ✅ **每 chunk ≈2.7 cm 的系统性动作偏置**（精度地板），不是链路、不是归一化、不是初态覆盖 | §4.3 + §4.5 |

**一句话判断（2026-11-26）**：

> 链路、归一化、步数、初态覆盖、离线可筛选——**五条候选解释已全部实测排除**，唯一存活的是
> 「蒸馏策略每 chunk 有 ~2.7 cm 的系统性动作偏置，闭环 2–3 个 query 内放大到 4–5 cm」。
> 这解释了 macro 的形态：粗接触容差任务 `close_drawer` 留存 71%，精定位任务（`pick`/`move`/`place`）留存 0–39%。
> **`§6 D1`（扩成功轨迹）不再做**：误差地板在 117–223 集的任务上几乎一致（0.034–0.042），不随数据条数变化，且偏置型误差不会被更多同类成功样本消除（§6）。

**可引用的核心数字**（同一 seeds 协议、50 seeds/任务、250 集）：

| | flashvla（MEAN_STD/8k/三路） | T0 = π0\*（in-process） | Δ |
| :--- | ---: | ---: | ---: |
| **macro（5 任务）** | **19.6%** | 56.4% | **−36.8 pp** |
| macro（终态口径） | 18.0% | — | — |
| `close_drawer`（唯一 71% 留存） | 48% | 68% | −20 pp |

---

## 0.1 现在从哪开始

> 前提：这份文档已包含全部结果与可复制命令，不需要回看 `f6_*` 系列。
> 仓库根 = `/home/zjr/workspace_tj/StreamingVLA/flashvla`。

| 若你要… | 做什么 | 去哪 |
| :--- | :--- | :--- |
| **只想要一个可引用的部署分数** | 已经拿到了（§0 表格 + §4.1），直接引用 | §4.1 |
| **想再提升分数** | 唯一对症的杠杆是 **on-policy（DAgger 类）修正**，设计草案见 §7 O4 | §7 |
| 想复核任一结论 | 附录 A 有全部复跑命令 | §A |
| ~~扩数据（D1）/ 补步数（D2）/ 增强（D3）~~ | **不做**，理由见 §6 | §6 |

**先别做的事**：不要再用离线 MAE 预筛任务（§4.3 已证方向相反）；不要用「初态覆盖」给 D1 立项（§4.4 已否）。

---

## 1. 手里有什么（可直接开跑，不用重新准备）

| 类型 | 路径 | 说明 |
| :--- | :--- | :--- |
| **主 ckpt（F1/F2 用）** | `runs/f7_s1_fractal8000_meanstd/checkpoints/008000/pretrained_model` | MEAN_STD + 三路 + 8000 步，41 G（含 2000/4000/6000/7500/8000） |
| 同配方 6000 步 | `runs/f7_s1_fractal8000_meanstd/checkpoints/006000` | 未评测（F2 只评了 8000） |
| 上一代 ckpt（QUANTILES） | `runs/f6_s5_3_fractal8000/checkpoints/{006000,008000}` | N1；34% / 28% |
| 历史 SIMPLER 锚点 | `/data/zjr/simpler_lerobot/runs/pi0_flashvla_fractal_success/checkpoints/012000` | 两路 / MEAN_STD / 12k / lr mult 1.0；close_drawer 46% |
| π0\* 参照（T0） | `third_party/SDN/eval_logs/p3_baseline/vanilla/seeds=2000-2049/<task>/` | `--method vanilla` 逐集 `qpos` |
| π0\* 训练数据来源 | `third_party/SDN/eval_logs/p1_dump/vanilla/seeds=*/<task>/dump/episode_*.npz` | 各任务采集 seed 区间不同，见 §4.4 表 |
| 评测代码 | `third_party/SDN/policy/pizero/evaluate.py` | `--method flashvla` / `vanilla` |
| 服务端 | `scripts/p3/flashvla_server.py` | `cold_start_burst=True`：排空在一次 query 内完成，不浪费 env 步 |
| **单任务评测** | `scripts/p3/eval_flashvla_task.sh` | **新增 `F7_LOGDIR`**：日志目录可覆盖（默认 `logs/`） |
| **五任务驱动（本轮新增）** | `scripts/p3/run_f1.sh` | `full` / `smoke` / `report`；聚合写 `F2_<mode>_gate.json` |

**本轮结果目录**

| 目录 | 内容 |
| :--- | :--- |
| `third_party/SDN/eval_logs/f7_p0/` | close_drawer × 50（P0 MEAN_STD 8000） |
| `third_party/SDN/eval_logs/f7_f2/008000/` | **五任务 × 50（F2 闸门）** |
| `third_party/SDN/eval_logs/f7_f3_indist/` | 实验 A：pick/move seeds 1000–1049 |
| `third_party/SDN/eval_logs/f7_f3_traj/` | 轨迹对比的 4 集学生 dump（301 M，可删） |
| `logs/05_f7_deploy/` | 本轮全部日志：训练/续训、链式、闸门、探针、轨迹、`05_f7_deploy/f2_gate/F2_full_gate.json` |
| `logs/05_f7_deploy/probe_tasks/*.json` | 4 个任务的离线探针（close_drawer 的见 `logs/05_f7_deploy/probe_f7s1_meanstd_8000.json`） |

### 1.1 T0 参照与本轮实测（seeds 2000–2049，50 seeds/任务）

| 任务 | T0（π0\*） | flashvla（008000） | Δ | **留存率** | 终态口径 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `google_robot_pick_coke_can` | 0.86 | **0.22** (11/50) | −64 pp | 26% | 20% (10/50) |
| `google_robot_move_near` | 0.74 | **0.14** (7/50) | −60 pp | 19% | 12% (6/50) |
| `google_robot_close_drawer` | 0.68 | **0.48** (24/50) | −20 pp | **71%** | 46% (23/50) |
| `google_robot_open_drawer` | 0.36 | **0.14** (7/50) | −22 pp | 39% | 12% (6/50) |
| `google_robot_place_apple_in_closed_top_drawer` | 0.18 | **0.00** (0/50) | −18 pp | 0% | 0% (0/50) |
| **macro** | **0.564** | **0.196** | **−36.8 pp** | 35% | 0.180 |

> 口径：`000_success_<rate>.log` 的文件名比例 = 「**曾**成功过一次」的集数；`000_episode_stats.json` 的
> `final_success` = 终态。两者在三个任务上差 1 集（例：`close_drawer` 48% vs 46%）。
> 引用时**必须写清是哪一个**，否则会和 F1 的 46% 打架。
> 分辨率：50 seeds 的 SE ≈ 4.9–7.1 pp，单任务 10 pp 以内不要解读；结论一律看 macro。

---

## 2. 三条口径（已核对，**不需要改代码**）

### 2.1 fractal 的 state/action 与 eval 天然一致

1. fractal 数据集来自 SDN 自己的 eval dump（`p1_dump/.../dump/episode_*.npz`）。
2. `sim_eval/simpler/convert_simpler_to_lerobot.py:259` 里 `observation.state = proprio[t]` 原样搬运。
3. `third_party/SDN/policy/pizero/dump_utils.py:7` 的 `proprio = obs['agent']['eef_pos']` 是 8 维原值 `[x,y,z,qw,qx,qy,qz,grip_openness]`。
4. eval 侧 `flashvla_client.py:150` 发送的也是 `proprio[:8]` 原值 → 同一管道，无错位。
5. 动作同理：dump 存的是「真正下发给 env 的 command」，client 原样回填。

> ⚠️ fractal **不要**加 `state_layout` 转换（那是 bridge 需要的）。

### 2.2 图像分辨率：**不存在**纵横比差异（F1 时期的怀疑已排除）

原表曾说「训练 512×640 / eval 480×640」——**不成立**。证据：

- `evaluate.py:218` 里 `image = get_image_from_maniskill2_obs_dict(env, obs)`，**`vanilla`（T0）与 `flashvla` 走同一个函数**；
- 训练数据的 `dump/meta` 记录 `camera: overhead_camera`、`image_shape [512, 640, 3]`、`fps 3`；
- 因此训练与评测是同一张 512×640 的 `overhead_camera` 图，同一 `resize_with_pad(224)`。
- `flashvla_server._build_batch` 的注释里那句「480x640」是 **bridge** 的遗留描述，不适用 fractal。

→ 纵横比不是本轮掉点的原因，**不要再往这个方向查**。

### 2.3 服务器/评测的非模型变量（已知行为，不要当 bug 调）

- `exec_steps=4`、`chunk_size=4`、`n_action_steps=4`：与训练一致；三者必须相等，否则 buffer 错位。
- `cold_start_mode=zero_delta` + `num_buffer_slots=4` + **`cold_start_burst=True`**：冷启动排空在**一次 query 内**完成
  （`flashvla_server.py:225-235`），**不占用 env 步**。→ 「episode 前几秒不动」不会发生；若真出现，才是 bug。
- server 会校验被服务的 ckpt 与 `--checkpoint` 是否同一路径，不匹配直接报错（`flashvla_client.py:100-105`）。
- 显存实测：训练峰值 **25.5 GB**；评测（server + SIMPLER 渲染，`--no-video`）只要 **≈11.8 GB**。
  → 在只剩 31 GB 的 GPU2 上（别人 isaac-sim 占 15 GB）两者都能跑，评测反而宽松。

---

## 3. F1 里程碑：归一化单变量 A/B（已完成，**假设成立**）

同一任务 `google_robot_close_drawer`、同 seeds（2000–2049）、同 server/client，只换 ckpt。

### 3.1 结果

| ckpt | 配方 | 成功率（曾/终态） | 近失手 0.05–0.08 | 0.08–0.12 | **>0.12（几乎没动）** | 中位 qpos | 离线运动维 MAE |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| π0\*（T0） | in-process，非蒸馏 | **68%** | 8 | 1 | 7 | 0.017 | — |
| 老锚点 012000 | **MEAN_STD** + 两路 + 12k | **46%** | 9 | 3 | 15 | 0.057 | 0.04544 |
| **P0 008000** | **MEAN_STD** + 三路 + 8k | **48% / 46%** | **5** | **2** | 19 | **0.066** | **0.04435** |
| N1 008000 | **QUANTILES** + 三路 + 8k | 34% | 4 | 7 | 22 | 0.107 | 0.04697 |
| N1 006000 | QUANTILES + 6k | 28% | 5 | 3 | 28 | 0.132 | 0.04712 |
| 臂③ 002000 | QUANTILES + 2k | 24% | 0 | 5 | 33 | 0.154 | 0.05984 |

### 3.2 结论

1. **QUANTILES 是闭环退化主因**：单变量 MEAN_STD 把 34% → 46–48%，与同为 MEAN_STD 的老锚点持平；中位 qpos 0.107 → 0.066。
2. **步数无罪**：8k MEAN_STD 已追平 12k MEAN_STD → D2（补 12000 步）作废。
3. **三路 vs 两路在闭环上未分离**（46% vs 46%），三路离线略优 → O1 降级（§7）。
4. 机制吻合：QUANTILES（p01/p99 → [−1,1]）在闭环里把 OOD state 饱和截断，智能体被自己带出训练分布后无法自救；
   「完全没动」22 → 19。

---

## 4. F2 里程碑：五任务闸门（已完成，**未过关**）

### 4.1 结果（`logs/05_f7_deploy/f2_gate/F2_full_gate.json`）

见 §1.1 表。判据：macro ≥ 0.544 为「部署口径不掉点」；**实测 0.196 → `gate_pass=false`**。

补齐的「非模型变量」核对（全部正常，不是 bug）：

- 5 个任务各 50 集**全部跑完**，无 Traceback，server 每次 `ready ~150 s`，`--reload-every 0`；
- 指令文本正确：`pick coke can` / `move orange near pepsi can` / `close bottom drawer` / `open bottom drawer` / `open top drawer`；
- **多阶段任务的指令逐步更新**（`evaluate.py:273-277` 每步重取 `get_language_instruction()`，client `step()` 每步带 instruction）
  → `place_apple` 的 0% 不是「没收到第二阶段指令」；
- 单相机口径与训练一致（§2.2）。

### 4.2 为什么这不是「某些任务没学会」

`close_drawer` 48% 与 `pick_coke_can` 22% 的差别不可能是「覆盖多少」——见 §4.3/§4.4 的两组实测。

### 4.3 逐任务离线探针：**离线筛选彻底出局**

同 ckpt、24 帧/任务、`scripts/p4/probe_actions.py`（含 cold-start burst 与 ckpt 自带的 postprocessor）：

| 任务 | 离线运动 6 维 MAE | `mae_overall` | `corr(dx)` | grip acc | 闭环留存 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `pick_coke_can` | 0.04199 | 0.09370 | 0.383 | 0.802 | 26% |
| `move_near` | **0.03628** | 0.08579 | **0.640** | 0.812 | 19% |
| `close_drawer` | **0.04435（最差）** | 0.09298 | 0.554 | 0.812 | **71%（最好）** |
| `open_drawer` | **0.03431** | 0.08666 | 0.502 | 0.802 | 39% |
| `place_apple` | **0.03472** | 0.08179 | 0.582 | 0.823 | 0% |

（mean-action baseline 0.157–0.183，五个任务都远好于它 → 头不是「塌成常数」。）

**读法**：闭环最惨的 4 个任务，离线 MAE **全部优于**唯一能跑的 `close_drawer`。
→ 离线指标与闭环能力在这条线上**方向相反**；任何「先看 MAE 再决定要不要花 GPU 评测」的做法都会误判。
→ 也否掉「蒸馏头只在 close_drawer 上调好了」这个假设。

### 4.4 实验 A：用**训练集自己的 seeds** 评（否掉「初态覆盖」）

| 任务 | in-dist（1000–1049） | OOD（2000–2049） | Δ | π0\* 在该采集区间 |
| :--- | ---: | ---: | ---: | ---: |
| `pick_coke_can` | **26%** (13/50) | 22% (11/50) | +4 pp | **89.3%**（1000–1149） |
| `move_near` | **28%** (14/50) | 14% (7/50) | +14 pp（≈1.75 SE） | **72.0%**（1000–1199） |

π0\* 在其余任务采集区间的自身成功率：`close_drawer` 79.5%（1000–1199）、`open_drawer` 44.6%（1000–1499）、
`place_apple` 23.4%（1000–1499）——与 T0 的 68/36/18% 同量级，**两个 seed 区间对 π0\* 难度相当**。

**读法**：回到模型训练时用过的初始状态，成功率几乎不动（+4/+14 pp），仍只有 26–28%，而老师是 72–89%。
→ **初始状态覆盖不是约束**；§6 里 D1 的立论（「训练集没有这些初态」）被直接证伪。
→ 若 D1 还有价值，只能是通过「更多数据降低 BC 误差」，而不是「覆盖更多初态」——而这一点被 §4.5/§6 的误差地板否定。

### 4.5 同种子轨迹对比：误差是**偏置**，不是抖动

方法：`pick_coke_can` seeds 1000–1003 用同一 ckpt 重跑并 `--dump-episodes --dump-images npz`，
与学生逐帧 `proprio[:, :3]` 对齐老师（`p1_dump/vanilla/seeds=1000-1149/.../episode_0100X.npz`）。

| seed | 老师 | 学生 | 偏离 @10 步 | @20 | @40 | 末帧 | 去趋势后残差 std |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 1000 | ✅ 第 18 步 | ❌ | **4.3 cm** | 3.4 | 6.6 | 15.1 | 0.7–2.0 cm |
| 1003 | ✅ 第 18 步 | ❌ | **5.3 cm** | 4.0 | 4.7 | 11.0 | 0.3–1.1 cm |
| 1001 | ❌（老师也失败） | ❌ | 10.5 | 11.9 | 14.1 | 6.6 | 1.4–3.1 cm |
| 1002 | ❌（老师也失败） | ❌ | 6.9 | 14.9 | 13.1 | 26.0 | 1.2–2.7 cm |

末 20 步的**有符号**偏置（x,y,z，cm）：seed 1000 = (−9.6, −11.5, −1.6)；seed 1003 = (+4.7, −7.6, +5.4)。
**偏置是残差的 5–10 倍**。夹爪时序也错：老师第 20 步已闭合（grip 0.47），学生第 40 步才半闭（0.174）、末段又张回 1.0。

**三条合起来的定量结论**

1. 开放环（demo 帧、on-manifold）误差就有 **2.7 cm**（`dx` 0.0266 / `dz` 0.0271，且 `dz` 是**过冲**：pred|dz| 0.0309 > gt|dz| 0.0232）；
2. 闭环里 **2–3 个 query 就累到 4–5 cm**，可乐罐半径 ≈3.3 cm → 机制上必然抓空；
3. 残差只有 0.3–2.0 cm → **加数据不会消掉偏置**；容差粗的抽屉任务（任何接触都能推上）留存 71%，精定位任务 0–39%。
   **这就是 macro 的全部形态。**

---

## 5. 分层定位表（**L1–L6 已全部排除**；留作以后出现 0% 时用）

| 层 | 查什么 | 判据 | 本轮状态 |
| :--- | :--- | :--- | :--- |
| **L1** | 模型本身还行吗 | `probe_actions.py` 运动维 MAE ≤0.05 | ✅ 五任务 0.034–0.044 |
| **L2** | 动作被 cold-start 吃掉？ | 前 12 步世界向量范数 | ✅ burst 在单次 query 内完成，不吃 env 步 |
| **L3** | state 串了？ | 逐维落在数据集 [q01,q99] | ✅ §2.1 |
| **L4** | 夹爪极性？ | dump 的 gripper vs 数据集末维 | ✅ 同号 |
| **L5** | 图像通道/朝向？ | 与数据集同 episode 比对 | ✅ 同函数同尺寸（§2.2） |
| **L6** | server/client 串了别的 ckpt？ | client 校验路径 | ✅ 不一致会直接报错 |

历史现场（别重复踩）：`logs/p4_eval_fractal.log` 是 0 字节 = 「起了评测但 server 没起」；
`eval_logs/p4_pi0star/bridge/*` 是 bridge 的烂结果（8/36/6/0%），**不要拿来推断 fractal**。

---

## 6. 「数据覆盖」假说已被实测否定 → **D1/D2/D3 不做**

原 §6 的立论与反证：

| 原假说 | 反证 |
| :--- | :--- |
| 「训练集没有评测用的初始状态」→ 扩 seeds 就能补 | **实验 A**：换成训练集自己的 seeds，成功率只 +4/+14 pp（26%/28% vs 22%/14%），仍远低于老师的 72–89%（§4.4） |
| 「只有成功轨迹，没有失败/恢复样本」→ 加失败样本 | 失败结构是**偏置漂移**（§4.5），且 `close_drawer` 也只有成功轨迹却能到 48% → 缺的不是「失败样本」这一种数据 |
| 「数据条数不够」→ 3 倍数据降误差 | 误差地板在 117–223 集的任务上几乎一致（0.034–0.042，§4.3），**不随条数变化**；偏置型误差不会被更多同类成功样本消除 |

**结论**：`D1`（采 seeds 1200–1599、≈7 h）**预期收益极低，不做**。`D2`（补 12000 步）已被 F1 否掉（§3.2）。
`D3`（增强）只在「误差是方差」时才有意义，而 §4.5 显示是偏置。

> 若将来仍要动数据，唯一有意义的形式是**带老师标签的 on-policy 数据**（DAgger），见 §7 O4 ——
> 它改变的不是「覆盖」而是「标签落在学生真正会走到的状态上」。

---

## 7. 可选臂（排序已按本轮结果更新）

| # | 臂 | 目的 | 代价 | 优先级 |
| :--- | :--- | :--- | :--- | :--- |
| **O4** | **on-policy 修正（DAgger 类）**：学生走、老师在同状态打标签，混合数据微调 | **直接消掉 §4.5 的偏置**——这是唯一对症的杠杆 | 采集 ≈1.5–2 h + 微调 5 h + 复评 2 h | **P0（若继续）** |
| O1 | MEAN_STD + **两路** + 8000 步 | 分离「三路 vs 两路」；但 F1 已显示两者闭环同分（46% vs 46%），只有离线差 2% | 5 h | P3（低） |
| O2 | state 通路换 AdamW | S6 已证 lr 不是瓶颈，只剩优化器形态 | 5 h | P3（低） |
| O3 | bridge 同构 A/B | 只在要部署 bridge 时需要 | 5 h + 2 h | — |
| ~~D1/D2/D3~~ | — | 见 §6 | — | **不做** |

### 7.1 O4 设计草案（未实施，最小可行版）

1. **同 loop 双策略**：`evaluate.py` 已有 `--method vanilla`（in-process π0\*）；再加一个 `--method dagg`：
   每 k 个 query，**学生**（flashvla server）出动作推进 env，同时用**老师**（in-process π0\*）在**学生当前 obs**上
   产生 4 步 chunk 标签，落盘成与 `p1_dump` 同格式的 npz（`proprio/action/world_vector/rot_axangle/gripper/instruction`）。
2. **数据口径必须与现有转换器一致**：`fps 3`、`proprio` = `obs['agent']['eef_pos'][:8]`、`action` = 真正下发的 command、
   `truncate_k: 2`；**但不要 `outcome_split`**（DAgger 数据的价值就在失败/偏离段，标签是老师的，不是结果的）。
3. **混合比例**：demo : dagg = 1 : 1（或 1 : 2），否则会忘掉 demo 分布。
4. **成本估算**：采集速度 ≈ 评测速度（50 集 ≈10–15 min）→ 每任务 300–500 集 ≈ 1–1.5 h；两卡（学生 server + 老师）显存各 ≈10 GB，
   一张 46 GB 卡能同时放下。
5. **判据**：`pick_coke_can` seeds 1000–1049 与 2000–2049 都必须升；**若只有 in-dist 升而 OOD 不动，说明仍有覆盖成分**。
   同时看 §4.5 的「@10 步偏离」是否从 4–5 cm 降到 <2 cm。

---

## 8. 环境与操作陷阱（SIMPLER 侧，都踩过）

1. **两个 env**：server 用 `py3.12 flashvla`，`evaluate.py` 用 `py3.10 sdn`。
2. **必须 `cd third_party/SDN`** 再 `python -m policy.pizero.evaluate`。
3. **端口 = `12400 + GPU 号`**；一张卡一个 server。
4. **`--reload-every 0`**：默认 8 会每 8 集重建 policy（对 flashvla 会重连 server）。
5. **`--no-video`**：省 ~30% 时间；显存也更低（≈11.8 GB）。
6. **一张卡一组**：server(≈10 GB) + SIMPLER(≈15–20 GB) 放同一张卡。
7. **`flock`**：`run_f1.sh` / `run_g3.sh` 都用全局锁防双开。
8. **`CUDA_VISIBLE_DEVICES` 对两个进程都要设**。
9. **（本轮新增）系统 `python3` 是 3.4**，不支持 f-string → 脚本里的聚合要用 `$ENVS/flashvla/bin/python`。
10. **（本轮新增）`timeout` 前面不能挂 env 赋值**：`timeout 5400 F7_LOGDIR=... script` 会 rc=127，
    必须写成 `F7_LOGDIR=... timeout 5400 script`。
11. **（本轮新增）中断续训**：直接重跑同一条 `train.sh` 命令即可，`auto_resume` 会读 `checkpoints/last`
    并恢复 optimizer/scheduler/rng；**不要删 `checkpoints/last`**。核对：进度条分母 = 剩余步数、
    `step:` 计数连续、`lr` 落在余弦轨迹上（若重启 warmup 会看到 2.6e-06 这种量级的 lr）。

---

## 9. 一页决策树（**已更新到 F2 完成之后**）

```
F1（完成）：close_drawer × 50 seeds
  π0* 68% | MEAN_STD/8k/三路 46-48% | MEAN_STD/12k/两路 46% | QUANTILES/8k 34% | 6k 28% | 2k 24%
  → 归一化是主因（单变量成立）；步数无罪
        │
        ▼
F2（完成）：5 任务 × 50 seeds = macro 19.6% vs T0 56.4%（gate 54.4%）
  close 48% | pick 22% | move 14% | open 14% | apple 0%
        │
        ├─ 逐任务离线探针 → 失败任务 MAE 反而更好 → 离线筛选出局
        ├─ 实验 A（训练 seeds）→ 26%/28% ≈ OOD → 初态覆盖出局
        └─ 同种子轨迹 → @10 步偏 4-5cm、残差 0.3-2cm → 偏置主导（精度地板 ~2.7cm/chunk）
        │
        ▼
结论：不是链路 / 不是归一化 / 不是步数 / 不是覆盖 / 不是头没学会
      = 每 chunk 2.7cm 系统性偏置，粗容差任务能活、精定位任务全灭
        │
        ├─ 想要「可引用分数」→ 已到手（§0/§4.1），引用时写清「曾/终态」口径
        └─ 想再提升 → 只有 on-policy（§7 O4）对症；D1/D2/D3 不做
```

---

## 10. 不要再做的事

1. 不要用 ≤3 pp 的夹爪二分类差异当判据（96 个二值 → 1 个样本 = 1.04 pp）。
2. 不要拿「同 step 但不同 `steps`」的 run 比绝对指标（调度器会压缩退火区间，`f6_s5.md` §4.0）。
3. 不要优化 `proprio_in_proj`（S6 已证位移放大 8× 而精度不变）。
4. 不要因为 `p4_pi0star/bridge` 的 8/36/6/0% 推断 fractal。
5. 不要给 fractal 加 `state_layout` 转换；也不要改成 `EDRSimplerAdapter` 的 xyzw+closedness。
6. **（本轮新增）不要用离线 MAE 预筛任务/臂**：本轮实测方向相反（最差 MAE 的任务闭环最好）。
7. **（本轮新增）不要再用「初态/seed 覆盖不足」给扩数据立项**：实验 A 直接测过（§4.4）。
8. **（本轮新增）不要把 `48%` 和 `46%` 混用**：前者是「曾成功」文件名口径，后者是 `final_success` 口径。

---

## 附录 A：本轮全部命令（可直接复跑）

```bash
cd /home/zjr/workspace_tj/StreamingVLA/flashvla
ENVS=/home/zjr/workspace_tj/StreamingVLA/miniconda_ygc/envs
CK=$PWD/runs/f7_s1_fractal8000_meanstd/checkpoints/008000/pretrained_model

# --- 训练（P0：MEAN_STD + 三路 + 8000 步）---
PATH=$ENVS/flashvla/bin:$PATH CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  nohup bash train/train.sh train/configs/pi0/simpler/_f7s1_meanstd.yaml 1 \
  > logs/05_f7_deploy/f7_s1_meanstd_8000_resume.log 2>&1 &
# 中断后重跑同一条命令即可（auto_resume 从 checkpoints/last 接力）

# --- Step 3 离线探针（单任务 / 逐任务）---
env -u PYTHONPATH -u BASH_ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 \
  $ENVS/flashvla/bin/python scripts/p4/probe_actions.py --ckpt "$CK" \
  --leaf /data/zjr/simpler_lerobot/fractal/google_robot_pick_coke_can/success \
  --samples 24 --seed 0 --device cuda --json logs/05_f7_deploy/probe_tasks/pick.json

# --- Step 4 单任务评测（close_drawer × 50）---
F7_LOGDIR=$PWD/logs/05_f7_deploy scripts/p3/eval_flashvla_task.sh --gpu 2 --ckpt "$CK" \
  --out "$PWD/third_party/SDN/eval_logs/f7_p0" \
  --task google_robot_close_drawer --n 50 --seed 2000 --tag meanstd_8k

# --- F2 五任务闸门（≈2 h，单卡）---
P3_F1_GPUS=2 P3_F1_CKPTS=008000 bash scripts/p3/run_f1.sh full
bash scripts/p3/run_f1.sh report          # 只聚合，不评测

# --- 实验 A：同种子对照（seeds 1000–1049）---
for t in google_robot_pick_coke_can google_robot_move_near; do
  F7_LOGDIR=$PWD/logs/05_f7_deploy/f3_indist scripts/p3/eval_flashvla_task.sh --gpu 2 --ckpt "$CK" \
    --out "$PWD/third_party/SDN/eval_logs/f7_f3_indist" --task $t --n 50 --seed 1000 --tag indist
done

# --- 轨迹对比：dump 学生 4 集（老师同 seed 的 dump 已在 p1_dump 里）---
# server 起好后：
cd third_party/SDN && CUDA_VISIBLE_DEVICES=2 $ENVS/sdn/bin/python -m policy.pizero.evaluate \
  --method flashvla --task google_robot_pick_coke_can --checkpoint "$CK" \
  --num-gpus 1 --n-trajs 4 --start-seed 1000 --reload-every 0 --no-video \
  --dump-episodes --dump-images npz \
  --result-root "$PWD/eval_logs/f7_f3_traj"    # 已在 third_party/SDN 内
```
