# F7 交接单：让 FlashVLA 在 SIMPLER 上跑通

> 总目标（用户口径）：**FlashVLA 在 SIMPLER 上能跑通并给出可引用的部署分数。**
> 上游：`f6_next_steps.md`（S0–S6 已完成，训练侧到顶）；S6 的实测结果与三条修正见 `f6_s5.md` §4。
> **最后更新：2026-11-25。**

---

## 0. 一页状态

| 项                                          | 状态                                                                                  | 证据                                                     |
| :------------------------------------------ | :------------------------------------------------------------------------------------ | :------------------------------------------------------- |
| 训练 S0–S6（fractal，三路 + π0\* 编码器） | ✅ 完成                                                                               | `f6_s5.md`（S5 A/B + S6 长跑）                         |
| 当前最好 ckpt                               | ✅ `runs/f6_s5_3_fractal8000/checkpoints/008000`                                    | 离线运动维 MAE 0.04697                                   |
| **SIMPLER 链路是否通**                | ✅ **通**（本轮实测 4 个 ckpt 全部跑完 50 seeds，无一次失败）                   | `eval_logs/f7_f1*/`、`eval_logs/p4_pi0star/fractal/` |
| **SIMPLER 部署分数**                  | ⚠️ **close_drawer 上 34%**（N1 8000，50 seeds），低于老锚点 46%，π0\* 是 68% | §3.1                                                    |
| 五任务闸门                                  | ⬜ 未跑（close_drawer 都还没到 46%，跑了也没有解释力）                                | —                                                       |

**一句话判断（2026-11-25 实测后更新）**：
链路没问题（4 个 ckpt × 50 seeds 全跑完，state/相机/夹爪口径都对），**问题是闭环分数**：
同一条 `close_drawer`×50 seeds 上，π0\* 68% > 老锚点（MEAN_STD/12000 步/两路）46% >
N1 8000（QUANTILES/三路）**34%** > N1 6000 28% > 臂③ 2000 24%。
→ **下一阶段的头号嫌疑是「归一化口径 QUANTILES」（F1）而不是结构、不是步数**，见 §3.3。

---

## 0.1 现在从哪开始（三步，按顺序执行）

> 前提：这份文档已经包含**全部可复制的命令**，不需要回看 `f6_*` 系列文档。
> 所有路径都以 `/home/zjr/workspace_tj/StreamingVLA/flashvla` 为仓库根；先 `nvidia-smi` 确认 GPU4/GPU2 空闲。

| 步           | 做什么                                                         | 命令在哪          | 耗时   | 判据                                                              |
| :----------- | :------------------------------------------------------------- | :---------------- | :----- | :---------------------------------------------------------------- |
| **①** | 跑 P0 臂：`MEAN_STD` + 三路 + 8000 步（yaml 已生成）         | §3.3.1 Step 0–2 | 5.0 h  | 起跑后三行核对（`state_path_lr_mult=3.0 -> 4 param tensor(s)`） |
| **②** | 同一任务同一 seeds 评 P0                                       | §3.3.1 Step 4    | 22 min | ≥46% / 38–45% / ≈34% → 三选一，按 §3.3 表分流                |
| **③** | 分流后：进五任务闸门（§4）或两路对照（§7 O1）或扩数据（§6） | §4 / §7 / §6   | 2–7 h | macro ≥ 0.544                                                    |

**不要跳步**：③ 的五任务闸门只有在 close_drawer ≥46% 时才有解释力（分母与 T0 同源）。

---

## 1. 手里有什么（可直接开跑，不用重新准备）

| 类型                           | 路径                                                                                                                                       | 说明                                                                                                              |
| :----------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------- |
| **主 ckpt A**            | `runs/f6_s5_3_fractal8000/checkpoints/008000/pretrained_model`                                                                           | 三路 + π0\* 编码器，8000 步，运动维 0.04697（最好）                                                              |
| **主 ckpt B**            | `runs/f6_s5_3_fractal8000/checkpoints/006000/pretrained_model`                                                                           | 同上；`mae_overall` 口径更好（0.09250 vs 0.09535），夹爪更稳                                                    |
| 备查 ckpt                      | ~~`runs/f6_s5_3_fractal8000_lr30/checkpoints/008000`~~                                                                                  | **已于 2026-11-25 删除**（lr30 判否）；见 §1.3，结论与探针 json 仍在                                       |
| **历史 SIMPLER 锚点**    | `/data/zjr/simpler_lerobot/runs/pi0_flashvla_fractal_success/checkpoints/012000`                                                         | 两路 / MEAN_STD / 12000 步 / `state_path_lr_mult=1.0`；close_drawer **46%**                               |
| π0\* 参照线                   | `logs/p3_overnight/gates/P2_t0.json`（汇总）+ `third_party/SDN/eval_logs/p3_baseline/vanilla/seeds=2000-2049/<task>/`（逐集 `qpos`） | T0 = `--method vanilla`（in-process π0\*，权重 `third_party/SDN/pretrained/open-pi-zero`），seeds 2000–2049 |
| 评测代码                       | `third_party/SDN/policy/pizero/evaluate.py`（`--method flashvla`）                                                                     | 已支持 google_robot 相机与 8 维 proprio                                                                           |
| 服务端                         | `scripts/p3/flashvla_server.py`                                                                                                          | 起服 ~135 s；`--ckpt` / `--port` / `--no-cold-burst` 等                                                     |
| **单任务评测（已入库）** | `scripts/p3/eval_flashvla_task.sh`                                                                                                       | 本次 F1 用的就是它；`--gpu/--ckpt/--out/--task/--n/--seed`                                                      |
| 五任务驱动模板                 | `scripts/p3/run_g3.sh`（bridge 版，4 个 widowx 任务）                                                                                    | 复制成 `run_f1.sh`，把内层换成上面的脚本，见 §4.1                                                              |

### 1.1 五任务的 T0 参照（`P2_t0.json`，seeds 2000–2049，50 seeds）

| 任务                                              |     T0（π0\*） | 备注                                                    |
| :------------------------------------------------ | --------------: | :------------------------------------------------------ |
| `google_robot_pick_coke_can`                    |            0.86 | 训练集 3984 帧 / 144 eps                                |
| `google_robot_move_near`                        |            0.74 | 14793 帧 / 223 eps（最大）                              |
| `google_robot_close_drawer`                     |            0.68 | 8010 帧 / 159 eps；**唯一已跑过 flashvla 的任务** |
| `google_robot_open_drawer`                      |            0.36 | 3783 帧 / 134 eps（最小）                               |
| `google_robot_place_apple_in_closed_top_drawer` |            0.18 | 20273 帧 / 117 eps                                      |
| **macro**                                   | **0.564** | 闸门 = macro − 2 pp = **0.544**                  |

> 分辨力：50 seeds 的 SE ≈ 3.5–7 pp（p≈0.5 时最大 7.1 pp）。**单任务差 <10 pp 不要解读**，
> 结论一律看 macro（5 任务平均）。

### 1.2 老锚点（46%）的离线锚点 —— 本轮补测，用来预测 F1 的天花板

| ckpt          | 配方                                               | 离线 6 运动维均值 |   `mae_overall` |     `corr(dx)` |       SIMPLER close_drawer |
| :------------ | :------------------------------------------------- | ----------------: | ----------------: | ---------------: | -------------------------: |
| 老锚点 012000 | 两路 / **MEAN_STD** / 12000 步 / lr mult 1.0 | **0.04544** | **0.08808** | **+0.558** |              **46%** |
| 新 A 008000   | 三路 / **QUANTILES** / 8000 步 / lr mult 3.0 |           0.04697 |           0.09535 |           +0.369 | **34%**（SE 6.6 pp） |
| 新 B 006000   | 同上（6000 步）                                    |           0.04712 |           0.09250 |           +0.369 |                        28% |
| 臂③ 002000   | 同上（2000 步）                                    |           0.05984 |           0.10343 |           +0.110 |                        24% |

**这张表是 F1 的预期管理**：老锚点在离线指标上**比新 ckpt 还好一点**（运动维 0.04544 vs 0.04697，
`corr(dx)` +0.558 vs +0.369），所以「新 ckpt 上线后 SIMPLER 大涨」不是合理预期。
老锚点自己的 SIMPLER 只有 46%（vs π0\* 68%）→ 反过来说明**离线 MAE 不是 SIMPLER 成功率的充分统计量**，
离线差 3% 在 SIMPLER 上可能差 10 pp（也可能是 ±SE 的噪声）。

> 注：老锚点是 MEAN_STD + 12000 步 + 两路，新 ckpt 是 QUANTILES + 8000 步 + 三路，**差三个变量**。
> F1 已测完且差异明显（46% vs 34%），所以第一件事就是把「归一化口径」单独拉出来（§3.3 的 P0）。

---

### 1.3 磁盘清理记录（2026-11-25，共释放 ≈150 G）

判据：**结论 / 探针 json / SIMPLER 结果已落盘，且后续施工单不会再读该 ckpt**。

| 已删除                                                          | 原占用 | 为什么可以删                                                  |
| :-------------------------------------------------------------- | -----: | :------------------------------------------------------------ |
| `runs/f6_s5_3_fractal8000_lr30`（整包）                       |   41 G | N2 判否（`f6_s5.md` §4.2），8 个探针 json 已留             |
| `runs/f6_s5_3_fractal2000`（整包）                            |   17 G | S5 臂③；离线 json + SIMPLER 24% 已记录                       |
| `runs/f6_s5_2_fractal2000`（整包）                            |   17 G | S5 臂②，已证有害                                             |
| `runs/f6_s2_smoke_fractal`、`f6_s2_ref2s_smoke_fractal`     |   34 G | S2/S3 冒烟，结论在 `f6_s2_s3.md`                            |
| `runs/f6_s4_bridge_smoke`                                     |   17 G | S4 冒烟，结论在 `f6_s4.md`；bridge 起点 ckpt 仍在 `ckpt/` |
| `runs/f6_s5_3_fractal8000/checkpoints/{002000,004000,007500}` |   24 G | 趋势点与 SIMPLER 24% 已记录；留 6000/8000 足够                |

**保留**：`runs/f6_s5_3_fractal8000/checkpoints/008000`（主 ckpt）、`.../006000`（F2 第二个 ckpt），
合计 17 G。若要复跑 2000/4000 的探针，只能重训（5 h）——2026-11-25 之后不要再假设这些路径存在。
起点 ckpt（`ckpt/pi0star_*`，77 G）**未动**，它们是所有臂的 `pretrained_path`。

## 2. 三条口径先对齐（已核对，**不需要改代码**）

### 2.1 fractal 的 state/action 与 eval 天然一致（这点比 bridge 简单）

证据链：

1. `p1_provenance.json` 显示 fractal 数据集的来源是 **SDN 自己的 eval dump**：
   `third_party/SDN/eval_logs/p1_dump/vanilla/seeds=1000-1199/<task>/dump/episode_*.npz`。
2. `sim_eval/simpler/convert_simpler_to_lerobot.py:259` 里 `observation.state = proprio[t]` **原样搬运**。
3. `third_party/SDN/policy/pizero/dump_utils.py:7` 里 `proprio = obs['agent']['eef_pos']`，是
   **8 维原值** `[x, y, z, qw, qx, qy, qz, grip_openness]`。
4. eval 侧 `flashvla_client.py:150` 发送的也是 `proprio[:8]` 原值 → **同一条管道，无错位**。
5. 动作同理：dump 存的是「真正下发给 env 的 command」（`evaluate.py:249` 拼接 world_vector /
   rot_axangle / gripper），grip ∈ [−1, 1]（数据集统计 `min=-1.0, max=1.078`），
   client 也原样回填 → 同口径。

> ⚠️ 与 **bridge** 不同：bridge 需要 `state_layout: bridge_euler7` 把
> `[x,y,z,qw,qx,qy,qz,grip]` 转成 `[x,y,z,rpy,grip]`（`f6_s4.md`）。
> **fractal 不要加这个转换**，加了反而错。ckpt 的 `policy_preprocessor.json` 也确认 fractal 只有
> `rename → to_batch → normalizer(QUANTILES) → …`，没有 layout step。

### 2.2 图像分辨率有一个已知但不致命的差异

| 项     | 训练                                                            | eval                                      |
| :----- | :-------------------------------------------------------------- | :---------------------------------------- |
| 相机名 | `observation.images.top`（数据集由 `overhead_camera` 存盘） | 同                                        |
| 分辨率 | 数据集落盘 512×640                                             | SIMPLER 给 480×640                       |
| 处理   | policy 内部 `resize_with_pad(224)` + `*2-1`                 | 同（`flashvla_server.py:_build_batch`） |

→ 两者都过同一个 `resize_with_pad`，只差输入纵横比（0.8 vs 0.75）。**先不处理**，
但把它记进「§6 若分数低」的候选解释里。

### 2.3 服务器/评测的「非模型」变量（已知行为，不要当 bug 调）

- `exec_steps=4`、`chunk_size=4`、`n_action_steps=4`：与训练一致。
- `cold_start_mode=zero_delta` + `num_buffer_slots=4`：server 在第一次 query 会**先排空 3 次冷启动**，
  期间返回**保持位姿**的 chunk。这是设计行为（`flashvla_server.py:222-230`）。
  → episode 前几秒动作范数≈0 **不是**坏了；见 §5 L2。
- server 会校验「被服务的 ckpt 与 `--checkpoint` 是否同一路径」，不匹配直接报错（`flashvla_client.py:100-105`）。

---

## 3. 里程碑 F1：单任务复测 —— **已执行（2026-11-25）**

同一任务 `google_robot_close_drawer`、同一 seeds（2000–2049，50 集）、同一 server/client 路径，
只换 ckpt。**4 个 ckpt 全部跑完，链路无一次失败**（server `load_s≈132 s`，单集 ≈22–25 s，单 ckpt ≈20 min）。

### 3.1 结果（成功率 / 失败结构）

| ckpt                                           | 配方                                 |                成功率 | 近失手 0.05–0.08 | 0.08–0.12 | **完全没动 >0.12** | 中位 qpos |
| :--------------------------------------------- | :----------------------------------- | --------------------: | ----------------: | ---------: | -----------------------: | --------: |
| π0\*（T0 参照）                               | in-process，MEAN_STD，非蒸馏         | **68%** (34/50) |                 8 |          1 |                        7 |     0.017 |
| 老锚点 `pi0_flashvla_fractal_success/012000` | **MEAN_STD** + 两路 + 12000 步 | **46%** (23/50) |                 9 |          3 |                       15 |     0.057 |
| **N1 8000（本轮主角）**                  | **QUANTILES** + 三路 + 8000 步 | **34%** (17/50) |                 4 |          7 |                       22 |     0.107 |
| N1 6000                                        | 同上，6000 步                        |           28% (14/50) |                 5 |          3 |                       28 |     0.132 |
| 臂③ 2000 `f6_s5_3_fractal2000`              | 同上，2000 步                        |           24% (12/50) |                 0 |          5 |                       33 |     0.154 |

> 成功判据：`qpos <= 0.05`（`CloseDrawerInSceneEnv.evaluate`，初始 0.2）。
> 50 seeds 的 SE ≈ 6.6 pp（p≈0.34）→ **34% 与 46% 差 12 pp ≈ 1.8 SE，单任务不足以判「更差」**，
> 但结合「完全没动」从 15 涨到 22，方向是明确的。

### 3.2 三条结论（都已核实）

1. **链路没问题**：state（8 维 raw `eef_pos`）、相机（`overhead_camera`）、夹爪（±1 命令）三条口径全部对上，
   4 个 ckpt 都能跑满 50 集，`--reload-every 0`、cold-start burst 都按设计工作。
   → §5 的 L2–L6 全部排除，**不用再查链路**。
2. **闭环分数随步数单调改善，但斜率很缓**：24%（2k）→ 28%（6k）→ 34%（8k），
   约 **+1.7 pp / 1000 步**。按此外推，靠加步数到 12000 也只有 ~40%，**不足以解释老锚点的 46%**。
3. **离线 MAE 能排名、不能定分**：四个 flashvla ckpt 的离线运动维 MAE 与闭环成功率**严格同序**
   （0.04544→46% > 0.04697→34% > 0.04712→28% > 0.05984→24%），
   但 0.003 的离线差距对应 12 pp 的闭环差距 → **离线探针只能做相对排名，不能预测绝对分数**。

### 3.3 由此得到的头号嫌疑：**归一化口径 QUANTILES（F1）**

两条配方在闭环上的差别是「MEAN_STD ↔ QUANTILES」，而离线 MAE 上两者几乎一样
（0.04544 vs 0.04697，差 3.4%）。机制上最自然的解释：

> QUANTILES（p01/p99 → [-1,1]）在**闭环**里会把超出训练分布的 state 直接**饱和截断**，
> 而 MEAN_STD 仍然线性保留尾部信息。智能体一旦被自己的动作带出训练状态分布，
> QUANTILES 下 state token 就「失去分辨率」，于是**无法把自己拉回来** ——
> 这与失败结构吻合：近失手数（0.05–0.08）在 π0\*/老锚点/N1 上分别是 **8/9/4，基本没变**；
> 而「完全没动」（>0.12）随配方劣化单调上涨：**7（π0\*）→ 15（老锚点）→ 22（8k）→ 28（6k）→ 33（2k）**。

> ⚠️ 这是**假设**，不是结论：老锚点还同时差「步数 12000 vs 8000」和「两路 vs 三路」。
> 但它是目前唯一能用一行配置改动（`normalization_mapping.STATE/ACTION: MEAN_STD`）验证、
> 且潜在收益 10 pp 量级的旋钮，所以排在第一位。

### 3.3.1 P0 施工单（可直接复制；yaml 已经生成好）

**Step 0 — 确认 yaml 在**（已生成，`diff` 应当只有 4 行）：

```bash
cd /home/zjr/workspace_tj/StreamingVLA/flashvla
diff train/configs/pi0/simpler/_f6s5_3_8000.yaml \
     train/configs/pi0/simpler/_f7s1_meanstd.yaml
# 期望：STATE/ACTION 两行 QUANTILES→MEAN_STD，output_dir/job_name 两行改名，其余无差异
```

要重新生成（幂等）：

```bash
sed -e 's/^    STATE: QUANTILES.*$/    STATE: MEAN_STD/' \
    -e 's/^    ACTION: QUANTILES.*$/    ACTION: MEAN_STD/' \
    -e 's|runs/f6_s5_3_fractal8000|runs/f7_s1_fractal8000_meanstd|' \
    -e 's/^job_name: f6-s5-3-fractal8000$/job_name: f7-s1-fractal8000-meanstd/' \
    train/configs/pi0/simpler/_f6s5_3_8000.yaml \
  > train/configs/pi0/simpler/_f7s1_meanstd.yaml
# 再补回文件头的注释（可选）：见已生成的版本
```

**Step 1 — 起跑**（5.0 h，`train.sh` 用裸 `python`，必须把 env 的 bin 放最前）：

```bash
cd /home/zjr/workspace_tj/StreamingVLA/flashvla
ls -d runs/f7_s1_fractal8000_meanstd 2>/dev/null   # 必须不存在，否则 auto_resume 会接管
PATH=/home/zjr/workspace_tj/StreamingVLA/miniconda_ygc/envs/flashvla/bin:$PATH \
CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup bash train/train.sh train/configs/pi0/simpler/_f7s1_meanstd.yaml 1 \
  > logs/f7_s1_meanstd_8000.log 2>&1 &
```

**Step 2 — 5 分钟内核对三行**（不对就立刻停）：

```bash
grep -nE "state_path_lr_mult=3.0 ->|Re-initialized|'STATE':" logs/f7_s1_meanstd_8000.log | head -5
```

- `state_path_lr_mult=3.0 -> 4 param tensor(s)`（不是 4 就是错的）
- config dump 里 `'STATE': 'MEAN_STD'`、`'ACTION': 'MEAN_STD'`、`use_proprio_stack=True`、`proprio_dim=0`
- 不应出现 `Re-initialized`
- 预期 2.26 s/step → ≈5.0 h；`epch` 在 8000 步 ≈5.04

**Step 3 — （可选，2.5 min）离线探针**，与 N1/老锚点保持同口径：

```bash
PATH=/home/zjr/workspace_tj/StreamingVLA/miniconda_ygc/envs/flashvla/bin:$PATH \
env -u PYTHONPATH -u BASH_ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 \
python scripts/p4/probe_actions.py \
  --ckpt runs/f7_s1_fractal8000_meanstd/checkpoints/008000/pretrained_model \
  --leaf /data/zjr/simpler_lerobot/fractal/google_robot_close_drawer/success \
  --samples 24 --seed 0 --json logs/p4/probe_f7s1_meanstd_8000.json
# 对照：N1 QUANTILES = 0.04697；老锚点 MEAN_STD = 0.04544（运动维均值）
```

**Step 4 — SIMPLER 评测**（22 min，先等训练结束腾卡；也可在 GPU2 上评 ckpt）：

```bash
cd /home/zjr/workspace_tj/StreamingVLA/flashvla
scripts/p3/eval_flashvla_task.sh \
  --gpu 4 \
  --ckpt "$PWD/runs/f7_s1_fractal8000_meanstd/checkpoints/008000/pretrained_model" \
  --out  "$PWD/third_party/SDN/eval_logs/f7_p0" \
  --task google_robot_close_drawer --n 50 --seed 2000 --tag meanstd_8k
```

**Step 5 — 读结果并分流**（`000_episode_stats.json` 里的 `success` / `qpos`）：

| 结果            | 含义                                                                                          | 下一步                                                                     |
| :-------------- | :-------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------- |
| **≥46%** | **QUANTILES 是闭环退化的主因** → 全线回退 MEAN_STD；S5/S6 的关键臂要在 MEAN_STD 下复核 | 直接进 §4 五任务闸门（2 ckpt 都评）                                       |
| 38–45%         | 归一化贡献一半、步数贡献一半                                                                  | 追加 `12000 步 + MEAN_STD` 一臂（改 `steps`/`num_decay_steps` 两行） |
| ≈34%（±SE）   | 归一化无罪 → 嫌疑转向「两路 vs 三路」与「数据覆盖」                                          | 先 §7 O1（MEAN_STD + 两路 8000 步），再 §6 D1                            |
| <30%            | 与 F1 判据同向恶化 → 先按 §5 复查链路                                                       | 停下写状态，别继续加臂                                                     |

> 若 P0 训练期间 GPU2 空闲，**可以同时起 §7 O1**（MEAN_STD + 两路 8000 步），
> 这样 P0 出结果时手上就有「三路 vs 两路」的同位对照，省 5 h 墙钟。
> O1 的 yaml = 在 `_f7s1_meanstd.yaml` 上再改两行：`use_proprio_stack: false` +
> `pretrained_path: .../ckpt/pi0star_fractal_flashvla_proprio`（编码器装在 `state_proj` 的两路起点 ckpt，见 S0-B）。

### 3.4 复跑命令（已跑通并入库，直接抄）

```bash
cd /home/zjr/workspace_tj/StreamingVLA/flashvla
# 一次一个 ckpt；GPU 号决定端口（12400+gpu）。≈22 min/ckpt。
scripts/p3/eval_flashvla_task.sh \
  --gpu 4 \
  --ckpt "$PWD/runs/f6_s5_3_fractal8000/checkpoints/008000/pretrained_model" \
  --out  "$PWD/third_party/SDN/eval_logs/f7_f1" \
  --task google_robot_close_drawer --n 50 --seed 2000 --tag n1_8k
# 结果：<out>/flashvla/seeds=2000-2049/<task>/000_success_<rate>.log
#       + 000_episodes.json / 000_episode_stats.json（成功数、qpos、speed）
```

脚本做的事（`scripts/p3/eval_flashvla_task.sh`，本次已入库）：
绝对路径校验 → 起 server（`--ckpt --port`）→ 真 `listen()` 探活（≈132 s）→
`evaluate.py --method flashvla --num-gpus 1 --reload-every 0 --no-video` → 打印结果文件。
`--ckpt` 必须是绝对路径（脚本会 `cd third_party/SDN`），且与服务端被服务的 ckpt 一致（client 会校验）。

---

## 4. 里程碑 F2：五任务 × 50 seeds 闸门

### 4.1 先补一个五任务驱动 `scripts/p3/run_f1.sh`

单任务已经跑通（`scripts/p3/eval_flashvla_task.sh`，§3.4），**剩下的只是「两个 ckpt × 5 任务」的外层循环**。
最省事的做法：复制 `scripts/p3/run_g3.sh`，把内层命令换成对 `eval_flashvla_task.sh` 的调用，
再**只改 7 处**（其余逐行不动，包括 flock、retry、聚合逻辑）：

| 行                                       | `run_g3.sh`                                                                                                 | `run_f1.sh`                                                                     |
| :--------------------------------------- | :------------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------- |
| `G3_TASKS`                             | 4 个 `widowx_*`                                                                                             | 5 个 `google_robot_*`（见 §1.1，顺序固定）                                     |
| `T0_WIDOWX_REFS="0.68 0.88 0.88 0.76"` | widowx 参照                                                                                                   | `"0.86 0.74 0.68 0.36 0.18"`（按上表顺序）                                      |
| `CKPT_ROOT`                            | `pi05_flashvla_bridge_overnight/...`                                                                        | `$FLASHVLA_ROOT/runs/f6_s5_3_fractal8000/checkpoints`                           |
| `G3_ROOT` 默认值                       | `eval_logs/p3_g3`                                                                                           | `eval_logs/f7_f1`                                                               |
| `PORT_BASE`                            | 12400                                                                                                         | 12400（保持）                                                                     |
| `G3_GPUS`                              | `4 0`                                                                                                       | `4 2`（现在这两张空闲；0/1/3/5/6/7 被别的项目占）                               |
| 环境变量名                               | `P3_G3_*`（`GPUS`/`CKPTS`/`N_TRAJS`/`START_SEED`/`PORT_BASE`/`TASK_TIMEOUT`/`ROOT`/`LOCK`） | 建议整体 `sed 's/G3/F1/g'` 改名成 `P3_F1_*`，避免和 bridge 的运行混用同一个锁 |
| `CKPTS`                                | `004000 002750`                                                                                             | `008000 006000`                                                                 |
| gate 名                                  | `G3_<mode>.json`                                                                                            | `F1_<mode>.json`（聚合脚本里改 `f"G3_{mode}..."`）                            |

- 聚合段（`run_g3.sh:180-234`）可以直接复用：它按 `000_success_<rate>.log` 抓字，
  把 `T0_WIDOWX_REFS` 换成 fractal 的 5 个参照就得到 macro / `delta_vs_t0_pp` / `gate_pass`。
- ⚠️ `T0_WIDOWX_REFS` 的含义是「T0 在同任务上的成功率」，改脚本时**顺序必须与 `TASKS` 一致**，
  否则 macro 会静默错位（这是最容易犯的错）。

### 4.2 跑法

```bash
cd /home/zjr/workspace_tj/StreamingVLA/flashvla
# 冒烟：1 任务（pick_coke_can，T0=0.86 最能分辨）× 10 seeds × 1 ckpt，≈8 min
P3_F1_GPUS="4" P3_F1_CKPTS="008000" bash scripts/p3/run_f1.sh smoke
# 正式：5 任务 × 50 seeds × 2 ckpt，2 卡并行，≈2–3 h
P3_F1_GPUS="4 2" bash scripts/p3/run_f1.sh full
```

- 时间预算：5 任务 ≈ 21+25+15+10+40 min ≈ **1.8 h/ckpt/卡**（按 §1.1 的 episode 长度估），
  2 个 ckpt 分给 GPU4 / GPU2 → **≈2 h 墙钟**。
- 显存：server ≈10 GB + SIMPLER 渲染 ≈15–20 GB → **单卡 46 GB 够，一张卡一组**。
- 输出：`logs/p3_overnight/gates/F1_full_<root>.json`（macro / 逐任务 delta / `gate_pass`）。

**判据**：`macro ≥ 0.544` 视为「部署口径不掉点」；`0.45–0.54` 视为「有差距但可用」；
`<0.45` 说明这条蒸馏路线在 SIMPLER 上不成立，必须先解决 §6 的数据覆盖问题。

---

## 5. 分层定位表（**本轮已完成 L1–L6，链路已排除**；留作以后出现 0% 时用）

**2026-11-25 结论**：4 个 ckpt × 50 集全部跑通、分数 24–34% 且与离线指标同序，
说明 state/相机/夹爪/服务端都没有串 → **L2–L6 一次通过，后续不必再查**。
若将来出现「全 0」或「分数与离线指标不同序」，按 L1→L6 顺序走。

| 层           | 查什么                           | 命令 / 指标                                                                                                            | 判据                                                               |
| :----------- | :------------------------------- | :--------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------- |
| **L1** | 模型本身还行吗                   | `scripts/p4/probe_actions.py --ckpt <同名 ckpt> --leaf .../google_robot_close_drawer/success --samples 24 --seed 0`  | 运动维 MAE ≤0.05 → 模型没问题，锅在链路                          |
| **L2** | 动作有没有被 `cold-start` 吃掉 | `--dump-episodes --dump-images npz` 跑 2 集，看前 12 步的世界向量范数                                                | 前 3 个 query 全 0 是设计行为；**超过 12 步仍为 0** 才是 bug |
| **L3** | state 有没有串                   | 在 `flashvla_server.act` 里打一行 `state`，与数据集 `observation.state` 的 q01/q50/q99 比（`meta/stats.json`） | 逐维落在 [q01,q99] 内                                              |
| **L4** | 夹爪极性                         | `--dump-episodes` 看 gripper 序列 vs 数据集 action 末维                                                              | pick 任务应先开后合；全程同号 = 极性反                             |
| **L5** | 图像通道/朝向                    | dump 里存 3 帧，和数据集 `observation.images.top` 同一 episode 比对                                                  | 同色、同向、非镜像                                                 |
| **L6** | 服务端/客户端串了别的 ckpt       | `logs/f7_f1_server_*.log` 里的 `ckpt=` 与 client 校验                                                              | 必须一致（不一致 client 会报错）                                   |

历史现场（别重复踩）：`logs/p4_eval_fractal.log` 是 **0 字节**，说明「起了评测但 server 没起或立刻死了」；
同期的 `eval_logs/p4_pi0star/bridge/*` 是 bridge 的烂结果（8/36/6/0%），**不要拿来推断 fractal**。

---

## 6. 若 P0（MEAN_STD）也无变化：缺口在**数据覆盖**

依据（§3 的失败结构 + 数据集事实）：

| 事实                     | 数值                                                                                                             |
| :----------------------- | :--------------------------------------------------------------------------------------------------------------- |
| fractal 数据集总规模     | 777 集 / 50,843 帧 / 5 任务；bs4×accum8 → 1 epoch = 1589 步（所以 8000 步 = 5.04 epoch，已充分过拟合该数据集） |
| close_drawer 的份额      | 159 集 / 8010 帧（**只占 16%**）                                                                           |
| 只含「成功轨迹」         | `p1_provenance.json`：`outcome_split: true`、叶子目录就叫 `success`                                        |
| 采集 seeds vs 评测 seeds | 采集 1000–1199，评测 2000–2049 → **初始状态分布不同**                                                   |
| 近失手率（0.05–0.08）   | π0\* 8/50、老锚点 9/50、N1 4/50（**量级相同**）                                                           |
| 完全没动率（>0.12）      | π0\* 7/50、老锚点 15/50、**N1 22/50**（随配方劣化单调上涨）                                               |

**读法**：模型已经学会了「怎么收抽屉」（近失手率一样），差的是「30% 的初始状态下能不能找到/走到抽屉」。
这正是「成功轨迹覆盖不足」的典型症状：训练集里没有这些初始状态，也没有任何失败/恢复样本。

### 6.1 三个候选动作（按性价比排序）

（现在这条路线是**第二顺位**：先做 §3.3 的 P0（归一化单变量），再决定要不要动数据。）

| #            | 动作                                                                                                                             | 代价                                   | 为什么值得                                                                         |
| :----------- | :------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------- | :--------------------------------------------------------------------------------- |
| **D1** | **扩成功轨迹**：再跑 `--method vanilla --dump-episodes` 采 seeds 1200–1599（≈400 集/任务），重新转 lerobot，重训同配方 | 采集 ≈1.5 h + 转换 ≈0.5 h + 训练 5 h | 直接把 close_drawer 从 159 集提到 ~500 集；近失手→成功最省力的路                  |
| **D2** | 同配方 12000 步（对齐老锚点的步数）                                                                                              | 2 h（GPU 空闲时）                      | 单变量排除「46% 只是步数不够」；但 S6 显示 6000 步后离线已饱和，**期望值低** |
| **D3** | 图像/状态增强，或 `state` 统计量口径 B（`f6_s4.md` §2.3）                                                                   | 1 天                                   | 只在 D1 之后仍卡在「没走到」时才有意义                                             |

**推荐顺序（2026-11-25 已更新）**：§3.3.1 的 P0（归一化单变量）→ 若 P0 也 ≈34%，再做 §7 O1（两路同位对照）
→ 两者都排除后才上 D1（扩数据）。**不要先做 D2/D3。**

### 6.2 D1 的具体施工（关键参数别改）

```bash
# 采集（teacher = π0* vanilla，与 fractal 数据集同一条管道）
cd /home/zjr/workspace_tj/StreamingVLA/flashvla/third_party/SDN
CUDA_VISIBLE_DEVICES=4 \
/home/zjr/workspace_tj/StreamingVLA/miniconda_ygc/envs/sdn/bin/python -m policy.pizero.evaluate \
  --method vanilla --task google_robot_close_drawer --checkpoint pretrained/open-pi-zero \
  --num-gpus 1 --n-trajs 100 --start-seed 1200 \
  --dump-episodes --dump-images npz \
  --result-root /home/zjr/workspace_tj/StreamingVLA/flashvla/third_party/SDN/eval_logs/f7_d1
# 转换
cd /home/zjr/workspace_tj/StreamingVLA/flashvla && \
/home/zjr/workspace_tj/StreamingVLA/miniconda_ygc/envs/flashvla/bin/python \
  sim_eval/simpler/convert_simpler_to_lerobot.py --help   # 参数与 P1 一致：fps 3 / truncate_k 2 / outcome_split
```

⚠️ 采集 seeds **必须与评测 seeds（2000–2049）不重叠**，否则闸门会泄漏。

---

## 7. 可选对照臂（不阻塞闸门，有卡再跑）

| #          | 臂                                                                                                                                                                                    | 目的                                                                                                                                                                                   | 代价           |
| :--------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------- |
| ~~P0~~    | **MEAN_STD + 三路 + 8000 步**                                                                                                                                                   | **已升级为下一步主任务，施工单见 §3.3.1**（yaml 已生成 `_f7s1_meanstd.yaml`）                                                                                                 | 5 h + 22 min   |
| O1         | **MEAN_STD + 两路 + 8000 步**                                                                                                                                                   | 与 P0 **只差 `use_proprio_stack`** → 在**同一归一化 + 同一 lr 轨迹**下分离「三路 vs 两路」；S5 的 ① 是压缩到 2000 步的短配方，不能直接比。**P0 无效时它升为 P1** | 5 h            |
| O1 的 yaml | 在 `_f7s1_meanstd.yaml` 上改两行：`use_proprio_stack: false`、`pretrained_path: .../ckpt/pi0star_fractal_flashvla_proprio`（S0-B 的两路起点），`output_dir`/`job_name` 改名 | —                                                                                                                                                                                     | —             |
| O2         | state 通路换 AdamW（不是 lr×10）                                                                                                                                                     | S6 已证 lr 不是瓶颈，只剩优化器形态没试过                                                                                                                                              | 5 h            |
| O3         | bridge 的同构 A/B（`f6_s4.md`）                                                                                                                                                     | 若要在 bridge 上部署                                                                                                                                                                   | 5 h + 2 h 评测 |

---

## 8. 环境与操作陷阱（SIMPLER 侧，都踩过）

1. **两个 env**：server 用 `py3.12 flashvla`，`evaluate.py` 用 `py3.10 sdn`（SIMPLER 的 py3.10 装不了 lerobot）。
2. **必须 `cd third_party/SDN`** 再 `python -m policy.pizero.evaluate`，否则 `policy` 包找不到。
3. **端口 = `12400 + GPU 号`**（`FLASHVLA_CONFIG.port_base`）；一张卡一个 server，多卡时 worker 各自连自己的。
4. **`--reload-every 0`**：默认 8 会每 8 集重建 policy（对 flashvla 会重连 server），评测时关掉。
5. **`--no-video`**：省 ~30% 时间；只在 §5 L5 需要看帧时开。
6. **一张卡一组**：server(≈10 GB) + SIMPLER(≈15–20 GB) 放同一张卡，别把两个 worker 挤一张卡。
7. **`flock`**：`run_g3.sh` 用全局锁防双开（端口冲突会静默串 episode），`run_f1.sh` 保留。
8. **CUDA_VISIBLE_DEVICES 对两个进程都要设**：server 和 client 各占一个 GPU 号，一旦不一致，client 会
   连到别的卡上的 server（或直接超时）。

---

## 9. 一页决策树（**已更新到 F1 完成之后**）

```
F1（已完成）：close_drawer × 50 seeds
  π0* 68% | 老锚点 MEAN_STD/12k/两路 46% | N1 QUANT/8k/三路 34% | 6k 28% | 2k 24%
  离线 MAE 与闭环严格同序；「完全没动」7 → 15 → 22 → 33
        │
        ▼
P0：MEAN_STD + 三路 + 8000 步（5 h + 20 min）
├─ ≥46%      → 归一化是主因 → 回退 MEAN_STD，重跑关键臂 + 进 §4 五任务闸门
├─ 38–45%    → 归一化一半、步数一半 → 12000 步 + MEAN_STD
└─ ≈34%      → 归一化无罪 → §7 O1（两路 8000 步）先分离结构，再 §6 D1（扩数据）

判据一律用 macro（5 任务）下结论；单任务 50 seeds 的 SE ≈ 6.6 pp，12 pp 以内不要写死结论。
```

**时间盒**：P0 5.5 h → 判据 → F2 2 h / D1 7 h。**任何一步超过 1.5× 预算就停下写状态**；
不要在 close_drawer 还没到 46% 时先跑五任务（分母不可比）。

---

## 10. 不要再做的事（沿自 S5/S6，SIMPLER 相关新增两条）

1. 不要再用 ≤3 pp 的夹爪二分类差异当判据（96 个二值 → 1 个样本 = 1.04 pp）。
2. 不要再拿「同 step 但不同 `steps`」的 run 比绝对指标（调度器会压缩退火区间，`f6_s5.md` §4.0）。
3. 不要再优化 `proprio_in_proj`（S6 已证位移放大 8× 而精度不变）。
4. **新增**：不要因为 `p4_pi0star/bridge` 的 8/36/6/0% 就推断 fractal 也不行——fractal 是 46%，
   两者差一个数量级，bridge 那批是另一条口径（见 `f6_s4.md`）。
5. **新增**：不要给 fractal 加 `state_layout` 转换（§2.1）；也不要为了「对齐 π0\*」而把 client 改成
   `EDRSimplerAdapter` 的 xyzw+closedness —— 训练数据不是那个口径。
