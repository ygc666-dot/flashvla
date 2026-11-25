# R1 正式基线结果：π0\* + SDN on SIMPLER（2026-11-19 实测）

> 运行日期：2026-11-19 13:29:30 → 20:20:34（**6 h 51 m**），单卡 **GPU4**（L40S 46 GB），
> 环境与权重见 `sdn_r1_result.md`，脚本为 `third_party/SDN/policy/pizero/scripts/run_sdn.sh`。
>
> 规模：9 个 SIMPLER 任务 × **50 episode**（seeds 0-49）= **450 episode**，
> **整体成功率 67.3%**（303/450）。
>
> 口径：论文 Table I/II 图注写的是 **50 rollouts/任务**（§V-A「each evaluated over 50 rollouts」），
> 但**该数字与其自身数值不自洽**（Table II 的 SDN 行有奇数百分比，base 行的小数位指向
> 3×100=300 episode），详见 **§10.1**。本轮的 50 episode 与图注一致，与"实际可能的 300"不一致。
> 与论文的配置逐项对照见 §8。
>
> §1–§7 是 `long_horizon=4` 的基线（67.3%）。2026-11-20 又补测：`long_horizon=5`（68.7%）、
> `=10`（65.8%）——**合并与影响分析见 §9**；全任务 `k=6` 臂（70.4%）——**见 §11**；
> 论文口径的讨论见 **§10**。

---

## 1. 运行配置

```bash
cd /home/zjr/workspace_tj/StreamingVLA/flashvla/third_party/SDN
conda activate sdn
export CUDA_VISIBLE_DEVICES=4 N_TRAJS=50
bash policy/pizero/scripts/run_sdn.sh 1 ./eval_logs/sdn_r1_baseline
```

| 项                          | 取值                                                                                  |
| :-------------------------- | :------------------------------------------------------------------------------------ |
| 方法                        | `--method sdn`（Stage 1 grounding filter + Stage 2 smoothness refinement）          |
| 候选数 N                    | `num_samples 12`                                                                    |
| 每任务 k / M / long_horizon | 脚本内 Table II 超参（下表逐任务列出），`long_horizon` 一律 4                       |
| 负样本模式                  | **`negative_mode zeros_bbox`（bbox-masking，pad=3）**，非 inpaint             |
| 掩码来源                    | `by grounded_sam_tracking`（Grounding-DINO + SAM2，config 默认）                    |
| 权重                        | `pretrained/open-pi-zero` 的 `bridge_beta` / `fractal_beta`（按任务域名自动选） |
| 并行度                      | `--num-gpus 1`（单卡，episode 不分片）                                              |
| 视频                        | 每 episode 一个 GIF（`ffmpeg` 7.1）                                                 |

---

## 2. 逐任务结果（50 episodes 每任务）

按论文表格顺序；`k/M` 为 SDN 超参，`horizon` 为该任务 episode 固定步数。

| 任务                                          |  k | M | 成功/总数 |          成功率 | 论文 π0\* | 论文 π0\*+SDN | Δ vs 论文 SDN | horizon | 平均首达步 |     耗时 |
| :-------------------------------------------- | -: | -: | :-------- | --------------: | ---------: | -------------: | -------------: | ------: | ---------: | -------: |
| google_robot_close_drawer                     | 10 | 5 | 39/50     | **78.0%** |       75.7 |             87 |          −9.0 |     116 |       37.5 | 52.4 min |
| google_robot_move_near                        | 10 | 5 | 36/50     | **72.0%** |       67.3 |             78 |          −6.0 |      80 |       23.1 | 38.4 min |
| google_robot_open_drawer                      | 10 | 5 | 24/50     | **48.0%** |       38.0 |             50 |          −2.0 |     116 |       58.5 | 51.4 min |
| google_robot_pick_coke_can                    | 10 | 3 | 43/50     | **86.0%** |       84.0 |             89 |          −3.0 |      80 |       29.1 | 36.4 min |
| google_robot_place_apple_in_closed_top_drawer | 10 | 3 | 10/50     | **20.0%** |       17.0 |             26 |          −6.0 |     200 |      174.5 | 90.2 min |
| widowx_carrot_on_plate                        | 10 | 3 | 28/50     | **56.0%** |       58.0 |             63 |          −7.0 |      60 |       34.3 | 30.3 min |
| widowx_put_eggplant_in_basket                 | 10 | 5 | 44/50     | **88.0%** |       86.0 |             92 |          −4.0 |     120 |       30.0 | 50.5 min |
| widowx_spoon_on_towel                         | 10 | 5 | 41/50     | **82.0%** |       80.7 |             90 |          −8.0 |      60 |       26.0 | 31.4 min |
| widowx_stack_cube                             |  6 | 3 | 38/50     | **76.0%** |       68.7 |             78 |          −2.0 |      60 |       28.1 | 28.9 min |

论文列取自 `third_party/SDN/README.md:70-73`（4×H100、100 episode/任务）。
「平均首达步」= 成功的 episode 中 `first_success`（env step）的均值，失败记 −1 不参与平均。

---

## 3. 汇总

| 口径                                  |       本机 50 ep | 论文 π0\*+SDN | 论文 π0\* |
| :------------------------------------ | ---------------: | -------------: | ---------: |
| Fractal（google_robot 5 任务，macro） |  **60.8%** |           66.0 |       56.4 |
| Bridge（widowx 4 任务，macro）        |  **75.5%** |           80.8 |       73.4 |
| 9 任务 macro                          |  **67.3%** |           72.6 |       63.9 |
| 450 episode micro                     | 67.3%（303/450） |             — |         — |

- 9 个任务全部**低于**论文 SDN 数字，逐任务 Δ 均值 **−5.2** 个百分点。episode 数与论文同为 50，
  所以差距不能归给样本量：更可能是 **单卡 L40S 取代 4×H100**、不同 seeds、以及每个任务实际用的
  (k, M, long_horizon) 组合与论文报告值不完全相同（论文只给搜索集，未逐任务公开组合，见 §8）。
- 相对论文 **π0\*** 行，本机仍有 8/9 个任务更高，只有 `widowx_carrot_on_plate`（56.0 vs 58.0）
  略低——但这是**跨机器比较**，不能当作「SDN 无效」的证据，见 §5。
- 域间差距与论文一致：Bridge 明显高于 Fractal，`place_apple_in_closed_top_drawer` 最难（20%）。

---

## 4. 资源与耗时实测

| 项                  | 实测                                                                    |
| :------------------ | :---------------------------------------------------------------------- |
| 总墙钟              | 6 h 51 m（13:29:30 → 20:20:34），其中逐任务累计 409.8 min              |
| 每 episode pace     | 33.9 s（widowx 短 horizon）～ 107.7 s（apple，200 步）                  |
| 显存峰值            | **≈20.7 GB**（google_robot 320×320 输入 + N=12 一次性批量采样） |
| 产物体积            | 1.6 GB（450 个 episode GIF + 9 份 `000_success_*.log`）               |
| 平均单 episode 开销 | ≈0.55 s / env step（含每 8 episode 一次 policy 重建）                  |

`--num-gpus 1` 已够用：分片只影响并行度，不改变单 episode 结果。

---

## 5. 失败模式（450 episode 中 147 个失败）

- **没有一例「达成后掉出」**：所有失败 episode 的 `first_success = −1`，即整局从未进入成功状态，
  失败都是「一直没做成」，而非「做成了又丢了」。
- 失败 episode 的 `steps` 与成功 episode 相同（每个任务 horizon 固定：60/80/116/120/200），
  说明环境不会因为成功提前终止，**walltime 由 horizon 长度决定，与成功率无关**。
- 失败集中度：`place_apple_in_closed_top_drawer` 40 个失败（占全部失败的 27%）、
  `open_drawer` 26 个，两者合计 45%；widowx 侧失败最多的是 `carrot_on_plate`（22 个）。

---

## 6. 产物位置

```
third_party/SDN/eval_logs/sdn_r1_baseline/
├── sdn/num_samples=12--knn_k=10--top_m=3--long_horizon=4--negative_mode=zeros_bbox/
│   ├── google_robot_pick_coke_can/
│   ├── google_robot_place_apple_in_closed_top_drawer/
│   └── widowx_carrot_on_plate/
├── sdn/num_samples=12--knn_k=10--top_m=5--.../
│   ├── google_robot_close_drawer/  move_near/  open_drawer/
│   └── widowx_put_eggplant_in_basket/  spoon_on_towel/
└── sdn/num_samples=12--knn_k=6--top_m=3--.../
    └── widowx_stack_cube/
```

每个任务目录含：`000_episodes.json`（逐 episode 成败）、`000_episode_stats.json`
（steps / first_success / qpos / speed）、`000_success_<rate>.log`（汇总）、`episode_*.gif`。
`eval_logs/`、`*.log`、`*.gif` 均在 `third_party/SDN/.gitignore` 内，子模块 `git status` 干净。

---

## 7. 缺口与下一步

1. **缺同机 vanilla 对照**：没有跑 `--method vanilla`，所以本文只能与论文数字比，无法给出
   「SDN 在本机带来多少增益」。要闭合这一点，在同一 GPU4 上补一轮
   `--method vanilla` × 9 任务 × 50 episode（预计 2.5–3 h，N=1 无候选开销）即可做同 seed 配对比较。
2. **episode 数已对齐论文**：本轮 `seeds=0-49` 正好是论文的 50 rollouts/任务。
   `N_TRAJS=100`（README 默认）只会进一步降低方差，属可选加固，非「对齐论文」的必要条件。
3. **负样本只覆盖 bbox-masking**：`inpaint`（LaMa）分支未做——`third_party/inpaint_anything/lama`
   的依赖与 `pretrained/big-lama` 权重都尚未准备。
4. 结果以 `000_success_<rate>.log` 为准，重跑同配置会覆盖同名目录（未跑完的任务会被删除重来）。

---

## 8. 与论文配置逐项对照

论文依据：`docs/biye_paper/papers/SDN - Selected Diffusion Noise.md`
（§III Stage 1/2、§V-A Benchmarks / Baselines / **c) Implementation Details**、Table I/II 图注、§V-D RQ4、Table VII）。

| 配置项 | 论文 | 本次运行 | 一致 |
| :--- | :--- | :--- | :--- |
| backbone | π0（SIMPLER 用官方预训练 ckpt） | open-pi-zero 重实现（README 称 π0\*）+ `bridge_beta` / `fractal_beta` | ≈ 同一套权重，命名不同 |
| benchmark | SIMPLER，5 Google Robot + 4 WidowX | 同 9 任务 | ✅ |
| rollouts | 图注 **50 / 任务**，但数字指向 3×100=300（§10.1） | 50 / 任务（seeds 0-49） | ⚠ 与图注一致、与实际不符 |
| 候选数 N | 12（不同噪声种子） | 12 | ✅ |
| Stage 1 邻域 k | 搜索 `k ∈ {6, 10}` | 逐任务取 6 或 10（仅 `stack_cube` = 6） | ✅ 在搜索集内 |
| Stage 1 保留数 M | 搜索 `M ∈ {3, 5}` | 逐任务取 3 或 5 | ✅ 在搜索集内 |
| Stage 2 extended chunk | 「比执行 horizon 更长」，长度搜索 `{4, 5, 10}` | `long_horizon=4`，而 π0\* 的 `horizon_steps=4`、`act_steps=4` → **等于不扩展** | ⚠ 取的是搜索集最小值 |
| 负样本扰动 | **zero-masking (bbox) 为主**，inpaint 只在 RQ4 | `negative_mode zeros_bbox`（bbox 填 0，`bbox_pad=3`） | ✅ |
| 掩码来源 | Grounding-DINO 首帧检测 + SAM2 跨帧跟踪 | `by grounded_sam_tracking`（同 DINO + SAM2） | ✅ |
| 采样 | π0\* 默认（beta flow、10 步 ODE，代码内定） | `flow_sampling=beta`、`num_inference_steps=10` | ✅ |
| 执行 horizon | `act_steps=4`（每 4 步推理一次） | 同 | ✅ |
| 环境实例 | 「All SIMPLER experiments use a single environment instance」 | 单进程单 env（`--num-gpus 1`） | ✅ |
| 硬件 | 4×H100 | **1×L40S**，driver 580.105.08 | ❌ 唯一确定的硬件差异 |
| seeds | 论文未说明 | 0-49 | — 不可核验 |
| bbox padding | 论文未给出 | `bbox_pad=3`（代码默认） | — 不可核验 |
| 每任务 (k, M, long_horizon) 具体组合 | 只给搜索集；README 说 Table II 行是「逐任务最优」 | 用 `run_sdn.sh` 内置 preset（自述按 seeds 0-99 的成功率选出） | ⚠ **无法核验是否与论文逐任务组合相同** |

结论：**所有论文明确写出的配置项都一致**（N=12、k∈{6,10}、M∈{3,5}、零值 bbox 掩码、
DINO+SAM2 跟踪、50 rollouts/任务、单环境实例、π0\* 原生 act_steps=4）。三处差异/不可核验：

1. **硬件**：4×H100 → 1×L40S（README 明示「on another GPU or driver a task can differ by several points」）；
   同时**样本量也可能有份**（图注 50 vs 数字指向 300，见 §10.1），两者共同解释 9 个任务一致的偏低；
2. **Stage 2 的 chunk 长度**：论文在 `{4,5,10}` 中搜索，基线固定 4，而 4 就是 π0\* 的执行 horizon，
   即 Stage 2 实际是在「未扩展」的 chunk 上算 JerkRMS。**该臂已于 2026-11-20 补齐并测完，见 §9**；
3. **逐任务的 (k,M,long_horizon) 组合**：论文与 README 都只公开搜索集，未公开每个任务最终用了哪个，
   因此「完全一致」只能在集合层面成立，无法逐位比对。§9 给出 `{4,5,10}` 三个组合的完整结果，
   可按「逐任务最优」口径与论文 72.6% 对齐。

---

## 9. long_horizon 消融：`{4, 5, 10}` 三组合并结果（2026-11-20 补测）

论文 §V-A c) 只说 Stage 2 的 extended chunk 长度在 `{4, 5, 10}` 里搜索，未公开逐任务取值；
§8 的基线只跑了 `4`（= π0\* 的 `horizon_steps`，等价于「不扩展」）。为此在同一张 **GPU4** 上
串行补跑 `5` 与 `10` 两臂，**其余一切不变**（N=12、逐任务 k/M、`zeros_bbox`、seeds 0-49），
因此三组可做**同 seed 配对**比较。

| 组 | 时间窗 | 目录 | 失败任务 |
| :--- | :--- | :--- | -: |
| `lh=4` | 11-19 13:29:30 → 20:20:34（6 h 51 m） | `eval_logs/sdn_r1_baseline` | 0 |
| `lh=5` | 11-19 20:40:06 → 11-20 03:11:44（6 h 32 m） | `eval_logs/sdn_lh5` | 0 |
| `lh=10` | 11-20 03:11:44 → 09:36:18（6 h 25 m） | `eval_logs/sdn_lh10` | 0 |

（三组串行由 `/tmp/sdn_lh_sweep.sh` 自动接力，日志 `eval_logs/sdn_lh_sweep.log`。）

### 9.1 主表：逐任务成功率（%，50 episode/格）

| 任务 | k/M | lh=4 | lh=5 | lh=10 | Δ(5−4) | Δ(10−4) | 该任务最优 |
| :--- | :--- | -: | -: | -: | -: | -: | :--- |
| google_robot_close_drawer | 10/5 | 78.0 | **82.0** | 78.0 | +4.0 | 0.0 | lh=5 |
| google_robot_move_near | 10/5 | 72.0 | 76.0 | **82.0** | +4.0 | +10.0 | lh=10 |
| google_robot_open_drawer | 10/5 | **48.0** | 30.0 | 26.0 | **−18.0** | **−22.0** | lh=4 |
| google_robot_pick_coke_can | 10/3 | 86.0 | **90.0** | **90.0** | +4.0 | +4.0 | lh=5 / 10 |
| google_robot_place_apple_in_closed_top_drawer | 10/3 | **20.0** | 12.0 | 10.0 | −8.0 | −10.0 | lh=4 |
| widowx_carrot_on_plate | 10/3 | 56.0 | 66.0 | **68.0** | +10.0 | +12.0 | lh=10 |
| widowx_put_eggplant_in_basket | 10/5 | 88.0 | **90.0** | 86.0 | +2.0 | −2.0 | lh=5 |
| widowx_spoon_on_towel | 10/5 | 82.0 | **86.0** | 84.0 | +4.0 | +2.0 | lh=5 |
| widowx_stack_cube | 6/3 | 76.0 | **86.0** | 68.0 | +10.0 | −8.0 | lh=5 |

### 9.2 汇总

| 口径 | lh=4 | lh=5 | lh=10 |
| :--- | -: | -: | -: |
| Fractal（5 任务 macro） | **60.8** | 58.0 | 57.2 |
| Bridge（4 任务 macro） | 75.5 | **82.0** | 76.5 |
| 9 任务 macro（= micro，各任务同 episode 数） | 67.3 | **68.7** | 65.8 |
| 总计 | 303/450 | **309/450** | 296/450 |
| 论文 π0\*+SDN | 72.6 | — | — |
| **逐任务最优组合的 macro** | **72.4** |  |  |
| 相对论文 Δ（逐任务最优口径） | −0.2 | | |

`lh=5` 的整体均值最高，`lh=10` 最低，但差距只有 2.9 个点（309 vs 296 个 episode）。

### 9.3 配对 McNemar 检验（同 seed，`b/c` = 后者赢/输的 episode 数）

| 对比 | 逐任务显著的（p<0.05） | 合并（450 episode） |
| :--- | :--- | :--- |
| lh=5 vs lh=4 | `open_drawer` 2/11, p=0.022（5 更差） | +50/−44, p=0.61 |
| lh=10 vs lh=4 | `open_drawer` 3/14, p=0.013（10 更差） | +47/−54, p=0.55 |
| lh=10 vs lh=5 | `stack_cube` 0/9, p=0.004（10 更差） | +34/−47, p=0.18 |

- **整体上三组没有可分辨的差异**（合并 p 均 > 0.5）。50 episode/任务时单任务 95% CI 约 ±14 个点，
  单任务 ±4～8 点的变化基本是噪声。
- 逐任务出现了 3 个 p<0.05 的格子，但一共做了 27 次比较（9 任务 × 3 对比），
  Bonferroni 阈值 0.05/27≈0.0019，**没有一个格子能过**。所以这些「显著」只能当线索，不当结论。
- 唯一在配对方向上也自洽的是 `open_drawer`：两个臂一起变差、效应量 18–22 点，
  且该任务本身基线就低（48%），说明它对 Stage 2 的 chunk 长度特别敏感。

### 9.4 影响分析：lh 到底改变了什么

1. **lh 几乎不改变推理成本，只改变 Stage 2 打分的窗口**：三组墙钟 6 h 51 m / 6 h 32 m / 6 h 25 m 基本一致
   （lh 越大反而略快，因为编译与稳定性差异大于 JerkRMS 窗口的额外开销）。所以「加长 lh」不是算力问题，
   而是**让 JerkRMS 看到更长的运动**——论文的动机是「更长 chunk 能暴露短窗口看不到的延迟振荡」。
2. **收益集中在 Bridge（WidowX）侧**：lh=5 把 Bridge 从 75.5 抬到 82.0（+6.5），
   `carrot_on_plate` +10、`stack_cube` +10、`spoon_on_towel` +4。这些任务都是**精细、短程、抓放为主**，
   4 步窗口里的动作本来就很平滑、区分度低；延长到 5 步后候选之间的加速度差异才显现，选择才有信号。
3. **代价集中在 Fractal 侧的抽屉任务**：`open_drawer` −18（lh=5）、−22（lh=10），
   `apple_in_closed_top_drawer` −8/−10。这两任务 horizon 长（116/200 步）、需要「先拉开再放」，
   **延长打分窗口会让 JerkRMS 短视**：它偏好的「更平滑」chunk 往往是「少动/慢动」，反而拖慢关键动作，
   与「smoothness 与语义正确性解耦」的已知风险一致。
4. **lh=10 在 Bridge 侧开始反噬**：`stack_cube` 86→68（10 vs 5，p=0.004，0/9 全输），
   而 `move_near`、`carrot_on_plate` 受益。说明最优窗口是**任务相关**的，不是全局常数。
5. **对论文数字的解释**：论文 Table II 的 72.6% 是「逐任务在 `{k}×{M}×{lh}` 搜索集上取最优」。
   本机若也按逐任务最优（把 §9.1 最后一列拼起来）得 **72.4%**，与论文 72.6% 只差 0.2 个点——
   这比任何单一 lh 都接近论文，**进一步支持「差异主要来自逐任务组合选择，而非硬件」**。
   但要注意这是**在同一批 seeds 上选最优**，属乐观估计（选择偏差），不能当作独立验证。
6. **实践建议**（本节只沿 lh 一维搜索；补上 k=6 臂后更新，见 §11.5）：
   - 只调 lh 时，`lh=5` 最好（68.7%）；
   - **别用 lh=10 当默认**：整体最低（65.8%），且在 `stack_cube` 上有统计显著的退化；
   - 若再开放 k，`k=6 / lh=4` 的 70.4% 是目前最好的**单一**配置（§11.3）。

---

## 10. 讨论：Table II 的口径与 k/M 的可复现性

§9 之后出现两个必须写清的问题：论文的 Table II 数字到底怎么来的，以及我们用的逐任务 (k, M, lh) 与论文差在哪。

### 10.1 论文 Table II 是「逐任务最优拼接」吗？——不能证实，且有反证

支持该说法的证据**全部在仓库侧，不在论文里**：

- `third_party/SDN/README.md:75`：「The paper row is the per-task best over k ∈ {6, 10} × M ∈ {3, 5} with N = 12.」
- `policy/pizero/scripts/run_sdn.sh:14-15`：预设超参是「Chosen per task … by the success rate on episode seeds 0-99」。

论文本身只公开搜索集（§V-A c：k∈{6,10}、M∈{3,5}、extended chunk ∈{4,5,10}），**从未说明 Table II 是逐任务选优的结果**。两处反证：

1. **论文内部冲突**：鲁棒性分析称「achieving the best performance of **74.6% at k = 6**」，
   而 Table II 的平均是 72.6%。若 Table II 是逐任务最优，它必须 ≥ 任何统一配置的 74.6%。
2. **数值精度对不上「50 rollouts」**：

| 现象 | 推论 |
| :--- | :--- |
| SDN 行含 87、89、63 三个**奇数** | 50 episode 的成功率只能是 2 的倍数 → **不可能是 50 rollouts** |
| Base 行的 75.7 / 67.3 / 80.7 / 68.7 全是 .7/.3 | 与「3 次运行取平均」吻合：227/300=75.67、202/300=67.33、242/300=80.67、206/300=68.67 |
| 若为 3×50=150 episode，则成功数需为 113.5 等非整数 | 故更可能是 **3 × 100 = 300 episode** |

即论文图注写的「50 rollouts」与其数字不自洽，聚合协议（几次运行、每次多少 episode、是否取整）
在论文里没有交代；SDN 行清一色整数又像被取整过。**结论：不能把「逐任务最优」当既定事实**，
只能表述为「仓库声称」。这也意味着 §8 里「episode 数与论文同为 50，差距不能归因于样本量」这句
应降级——若论文实际是 300 episode，我们的 50 episode 方差更大，样本量重新成为差值的候选解释。

### 10.2 本轮 (k, M, lh) 是怎么定的

**没有做任何搜索**，直接照抄 `run_sdn.sh:16-26` 的 preset（`lh` 全为 4，§9 只补了 lh=5/10 两臂）：

| 任务 | k | M | lh |
| :--- | -: | -: | -: |
| close_drawer / move_near / open_drawer | 10 | 5 | 4 |
| pick_coke_can / place_apple / carrot_on_plate | 10 | 3 | 4 |
| put_eggplant_in_basket / spoon_on_towel | 10 | 5 | 4 |
| stack_cube | **6** | 3 | 4 |

即 8/9 任务用 k=10，M=5 与 M=3 各占 5 / 4 个任务。

### 10.3 与论文的具体差异（4 条）

1. **逐任务取值不可核验**：论文未公开每任务用哪个 (k, M, lh)，我们没有可比对的“标准答案”，
   只能在集合层面声称“落在搜索空间内”。
2. **选择集泄漏**：preset 按 seeds 0-99 的成功率挑，而我们评估用 seeds 0-49 ⊂ 0-99。
   → lh=4 那批数字是**在挑超参用过的样本上**测的，偏乐观；论文是否同样泄漏未知。
3. **lh 被固定，且仓库自相矛盾**：`run_sdn.sh` 全部写 4（= π0\* 的 `act_steps`，等价不扩展），
   但 `run_ablations.sh:8` 注释举例 `long_horizon 5` 来「match per-task settings」，
   暗示真实逐任务设置里至少部分任务是 5。§9 的实测（lh=5 在 5/9 任务最优、Bridge +6.5）与之一致。
4. **搜索网格覆盖不足**：论文每任务 4(k×M) × 3(lh) = 12 个组合，我们每任务只有 1 个（lh 补测后 3 个）。
   §9.2 的 72.4% 只是沿 lh 一维拼接的结果，不等于完整网格的逐任务最优。

补充一处方向性异常：论文说统一 k=6 最优（74.6%），而 preset 里 8/9 任务用 k=10。
若两者同协议，preset 不应系统性偏离论文的最优方向 → preset 的选取条件与论文不同。

### 10.4 补充臂：k=6（已完成，结果见 §11）

为直接检验论文「k=6 更优」的说法，2026-11-20 **10:58–17:51** 在 GPU4 后台跑完 k=6 臂：

| 项 | 设置 |
| :--- | :--- |
| 规模 | 9 任务 × 50 episode（seeds 0-49） |
| k | **6**（全部任务） |
| M | 沿用 preset（5/5/5/3/3/3/5/5/3） |
| lh | 4（沿用） |
| 目录 / 日志 | `eval_logs/sdn_k6/`、`eval_logs/sdn_k6_sweep.log` |
| 预计 | ≈6.5 h（参考 lh 两臂 6 h 25–32 m），约 17:30 结束 |

`widowx_stack_cube` 的 preset 本就是 k=6/M=3/lh=4，与本轮基线**同配置同 seeds**，
因此它的结果应当逐 episode 完全重现（基线 86.0%）——可顺带验证这套评测的**确定性**；
若不一致，说明存在非确定性来源（如 torch.compile / 采样噪声），后续所有配对比较都要打折扣。

### 10.5 若要严格对齐论文还差什么

- 完整网格：4(k×M) × 3(lh) = 12 组合/任务 × 50 episode = 5400 episode，单卡约 **78 h**；
  仅补 k×M 四宫格（lh=4）则 4 × 9 × 50 = 1800 episode ≈ **26 h**。
- 即便补齐，仍受两点限制：**单卡 L40S vs 4×H100**、以及论文聚合协议不明（见 10.1），
  所以本机的用途是"同协议下的相对比较"，不宜宣称复现论文绝对值。

---

## 11. k=6 补充臂结果（2026-11-20 跑完）

§10.4 启动的 k=6 臂已完成：**9 任务 × 50 episode，18:00 前结束，0 个任务失败**。
k 全任务固定为 6，M 与 lh 沿用 preset（M=5/5/5/3/3/3/5/5/3，lh=4），
因此除 `stack_cube`（preset 本就是 k=6）外，其余 8 个任务构成**只改 k 的干净对照**（同 seeds 0-49）。

| 组 | 时间窗 | 目录 |
| :--- | :--- | :--- |
| k=6 臂 | 11-20 10:58:19 → 17:51:50（6 h 54 m） | `eval_logs/sdn_k6`（日志 `eval_logs/sdn_k6_sweep.log`） |

### 11.1 主表：k=6 对比 preset（k=10）

| 任务 | preset k/M | preset 成功率 | k=6 成功率 | Δ | 配对 McNemar（6 vs 10） |
| :--- | :--- | -: | -: | -: | :--- |
| google_robot_close_drawer | 10/5 | 78.0 | **82.0** | +4.0 | 3/1, p=0.63 |
| google_robot_move_near | 10/5 | 72.0 | **78.0** | +6.0 | 5/2, p=0.45 |
| google_robot_open_drawer | 10/5 | 48.0 | 48.0 | 0.0 | 5/5, p=1.00 |
| google_robot_pick_coke_can | 10/3 | 86.0 | **90.0** | +4.0 | 5/3, p=0.73 |
| google_robot_place_apple_in_closed_top_drawer | 10/3 | **20.0** | 16.0 | −4.0 | 3/5, p=0.73 |
| widowx_carrot_on_plate | 10/3 | 56.0 | **66.0** | +10.0 | 14/9, p=0.41 |
| widowx_put_eggplant_in_basket | 10/5 | 88.0 | **90.0** | +2.0 | 3/2, p=1.00 |
| widowx_spoon_on_towel | 10/5 | 82.0 | **88.0** | +6.0 | 4/1, p=0.38 |
| widowx_stack_cube | 6/3 | 76.0 | 76.0 | 0.0 | 0/0（同配置，见 11.2） |

### 11.2 确定性校验：通过

`stack_cube` 在两条臂里是**完全相同的配置**（k=6 / M=3 / lh=4 / seeds 0-49），结果：

- 成功率 38/50 = 76.0%，两条臂一致；
- **逐 episode 比对，0 个 seed 不一致**（`000_episodes.json` 完全相同）。

这证明这套评测（含 `torch.compile`、beta 噪声采样、SAM2/DINO 负样本生成）在固定 seeds 下是
**可复现的确定性流程**，因此 §9 与本节的所有同 seed 配对比较都成立。

代码版本备注：本臂跨越了 `policy/pizero/evaluate.py` 的一次改动（14:34 加入 `--start-seed` /
`--dump-episodes` 的数据采集旁路，供 P1 用）。前后各有任务（旧版：open_drawer / apple / move_near /
stack_cube / carrot；新版：spoon / eggplant / coke / close）。核对该 diff 后确认：改动全部在
dump 旁路与 `start_seed`（默认 0，seeds 不变）上，成功判定、episode 循环与 SDN 选择逻辑未动，
因此跨版本结果仍可比；`stack_cube` 的逐 episode 一致是这一点的直接旁证。

### 11.3 汇总

| 口径 | preset（8×k10 + 1×k6） | k=6 臂（全 k6） | Δ |
| :--- | -: | -: | -: |
| Fractal（5 任务 macro） | 60.8 | **62.8** | +2.0 |
| Bridge（4 任务 macro） | 75.5 | **80.0** | +4.5 |
| 9 任务 macro（= micro） | 67.3 | **70.4** | **+3.1** |
| 总计 | 303/450 | 317/450 | +14 episode |
| 合并配对检验（8 个改 k 的任务） | — | +42/−28 | p=0.12 |

- 方向与论文「统一 k=6 时最好」**一致**（论文报 74.6%），8 个任务里 6 涨、1 平、1 跌。
- 但幅度不显著（p=0.12），且我们只测了 M 固定为 preset 值的一档，**不等于**论文那个 74.6% 的完整条件。
- `place_apple_in_closed_top_drawer` 是唯一变差的（20→16，−4 点 = 2 个 episode），属噪声级别。

### 11.4 逐任务最优拼接：72.7%，与论文 72.6% 几乎重合

把目前测过的 4 个组合（k∈{6,10} × lh∈{4,5,10}，其中 k=6 只有 lh=4）按**逐任务取最优**拼起来：

| 任务 | 已测组合里最优 | 取值 | 拼接结果 |
| :--- | :--- | :--- | -: |
| close_drawer | 82 | k6 / lh4 = k10 / lh5 | 82 |
| move_near | 82 | k10 / lh10 | 82 |
| open_drawer | 48 | k10 / lh4 = k6 / lh4 | 48 |
| pick_coke_can | 90 | k6 或 k10(lh5/10) | 90 |
| place_apple | 20 | k10 / lh4 | 20 |
| carrot_on_plate | 68 | k10 / lh10 | 68 |
| put_eggplant | 90 | k6 / lh4 或 k10 / lh5 | 90 |
| spoon_on_towel | 88 | k6 / lh4 | 88 |
| stack_cube | 86 | k10 / lh5 | 86 |
| **macro** | | | **72.7** |

**72.7% vs 论文 72.6%**——只差 0.1 个点。这是本项目目前最强的间接证据：
论文的 Table II 数值只能由「逐任务在搜索集中取最优」复现出来，任何单一固定配置都到不了
（单配置最好的是 k=6/lh=4 的 70.4%）。
但同时要提醒：这个 72.7% 是**在同一批 seeds 上挑出来的**，属乐观估计，不能当独立验证
（对应 §10.1 的结论：论文大概率也是同样的选择方式，且其聚合协议仍不透明）。

### 11.5 更新后的结论

1. **评测是确定性的**（11.2），所以本文件所有同 seed 配对比较可信。
2. **k=6 优于 preset 的 k=10**：整体 67.3 → 70.4（+3.1），Bridge 提升最明显（+4.5）；
   若要一个固定配置，**`k=6, M=preset, lh=4` 是当前最优的单一方案**。
3. **单配置天花板 ≈70%**，要摸到论文 72.6% 必须逐任务选优（11.4 得 72.7%）。
4. 仍未测的组合：`k=6` × `lh∈{5,10}`（2 组合 × 9 任务 × 50 episode = 900 episode，单卡 ≈13 h）。
   补齐后才是完整的 4(k×M？不，M 也未搜) × 3(lh) 覆盖；严格全网格是
   4(k×M) × 3(lh) = 12 组合/任务，当前已覆盖 4/12。
