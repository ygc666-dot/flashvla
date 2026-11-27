# 串行基线与时间线分析（要点一 / 技术路线(2)）执行方案

> 对应开题报告：**技术路线（2）串行基线与时间线分析**、**研究内容要点一**。
> 目标产出：一张"单步控制周期"的**分阶段时间线**（视觉预处理 / 视觉语言编码 / 动作去噪 / 动作执行 / 阶段间停顿），
> 以及**块边界停顿的持续时长与分布**，据此回答一个问题：**哪些环节值得流水线化，理论上限是多少**。
>
> 现状：`π0.5`（`ckpt/pi05_libero`，d=0 同步）与 `FlashVLA`（`ckpt/flashvla-pi05-libero`，d=1 异步）已按 seed
> 1000–5000 跑过多轮（FlashVLA 5 个 seed 全齐；π0.5 基线 1000/2000 完整、3000–5000 仍在补），成功率与 Time/Step 已有结论；
> **但没有任何分阶段计时数据**（现有 `.log` 里只有 `transition / internal` 两个聚合量，`eval_info.json` 里只有
> `walltime / steps`）。所以本文档的计时实验是**"在已有主表之上补一层时间线"**，不改动、不重跑主表统计。

---

## 0. 结论先行：这份方案要交付什么

| # | 交付物 | 形态 | 用在哪里 |
| :- | :-- | :-- | :-- |
| D1 | 五段计时口径 + 代码埋点 | `sim_eval/libero/timeline.py`（新）+ `eval.py` 约 10 行门控改动 | 论文 §方法前 / 实验设置 |
| D2 | 逐 episode 逐 step 原始记录 | `results_timeline/{variant}/.../*.jsonl` | 所有图表的唯一数据源 |
| D3 | 分阶段占比表（4 套件 × 3 变体） | 表 1–表 3 | 论文"瓶颈定位"一节 |
| D4 | 时间线甘特图 + 堆叠柱 + 停顿 CDF | 图 1–图 4 | 同上 |
| D5 | 流水线适配性判定表 + 理论上限 | 表 4（含公式） | 直接支撑要点二/三的方案选择 |

一句话：**用已有的多种子成功率/Time-Step 做主表，用本文档的 200 episode × 3 次重复做"时间都花在哪"的剖面**，
两者通过 `Time/Step` 对齐校验（误差 < 5%）。

---

## 1. 已有数据盘点（哪些能直接用，哪些必须新测）

### 1.1 已经有的（截至 2026-11-13）

| 变体 | 路径 | seed 覆盖 | 每 (seed, suite) 规模 |
| :-- | :-- | :-- | :-- |
| π0.5 基线（d=0，10 步去噪） | `results_pi05_baseline/ov0/seed*/` | **1000/2000 完整**，3000–5000 部分完成（3000 仅日志，4000 缺 goal/10，5000 仅 spatial） | 10 task × 50 ep = 500 |
| FlashVLA（d=1，1 步流式） | `results/ov1/seed*/` | **1000–5000 全部完整** | 10 task × 50 ep = 500 |

已有字段（`eval_info.json: per_task[*].metrics`）：`successes / episode_steps / episode_walltimes_s`（逐 episode），
`per_group / overall` 里有 `pc_success / avg_episode_steps / avg_episode_walltime_s / avg_time_per_action_step_ms`。
`.log` 里有每条 rollout 的 `transition / internal` 聚合（mean/p50/p95/max）。

已能直接算出的关键量（seed 1000 为主线，均见 `docs/pi05_baseline_vs_flashvla_seed1000.md`）：

| 指标 | π0.5 基线 | FlashVLA d=1 | 用途 |
| :-- | :--: | :--: | :-- |
| Time/Step（含 env.step + 渲染） | 55.6–86.1 ms | 34.3–47.0 ms | 时间线总量校验锚点 |
| `transition`（阻塞一次完整推理） | 108.6–113.8 ms | 6.9–11.2 ms | 块边界停顿的直接观测量 |
| `internal`（取缓存动作） | 0.01 ms | 0.01 ms | "纯动作执行"下界 |
| 反推 仿真+渲染 / 步 | 29.1–64.2 ms | 32.1–45.6 ms | 异步不可消除的地板 |

### 1.2 缺的（必须新测）

1. 没有 `视觉预处理 / 视觉语言编码 / 动作去噪 / 动作执行` 的**分项耗时**——只有"取动作"与"执行"两段。
2. 没有**逐 step** 记录（`eval.py` 收集了 `step_latencies_ms` 但只打印成 4 行聚合，且只在 rollout 级）。
3. 没有**块边界停顿的分布**（只有 transition 的 mean/p95，且混了冷启动、按 rollout 平均后丢失了"每 5 步一次"的结构）。
4. 没有 **`env.step` 内部** 物理 vs 渲染的拆分。
5. 没有 FlashVLA **冷启动段 vs 稳态段** 的分离（`num_buffer_slots=4` → 前 3 次推理返回 `None` 走 hold-pose）。

> 因此：**主表（成功率 / Time-Step）不重跑**；只补一轮"计时专用"的小规模运行（§4），产出 §0 的 D2–D5。

---

## 2. 计时口径定义（核心，先定死再写代码）

### 2.1 一个控制周期的物理结构

```
                ┌──────────── 一个控制步 (step) ────────────────────────────────────┐
t_start ──▶ 观测整理 ──▶ tokenize ──▶ 视觉变换 ──▶ [推理] ──▶ 后处理 ──▶ env.step ──▶ t_end
             T_obs        T_tok        T_vis      T_infer      T_post     T_exec
                                                     │                       └─ 物理+渲染+success
                                                     ├─ 编码 T_enc  (SigLIP + projector + token embed)
                                                     ├─ 预填 T_pre  (VLM prefill, KV cache)
                                                     └─ 去噪 T_den  (action expert × n_steps)
             └───────────────── T_gap（未归属残余：Python 调度、同步等待）────────────────┘
```

- **d=0（同步）**：`T_infer` 整段阻塞在 chunk 边界步上 → 该步的 `T_step = T_obs+T_tok+T_vis+T_infer+T_post+T_exec`，**块边界停顿 = T_infer 的阻塞部分**。
- **d=1（异步）**：`act()` 在边界步只做"launch"（约 1–3 ms），GPU 在后台跑 → 必须用 **in-flight 窗口** 度量真实推理时长（§3.5），并按 `stall = max(0, T_inflight − d · T_step)` 判断是否真的"藏住了"。

### 2.2 五个计时项 → 八个原始计数器（映射表）

开题报告要求的是五项，但直接按五项测会丢掉两个可优化点（tokenize 可缓存、render 是地板），所以底层记 8 个，对外仍按五项汇报：

| 报告口径（5 项） | 原始计数器（8 项） | 起止点（代码位置） | 性质 |
| :-- | :-- | :-- | :-- |
| **① 视觉预处理** | `T_obs` 观测整理 | `eval.py:164-170`（`preprocess_observation` + `add_envs_task` + `env_preprocessor`） | CPU/numpy，每步必做 |
| | `T_tok` 语言 tokenize | `eval.py:172`（`preprocessor` 整条：`normalizer` → `prepare_language` → `tokenizer_processor`(PaliGemma `max_length=200`) → `device_processor`） | **CPU，同一 episode 内指令不变 → 可缓存**；tokenizer 子项用 §3.7 微基准单列 |
| | `T_vis` 视觉变换 | `policy.prepare_images`（`modeling_pi05.py:1215` / `_flashvla.py:1282`，`resize_with_pad` 到 224 + 归一化到 [-1,1]） | GPU，小张量 |
| **② 视觉语言编码** | `T_enc` 前缀嵌入 | `prefix_embedder`（in-model profile 的 `encode`） | GPU |
| | `T_pre` VLM 预填充 | `for layer in self.layers: ... use_cache=True`（in-model profile 的 `prefill`） | GPU |
| **③ 动作去噪** | `T_den` 去噪循环 | `sample_actions` 的 `for _ in range(num_steps)`（10 步 Euler，profile 的 `action`）；FlashVLA 为 1 步 `_steady_streaming` | GPU，**占单步推理约 75%（待表 2 验证）** |
| **④ 动作执行** | `T_post` 动作后处理 | `eval.py:182-188`（unnormalize + `.cpu().numpy()` + `env_postprocessor`） | CPU + **隐式同步点** |
| | `T_exec` 环境步进 | `eval.py:191`（`env.step`）；内部再拆 `T_physics` / `T_render` / `T_check`（§3.4） | MuJoCo + EGL，**不可消除地板** |
| **⑤ 阶段间停顿** | `T_gap` 阶段间空隙 | `T_step_total − Σ(上表)`，逐 step 计算 | 调度/同步开销 |
| **（派生）块边界停顿** | `T_stall` | d=0：`T_infer_host`（阻塞窗口）；d=1：`max(0, T_inflight − d·T_step_async)` | 论文"停顿"指标 |

**一致性校验（硬性）**：对每个 episode，
`Σ[T_obs+T_tok+T_vis+T_infer+T_post+T_exec+T_gap] ≈ episode_walltime_s`，逐套件误差 **< 5%**；
且 `Σ/T_step` 应与已有主表的 `avg_time_per_action_step_ms` 同量级（允许 5–10% 因计时开销）。
**过不了这条校验，说明漏了阶段，不许进入分析。**

### 2.3 派生指标定义（写进论文的公式）

设每个 chunk 产出 `n_a = n_action_steps = 5` 个可执行动作、一次推理耗时 `T_infer`，则：

```
单步分摊推理成本        T_infer/n_a
同步每步                T_step_sync = T_obs + T_tok + T_vis + T_infer/n_a + T_post + T_exec
同步块边界步            T_step_bound = T_step_sync + T_infer·(1 − 1/n_a)     ← 停顿就发生在这里
异步每步（理想重叠）    T_step_async = T_obs + T_tok + T_vis + T_post + T_exec
理论加速上限            S_max = (T_env_floor + T_infer/n_a) / T_env_floor,   T_env_floor = T_exec + T_obs + T_tok + T_vis + T_post
停机判据                T_stall = max(0, T_inflight − d·T_step_async)
停顿占比                R_stall = Σ T_stall / episode_walltime
```

用 seed 1000 的已有数字代入即得 `S_max ≈ (30+22)/(30+2) ≈ 1.6×`——**这正是实测的 1.48–1.60×**。
本轮时间线的任务就是把这几个数**直接测出来**（而不是从 `transition/5` 反推），并给出各套件、各变体的真实值。

---

## 3. 埋点实现（最小侵入）

### 3.1 新增 `sim_eval/libero/timeline.py`

一个类 + 一个管理器子类，不碰上游 `flashvla/` 包：

```python
# sim_eval/libero/timeline.py
import time, torch, numpy as np
from flashvla.async_manager import AsyncStreamingActionManager

class StepRecorder:
    """逐 step 记录 8 个原始计数器 + 派生量，episode 结束落一行 JSONL。"""
    def __init__(self, meta):          # meta: suite/task_id/seed/episode_ix/variant/d/n_a/chunk_size
        self.meta, self.steps = meta, []
        self._t = {}
    def tic(self, name):  self._t[name] = time.perf_counter()
    def toc(self, name):  return (time.perf_counter() - self._t[name]) * 1000.0
    def ev(self):         return torch.cuda.Event(enable_timing=True)   # 仅 GPU 段
    def push(self, **kv): self.steps.append({**kv, "step": len(self.steps)})
    def to_json(self):    return {**self.meta, "steps": self.steps}


class TimedAsyncStreamingActionManager(AsyncStreamingActionManager):
    """在 launch / 消费 两处加 CUDA event，量出异步 in-flight 窗口。"""
    def __init__(self, *a, recorder=None, **kw):
        super().__init__(*a, **kw)
        self.rec, self._launch_ev, self._inflight = recorder, None, None

    def _launch_inference(self, obs):
        ev0, ev1 = self.rec.ev(), self.rec.ev()
        ev0.record()
        out = super()._launch_inference(obs)      # 编译+CUDA graph 下这是非阻塞 dispatch
        ev1.record()                              # 同一条 default stream → 记录的是真实 GPU 区间
        self._launch_ev = (ev0, ev1)              # 先存着，等真正被消费时再读
        return out

    def act(self, obs):
        out = super().act(obs)                    # 内部完成 promote 或 .cpu().numpy() 的隐式同步
        # 消费判据：pending 已存在且 next_chunk 已被 promote 掉（异步），
        # 或本次就是 d=0/冷启动的同步取用 → chunk 一定已经落到 CPU。
        if self._launch_ev is not None and self.next_chunk is None:
            e0, e1 = self._launch_ev
            self._inflight = e0.elapsed_time(e1)  # 事件必已完成，无需额外 sync
            self._launch_ev = None
        return out
```

要点：

- **GPU 段一律用 `torch.cuda.Event`**（`enable_timing=True`），CPU 段用 `perf_counter`；两者不混算。
- event 只在 **编译区之外** 使用（见 3.3），否则会打断 dynamo 图/CUDA Graph capture。
- `_launch_inference` 的两次 `record()` 夹住的是 **kernel 真实执行区间**，不受 Python 何时返回影响 → 异步模式下也能拿到 `T_infer`。
- **绝对不能在 launch 之后立刻 `synchronize()` 取值**：那会把重叠彻底破坏（测出来的是"加了同步的同步版本"）。正确顺序是"记录 → 存 → 在消费点（本来就要同步的地方）读 `elapsed_time`"。合法读点只有 `_promote_next_chunk()` 里的 `.cpu().numpy()` 与 d=0/冷启动路径的 `.cpu().numpy()`。
- 实践上更省事的等价做法：不改 `AsyncStreamingActionManager`，而是在 `eval.py` 里对 `act()` 前后各 record 一个 event，并在**同一次 `act()` 返回后**判断 `action_manager.next_chunk is None`（说明 pending 已被取走、同步已发生）再读 `elapsed_time`；结果与上面一致，代码更少。§3.1 的类只是为了把这段逻辑收敛在一个文件里。

### 3.2 `eval.py` 的门控改动（约 10 行，默认关闭）

保留原行为，只在 `TIMELINE=1` 时启用；改 4 个位置：

| 位置 | 现状 | 改动 |
| :-- | :-- | :-- |
| `rollout()` 开头（`:134`） | `action_manager = AsyncStreamingActionManager(...)` | `Cls = TimedAsyncStreamingActionManager if TIMELINE else AsyncStreamingActionManager`，并构造 `StepRecorder` |
| 循环体 `:164-172` | 连续 4 次预处理 | 前后插 `rec.tic("obs")/toc`、`tic("tok")/toc`（把 `env_preprocessor` 归 `obs`，`preprocessor` 归 `tok`） |
| `:177-180` | `_t0=perf_counter(); act(); latency` | 保留原计时，另外把 `T_infer_host`、`T_inflight` 写入 recorder |
| `:182-191` | 后处理 + `env.step` | 拆 `T_post` / `T_exec`；`T_gap = T_step_total − Σ` |
| `rollout()` 结尾（`:228`） | 返回 dict | 返回 `rec.to_json()` 或直接 append 到 `--output_dir/timeline_ep{N}.jsonl` |

> 若坚持"仓库零改动"，就走 `docs/howtodo.md` §7 的副本方式（`cp eval.py eval_timeline.py` 再打补丁）。
> 但副本会与主线漂移，**建议门控改动并提交一次 commit**，论文复现说明里只需一句 `TIMELINE=1`。

### 3.3 模型内 `T_enc / T_pre / T_den` 的可用性边界（重要）

仓库里 **已经有** in-model profile（`modeling_pi05.py:966-1034`、`modeling_pi05_flashvla.py:891-964`），返回
`{encode, prefill, action, total}`。但它有两个硬约束：

1. **`compile_model=True` 时不能用**：`sample_actions` / `_cold_start` / `_steady_streaming` 被 `torch.compile`
   （`modeling_pi05.py:581`、`_flashvla.py:503-504`）整体编译，内部插 `cuda.Event` 会造成 graph break 与二次特化，
   `benchmarks/benchmark_latency.py:314-334` 已明确因此禁用该路径。**在 CUDA Graph 里读 event 更是无意义**（`_steady_streaming` docstring 已注明）。
2. `profile=True` 会改变返回值签名（返回 tuple），不能直接塞给 `async_manager`。

因此分三档执行：

| 档 | 配置 | 能拿到什么 | 用途 |
| :-- | :-- | :-- | :-- |
| **A（主口径）** | 与主表一致：`compile_model=true, fuse=true, d=0/1` | 只有外部总量（`T_vis / T_infer_host / T_post / T_exec / T_gap`）+ 异步 in-flight | 表 1 的主数据、图 1 甘特图 |
| **B（阶段拆解）** | 额外跑一次 **`compile_model=false`** 的 `π0.5`，`profile=True` | `T_enc / T_pre / T_den` 的真实占比 | 表 2"推理内部拆解"、验证"去噪≈75%"这一说法 |
| **C（可选交叉验证）** | `torch.profiler`（CUDA activity）包住 50–100 步 | kernel 级热点；FlashVLA 单步 1 次前向的内部构成 | FlashVLA 稳态（总时 ~2 ms，只有 profiler 分得开） |

- A 与 B 的 `T_infer` 总量要**交叉对齐**（compile 只改变总量，不改变 `T_den/T_infer` 的比例级），差值写进论文作为"编译收益"。
- B 档只需 1 套件（`libero_10`，episode 最长、占比最稳）× 10 task × 5 ep 即可。

### 3.4 `env.step` 内部再拆：物理 / 渲染 / success

LIBERO 的 `env.step`（`lerobot/envs/libero.py:322`）→ `self._env.step(action)` → robosuite `_get_observations()` 内部做 EGL 渲染。
不动源码，只在 episode 开始时挂一个计时包装（`rollout()` 里 `env.envs[0]` 为 `SyncVectorEnv` 的子环境）：

```python
inner = env.envs[0]._env.env                    # robosuite MujocoEnv
_orig = inner._get_observations
def _timed(*a, **kw):
    t0 = time.perf_counter(); out = _orig(*a, **kw)
    rec.add("render", (time.perf_counter() - t0) * 1000.0)
    return out
inner._get_observations = _timed
```

于是：`T_exec = T_physics + T_render + T_check`，其中 `T_physics = T_exec − T_render − T_check`。
`T_check = env.check_success()` 可单独用 `perf_counter` 包（`libero.py:330`）。
这一步直接回答开题报告"区分渲染、通信与策略推理耗时"和"仿真渲染干扰"的风险项。

### 3.5 异步模式的正确测法（最容易做错的地方）

d=1 时 `act()` 的 wall-clock 只有 1–3 ms（launch 完就返回），**直接量 `act()` 会得出"推理只要 2 ms"的错误结论**（现有 `internal=0.01 ms` 就是这么来的，它量的是"取缓存"，不是推理）。正确做法：

1. **in-flight 窗口**（§3.1 的两 event 夹逼）：`T_inflight` = 该次推理 kernel 的真实 GPU 时长。
2. **停机判据**：`T_stall = max(0, T_inflight − d·T_step_async)`。注意 `d=1`、`n_a=5` 时下次推理在**倒数第 1 步**launch，留给 GPU 的窗口只有约 `1×T_step_async ≈ 35 ms`；若 `T_inflight > 35 ms` 就会在 promote 处同步等待 → 这就是 `transition` 尾部 `max` 达到 57–63 ms 的原因（见 `eval_report_ov1_seed1000.md` §4.2），必须在图 3 的 CDF 里体现出来。
3. **冷启动分离**：FlashVLA `num_buffer_slots=4` → 前 3 次 `predict_action_chunk` 返回 `None`（hold-pose），第 4 次才给首个可执行块。统计时**必须把前 3 次记作 `phase=cold_start`**，稳态只算 `phase=steady`；否则短 episode（如 goal 100 步）会被冷启动污染 ~10%。

### 3.6 开关与命令

沿用 `eval.sh`，全部通过环境变量（脚本会把 env 透传给 `eval.py`，**无需改 `eval.sh`**）：

```bash
conda activate flashvla && cd /home/zjr/StreamingVLA/flashvla
export MUJOCO_GL=egl

# A 档：π0.5 同步基线（与主表同配置：compile+fuse）
TIMELINE=1 TIMELINE_DIR=results_timeline/pi05_ov0 \
GPUS=2 SUITES="libero_spatial libero_object libero_goal libero_10" \
N_EPISODES=5 COMPILE_MODEL=true FUSE=true \
POLICY_PATH=$PWD/ckpt/pi05_libero \
bash sim_eval/libero/eval.sh 0 1000

# A 档：FlashVLA d=1（异步）
TIMELINE=1 TIMELINE_DIR=results_timeline/fv_ov1 \
GPUS=2 SUITES="libero_spatial libero_object libero_goal libero_10" \
N_EPISODES=5 \
POLICY_PATH=$PWD/ckpt/flashvla-pi05-libero \
bash sim_eval/libero/eval.sh 1 1000

# A 档：FlashVLA d=0（只关异步，分离"更快解码"与"重叠"）
TIMELINE=1 TIMELINE_DIR=results_timeline/fv_ov0 \
GPUS=2 SUITES="libero_spatial libero_object libero_goal libero_10" \
N_EPISODES=5 COMPILE_MODEL=true FUSE=true \
POLICY_PATH=$PWD/ckpt/flashvla-pi05-libero \
bash sim_eval/libero/eval.sh 0 1000

# B 档：π0.5 不编译 + in-model profile（拿 encode/prefill/denoise 拆解）
TIMELINE=1 TIMELINE_PROFILE_B=1 TIMELINE_DIR=results_timeline/pi05_ov0_nocompile \
GPUS=2 SUITES="libero_10" N_EPISODES=5 COMPILE_MODEL=false FUSE=false \
POLICY_PATH=$PWD/ckpt/pi05_libero \
bash sim_eval/libero/eval.sh 0 1000
```

原则：**计时跑与主表跑并行不得在同卡同时进行**（见 §8）；`EVAL_MAX_RENDERED` 必须为 0。

### 3.7 tokenize 单列（离线微基准，5 行）

`preprocessor` 是一条不可打断的 pipeline，运行中拆不开 tokenizer 的份额。用同一份 `preprocessor` 做离线标定即可：

```python
obs = dict(observation)                                   # 一个真实 preprocessed 之前的 obs
for _ in range(20): preprocessor(obs)                     # 预热
t0 = time.perf_counter()
for _ in range(200): preprocessor(obs)
t_pp = (time.perf_counter() - t0) / 200 * 1000            # ms/步 = T_tok 总量
```

再对比"仅 tokenizer 步骤"（把 `tokenizer_processor` 单独取出、对同一句指令调用 200 次）得到 `t_tok_only`。
若 `t_tok_only / t_pp > 0.6`，说明 `T_tok` 主要是重复 tokenize，**可缓存**的结论成立（此项与 `T_exec` 无关，成功率零风险）。

---

## 4. 实验矩阵（规模要小、可重复）

| 编号 | 变体 | d | 编译 | 套件 | 规模 | 重复 | 预估 |
| :-- | :-- | :-: | :-: | :-- | :-- | :-: | :-- |
| V1 | π0.5 基线 | 0 | on | 4 套件 | 10 task × 5 ep = 200 ep | ×3 | 3 × ~20 min |
| V2 | FlashVLA | 1 | on | 4 套件 | 200 ep | ×3 | 3 × ~14 min |
| V3 | FlashVLA | 0 | on | 4 套件 | 200 ep | ×3 | 3 × ~20 min |
| V4（B 档） | π0.5 不编译 + profile | 0 | off | `libero_10` | 50 ep | ×2 | 2 × ~8 min |
| V5（可选） | π0.5 块长/去噪步数敏感性 | 0 | on | `libero_10` | 50 ep × {n_a=5,10} × {n_steps=10,5} | ×1 | ~30 min |

- **为什么 5 ep/task 就够**：本轮目标是"时间的结构占比"，不是成功率。200 episode 已足以把 `T_infer/T_exec` 的均值误差压到 ±3%（同一 task 内方差远小于跨任务）；成功率仍引用已有 500 ep × 5 seed 的主表，**不复用本轮的 SR**（样本量不同，禁止混表）。
- **为什么必须 3 次重复**：共享 L40S 节点上 `spatial` 已两次出现 1.4–2× 的仿真开销抬升（`pi05_baseline_vs_flashvla_seed1000.md` §5.1）。三次取中位数，并记录三次数值的离散度作为"环境噪声"指标。
- **seed 固定 1000**：与已有主表同 seed，便于把逐 episode 的 `walltime/steps` 与主表同 episode 对齐（配对）。
- V5 的用途：给出"chunk 越长 → 分摊越低但边界停顿越高"的定量曲线，作为选用流式解码（要点二）的直接论据。

---

## 5. 数据产物与 schema

```
results_timeline/
├── pi05_ov0/seed1000/{libero_spatial,...}/timeline_ep0000.jsonl ... （V1）
├── fv_ov1/seed1000/...                                            （V2）
├── fv_ov0/seed1000/...                                            （V3）
├── pi05_ov0_nocompile/seed1000/libero_10/...                       （V4）
└── summary/
    ├── stages_by_suite.csv        # 表 1/表 2 的来源
    ├── stall_distribution.npz     # 图 3 的原始数组
    └── timeline_check.json        # Σ 段 vs Time/Step 的校验结果
```

每行（一个 episode）：

```json
{"variant":"pi05","d":0,"suite":"libero_10","task_id":8,"seed":1000,"episode_ix":3,
 "n_action_steps":5,"chunk_size":50,"num_inference_steps":10,"compile":true,
 "success":false,"n_steps":520,"walltime_s":26.41,
 "steps":[{"step":0,"phase":"cold_start","chunk_index":0,"is_boundary":true,
           "T_obs":1.9,"T_tok":3.4,"T_vis":1.2,"T_enc":null,"T_pre":null,"T_den":null,
           "T_infer_host":112.6,"T_inflight":110.9,"T_post":1.1,"T_exec":33.8,
           "T_render":26.4,"T_check":0.7,"T_gap":0.8,"T_step_total":155.0,
           "T_stall":0.0}]}
```

- `T_enc/T_pre/T_den` 只在 B 档非空（`null` 表示该步未拆解，别填 0，否则聚合会被污染）。
- `is_boundary`：`chunk_index==0 or _should_launch_next()`，与 `eval.py:174-176` 的 `will_trigger_inference` 同义。
- 落盘用 `append + flush`，单文件 < 500 KB；episode 长（520 步）时约 200 KB。

---

## 6. 分析产出（表 + 图，逐项说明"它证明什么"）

### 表 1：分阶段耗时（4 套件 × 3 变体，中位数 [p10, p90]，单位 ms/步）

| 阶段 | π0.5 d=0 | FlashVLA d=0 | FlashVLA d=1 |
| :-- | :--: | :--: | :--: |
| ① 视觉预处理 | | | |
| ② 视觉语言编码 | | | |
| ③ 动作去噪（分摊/步） | | | |
| ④ 动作执行（物理/渲染/check） | | | |
| ⑤ 阶段间停顿（gap + 边界 stall） | | | |
| **合计（=Time/Step）** | | | |
| 校验：与主表 Time/Step 的偏差 | | | < 5% |

**证明**：谁占了时间的大头；异步把哪一列抹掉；哪一列是地板。

### 表 2：推理内部拆解（B 档，`libero_10`，π0.5 10 步去噪）

| `T_enc` | `T_pre` | `T_den` | `T_den/T_infer` | 结论 |
| :--: | :--: | :--: | :--: | :-- |
| ms | ms | ms | **≈75%?** | 与开题报告"动作解码占单步推理约 75%"的论断对照 |

**证明**：把改造火力集中在去噪循环上是合理的（也是 FlashVLA 只推进一步去噪的动机）。

### 表 3：块边界停顿分布（每 chunk 一次）

| 变体 | 停顿次数/ep | mean | p50 | p90 | p99 | max | `R_stall` |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| π0.5 d=0（理论值 = `T_infer`） | ~n_steps/5 | | | | | | |
| FV d=0 | | | | | | | |
| FV d=1（理论值 = `max(0,T_inflight−d·T_step)`） | | | | | | | |

**证明**：停顿是"周期性尖峰"而非均匀开销 → 解释了为什么 Time/Step 只能反映平均、必须看分布；同时给出 `R_stall`（停顿时长占 episode 墙钟的比例），这是论文里"响应延迟/流畅性"最直接的量化。

### 图 1：单步时间线甘特图（三种变体并排，各取稳态连续 15 步）

横轴 = 时间（ms），每行一个 step，色块 = 五阶段，空白 = gap。左侧标注 `chunk 边界`。
**证明**：一眼看出 d=0 的"5 步里 1 步巨长"锯齿，与 d=1 的"每步几乎等长"。

### 图 2：分阶段堆叠柱（套件 × 变体）

每柱一根 `Time/Step`，按五段堆叠。**证明**：加速比 1.5–1.6× 的来源分解（推理被压掉，环境地板不动）。

### 图 3：块边界停顿 CDF + 尾部放大

三条曲线（V1/V2/V3），横轴停顿毫秒（对数），纵轴累计比例。**证明**：d=1 的 p99 尾巴来自 GPU 排队/干扰，而非机制失效（与 `transition max=57 ms` 对得上）。

### 图 4：加速上限瀑布图

从 `T_step_sync` 出发，逐项减去"被异步消除的推理分摊"，落到 `S_max`，标注实测加速比。
**证明**：把"为什么只有 1.6×"从口头解释变成可复算的分解。

---

## 7. 判定规则：哪些环节适合流水线化

对每个阶段打 4 个分（0/1/2），加权得 `P_stage`（流水线收益潜力）：

| 判据 | 含义 | 视觉预处理 | 视觉语言编码 | 动作去噪 | 动作执行 |
| :-- | :-- | :--: | :--: | :--: | :--: |
| ① 阻塞占比 | 其在 `T_step_sync` 中的份额 | 中（~15%，含 tokenize） | 低-中（~10%） | **中-高（~25–30%）** | 高（~45%）但不可压 |
| ② 可离线/异步 | 能否在"执行当前动作"时并行完成 | 是（CPU 可与 GPU 并行） | 是 | **是** | 否（它是消费者） |
| ③ 陈旧容忍度 | 用旧观测完成是否会引入失配 | 高（可缓存） | 低（依赖当前帧） | **低→正是失配成本来源** | — |
| ④ 零风险可消除项 | 是否可无损优化 | **tokenize 可缓存（指令不变）** | 可复用 VLM KV cache | 需联合解码 | 渲染可降分辨率/减视角（会改成功率，需单列） |

预期结论（供论文写作定调，**以实测为准**）：

1. **动作去噪是唯一值得"结构性重叠"的对象**——它占推理 ~75%、占 `T_step_sync` ~25–30%（**待表 1 实测**），且不依赖执行结果；但它依赖"当前观测"，因此重叠必然带来 d 步的状态失配 → 这正是要点二/三要处理的成本项，须在时间线上把"延迟收益"与"失配成本"分开计量（后者用成功率/步数变化表征）。
2. **`T_tok` 是白捡的**：同一 episode 内语言指令恒定，`tokenizer_processor` 每步重复 tokenize 200 token，测得 `T_tok` 后可直接缓存，**成功率零风险**，可作为一个干净的工程改进点（若数值可观，可写进要点三）。
3. **`T_exec`（渲染为主）是地板**：与策略无关，任何异步都压不掉。它决定了 `S_max`，也决定了"要在 4090/5090 上才能复现论文 2.4×"这一结论的合理性（分母变小 → 比值变大）。
4. **视觉语言编码不宜重叠**：前缀依赖当前帧，重叠即等价于提前观测；若要做，必须像 StreamingVLA 那样显式处理早期观测失配。
5. **`T_gap` 是"看不见的税"**：若测得 >3 ms/步，说明 bat=1 的小算子/Python 调度有优化空间（`OMP_NUM_THREADS`、torch.compile 覆盖范围），可单列一条工程注记。

输出 `表 4：流水线适配性判定`，并在文末给出"改造顺序建议"：`T_tok 缓存 → 去噪重叠（FlashVLA 路线）→ 调度改进（要点三）`。

---

## 8. 预算与口径纪律

| 项 | 数值 |
| :-- | :-- |
| 计时总机时（V1–V4，含 3 次重复） | ~2.5–3.5 h（单卡串行） |
| 落盘 | < 100 MB |
| 分析脚本运行 | < 1 min |

**必须遵守的口径纪律**（否则数据作废）：

1. `--eval.batch_size=1`、`--env.max_parallel_tasks=1`、`EVAL_MAX_RENDERED=0`、`OMP_THREADS=16` 全程不变。
2. **计时跑不得与主表跑同卡并发**；记录开始/结束时间与 `nvidia-smi` 同卡邻居数，写进 `summary/timeline_check.json`。
3. 丢弃每 episode 前 `n_action_steps` 步（冷启动/图捕获余波），但**冷启动单独成类统计**，不要混入稳态均值。
4. `T_enc/T_pre/T_den` 只在 B 档出现，聚合时按 `not null` 过滤，禁止把 `null` 当 0。
5. 三次数值若极差 > 30%（尤其 spatial），以**中位数**为准并标注为"高噪声，仅用于结构分析，不用于绝对时延对比"。
6. 所有百分比以 `T_step_total` 为分母，不用 `T_infer` 为分母（避免与"推理占比"混淆，两个口径都要报时须显式标注）。

---

## 9. 验收清单（DoD）

- [ ] `TIMELINE=1` 时产出 `timeline_ep*.jsonl`，字段齐全（§5），`TIMELINE=0` 时行为与原版**逐字节一致**（跑 1 个 task × 1 ep 对比 `eval_info.json`）。
- [ ] 校验通过：`Σ 五段 + gap` 与 `episode_walltime_s` 偏差 < 5%，写入 `timeline_check.json`。
- [ ] 表 1 的 `Time/Step` 与主表同套件同 seed 偏差 < 10%（差异解释为计时开销 + 5 ep 采样）。
- [ ] 表 2 给出 `T_den/T_infer`，可与"75%"对照。
- [ ] 表 3 给出 `R_stall`，d=1 的 `T_stall` 中位数 ≈ 0（重叠生效），且 p99 与 `.log` 的 `transition max` 一致。
- [ ] 图 1–4 落盘（`docs/figures/timeline_*.png`），可在论文正文直接引用。
- [ ] 表 4 判定表 + "改造顺序建议"成文，作为要点二（FlashVLA 复现）与要点三（策略改进）的输入。
- [ ] 文档回填：把实测值补进本文档 §0/§2.3 与 `docs/pi05_baseline_vs_flashvla_seed1000.md` 的"下一步建议"第 2 条。

---

## 10. 与后续两步的接口

1. **要点二（FlashVLA 复现）**：时间线给出 `T_infer` 的真实值 → 直接换算"1 步去噪"分摊到每步的残余，可预测 d=1 的 `Time/Step` 下界，用于判定复现是否到位（预测 vs 实测）。
2. **要点三（策略改进）**：`T_stall` 与 `R_stall` 是改进的**基线画像**；改进后只要看这两条曲线是否左移/下移即可，不必再重跑全量主表。特别是 `d` 的动态化（缓冲区将尽时提前 launch）本质上就是在压 `T_stall` 的尾部——图 3 的 p90–p99 区间就是它的优化目标。
