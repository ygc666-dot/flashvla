# FlashVLA × LIBERO 运行方案（四套件评测 + 时延/成功率统计）

> 目标：在 LIBERO 四个测试套件（`libero_spatial` / `libero_object` / `libero_goal` / `libero_10`）上分别运行官方开源检查点
> `z-lab/flashvla-pi05-libero`，并统计 **成功率、动作执行时间（Time/Step）、动作间延迟（取动作 / 推理等待延迟）**。
> 本文所有命令均在 flashvla 仓库内执行。仓库里只有一处必要改动：`sim_eval/libero/eval.sh` 增加了 **LIBERO 非交互初始化 + preflight 检查**（见 2.7），否则后台任务必然 `EOFError` 全崩。时延细粒度落盘的可选增强见第 7 节，采用“复制副本”方式，不改仓库。

---

## 0. 结论速览（先做什么）

| 步骤 | 命令   | 说明                                                                                                                   |
| :--- | :----- | :--------------------------------------------------------------------------------------------------------------------- |
| 0    | 前置   | ① 申请 `google/paligemma-3b-pt-224` 访问权 + `hf auth login`（见 2.6）② 生成 `~/.libero/config.yaml`（见 2.7） |
| 1    | 建环境 | `conda env create -f environment.yml` + `pip install --no-build-isolation -e ".[libero]"`                          |
| 2    | 冒烟   | 单套件 `libero_spatial`，`N_EPISODES=2`，seed 用**位置参数**传（见第 4 节 `SEEDS` 行）                     |
| 3    | 正式跑 | `GPUS=0 SUITES="libero_spatial libero_object libero_goal libero_10" bash sim_eval/libero/eval.sh 1 1000`             |
| 4    | 取指标 | 读 `eval_info.json`（成功率 / Time/Step）+ 读 `.log`（动作间延迟）                                                 |

> 第 0 步是最容易漏、且失败信息最不直观的两步：漏了会分别得到 `EOFError`（10 秒内全部失败）和 `401 GatedRepoError`（约 4 分钟后全部失败）。排障入口见第 10 节。

仓库里已经内置了你要的三类指标的采集代码，**不需要自己写计时逻辑**，只需要知道去哪里读（第 6 节）。

---

## 1. 系统 / 硬件前提

- **必须 Linux**（原生 Ubuntu 22.04/24.04 或 WSL2）。
  - 原因：`sim_eval/libero/eval.sh` 用到了 `flock`、`mktemp`、bash 数组；LIBERO/MuJoCo 需要 EGL 无头渲染（`MUJOCO_GL=egl`）。
  - 你当前工作区是 Windows：原生 Windows 下 **跑不起来**（无 `flock`、EGL vendor 路径为 `/usr/share/glvnd/...`）。建议直接用 Linux 服务器；若只有本机，再考虑 WSL2 + NVIDIA CUDA on WSL（需在 WSL 内安装 NVIDIA 用户态驱动，`/usr/share/glvnd/egl_vendor.d/10_nvidia.json` 需存在，否则退回软件渲染、速度慢 5~10 倍且时延数据不可用）。
- **GPU**：单卡 ≥ 16 GB 显存即可（π0.5 bf16 约 6~7 GB + `torch.compile`/CUDA Graph 工作区）。论文数据基于 RTX 4090/5090。
- **驱动**：`environment.yml` 装的是 `torch==2.9.1+cu128`，需要支持 CUDA 12.8 的驱动（Linux ≥ 570）。
- **磁盘**：HF 缓存需 ~10 GB（模型 ~7 GB）。**注意**：`benchmarks/` 那套延迟基准会下载 `HuggingFaceVLA/libero` 数据集（几十 GB），本文主线流程 **不需要** 下载数据集。
- **网络**：需访问 huggingface.co。若不通，用镜像：`export HF_ENDPOINT=https://hf-mirror.com`。

---

## 2. 环境搭建

```bash
cd /path/to/flashvla                 # 仓库根目录（含 environment.yml / pyproject.toml）

# 2.1 基础环境（lerobot 0.5.1 + torch 2.9.1+cu128，并已 pip install -e .）
conda env create -f environment.yml
conda activate flashvla

# 2.2 验证 flashvla 可导入（eval.sh 会检查这一项）
python -c "import flashvla, lerobot; print('ok')"

# 2.3 LIBERO 额外依赖（基础环境不含）
conda install -y cxx-compiler make
CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install --no-build-isolation -e ".[libero]"
```

> `--no-build-isolation` 是必须的：lerobot 依赖通过 pip 拉进来的 `cmake` 是 Python 入口脚本，在 pip 的隔离构建环境里不可见，会导致构建失败。

```bash
# 2.4 无头渲染（每个跑 eval 的 shell 都要设置，建议写进 ~/.bashrc 或每个运行命令前 export）
export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

# 2.5 可选：指定 HF 缓存位置（模型较大，避免撑爆根目录）
export HF_HOME=/data/hf_cache
```

> `eval.sh` 现在会用 `${MUJOCO_GL:-egl}` / `${__EGL_VENDOR_LIBRARY_FILENAMES:-...}` 兜底，即未设置时自动填上面的默认值、已设置时不覆盖；写进 `~/.bashrc` 仍然推荐（直接跑 `eval.py` 时不经过脚本兜底）。

### 2.6 HF 认证：`google/paligemma-3b-pt-224` 是 gated 仓库（必需）

只下载官方检查点**不够**。检查点里的 `policy_preprocessor.json` 固定写着
`tokenizer_processor.tokenizer_name = google/paligemma-3b-pt-224`，该仓库需要申请访问权并登录，否则在
`make_pre_post_processors()` 阶段炸掉：

```
ValueError: Failed to instantiate processor step 'tokenizer_processor' ...
401 Client Error ... Access to model google/paligemma-3b-pt-224 is restricted.
```

- 只有 **tokenizer** 走这个仓库；模型权重全部在本地 ckpt 里（`make_policy()` 先于它以成功），所以不需要下整仓权重。
- 本地环境的 `huggingface_hub` 是 **1.33.0**，CLI 主命令是 `hf`；`huggingface-cli` 仅是没有独立安装的 shim，它 `import` 后转发到 `hf`（会打印 deprecation 警告），新写的命令一律用 `hf`。
- `hf-mirror.com` 也返回 401（gated 无法用镜像绕过），必须真登录。

```bash
# 先到 https://huggingface.co/google/paligemma-3b-pt-224 申请访问，获批后用同一账号登录
hf auth login          # 粘贴 https://huggingface.co/settings/tokens 的 read token
hf auth whoami         # 确认登录账号 = 获批账号

# 验证 gated 可读（打印出缓存路径即成功）
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('google/paligemma-3b-pt-224','tokenizer_config.json'))"

# 只预下载 tokenizer（几 MB，别下整仓）
hf download google/paligemma-3b-pt-224 \
  --include "tokenizer.model" "tokenizer_config.json" "special_tokens_map.json" "added_tokens.json" "config.json"
```

> token 写在 `~/.cache/huggingface/token`，后台 worker 子进程也能读到，比 `export HF_TOKEN=` 更稳（两者皆可）。
> 同时确认没有设 `HF_ENDPOINT=https://hf-mirror.com`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，否则 gated 校验仍然过不去。

### 2.7 LIBERO 的 `~/.libero/config.yaml`（缺失会导致全部任务 10 秒内失败）

`libero/libero/__init__.py` 在**模块导入时**就检查 `~/.libero/config.yaml`，缺失则 `input()` 交互式询问数据集路径。
`eval.sh` 的任务跑在后台子进程、stdin 无 TTY，`input()` 直接 `EOFError`：

```
Do you want to specify a custom path for the dataset folder? (Y/N): EOFError: EOF when reading a line
  File ".../lerobot/envs/libero.py", line 29, in <module>
    from libero.libero import benchmark, get_libero_path
```

**eval.sh 已内置非交互处理**（缺失时自动生成 + preflight 校验），无需手工干预；内容如下，手工创建也等价：

```yaml
# ~/.libero/config.yaml（路径按实际 conda env 填写）
assets: /home/zjr/StreamingVLA/miniconda_ygc/envs/flashvla/lib/python3.12/site-packages/libero/libero/assets
benchmark_root: /home/zjr/StreamingVLA/miniconda_ygc/envs/flashvla/lib/python3.12/site-packages/libero/libero
bddl_files: /home/zjr/StreamingVLA/miniconda_ygc/envs/flashvla/lib/python3.12/site-packages/libero/libero/bddl_files
datasets: /home/zjr/StreamingVLA/miniconda_ygc/envs/flashvla/lib/python3.12/site-packages/libero/datasets
init_states: /home/zjr/StreamingVLA/miniconda_ygc/envs/flashvla/lib/python3.12/site-packages/libero/libero/init_files
```

- `LIBERO_CONFIG_PATH` 可覆盖默认位置（默认 `$HOME/.libero`）。
- 不要在后台/重定向场景下靠 `python -c "import libero.libero"` 初始化（照样 EOF），要么手写 yaml，要么用 `eval.sh` 的 bootstrap。
- 首次建环境还会从 HF 下载 **`lerobot/libero-assets`（586 个文件）到 `~/.cache/libero/assets`**，约几分钟；之后复用。日志里的 `Fetching 586 files: ...` 是正常进度条，不是卡死。这与第 8 节那个几十 GB 的数据集**不是一回事**。

**验证（2 分钟内给结果，且不会触发交互提问）：**

```bash
python - <<'PY'
import os
from libero.libero import get_libero_path
for k in ("bddl_files", "init_states", "datasets", "assets"):
    p = get_libero_path(k)
    print(f"{k:12s} {'OK     ' if os.path.isdir(p) else 'MISSING'} {p}")
from huggingface_hub import hf_hub_download
print("tokenizer   OK     ", hf_hub_download("google/paligemma-3b-pt-224", "tokenizer_config.json"))
PY
```

> 注意：原来的检查 `python -c "from lerobot.envs.factory import make_env"` **无效**——`lerobot/envs/factory.py` 是在 `make_env()` 函数体内才 `from lerobot.envs.libero import ...`，仅导入 factory 不会触发 libero 导入，也发现不了缺 config / gated tokenizer 的问题。

---

## 3. 检查点

- 官方 LIBERO 检查点：**`z-lab/flashvla-pi05-libero`**（策略类型 `pi05-flashvla`，由 HF 仓库里 `config.json` 自动推断，无需手动指定 `--policy.type`）。
- 首次运行会自动从 HF 下载；也可先手动预下载，避免多卡并发重复拉（`huggingface_hub` 1.x 的 CLI 是 `hf`，`huggingface-cli` 已废弃）：

```bash
hf download z-lab/flashvla-pi05-libero --local-dir ./ckpt/flashvla-pi05-libero
# 之后用本地路径跑：POLICY_PATH=/abs/path/to/ckpt/flashvla-pi05-libero
```

- 完整 ckpt 应包含 8 个文件：`config.json` / `model.safetensors` / `train_config.json` / `README.md`
  + `policy_preprocessor.json` / `policy_preprocessor_step_2_normalizer_processor.safetensors`
  + `policy_postprocessor.json` / `policy_postprocessor_step_0_unnormalizer_processor.safetensors`。
- **权重不需要 HF 认证**（`make_policy()` 阶段全部从本地 ckpt 读取）；但 `policy_preprocessor.json` 里的 tokenizer 指向 gated 仓库，所以**仍需完成 2.6 的登录**，否则评测会在 policy 加载成功之后才失败。
- `POLICY_PATH` 建议写绝对路径：`eval.sh` 会 `cd` 到仓库根目录再起子进程，相对路径容易踩坑。

---

## 4. 跑之前必须知道的几个开关

全部来自 `sim_eval/libero/eval.sh` 与 `flashvla/async_manager.py`：

| 变量 / 参数                          | 默认                         | 含义                                                                                                                                                                                          |
| :----------------------------------- | :--------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OVERLAP_STEPS`（第 1 个位置参数） | `1`                        | 论文的 `d`。`0`=同步（每个 chunk 边界阻塞等推理）；`≥1`=异步重叠。**必须 `0 ≤ d ≤ n_action_steps`**                                                                          |
| `SUITES`                           | 四套件全列                   | 空格分隔的套件名                                                                                                                                                                              |
| `N_EPISODES`                       | `50`                       | **每个 task** 的 episode 数（每套件 10 个 task → 500 episodes/套件）                                                                                                                   |
| `N_ACTION_STEPS`                   | `5`                        | 每次推理实际执行的动作数（论文 LIBERO 设置）                                                                                                                                                  |
| `GPUS`                             | `1`                        | **默认用的是物理 1 号卡！单卡机器必须显式写 `GPUS=0`**                                                                                                                                |
| `SEEDS`（位置参数）                | `1000 2000 3000 4000 5000` | 种子；论文结果为 5 次平均。**必须用位置参数**（`bash .../eval.sh 1 1000`）；写 `SEEDS="1000"` 环境变量当前**不生效**，会被覆盖成 5 个默认种子（已知 bug，见第 10 节第 11 条） |
| `EPISODE_LENGTH`                   | 未设置                       | 未设置时用 lerobot 的 `TASK_SUITE_MAX_STEPS`（280/280/300/520）                                                                                                                             |
| `COMPILE_MODEL`                    | `d>0` 时自动 `true`      | 异步模式**必须**开编译，否则 launch 是阻塞的，重叠失效                                                                                                                                  |
| `FUSE`                             | 跟随 `COMPILE_MODEL`       | `fuse_qkv` / `fuse_gate_up`，只影响速度不影响成功率                                                                                                                                       |
| `OMP_THREADS`                      | `16`                       | 观察预处理的小张量 CPU 算子在众核机器上会被 OpenMP 拖慢（+8 ms/step），固定 16 线程可复现论文时延                                                                                             |
| `SKIP_STALE_ACTIONS=1`             | 关                           | RTC 重对齐：跳过已执行过的 `d` 个陈旧动作。**做时延对比实验时保持关闭**（默认），与论文主表一致                                                                                       |
| `EVAL_MAX_RENDERED`                | `0`                        | `>0` 则录视频（默认 `EVAL_RECORD_ALL=0` → 只录失败案例）。**计时时保持 0**，录视频会污染时延                                                                                       |
| `EVAL_RECORD_ALL=1`                | 关                           | 录制所有 episode（含成功）                                                                                                                                                                    |

> 计时可信度的两个硬约束（eval.sh 已保证）：`--eval.batch_size=1`、`--env.max_parallel_tasks=1`。
> `batch_size=1` 时一次 rollout = 一个 episode，代码里 episode walltime 才是精确的（`eval.py` 注释明确说明）；并行 task 会让多进程的 env.step 互相抢占，时延数据作废。

---

## 5. 执行流程

打开flashvla虚拟环境：

source miniconda_ygc/bin/activate

### 5.1 冒烟测试（先跑这个，确认管线通）

```bash
conda activate flashvla
cd /path/to/flashvla

# 单套件、2 episodes/task、异步 d=1、只跑 seed 1000
# 首次约 15~30 分钟：assets 下载（586 文件）+ tokenizer 下载 + torch.compile(max-autotune) + rollout
GPUS=0 SUITES="libero_spatial" N_EPISODES=2 \
POLICY_PATH=/home/zjr/StreamingVLA/flashvla/ckpt/flashvla-pi05-libero \
bash sim_eval/libero/eval.sh 1 1000
```

> 末尾的 `1 1000`：`1` 是 `overlap_steps`，`1000` 是位置参数形式的 seed。**不要**写成 `SEEDS="1000"`（会被覆盖成 5 个种子，事倍功半）。
> 单卡机器把 `GPUS=0` 换成自己的卡号；`GPUS` 默认值是物理 1 号卡。

成功后确认三件事：

1. `$OUTPUT_ROOT/ov1/seed1000/libero_spatial/eval_info.json` 存在；
2. 日志里有 `[AsyncStreamingActionManager] warmup done in ...`；
3. 日志里有 `[rollout latency: overlap_steps=1]` 及其 3 行统计。

**失败时先按「每个 job 挂掉耗时」定位**（日志：`$OUTPUT_ROOT/ov{1}/seedXXXX/<suite>.log`）：

| 现象                                            | 首个 Traceback                                                                                       | 原因                              | 处理     |
| :---------------------------------------------- | :--------------------------------------------------------------------------------------------------- | :-------------------------------- | :------- |
| 每个 job 约 10 s 就失败                         | `EOFError: EOF when reading a line` @ `libero/libero/__init__.py`                                | 缺 `~/.libero/config.yaml`      | 见 2.7   |
| 每个 job 约 4 min 后失败                        | `ValueError: Failed to instantiate processor step 'tokenizer_processor'` / `GatedRepoError: 401` | paligemma tokenizer 未认证        | 见 2.6   |
| 日志停在 `Fetching 586 files: ...`            | 无 Traceback                                                                                         | 首次下载 LIBERO assets（正常）    | 等待即可 |
| 末尾大量 `EGLError: <exception str() failed>` | `EGLGLContext.__del__`                                                                             | 解释器退出时销毁 EGL 上下文，噪声 | 忽略     |

```bash
# 一步定位首个真实错误
grep -n -E "Traceback \(most recent call last\)|Error" $OUTPUT_ROOT/ov1/seed1000/libero_spatial.log | head
```

> `eval.sh` 启动时的 preflight 已经会提前拦住「缺 config / bddl 目录不存在」，并对「assets 缺失、首次会下载」给出 WARNING；看到 `[preflight] libero paths OK` 才说明前置条件齐全。

### 5.2 正式：四套件 × 单种子（推荐先跑这一轮）

```bash
flashvla:
export OUTPUT_ROOT=/home/zjr/StreamingVLA/flashvla/results_pi05_baseline
# 别用 /tmp，重启会丢
GPUS=2 \
SUITES="libero_spatial libero_object libero_goal libero_10" \
N_EPISODES=50 \
POLICY_PATH=/home/zjr/StreamingVLA/flashvla/ckpt/flashvla-pi05-libero \
bash sim_eval/libero/eval.sh 1 1000   

基线pi05:
export OUTPUT_ROOT=/home/zjr/StreamingVLA/flashvla/results_pi05_baseline
GPUS=2 \
SUITES="libero_spatial libero_object libero_goal libero_10" \
N_EPISODES=50 \
COMPILE_MODEL=true FUSE=true \
POLICY_PATH=/home/zjr/StreamingVLA/flashvla/ckpt/pi05_libero \
bash sim_eval/libero/eval.sh 0 1000# d=1 异步

后台
export OUTPUT_ROOT=/home/zjr/StreamingVLA/flashvla/results_source_free
nohup env GPUS=3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\
SUITES="libero_spatial libero_object libero_goal libero_10" \
N_EPISODES=50 \
COMPILE_MODEL=true FUSE=true \
POLICY_PATH=/home/zjr/StreamingVLA/flashvla/ckpt/pi05_libero \
bash sim_eval/libero/eval.sh 0 1000 \
> /home/zjr/StreamingVLA/flashvla/logs/pi05_ov0_seed1000.log 2>&1 & \
echo "LAUNCHED PID=$!"
```

产出 4 个 `eval_info.json`（每套件 500 episodes，合计 2000 episodes = 论文「一次 run」的规模）。

### 5.3 论文级复现：四套件 × 5 种子 × 三种 d

```bash
# d=0（同步基线）—— COMPILE_MODEL 会自动 false
GPUS="0 1 2 3" OUTPUT_ROOT=/data/libero_eval bash sim_eval/libero/eval.sh 0

# d=1 / d=2（异步）
GPUS="0 1 2 3" OUTPUT_ROOT=/data/libero_eval bash sim_eval/libero/eval.sh 1
GPUS="0 1 2 3" OUTPUT_ROOT=/data/libero_eval bash sim_eval/libero/eval.sh 2
```

- 多 GPU 时脚本用 `flock` 从共享队列原子取任务（pull-based），`libero_10` 比 `libero_goal` 长约 50% 也不会让快卡空转。
- **断点续跑**：已存在 `eval_info.json` 的 `(seed, suite)` 会自动跳过，随时可中断重跑。
- 脚本结尾会用 `python3` 打印 SUMMARY 表（每 seed × 每套件的成功率、每套件 mean±std、总墙钟）。

### 5.4 只跑某一个套件（调试 / 补跑）

```bash
python sim_eval/libero/eval.py \
  --policy.path=z-lab/flashvla-pi05-libero \
  --policy.compile_model=true --policy.fuse_qkv=true --policy.fuse_gate_up=true \
  --policy.n_action_steps=5 --policy.device=cuda \
  --env.type=libero --env.task=libero_goal \
  --env.max_parallel_tasks=1 --eval.batch_size=1 --eval.n_episodes=50 \
  --inference_overlap_steps=1 --seed=1000 \
  --output_dir=/data/libero_eval/ov1/seed1000/libero_goal
```

---

## 6. 指标统计（成功率 / 动作执行时间 / 动作间延迟）

### 6.1 输出到哪里

```
$OUTPUT_ROOT/
└── ov1/                          # ov{d}
    └── seed1000/
        ├── libero_spatial/
        │   ├── eval_info.json    # 结构化指标
        │   └── libero_spatial.log# stdout+stderr（时延明细在这里）
        ├── libero_object/ ...
        ├── libero_goal/   ...
        └── libero_10/     ...
```

### 6.2 成功率 + 动作执行时间 → `eval_info.json`

`eval_info.json` 三个顶层键：`per_task` / `per_group` / `overall`。每个都有：

| 字段                                        | 含义                                                                                      |
| :------------------------------------------ | :---------------------------------------------------------------------------------------- |
| `pc_success`                              | **成功率 %**（论文主表指标）                                                        |
| `avg_episode_steps`                       | 平均每个 episode 执行的低层动作步数                                                       |
| `avg_episode_walltime_s`                  | 平均 episode 墙钟时间                                                                     |
| `avg_time_per_action_step_ms`             | **动作执行时间（Time/Step，ms）** = 平均 episode 墙钟 / 平均步数，含渲染与 env.step |
| `n_episodes` / `eval_s` / `eval_ep_s` | 样本数、总耗时、每 episode 耗时                                                           |
| `avg_sum_reward` / `avg_max_reward`     | 参考                                                                                      |

`per_task[*].metrics` 里还有**逐 episode** 的 `successes` / `episode_steps` / `episode_walltimes_s`，可做分布与显著性检验。

一键汇总（heredoc，不落文件）：

```bash
python - "$OUTPUT_ROOT/ov1" <<'PY'
import json, glob, os, sys, statistics as st
root = sys.argv[1]
print(f"{'suite':>16} {'seed':>6} {'SR%':>7} {'steps':>7} {'wall_s':>8} {'Time/Step(ms)':>14}")
agg = {}
for f in sorted(glob.glob(f"{root}/seed*/*/eval_info.json")):
    seed = os.path.basename(os.path.dirname(os.path.dirname(f)))
    suite = os.path.basename(os.path.dirname(f))
    o = json.load(open(f))["overall"]
    print(f"{suite:>16} {seed:>6} {o['pc_success']:7.2f} {o['avg_episode_steps']:7.1f} "
          f"{o['avg_episode_walltime_s']:8.2f} {o['avg_time_per_action_step_ms']:14.2f}")
    agg.setdefault(suite, []).append(o)
print()
for s, v in agg.items():
    sr = [x['pc_success'] for x in v]; ts = [x['avg_time_per_action_step_ms'] for x in v]
    print(f"{s:>16}  SR={st.mean(sr):.2f}±{st.stdev(sr) if len(sr)>1 else 0:.2f}%   "
          f"Time/Step={st.mean(ts):.2f}ms")
allv = [x for v in agg.values() for x in v]
print(f"{'AVG':>16}  SR={st.mean([x['pc_success'] for x in allv]):.2f}%   "
      f"Time/Step={st.mean([x['avg_time_per_action_step_ms'] for x in allv]):.2f}ms")
PY
```

对照论文（单卡 4090，2 views，batch=1，omp=16）：

| d            | Spatial | Object | Goal | Long | Avg SR | Time/Step |
| :----------- | :-----: | :----: | :--: | :--: | :----: | :-------: |
| π0.5 基线   |  98.8  |  98.2  | 98.0 | 92.4 |  96.9  |  53.8 ms  |
| FlashVLA d=0 |  98.6  |  99.0  | 97.8 | 96.2 |  97.9  |    –    |
| FlashVLA d=1 |  98.8  |  99.6  | 97.6 | 95.4 |  97.8  |  22.1 ms  |
| FlashVLA d=2 |  99.0  |  99.8  | 97.2 | 97.4 |  98.3  |  20.6 ms  |

### 6.3 动作间延迟 → `.log` 里的 rollout 时延统计

`sim_eval/libero/eval.py` 在**每个 episode（rollout）**结束时打印：

```
[rollout latency: overlap_steps=1]
  transition: n=45 mean= 24.31ms p50= 23.90ms p95= 27.10ms max= 31.20ms
  internal  : n=180 mean=  0.08ms p50=  0.07ms p95=  0.12ms max=  0.90ms
  transition/internal mean ratio: 304.0x (small ratio = async hides inference latency well; ...)
```

- 计时点：`action_manager.act(observation)` 的**纯取动作耗时**（不含 `env.step`，不含渲染）。
- `transition` = **触发下一次推理的那一步**（`chunk_index==0` 或 `_should_launch_next()`），即“动作间延迟/推理等待”发生在这些步上；
- `internal` = 直接取缓存动作的步 = **纯动作执行开销**（近乎为 0，说明异步把推理藏住了）；
- 已自动跳过每个 rollout 的前 `n_action_steps` 步（冷启动段）再统计；
- `ratio` 越小说明异步重叠越成功（`d=0` 时 transition 会明显更慢）。

批量抽取（heredoc，不落文件）：

```bash
python - "$OUTPUT_ROOT/ov1" <<'PY'
import re, glob, os, sys, numpy as np, collections
root = sys.argv[1]
pat = re.compile(r"^  (transition|internal)\s*: n=(\d+) mean=\s*([\d.]+)ms p50=\s*([\d.]+)ms p95=\s*([\d.]+)ms max=\s*([\d.]+)ms")
per = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(f"{root}/seed*/*.log"):
    suite = os.path.basename(f)[:-4]
    for line in open(f, errors="ignore"):
        m = pat.match(line.rstrip())
        if m:
            per[suite][m.group(1)].append(float(m.group(3)))   # mean 值
print(f"{'suite':>16} {'kind':>11} {'n_rollouts':>10} {'mean':>8} {'p50':>8} {'p95':>8} {'max':>8} (ms)")
for suite, kd in sorted(per.items()):
    for kind, xs in kd.items():
        a = np.array(xs)
        print(f"{suite:>16} {kind:>11} {a.size:>10} {a.mean():8.2f} "
              f"{np.percentile(a,50):8.2f} {np.percentile(a,95):8.2f} {a.max():8.2f}")
PY
```

只想要原始行，直接 grep 也行：

```bash
grep -h -A3 "rollout latency" $OUTPUT_ROOT/ov1/seed*/*.log | grep -E "transition|internal" | head -40
```

### 6.4 三个指标的关系（写论文/报告时别混用）

| 指标                                   | 来源                                            | 包含什么                                                  |
| :------------------------------------- | :---------------------------------------------- | :-------------------------------------------------------- |
| **动作执行时间 Time/Step**       | `eval_info.json: avg_time_per_action_step_ms` | episode 墙钟 / 步数，**含渲染 + env.step + 取动作** |
| **动作间延迟（取动作延迟）**     | `.log` 的 `transition` / `internal`       | 仅 `act()` 耗时，**异步重叠效果好坏的判据**       |
| **纯推理延迟 Inference Latency** | `benchmarks/benchmark_latency.py`             | 单次策略调用的端到端推理耗时（见第 8 节，可选）           |

---

## 7. （可选）把逐步延迟落盘成 JSON

`eval.py` 已采集 `step_latencies_ms` / `step_is_transition`，但只打印、**没写进 `eval_info.json`**。若你要做逐步分布/直方图分析，按下面方式做，**不改动仓库文件**：

```bash
mkdir -p /data/myeval
cp flashvla/sim_eval/libero/eval.py /data/myeval/eval_timing.py   # 副本，仓库保持原样
```

在 `/data/myeval/eval_timing.py` 副本上打 3 处补丁（**仓库内的 `sim_eval/libero/eval.sh` 会自动用它吗——不会**，所以这一步要么用第 5.4 节的 `python ...` 命令直接跑副本，要么把副本路径替换掉 eval.sh 的调用；推荐前者）：

1. `eval_policy()` 的返回值里，把 rollout 级数组并入 per-episode 记录：

   ```python
   # 在 for batch_ix in progbar: 循环内，rollout_data 之后
   if "step_latencies_ms" in rollout_data:
       ep_latencies.extend([rollout_data["step_latencies_ms"]] * env.num_envs)
       ep_is_transition.extend([rollout_data["step_is_transition"]] * env.num_envs)
   ```

   （在 `sum_rewards = []` 附近加 `ep_latencies = []` / `ep_is_transition = []`）
2. `info["per_episode"]` 的每个 dict 增加 `"step_latencies_ms"` / `"step_is_transition"` 两个键。
3. `eval_one()` 的 `TaskMetrics` 透传这两个列表（或在 `eval_policy_all` 里直接从 `per_episode` 汇总）。

然后：

```bash
python /data/myeval/eval_timing.py \
  --policy.path=z-lab/flashvla-pi05-libero \
  --policy.compile_model=true --policy.fuse_qkv=true --policy.fuse_gate_up=true \
  --policy.n_action_steps=5 --policy.device=cuda \
  --env.type=libero --env.task=libero_goal \
  --env.max_parallel_tasks=1 --eval.batch_size=1 --eval.n_episodes=50 \
  --inference_overlap_steps=1 --seed=1000 \
  --output_dir=/data/libero_eval/timing/ov1/seed1000/libero_goal
```

> 注意：`eval_timing.py` 需从仓库根目录运行（`from flashvla.async_manager import ...` 依赖包可导入），或 `PYTHONPATH` 指到仓库根。

---

## 8. （可选）纯推理延迟基准

只想测“一次策略调用多快”（不跑仿真），用仓库自带基准：

```bash
python benchmarks/benchmark_latency.py --config_path=benchmarks/configs/latency_flashvla.yaml
```

- 需要把 yaml 里的 `policy.pretrained_path` 改成 `z-lab/flashvla-pi05-libero`（默认写的是 `lerobot/pi05_base`）。
  不改文件的话用 CLI 覆盖：`--policy.pretrained_path=z-lab/flashvla-pi05-libero --policy.type=pi05-flashvla`。
- **会下载 `HuggingFaceVLA/libero` 数据集**（`streaming: false`，几十 GB）。磁盘紧张可跳过本节，时延以 6.3 的 `transition` 为准。
- `--num_views=2`（LIBERO 是 2 视角：third-person + wrist）。结果写到 `output_file`（mean/median/std/p50/p90/p95/p99/fps）。

---

## 9. 时间预算与实验规划

单卡（按论文 Time/Step ≈ 20~55 ms、平均 150~250 步/episode 估）：

| 规模                        | episodes | 预估墙钟                                                                                                    |
| :-------------------------- | :------: | :---------------------------------------------------------------------------------------------------------- |
| 冒烟（1 套件 × 2 ep/task） |    20    | 首次 ~15–30 min（assets 586 文件 + tokenizer + `compile max-autotune` + rollout）；缓存就绪后 ~5–10 min |
| 四套件 × 1 seed            |  2,000  | ~2–4 h                                                                                                     |
| 四套件 × 5 seeds × 1 个 d |  10,000  | ~10–20 h                                                                                                   |
| 三种 d（0/1/2）× 5 seeds   |  30,000  | ~30–60 h（4 卡并行可压到 ~1/4）                                                                            |

建议节奏：

1. 冒烟（5.1）→ 确认成功率落在 90%+、日志有 latency 统计；
2. 四套件 × 1 seed × d∈{0,1,2}（约 6–12 h）→ 先拿到可比的三组 Time/Step 与延迟数据；
3. 有余力再扩到 5 seeds 做 mean±std（论文口径）。

---

## 10. 常见坑

1. **默认跑在 1 号卡**：`eval.sh` 里 `GPUS:=1`。单卡机器一定写 `GPUS=0`。
2. **`d>0` 必须开编译**：`COMPILE_MODEL` 逻辑自动处理，别手动设成 `false` 又开异步，否则 launch 阻塞、重叠失效、时延反而更差。
3. **`d` 超界**：`0 ≤ d ≤ n_action_steps`（默认 5）。`AsyncStreamingActionManager.__init__` 会直接抛 `ValueError`。
4. **没设 `MUJOCO_GL=egl`**：MuJoCo 起不来或退回软件渲染，Time/Step 会离谱地大（几百 ms）。
5. **多核机器时延偏高**：众核 CPU 上 OpenMP 同步税可达 +8 ms/step。脚本默认 `OMP_THREADS=16`；改机器时固定同一值再横向对比。
6. **开了录视频 / 并行 task**：`EVAL_MAX_RENDERED>0` 或 `--env.max_parallel_tasks>1` 都会污染时延，计时时保持默认。
7. **冷启动污染**：前 `num_buffer_slots-1` 次 `predict_action_chunk` 返回 `None`（hold-pose 动作）。eval 脚本已做 `warmup()`（预捕获 CUDA Graph）并跳过前 `n_action_steps` 步统计，别自己去掉 warmup。
8. **`SKIP_STALE_ACTIONS`**：默认关闭；开启会做 RTC 重对齐（丢弃 `d` 个陈旧动作），会改变成功率与执行步数，属于另一个实验变量，别和主表混着比。
9. **Windows 本机**：原生不可跑（缺 `flock`、EGL 路径）；WSL2 可行但有风险。优先 Linux。
10. **断点续跑**：重复的 `(seed, suite)` 会自动 skip。想重跑必须先 `rm -rf $OUTPUT_ROOT/ov{d}/seed{xxxx}/{suite}`。
11. **`SEEDS="1000"` 环境变量不生效**（脚本 bug）：`eval.sh` 里是 `if [ "$#" -gt 0 ]; then SEEDS=("$@") else SEEDS=(1000 2000 3000 4000 5000) fi`。你传了位置参数 `1`（overlap）后 `shift`，`$#` 变 0 → 走 `else` → 环境变量被 5 个默认种子覆盖。**表现**：banner 里 `seeds:` 显示 5 个、耗时是预期的 5 倍。**规避**：种子用位置参数 `bash sim_eval/libero/eval.sh 1 1000`；或把该 `if` 改成「位置参数 > `SEEDS` 环境变量 > 默认」。
12. **缺 `~/.libero/config.yaml`**：`import libero` 时 `input()` → 后台任务 `EOFError`，每个 job 10 秒内 `rc=1`。`eval.sh` 已自动生成 + preflight 校验（见 2.7）；直接调 `eval.py` 时仍需自己保证。
13. **paligemma tokenizer 是 gated 仓库**：`google/paligemma-3b-pt-224` 未认证时，环境都建好了才在 `make_pre_post_processors()` 抛 `401 GatedRepoError`（每个 job 约 4 分钟后失败）。先 `hf auth login`（见 2.6）；`hf-mirror` 绕不过 gated。
14. **首次会下载 LIBERO assets**：`lerobot/libero-assets` 586 个文件 → `~/.cache/libero/assets`，几分钟，`Fetching 586 files` 不是卡死。与此相对，第 8 节的 `HuggingFaceVLA/libero`（几十 GB）主线**不需要**。
15. **CLI 用 `hf`**：本地 `huggingface_hub==1.33.0`，`huggingface-cli` 只是转发到 `hf` 的废弃 shim（会有 deprecation 警告，且子命令/参数可能与老教程不同）；`hf auth login`、`hf download <repo> --include ...` 是当前写法，老教程里的 `huggingface-cli download` 用法照抄容易踩坑。
16. **日志末尾的 `EGLError: <exception str() failed>` / `Exception ignored in: EGLGLContext.__del__`**：解释器退出时销毁 EGL 上下文的噪声，不是失败原因，别顺着它查。

---

## 11. 最小可执行清单（复制即用）

```bash
conda activate flashvla && cd /path/to/flashvla
export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export OUTPUT_ROOT=/data/libero_eval
export POLICY_PATH=/abs/path/to/ckpt/flashvla-pi05-libero

# 0) 前置：gated tokenizer 认证（只做一次）+ 快速自检
hf auth login
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('google/paligemma-3b-pt-224','tokenizer_config.json'))"

# 1) 冒烟：单套件、2 ep/task、d=1、只跑 seed 1000（seed 必须是位置参数）
GPUS=0 SUITES="libero_spatial" N_EPISODES=2 bash sim_eval/libero/eval.sh 1 1000

# 2) 正式：四套件 × 5 seeds，d=0/1/2（多卡把 GPUS 改成 "0 1 2 3"；种子仍走位置参数）
for D in 0 1 2; do
  GPUS=0 SUITES="libero_spatial libero_object libero_goal libero_10" N_EPISODES=50 \
  bash sim_eval/libero/eval.sh $D 1000 2000 3000 4000 5000
done
```

> `~/.libero/config.yaml` 由 `eval.sh` 自动生成；启动时先看 preflight 输出，出现 `[preflight] libero paths OK` 再往下等结果。
> 想只跑某一个套件补数据：把 `SUITES` 改成单个套件名，并换 `OUTPUT_ROOT`，避免覆盖已有 `eval_info.json`（已完成的 `(seed, suite)` 会被 skip，其实覆盖也无害）。
