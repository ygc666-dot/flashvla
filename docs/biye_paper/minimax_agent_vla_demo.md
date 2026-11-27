# MiniMax-Agent-VLA-Demo 详解与「远端主机 ↔ 用户主机」通信方案

> 目标读者：要做一个「网页里下指令 → 远端 VLA 跑仿真 → 浏览器看到机器人动」的系统的开发者。
> 本文只回答两件事：
> 1. `MiniMax-Agent-VLA-Demo` 这个 agent **到底是什么、吃什么、吐什么、怎么跑、怎么配**；
> 2. 基于它，**远端主机（本机）与用户主机**之间该怎么通信（协议、线程模型、带宽、安全、改造清单）。
>
> 所有结论来自本仓库/本机实测，凡未验证处均标注 **未验证**。
> 证据基准：`/home/zjr/StreamingVLA`，环境 `/home/zjr/StreamingVLA/miniconda_ygc/envs/flashvla`（Python 3.12 / torch 2.9.1+cu128 / lerobot 0.5.1）。
> 实测与撰写日期：2026-11-05。

---

## 0. 三句话结论

1. **它不是服务，是一个单进程、单会话的命令行 Agent**：`stdin` 读自然语言 → MiniMax LLM 拆解 → 调 `execute_task`（VLA 跑满 280 步）→ 调 `get_scene_info`（外部 MCP 视觉模型看图）→ 打印文字结果。**没有任何 HTTP/RPC 接口、没有回调、没有事件流、没有取消机制**。
2. **它的价值在于「编排骨架」而不是「工程质量」**：LLM 工具调用循环（`AGENT_TOOLS` + `run_agent`）、LLM↔VLA↔VLM 的分层职责、以及 LIBERO 环境的启动与查看器同步代码，这三块是可以直接复用的；其余（阻塞式执行、无鉴权、硬编码任务白名单、obs 四元数顺序 bug）在服务化时都要改。
3. **通信方案不该是「给 agent 加个 HTTP 客户端」**：模型、仿真、LLM 全在同一台机器上，正确形态是 **「远端一个进程组（仿真线程 + Agent 线程 + Web 服务线程）+ 用户主机只跑浏览器」**，浏览器拿到的只有 JPEG 帧 + JSON 状态 + Agent 事件流。

---

## 1. 目录与「哪些代码是真的在跑」

```
/home/zjr/StreamingVLA/MiniMax-Agent-VLA-Demo/
├── agent_mode.py        ← 主入口（Agent 模式：LLM + VLA + MCP 视觉验证）
├── run_robot.py         ← 备用入口（简单模式：只有 VLA，无 LLM）
├── .env                 ← 实际生效的配置源（LLM_BASE_URL / LLM_API_KEY / LLM_MODEL）
├── requirements.txt     ← 依赖清单（torch/libero/robosuite/mcp/anthropic/...）
├── README.md / README_EN.md
├── demo.webp            ← 演示动图
├── lerobot_back/        ← ⚠️ 无效目录（见下）
└── lerobot/             ← 已被 git 删除（git status 显示为 D），即「不随项目分发」
```

**关于 `lerobot_back/` 无效这件事**（你提到的关键点），事实链如下：

- `git status` 显示 `lerobot/**` 全部为 `D`（删除），`lerobot_back/` 是未跟踪目录 —— 即这个目录是从 `lerobot/` 重命名来的**备份/残骸**，不在 `sys.path` 上，`agent_mode.py` 也不做任何 `sys.path.insert`（旧版曾 `sys.path.insert(0, "/data1/devin/robot1/lerobot/src")`，本次修改已删除）。
- `agent_mode.py:45-48` 的 import 是：

  ```python
  from lerobot.envs.libero import LiberoEnv
  from lerobot.policies.pi05.modeling_pi05 import PI05Policy
  from lerobot.policies.factory import make_pre_post_processors
  from lerobot.envs.utils import preprocess_observation
  ```

- 这些模块**实际来自 conda 环境已安装的 lerobot 0.5.1**：
  `/home/zjr/StreamingVLA/miniconda_ygc/envs/flashvla/lib/python3.12/site-packages/lerobot/`
- 因此：**不要用 `lerobot_back/`，也不要把它加进 `PYTHONPATH`**（会和已安装版本冲突，出现难查的 `ImportError`/行为差异）。只用 `conda activate flashvla`。

> 结论：这个项目**必须**在 `flashvla` 这个 conda 环境里跑，依赖的是 site-packages 的 lerobot，而不是仓库内的拷贝。

---

## 2. 作用：它到底在编排什么

### 2.1 四层职责

```
        ┌─────────────────────────────────────────────────────────┐
用户 ──▶│ L1 规划层  MiniMax LLM（Anthropic 兼容接口，tool-use）    │
        │   输入：自然语言 + 系统提示 + 历史；输出：tool_use 决策  │
        └───────────────┬─────────────────────────────────────────┘
                        │ tool: execute_task(task)  /  get_scene_info()
        ┌───────────────▼─────────────────────────────────────────┐
        │ L2 执行层  Pi05 / FlashVLA 策略（7 维 delta-EEF 动作）    │
        │   每步：obs → 预处理 → policy → 后处理 → env.step()      │
        └───────────────┬─────────────────────────────────────────┘
        ┌───────────────▼─────────────────────────────────────────┐
        │ L3 仿真层  LIBERO(robosuite/MuJoCo) OffScreenRenderEnv   │
        │   观测：2 路 256×256 RGB + eef/gripper/joints 状态       │
        └───────────────┬─────────────────────────────────────────┘
                        │ 渲染帧截图 → /tmp/scene_capture_*.png
        ┌───────────────▼─────────────────────────────────────────┐
        │ L4 验证层  MiniMax MCP（stdio 子进程 uvx）understand_image│
        │   输入：截图 + 提示；输出：任务是否完成的自然语言判断    │
        └─────────────────────────────────────────────────────────┘
```

- **L1 规划**：把「把碗放到炉子上」这类意图映射到 10 条**受支持任务字符串**之一；不支持的（如 close drawer）直接拒绝。
- **L2 执行**：不做任何语义理解，只吃 7 维动作回归。它不知道「任务成功」——`execute_task` 永远跑满 280 步并返回 `{"steps": 280, "message": "..."}`。
- **L3 仿真**：LIBERO `libero_goal / libero_object / libero_spatial` 套件，Franka Panda + OSC_POSE。
- **L4 验证**：把当前帧发给一个**外部视觉模型**（MCP server `minimax-coding-plan-mcp`），让 LLM 判断「刚才那个任务成了没有」。这是整套系统里唯一「跨网络、非确定、耗时秒级」的环节。

### 2.2 它「做了什么、没做什么」

| 做了 | 没做（服务化时必须补） |
| :--- | :--- |
| LLM tool-use 循环（含 thinking/text/tool_use 三种 block 解析） | 无 HTTP/gRPC/WS 服务 |
| 任务白名单校验 + 失败重试 1 次（写在 system prompt 里让 LLM 重试） | 无回调/事件流（执行期间外部拿不到任何进度） |
| 每个任务前重置机械臂位姿（`_reset_robot_pose`） | 无取消（`cancel`）机制，只能 Ctrl-C |
| 保留会话历史（`conversation_history`），支持「继续」 | 无多会话/多用户隔离（全部是实例属性单例） |
| 可选 MuJoCo passive viewer 实时同步 | 无鉴权、无日志落盘、无结构化结果 |
| 模型 warmup（避免第一次推理抖动） | 无帧输出接口（画面只进 viewer 或 `/tmp` PNG） |

---

## 3. 输入

### 3.1 分层的输入契约（这张表是后续做接口的规格书）

| 层 | 输入 | 具体形态 | 代码位置 |
| :--- | :--- | :--- | :--- |
| 进程 | 环境变量 / `.env` | `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`、`PI05_MODEL_PATH`、`HEADLESS`、`DISPLAY`、`CUDA_VISIBLE_DEVICES` | `agent_mode.py:8-29,53` |
| 启动交互 | `stdin` 套件选择 | `1/2/3` → `libero_object / libero_spatial / libero_goal`（默认 3） | `agent_mode.py:671-683` |
| 会话交互 | `stdin` 命令行 | 自然语言（中文/英文）、`reset`、`quit`、`继续` | `agent_mode.py:593-629` |
| L1 | `messages`（Anthropic 格式） | `conversation_history`，含 user / assistant(content blocks) / user(tool_result) | `agent_mode.py:531-589` |
| L1 | `tools` | 两个工具：`execute_task{task}`、`get_scene_info{}` | `agent_mode.py:56-80` |
| L1 | `system` | 场景名 + 当前 `task_description` + 10 条任务白名单 + 规则 | `agent_mode.py:493-529` |
| L2 | 预处理后的批 | `observation.images.image`(1,3,256,256)、`observation.images.image2`、`observation.state`(1,8)、`observation.language.tokens/attention_mask` | `agent_mode.py:239-257` |
| L3 | 环境观测（reset/step 返回） | `{pixels:{image(256,256,3 uint8), image2}, robot_state:{eef:{pos(3),quat(4),mat(3,3)}, gripper:{qpos(2),qvel(2)}, joints:{pos(7),vel(7)}}}` | `lerobot/envs/libero.py:245-295` |
| L4 | 截图 + 提示词 | `/tmp/scene_capture_<ms>.png` + 中文/英文校验问句；经 `uvx minimax-coding-plan-mcp` 的 `understand_image` 工具 | `agent_mode.py:365-423` |

**观测实测样本**（本机 `libero_goal` task 0，本次跑出来的真实值）：

```
task              = "open the middle drawer of the cabinet"
pixels            = {'image': (256,256,3) uint8, 'image2': (256,256,3) uint8}
eef_pos           = [-0.21681,  0.00942,  1.17117]
eef_quat (xyzw)   = [ 0.99960,  0.00155, -0.02823,  0.00031]   ← 顺序是 (x,y,z,w)！
gripper_qpos      = [ 0.03872, -0.03872]
```

### 3.2 状态向量 `observation.state`（8 维）的构造

`agent_mode.py:239-257` 手工拼：

```
state = [ eef_pos(3) , axisangle(eef_quat)(3) , gripper_qpos(2) ]   # 共 8 维
```

> ⚠️ **这里有一个已经确认的 bug（详见 §8.1）**：`_quat_to_axisangle` 把四元数当作 `(w,x,y,z)`，但 robosuite 的 `robot0_eef_quat` 是 `(x,y,z,w)`。实测：正确换算给出 `[3.1397, 0.0049, -0.0887]`（与 ckpt 训练统计均值 `[2.972, -0.220, -0.126]` 同量级），而当前实现给出 `[0.0031, -0.0565, 0.0006]`（≈0），属于「静默错，机器人会动但姿态维度全错」。

---

## 4. 输出

| 层 | 输出 | 形态 | 位置 |
| :--- | :--- | :--- | :--- |
| L1 | Agent 文本 | 终端打印 + 作为 `run_agent` 返回值 | `agent_mode.py:559-574` |
| L2 | 动作 | 每步一个 `(7,)` float，`[-1,1]` 归一化 `[dx,dy,dz,dax,day,daz,gripper]` | `agent_mode.py:285-295` |
| L3 | 场景变化 | MuJoCo 内部状态；可选 `viewer` 窗口同步（需 `DISPLAY`） | `agent_mode.py:297-303, 656-664` |
| L4 | 工具结果 JSON | 字符串化后作为 `tool_result` 回灌给 LLM | `agent_mode.py:458-484` |
| 副作用文件 | 截图 | `/tmp/scene_capture_<epoch_ms>.png`（长期运行会堆积） | `agent_mode.py:388-392` |
| 持久化 | **无** | 不落盘、不返回结构化 success、无 metrics | — |

**两个工具的输出契约（服务化时建议直接沿用字段名）**：

```jsonc
// execute_task
{ "steps": 280, "message": "Executed 280 steps. Use get_scene_info to verify result." }

// execute_task 参数不合法
{ "error": "Task 'xxx' is NOT supported. Only these 10 tasks are available: [...]" }

// get_scene_info
{ "visual_analysis": "<VLM 的自然语言判断>",
  "image_path": "/tmp/scene_capture_1730000000123.png",
  "verified_task": "put the bowl on the stove" }   // 无任务时为 null

// get_scene_info 失败
{ "error": "Scene capture failed: ..." }  |  { "error": "MCP error: ..." }
```

**注意**：`execute_task` 的返回值**故意不含 success**（工具描述里明确写了 `Does NOT return success/failure - use get_scene_info to verify result`）。但 `env.step` 的 `info` 里其实**有** `is_success`（`lerobot/envs/libero.py:327-341`，`self._env.check_success()`）。这是后面做 Web 时最值得改的一处：**用仿真器的真值替代 VLM 猜测，能省掉每轮 3~10 秒的网络往返和不确定性**。

---

## 5. 用法

### 5.1 环境（本机已就绪）

```bash
conda activate /home/zjr/StreamingVLA/miniconda_ygc/envs/flashvla   # 或 source .../bin/activate
cd /home/zjr/StreamingVLA/MiniMax-Agent-VLA-Demo
```

依赖**全部已安装**（实测 `importlib.util.find_spec`）：`torch 2.9.1+cu128`、`lerobot 0.5.1`、`libero`、`robosuite`、`mujoco`、`anthropic`、`mcp`、`python-dotenv`、`flashvla`（editable）。`uvx` 在 `/home/zjr/.local/bin/uvx`（MCP 需要）。

无需再 `pip install -r requirements.txt`（里面没有任何固定版本，重装容易把 torch 打坏）。

### 5.2 启动（Agent 模式）

```bash
export CUDA_VISIBLE_DEVICES=2          # 本机只有 2 号卡有余量（见 §8.7）
unset DISPLAY                          # 无 X11 → 走 headless
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python agent_mode.py
```

- 启动后先问 `Select task suite`（`1/2/3`，默认 3 = `libero_goal`）。
- `HEADLESS` 的判定逻辑在 `agent_mode.py:17`：`HEADLESS=1` 或 `DISPLAY` 为空 → 不起 viewer，只走 EGL 离屏渲染；反之 `mujoco.viewer.launch_passive` 开窗口并每步 `sync()`。
- 首次加载：模型 **148.9 s**（8.5 GB 权重 + tokenizer）+ 编译预热 **59.1 s** + 环境 ~1.5 s ≈ **3.5 min**（§7.6 实测）。
- 默认跑的是 FlashVLA（异步 `d=1` + compile + fuse）；切回 π0.5 基线见 §7.1。

### 5.3 交互

```
🎯 Agent> open the middle drawer of the cabinet     # 支持中文意图，会映射到白名单
🎯 Agent> 把碗放到炉子上
🎯 Agent> 继续                                       # 沿用上一轮上下文
🎯 Agent> reset                                      # env.reset() + 清空对话历史
🎯 Agent> quit
```

### 5.4 简单模式（无 LLM，只跑 VLA）

```bash
python run_robot.py     # 直接把一句话指令交给 VLA 跑 280 步
```

`run_robot.py` 与 `agent_mode.py` 共用同一套 obs 处理（含同一个四元数 bug）和 viewer 同步逻辑；差别只是没有 LLM 层，指令由人直接给。

### 5.5 可执行的任务（`libero_goal`，白名单硬编码）

| # | task 字符串（必须逐字一致） |
| :-- | :--- |
| 1 | `open the middle drawer of the cabinet` |
| 2 | `put the bowl on the stove` |
| 3 | `put the wine bottle on top of the cabinet` |
| 4 | `open the top drawer and put the bowl inside` |
| 5 | `put the bowl on top of the cabinet` |
| 6 | `push the plate to the front of the stove` |
| 7 | `put the cream cheese in the bowl` |
| 8 | `turn on the stove` |
| 9 | `put the bowl on the plate` |
| 10 | `put the wine bottle on the rack` |

> ⚠️ 这份白名单写在两处（`agent_mode.py:463-474` 和 system prompt `agent_mode.py:498-513`），且**只对 `libero_goal` 成立**。若在启动时选 `1=libero_object` / `2=libero_spatial`，`process_tool_call` 会把该套件的正确任务全部判为「NOT supported」——**这两个套件目前实际不可用**。

---

## 6. 配置总表

| 配置 | 位置 | 默认 / 当前值 | 说明 |
| :--- | :--- | :--- | :--- |
| `LLM_BASE_URL` | `.env` | `https://api.minimax.cn/anthropic` | Anthropic 兼容端点；同时会镜像到 `ANTHROPIC_BASE_URL`（MCP 子进程也读） |
| `LLM_API_KEY` | `.env` | `sk-api-...`（明文） | 同时镜像到 `ANTHROPIC_API_KEY` |
| `LLM_MODEL` | `.env` | `MiniMax-M3.1-Flash` | ⚠️ 代码里还有 `temperature=1.0` 与 `max_tokens=4096` |
| `PI05_MODEL_PATH` | env / `agent_mode.py:63` | `/home/zjr/StreamingVLA/flashvla/ckpt/flashvla-pi05-libero` | 一键切 ckpt（改回 `ckpt/pi05_libero` 即基线），见 §7 |
| `FLASHVLA_OVERLAP_STEPS` / `PI05_COMPILE` / `PI05_FUSE` / `FLASHVLA_N_ACTION_STEPS` / `FLASHVLA_COLD_START` / `SKIP_STALE_ACTIONS` | env / `agent_mode.py:73-76` | 见 §7.4 | FlashVLA 专用：异步重叠 `d`、编译/融合、执行步数、冷启动模式 |
| `HEADLESS` | env / `:17` | 自动（`DISPLAY` 空即 headless） | `0` 强开窗口，`1` 强关 |
| `DISPLAY` / `MUJOCO_GL` | env | `:2`（VNC）/ `egl` | 无头必须 `egl` |
| `CUDA_VISIBLE_DEVICES` | env | 无（默认 0 号卡） | 本机 0 号卡被占满，必须显式指定 |
| `suite_name` | `main()` 的输入 | `libero_goal` | `initialize()` 默认值却是 `libero_object` |
| `max_steps` | `execute_task(max_steps=280)` | 280 | 实际预算 = `max_steps + 冷启动步数`，并受 `env._max_episode_steps` 截断（libero_goal=300） |
| `max_iterations` | `run_agent(max_iterations=10)` | 10 | LLM 工具循环上限 |
| `temperature` | `run_agent` | 1.0 | anthropic>=2 需走 `extra_body`（代码已兼容，`:544-548`） |
| `MINIMAX_API_HOST` | `:333` | `https://api.minimaxi.com` | MCP 子进程用，**与 LLM_BASE_URL 域名不同（cn vs .com）** |
| MCP 命令 | `:337-339` | `uvx minimax-coding-plan-mcp -y` | 依赖外网 + `uvx` |
| viewer 机位 | `:660-663` | az 135 / el -25 / dist 1.8 | 只影响窗口观感，不影响推理 |

---

## 7. 模型切换与异步加速

`agent_mode.py` 现在**同时支持两个 ckpt**（π0.5 基线与 FlashVLA），由 `PI05_MODEL_PATH` 一键切换，策略类 / 处理器工厂 / 配置类型全部按 ckpt 自动探测，不需要改代码。

### 7.1 一键切换

```bash
# 默认：FlashVLA（异步 d=1 + torch.compile + fusion）
python agent_mode.py

# 切回 π0.5 基线（同步、无冷启动、n_action_steps=50）
PI05_MODEL_PATH=/home/zjr/StreamingVLA/flashvla/ckpt/pi05_libero python agent_mode.py
```

启动时会打印实际生效的配置，先看这三行再等结果：

```
  ℹ️  torch.compile + fusion 已开启：首次启动需约 3~5 分钟预热（只付一次）
  ⚙️  FlashVLA: n_action_steps=10 chunk_size=10 buffer_slots=4 cold_start=zero_delta compile=True fuse=True
  ⚡ Async overlap: d=1 (skip_stale_actions=False)
Model on cuda (148.9s)  type=pi05-flashvla n_action_steps=10
```

### 7.2 两个 ckpt 的差异（读 `ckpt/*/config.json` 实测）

| 字段 | `ckpt/pi05_libero`（基线） | `ckpt/flashvla-pi05-libero`（默认） |
| :--- | :--- | :--- |
| `type` | `pi05` | `pi05-flashvla` |
| `chunk_size` / `n_action_steps` | 50 / 50 | 10 / 10 |
| `num_buffer_slots` | — | 4（总视界 40） |
| `cold_start_mode` | — | `zero_delta`（前 3 次推理=保持不动，即前 30 步机器人静止） |
| `compile_model` | **True**（构造即 `torch.compile`，`modeling_pi05.py:586`） | False（但异步场景要手动开成 True） |
| 图像输入键 | `empty_camera_0`(3,224,224)、`image2`、`image` | `image2`、`image` |
| 缺失图像键行为 | `_preprocess_images` 只取 batch 里存在的键，其余置 mask（`modeling_pi05.py:1154-1161`）→ 2 路相机也能跑，但 `empty_camera_0` 与训练口径不一致 | 与训练完全一致 |
| state 归一化统计 | 8 维 `mean/std` | **与左侧完全相同**（同一套数据统计） |

### 7.3 为什么必须「成对切换」

只改路径、或只换策略类都会失败，两条报错都实测过：

| 漏改项 | 报错 |
| :--- | :--- |
| 不 import FlashVLA 的配置模块 | `DecodingError: Couldn't find a choice class for 'pi05-flashvla' in <class 'lerobot.configs.policies.PreTrainedConfig'>` |
| 换策略类但仍用 `lerobot.policies.factory` | `ImportError: Processor step 'flashvla_pi05_prepare_language_processor_step' not found in registry` |

所以三件东西必须一起动（`agent_mode.py` 里集中在 `load_policy_and_processors()`）：

```python
if is_flashvla:
    from flashvla.policies.pi05.configuration_pi05 import PI05FlashVLAConfig      # noqa: F401  注册类型
    from flashvla.policies.pi05.modeling_pi05_flashvla import PI05FlashVLAPolicy as PolicyClass
    from flashvla.policies.factory import make_pre_post_processors as make_processors
else:
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy as PolicyClass
    from lerobot.policies.factory import make_pre_post_processors as make_processors
```

**明确不需要改的**（都已实测）：`_process_obs` / state 8 维构造（两 ckpt 共用同一套统计，§8.1 的四元数修复对两者都正确）、图像键（`image`=agentview、`image2`=wrist，与 `LiberoEnv` 一致）、dtype（`cfg.dtype=bfloat16` 只是模型内部 `to_bfloat16_for_selected_params` 的混合精度，**fp32 图像直接喂没问题**）、后处理调用方式（`postprocessor(action)` 传裸 `(1,7)` tensor，返回 `(1,7)`）。

### 7.4 开关总表

| 环境变量 | 默认 | 作用 |
| :--- | :--- | :--- |
| `PI05_MODEL_PATH` | `.../ckpt/flashvla-pi05-libero` | 一键切 ckpt（改回 `.../ckpt/pi05_libero` 即基线） |
| `FLASHVLA_OVERLAP_STEPS` | `1` | 论文的 `d`。`0`=同步；`1..n_action_steps`=异步重叠。**仅 FlashVLA 生效** |
| `PI05_COMPILE` | 跟随 `d`（`d>0` → `True`） | `torch.compile`；**异步不编译等于白开**（launch 仍阻塞） |
| `PI05_FUSE` | 跟随 `PI05_COMPILE` | qkv / gate-up 权重融合，只影响速度 |
| `FLASHVLA_N_ACTION_STEPS` | ckpt 的 `10` | 每次推理执行多少步；论文 LIBERO 用 `5`（必须 ≤ `chunk_size=10`） |
| `FLASHVLA_COLD_START` | `zero_delta` | delta-EEF 环境必须用 `zero_delta`；`current_state` 是给绝对关节位置环境（RoboTwin）的 |
| `SKIP_STALE_ACTIONS` | `0` | RTC 重对齐（推进时跳过 `d` 个陈旧动作）。做时延对比实验时保持关闭 |

### 7.5 异步重叠与冷启动

- **重叠语义**：`d=1` 表示在当前 chunk 还剩 1 步时启动下一次推理，中间步骤直接取缓存动作。实测 `manager.act()` 的慢步恰好落在 **第 8 / 18 / 28 / 38 / 48 / 58 步**（`n_action_steps - d = 9`，0-indexed 即 8），可用来验证重叠是否真的生效。
- **必须编译**：不编译时 launch 是同步调用，慢步只是把阻塞从 chunk 边界挪了一步，收益为零 —— 所以 `PI05_COMPILE` 默认跟随 `d`（与 `eval.sh` 的 `COMPILE_MODEL := d > 0` 同逻辑）。
- **冷启动 30 步**：FlashVLA 前 `num_buffer_slots-1 = 3` 次推理返回 hold 动作（`zero_delta` → 真实动作为 0），即前 `3 × 10 = 30` 步机器人完全不动，占总预算 10.7%。
- **预算自动补偿**：`execute_task` 里 `budget = max_steps + cold_steps`，并被 `env._max_episode_steps`（libero_goal=300）截断，冷启动不再吃任务步数；日志会显式打印：

  ```
  ❄️ 冷启动：前 30 步机器人保持不动（cold_start_mode=zero_delta，3 次推理返回 hold 动作）
  🎯 步数预算 70（max_steps=40 + 冷启动 30，episode 上限 300）
  ♻️ 冷启动结束，第 30 步起为稳态推理
  step 30: action=[ 0.4346  0.1187 -0.0332]
  ```

### 7.6 实测数据（2026-11-05，L40S，GPU 2，FlashVLA + compile + fuse + d=1）

| 指标 | 实测值 | 备注 |
| :--- | :--- | :--- |
| 启动总耗时 | **211.4 s** | 加载 148.9 s + compile 预热 59.1 s（比 manifest 记的 223 s 预热快，卡/负载差异） |
| 冷启动 | **前 30 步动作全 0** | 3 次 hold 推理 |
| 单步预处理（`_process_obs`+preprocessor 含 tokenizer） | **p50 1.9 ms**（max 270 ms，仅首次） | 不是瓶颈 |
| `manager.act()` 取动作 | **p50 0.0 ms**，每 10 步一次 launch（>10 ms） | 编译后单次推理 ~50.7 ms |
| `env.step` | **p50 21.5 ms** | EGL 离屏，2 视角 256×256 |
| **单步合计** | **p50 23.6 ms** | 与论文 LIBERO Time/Step 22.1 ms 同量级 |
| 端到端 70 步 rollout | **5.9 s** | 含切换到 task 9 的 env 重建（~1.5 s）与 reset |
| 不编译对照 | 推理 p50 80.0 ms（manifest） | 摊薄后单步 ≈ 30 ms（估算），且重叠失效、launch 步阻塞 80 ms |

### 7.7 取舍与注意事项

1. **首次启动 3.5 min**（148.9 + 59.1 s）。只想快速调试时用 `PI05_COMPILE=0`，启动降到 ~150 s，但异步收益消失。
2. **显存峰值 ~8.3 GiB**：本机只有 2 号卡有余量，务必 `CUDA_VISIBLE_DEVICES=2`。
3. **pi05 分支只做了桩验证**（dispatch / cfg 覆盖 / 处理器链都通过，`n_action_steps=50`、`compile=True` 与改动前一致），完整加载未跑；FlashVLA 分支是真机端到端跑通的。
4. **`run_robot.py` 未改**：仍是旧的四元数实现、只支持 pi05。要走简单模式需要单独同步本次改动。
5. **warmup 不能省**：`AsyncStreamingActionManager.warmup()` 会穿过冷启动抓 CUDA graph 并把 buffer 复原；若跳过，编译/CUDA graph 捕获会掉在第一个任务的第 30 步附近，表现为「先卡 10~30 s 然后突然动起来」。

---

## 8. 已知问题与坑（都带证据）

> 优先级：**8.0 > 8.2 > 8.1 > 8.3 ...**。8.0 会让「任何非 0 号任务」的成功判定永久为假，且症状极具迷惑性（画面正常、机器人会动、只是"总也做不成"）。
>
> **修复状态（2026-11-05）**：8.0 / 8.1 / 8.3 / 8.4 / 8.5 **已在 `agent_mode.py` 中修复**，条目标题带 ✅；8.2 **部分修复**（成功早停 + 结构化返回值已落地，per-step 回调与取消仍未做）；8.6 / 8.8 **未动**。
> `run_robot.py` **未修改**，仍带 8.1 的四元数缺陷。

### 8.0 【最致命】`task_id` 硬编码为 0 —— 环境永远在别的任务上跑 ✅ 已修复

> **修前**：`agent_mode.py:142-148` 恒用 `task_id=0`，`LiberoEnv` 构造后不再改变。
> **修后**：`_task_id_for()`（`agent_mode.py:195-201`）+ `_build_env()`（`:203-236`），`execute_task` 入口先映射任务再按需重建 env（`:265-271`）。

`agent_mode.py`（修前）：

```python
self.env = LiberoEnv(task_suite=task_suite, task_id=0, task_suite_name=suite_name, ...)
```

`task_id` 在**构造时绑定**，之后永不改变。后果三条（实测 `libero_goal`）：

1. **场景初始状态是 0 号任务的**（"open the middle drawer of the cabinet"），而 `init_states` 也是 0 号任务的 50 条。两个任务物体位置只差 1~2 cm（下表），所以画面「看起来没问题」，极难发现。

   | 物体 | task_id=0 初始 xpos | task_id=9（bottle on rack） |
   | :--- | :--- | :--- |
   | wine_bottle_1 | (-0.210, -0.052, 0.899) | (-0.194, -0.037, 0.899) |
   | wine_rack_1 | (-0.269, -0.253, 0.920) | (-0.254, -0.269, 0.920) |
   | flat_stove_1 | (-0.419, +0.214) | (-0.401, +0.213) |

2. **成功判定属于 0 号任务**：`LiberoEnv.step` 里的 `is_success` 来自 `_check_success()`，而 0 号任务的 BDDL 目标是
   `(:goal (And (Open wooden_cabinet_1_middle_region)))`。
   → 你让它「把酒瓶放架子上」，无论做得多好，`info["is_success"]` **永远是 False**。
3. **`env.task_description` 是错的**：system prompt 里 `Current scene:` 会写成 "open the middle drawer of the cabinet"（你日志里 LLM 的 thinking 已经自己发现了这一点，但它无权改环境）。

**修法（已落地）**：先用指令反查 `task_id`，再按任务构造环境（`LiberoEnv` 不支持换任务，必须重建或预建）：

```python
task_id = next(i for i, t in enumerate(task_suite.tasks) if t.language == task)
if task_id != self.task_id:                 # 需要重建 env（模型不用重载）
    self.env.close()
    self.env = LiberoEnv(task_suite=task_suite, task_id=task_id,
                         task_suite_name=self.suite_name,
                         render_mode="rgb_array", obs_type="pixels_agent_pos")
    self.obs, _ = self.env.reset()
```

实现细节里有两个必须知道的副作用：

1. MuJoCo 的 `model/data` 指针会随 env 重建而更换，所以 `_build_env()` 会**先关掉 passive viewer** 并置 `_viewer_lost`，`run_interactive()` 的循环条件因此从 `viewer.is_running()` 改成 `viewer.is_running() and not self._viewer_lost`（headless 下无影响）。
2. 重建只需 ~1.4 s（env 构造 + reset），**模型不重载**，所以任务间切换的代价可以忽略。

### 8.1 `_quat_to_axisangle` 四元数顺序错（高危，静默错）✅ 已修复

> **修后**（`agent_mode.py:164-183`）：按 `(x,y,z,w)` 取 `w = quat[..., 3]`，与 lerobot `EnvProcessorPipeline._quat2axisangle` 对齐；零旋转走 mask 分支。
> **验收**（真实 `LiberoEnv` 重置帧）：`state = [..., 3.13982, 0.00305, -0.09100, ...]`，MEAN_STD 归一化 `[..., 0.49, 0.25, 0.11, ...]` —— 从 −8.62σ 回到 ±3σ 内。
> ⚠️ `run_robot.py:26-35` **仍是 w-first**，走简单模式时同样要改。

- 修前代码：`agent_mode.py:170-178`（`run_robot.py:26-35` 同款），按 `(w,x,y,z)` 解析。
- 事实：robosuite `robot0_eef_quat` 由 `T.convert_quat(get_body_xquat(...), to="xyzw")` 产生 → **`(x,y,z,w)`**（`robosuite/robots/single_arm.py:308-309`）。lerobot 官方 `EnvProcessorPipeline._quat2axisangle` 也是 `w=quat[:,3]`（`lerobot/processor/env_processor.py:114-152`）。
- 实测（`libero_goal` task 0 重置后同一帧）：

  | 实现 | 结果 |
  | :--- | :--- |
  | 当前 `agent_mode.py`（w-first） | `[ 0.0031, -0.0565, 0.0006]` ← 近似零，姿态信息丢失 |
  | lerobot 官方（xyzw） | `[ 3.1397,  0.0049, -0.0887]` |
  | ckpt 训练统计均值 | `[ 2.9722, -0.2205, -0.1256]`（std `[0.344, 0.907, 0.325]`） |

  即：当前实现的 state 经 MEAN_STD 归一化后落在 `-8σ` 之外，与训练分布完全不匹配。
- 修法（1 行）：改用 `w = quat[..., 3]`，或直接复用 lerobot 的 `EnvProcessorPipeline`。
- **含义**：在动工前必须先修这条，否则「成功率不对」会被误判成模型问题。

### 8.2 `execute_task` 是黑盒阻塞调用 🟡 部分修复

> **修后**（`agent_mode.py:258-337`）：改为逐步读 `info["is_success"]`，成功即 `break`；`done/truncated` 也退出；返回 `{"steps", "success", "message"}`，并写入 `last_task_success / last_task_steps`。LLM 侧规则同步改成「`success=true` 就不必再调 `get_scene_info`」。
> **仍未做**（Web 化必需）：per-step 回调、超时、`threading.Event` 取消。现在仍是一次调用占满整个 episode 时长，前端拿不到中途进度。

修前：`agent_mode.py:259-314` 跑满 280 步才返回，**没有 per-step 回调、没有超时、没有取消**。对 Web 场景是致命限制——前端在 8~30 秒内拿不到任何进度。必须加 `on_step` 回调 + `threading.Event` 取消。

### 8.3 任务白名单与套件不匹配 ✅ 已修复

见 §5.5。修前 `libero_object` / `libero_spatial` 会被硬编码白名单全拒。
**修后**：`process_tool_call`（`agent_mode.py:488`）与 system prompt（`:507-510`）都改为从 `self.task_suite.tasks[*].language` 动态生成。

### 8.4 `env.auto_reset = False` 是空操作 ✅ 已修复

`agent_mode.py:149`（修前）设了这个属性，但**安装的 lerobot 0.5.1 里不存在 `auto_reset`**（全仓 grep 无命中）；`LiberoEnv.step` 在 `terminated`（含 `is_success`）时会**无条件 `self.reset()`**（`lerobot/envs/libero.py:327-341`）。后果：任务成功瞬间场景被重置，Agent 后续的 `get_scene_info` 看到的是「新场景」，验证结论会自相矛盾。

**修后**：删掉这条无效赋值；`execute_task` 在 `is_success` 或 `done/truncated` 的第一时间就 `break`，因此 env 内部那次自动 reset 不再影响后续判断（下一个 attempt 本来就要 `env.reset()`）。

### 8.5 `make_pre_post_processors` 的第一个参数传错类型（侥幸能跑）✅ 已修复

`agent_mode.py:134-136`（修前）传的是 `self.policy.config.type`（字符串），而签名要求 `PreTrainedConfig`（`lerobot/policies/factory.py:243`）；因为 `pretrained_path` 非空时只走 `isinstance` 分支判断，字符串恰好没触发报错。

**修后**（`agent_mode.py:136-142`）：`make_pre_post_processors(self.policy.config, MODEL_PATH)`。已验证处理器链可正常加载：pre = `[RenameObservations, AddBatchDimension, Normalizer, Pi05PrepareStateTokenizer, Tokenizer, Device]`，post = `[Unnormalizer, Device]`。

### 8.6 MCP 视觉验证是「外网 + 子进程 + 秒级」

- 每调用一次 `get_scene_info` 会 `uvx` 拉起一个 MCP server 子进程（`agent_mode.py:316-363`），失败时静默返回 `"MCP error: ..."` 字符串给 LLM（LLM 会把错误当成「验证失败」继续重试）。
- 域名不一致：LLM 用 `api.minimax.cn`，MCP 用 `api.minimaxi.com`（`:333`）。任一不可达 → 验证层失效。
- **推荐替换**：用 `env.step` 返回的 `info["is_success"]`（LIBERO 的 BDDL 谓词真值）做判定，毫秒级且确定。VLM 验证留作「可选增强」。

### 8.7 硬件与并发约束（本机实测）

| 项 | 事实 |
| :--- | :--- |
| GPU | 8×L40S（46 GB/卡）；0/1/3~7 号卡被 VLLM 占 40~43 GB；**仅 2 号卡有余量（~40 GB），且同时被 Isaac Sim 占用** |
| 单实例显存 | FlashVLA 峰值 **8.29 GiB** |
| 建议 | 只跑 1 个实例，`CUDA_VISIBLE_DEVICES=2`；p99 抖动来自共享卡，前端做插值平滑 |
| 主机 | `172.30.1.26`（内网），SSH 22 已监听 |
| 端口 | `8000` 空闲；`9999`（RoboTwin server 默认）也空闲 |

### 8.8 其它

- `_reset_robot_pose`（`:180-237`）直接改 `sim.data.qpos/qvel` 与控制器 `goal_pos/goal_ori`，跨 robosuite 版本极脆；正常路径应优先 `env.reset()`。
- 明文密钥：`.env` 与 `README.md` 里都有**真实 API Key**。这个仓库如果给出去，等于给出账单。→ 轮换 + 改成模板变量。
- `analyze` 提示词里硬编码了「bowl/plate/wine bottle/cream cheese」（`:397-412`），换套件后提示词与新场景不匹配。
- 所有输出都是 `print`，无级别、无时间戳、无结构化字段 → 服务化时统一改成 `emit(event)`。

---

## 9. 远程主机 ↔ 用户主机 通信方案（基于这个 Agent）

### 9.0 结论先行

> **不要**给 `agent_mode.py` 加一个「HTTP 客户端」去连远端推理服务——模型、仿真、LLM 全在 `172.30.1.26` 这台机器上，多一跳 HTTP 只会把 192 KB/帧的图变成 base64 垃圾。
> **要做的**是：把这台机器变成「一个进程组 + 一个 Web 服务」，用户主机只跑浏览器，通过 **WebSocket 收「帧 + 状态 + Agent 事件」，通过 REST 下「指令 + 重置 + 取消」**。

```
┌──────────────────────── 远端主机 172.30.1.26 ────────────────────────┐
│                                                                      │
│  ┌──────────────┐   cmd queue   ┌───────────────────────────────┐    │
│  │ FastAPI      │──────────────▶│ AgentWorker (线程)            │    │
│  │ + uvicorn    │◀──────────────│  anthropic.SyncClient         │    │
│  │ (asyncio 主) │   event queue │  conversation_history         │    │
│  │              │               │  调工具 → 投递 SimCmd         │    │
│  │  WS /ws      │               └───────────┬───────────────────┘    │
│  │  REST /api/* │                           │ SimCmd(execute/reset)   │
│  │  GET /       │               ┌───────────▼───────────────────┐    │
│  └──────┬───────┘   frame queue │ SimWorker (线程, 独占)         │    │
│         │                       │  LiberoEnv + Policy + Manager │    │
│         │                       │  10 Hz 循环: step → 渲染 → 推 │    │
│         │                       └───────────────────────────────┘    │
└─────────┼────────────────────────────────────────────────────────────┘
          │  WebSocket: JPEG 帧 + JSON 事件
┌─────────▼─────────┐
│ 用户主机 浏览器    │  index.html + app.js（原生，无构建）
└───────────────────┘
```

**为什么必须三个线程**（不是风格问题，是硬约束）：

1. MuJoCo 的 `MjModel/MjData` 与 robosuite 环境**不是线程安全的**，且 `env.step` 必须与创建 GL context 的线程同源（EGL 上下文绑定线程）→ SimWorker 独占一个线程，**所有** `env.*` 与 `policy.*` 调用都在这个线程里。
2. `anthropic` 同步客户端一次调用 1~5 秒（MiniMax 云端）→ 放在 asyncio 事件循环里会直接卡死所有 WS 连接。
3. uvicorn 需要一个不阻塞的 loop → 它只做「收指令 / 广播事件」，永不碰 env/policy。

### 9.1 协议设计

#### REST（控制面，幂等、可重放）

| 方法 | 路径 | 请求 | 响应 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| GET | `/`、`/app.js` | — | 静态文件 | 前端与后端同源，免 CORS |
| GET | `/api/state` | — | `{suite, task, step, success, eef_pos, eef_quat, gripper, joints}` | 给「后加入的页面」做首帧快照 |
| GET | `/api/tasks` | — | `[{id, text}]` | 当前套件支持的任务（替代硬编码白名单） |
| POST | `/api/command` | `{"text":"把碗放到炉子上"}` | `202 {queued:true}` | 交给 AgentWorker |
| POST | `/api/direct` | `{"task":"put the bowl on the stove"}` | `202` | 跳过 LLM，直接跑 VLA（调试用） |
| POST | `/api/reset` | — | `202` | `env.reset()` + `manager.reset()` + 清空对话 |
| POST | `/api/cancel` | — | `202` | 置 `cancel_event`，SimWorker 下个 step 退出 |

> 只用 POST 触发、GET 查状态；**所有会改环境的行为都进队列**，handler 立即返回 `202`，不阻塞。

#### WebSocket `/ws`（数据面）

服务端 → 客户端（每帧/每事件一条消息；文本 JSON + 二进制帧分离）：

| type | 字段 | 频率 | 用途 |
| :--- | :--- | :--- | :--- |
| `hello` | `suite, task, fps, action_dim, frame_size` | 连接时 1 次 | 前端初始化画布 |
| `frame` | 二进制 JPEG（或 JSON+`data`base64） | 10 Hz | 机器人画面 |
| `state` | `step, eef_pos[3], eef_quat[4], gripper[2], joints[7], success` | 10~30 Hz | 3D 视图/数值面板 |
| `status` | `phase: idle\|planning\|executing\|verifying\|done, text` | 变化时 | 顶部状态条 |
| `agent` | `role: assistant\|tool_use\|tool_result, text` | 变化时 | 对话/推理流 |
| `episode_end` | `success: bool, steps: int, wall_ms: int` | 每 episode | 结果卡片 |

客户端 → 服务端：`{"type":"command","text":"..."}`、`{"type":"reset"}`、`{"type":"cancel"}`、`{"type":"ping"}`。

示例：

```jsonc
// S→C
{"type":"hello","suite":"libero_goal","task":"open the middle drawer of the cabinet","fps":10,"frame_size":[256,256]}
{"type":"status","phase":"executing","text":"execute_task: put the bowl on the stove (step 120/280)"}
{"type":"state","step":120,"eef_pos":[-0.21,0.01,1.17],"eef_quat":[0.9996,0.0016,-0.0282,0.0003],"gripper":[0.039,-0.039],"success":false}
{"type":"episode_end","success":true,"steps":168,"wall_ms":18200}
// C→S
{"type":"command","text":"把碗放到炉子上"}
```

**帧编码三选一（推荐 A）**：

| 方案 | 单帧体积 | 实现成本 | 评价 |
| :--- | :--- | :--- | :--- |
| **A. WS 二进制 JPEG** | 256×256 q75 ≈ 12~25 KB → 10 Hz ≈ **0.2~0.3 MB/s** | 中（前端 `createImageBitmap(blob)`） | ✅ 推荐；与事件同连接，易做背压 |
| B. JSON + base64 | ×1.33 | 低 | 可用；多 33% 带宽 + 两次编解码 |
| C. MJPEG（`multipart/x-mixed-replace`） | 同上 | **最低**（`<img src="/stream">` 即可） | ✅ 兜底首选：不需要 `websockets` 包，纯 HTTP |

> 本机 **`websockets` 未安装**（`fastapi` 也未安装），而 `uvicorn` / `starlette` / `aiohttp` 都在。所以两条路：
> - 走 A：`pip install fastapi websockets`（走代理 `http_proxy=127.0.0.1:7890`）；
> - 走 C 或 aiohttp：**零安装**，今天就能跑。**建议先用 C 打通，再升级到 A。**

#### 背压与节流（不写这段一定会「越跑越卡」）

- 每个 WS 连接一个 `asyncio.Queue(maxsize=2)`；投递时 `put_nowait` 失败就**丢旧帧**（丢帧不丢状态）。
- 渲染与推理解耦：`manager.needs_observation()` 为 False 的步**可以完全不渲染**（FlashVLA 一个 chunk 只需 1 帧），只在「要推理」和「要显示」时渲染 → 带宽与 CPU 双降。实测 `need_render : total ≈ 1 : n_action_steps`。
- 前端用 `requestAnimationFrame` + 线性插值，把 10 Hz 帧补到 60 fps 视觉；`state` 里的 `eef_pos` 可直接驱动 three.js 里的 Panda（第二阶段再做）。

### 9.2 从 `agent_mode.py` 到服务的改造清单（逐条）

> 进度（2026-11-05）：**#4 / #6 / #7 已随 §8 的修复落地**；#2 只完成了「`max_steps` 参数 + 成功早停」，回调与取消仍未做；其余未动。

| # | 现状 | 改法 |
| :--- | :--- | :--- |
| 1 | `print(...)` 散布全局 | 抽 `emit(type, **fields)`，同时投递给 `console_logger` 与 `event_queue` |
| 2 | `execute_task` 无进度 | 加 `on_step(step, obs, info, action)` 回调（每步或每 N 步）、`cancel_event: threading.Event`、`max_steps` 参数 |
| 3 | `_interaction_loop` 读 stdin | 换成 `cmd_queue.get()`；`quit` → 关闭服务 |
| 4 | 无 success 判定 | 用 `info["is_success"]` 发 `episode_end`；`auto_reset=False` 删掉（无效），改为服务层控制 reset |
| 5 | `get_scene_info` 走 MCP | 保留为可选工具；默认用 `is_success`；返回里加 `image_path` 供前端展示「验证帧」 |
| 6 | 白名单硬编码 | 从 `task_suite.tasks[i].language` 动态生成（`LiberoEnv` 里已有 `task_description`） |
| 7 | 四元数 bug | 改 `w=quat[...,3]`，或整体改用 `EnvProcessorPipeline` |
| 8 | 单会话 | 加 `session_id`；先做单用户 + 全局锁（一个 SimWorker 只能跑一个 episode） |
| 9 | 无鉴权 | `?token=` 校验 + 只监听 `127.0.0.1`，靠 SSH 隧道对外 |
| 10 | 无落地 | `results/<ts>.jsonl` 记录每个 episode 的 steps/success/latency（对齐 `eval_info.json` 字段） |

### 9.3 最小骨架（可直接落地）

服务端（`server.py`，约 90 行；用 aiohttp 则连 `pip install` 都省了）：

```python
import asyncio, io, queue, threading, json, time
from dataclasses import dataclass
import numpy as np
from PIL import Image
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI()
cmd_q: "queue.Queue[str]" = queue.Queue()      # HTTP → AgentWorker
frame_q: "asyncio.Queue[bytes]" = None          # SimWorker → WS（在 loop 里创建）
LOOP: asyncio.AbstractEventLoop = None
latest = {"state": {}, "frame": None}

def push_frame(jpg: bytes):
    # 线程安全地向 asyncio 队列投递；满了就丢最旧帧
    def _put():
        if frame_q.full():
            try: frame_q.get_nowait()
            except asyncio.QueueEmpty: pass
        frame_q.put_nowait(jpg)
    LOOP.call_soon_threadsafe(_put)

def sim_worker(env, policy, pre, post, manager):
    while True:
        text = cmd_q.get()                      # 阻塞等指令（由 AgentWorker 或 /api/direct 投递）
        if text == "__quit__": return
        manager.reset(); policy.reset()
        for step in range(280):
            obs = env.obs
            proc = build_batch(obs, text, pre, policy)       # 见 §3.2，注意 rot180 与 xyzw
            action = post(manager.act(proc))                 # d=0/1 由 manager 决定
            obs, _r, _term, _trunc, info = env.step(action.cpu().numpy().squeeze())
            if step % 3 == 0:                                # 10 Hz 显示 / 30 Hz 物理
                img = obs["pixels"]["image"][::-1, ::-1]     # 与 LiberoEnv.render() 一致
                buf = io.BytesIO(); Image.fromarray(img).save(buf, "JPEG", quality=75)
                push_frame(buf.getvalue())
                latest["state"] = pack_state(obs, step, info)
            if info.get("is_success") or CANCEL.is_set():
                break

# 需自行实现的 3 个纯函数（只做数学，别夹带状态）：
#   build_batch(obs, task, pre, policy) -> 已上 device 的批（见 §3.2/§8.1）
#   pack_state(obs, step, info)         -> {"step","eef_pos","eef_quat","gripper","joints","success"}
#   CANCEL                              -> threading.Event()，由 /api/cancel 置位

@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    await sock.send_json({"type": "hello", "fps": 10})
    while True:
        jpg = await frame_q.get()
        await sock.send_bytes(jpg)              # 二进制帧
        await sock.send_json({"type": "state", **latest["state"]})
```

前端（`index.html`，无构建）：

```html
<img id="cam" width="384">                       <!-- 或走 MJPEG: <img src="/stream"> -->
<pre id="log"></pre>
<script>
const ws = new WebSocket(`ws://${location.host}/ws`);
ws.binaryType = "blob";
ws.onmessage = (e) => {
  if (typeof e.data === "string") { /* JSON 事件：state/agent/status/episode_end */ return; }
  document.getElementById("cam").src = URL.createObjectURL(e.data);
};
const send = (o) => ws.send(JSON.stringify(o));
function submit(){ send({type:"command", text: document.querySelector("#in").value}); }
</script>
```

### 9.4 用户主机怎么连（三选一）

| 方式 | 命令/地址 | 适用 | 注意 |
| :--- | :--- | :--- | :--- |
| **A. 同内网直连** | 浏览器开 `http://172.30.1.26:8000` | 两台机器在同一局域网 | 防火墙放行 8000（**未验证**）；服务需监听 `0.0.0.0` |
| **B. SSH 隧道（推荐）** | `ssh -N -L 8000:127.0.0.1:8000 zjr@172.30.1.26`，然后开 `http://127.0.0.1:8000` | 跨网段/跨城；服务只监听 `127.0.0.1` | 零额外软件、天然加密、不需要动防火墙 |
| **C. 反代/内网穿透** | `frpc` / `cloudflared tunnel` | 要给外部演示 | **必须先加 token 鉴权**，此 demo 当前无鉴权 |

**安全红线**：
1. `LLM_API_KEY` 只留在服务端 `.env`，**永不下发浏览器**；前端拿到的最多是「已连接的模型名」。
2. 现状 `.env` 与 `README.md` 里是**真实可用 key**（`sk-api-b36...`），已在 git 历史中 → **建议立即在 MiniMax 控制台轮换**，并改成 `LLM_API_KEY=${MINIMAX_KEY}` 形式。
3. 服务默认 `host=127.0.0.1`，对外一律通过 SSH 隧道，避免把「能操控机器人 + 烧 GPU + 烧 API 额度」的接口暴露在 172.30.x 网段上。

### 9.5 带宽与时延预算（本机实测数字代入）

| 环节 | 耗时 | 备注 |
| :--- | :--- | :--- |
| `env.step`（含 2×256×256 渲染） | **21.5 ms**（agent 循环内 p50）/ 26.5 ms（独立测） | p50，EGL 离屏 |
| FlashVLA 推理 / 次 | **80 ms**（不编译）/ 50.7 ms（编译） | 每 `n_action_steps=10` 步一次 → 摊到每步 5~8 ms，`d>0` 时被重叠隐藏 |
| JPEG 编码 256×256 q75 | ~2~5 ms | Pillow |
| LAN WS 传输 | < 1 ms | 帧 12~25 KB |
| 单步合计（10 Hz 节拍） | **~24 ms**（§7.6 实测 23.6）→ 可跑 40 Hz，节流到 10 Hz 很轻松 | |
| LLM 一次工具循环 | 1~5 s | 云端，不可控；必须有「planning」状态提示 |
| MCP 视觉验证 / 次 | 3~10 s（**未精确测量**） | 换成 `is_success` 可降到 0 |
| 模型加载 / torch.compile 预热 | 148.9 s / 59.1 s（§7.6 实测；manifest 记 145.6 / 223） | 只在启动时付一次 |

> 结论：**延迟大头是 LLM 与 MCP，不是模型也不是网络**。所以：
> - 首屏体验优化点 = 「进页面先给一张静态帧 + 一个 (空闲) 状态条」；
> - 交互体验优化点 = 「LLM 规划阶段用打字机效果展示 thinking/text，别让画面静止 5 秒没有反馈」；
> - 真正影响「机器人看起来流畅」的 = 10 Hz 节拍 + 前端插值。

### 9.6 落地顺序与验收

| 阶段 | 内容 | 验收标准 |
| :--- | :--- | :--- |
| P0（半天） | MJPEG 或 WS 推 `obs["pixels"]["image"]`，`/api/direct` 直跑固定任务 | 浏览器能看到机器人在动；日志里 step 递增；无内存增长 |
| P1（1 天） | 接入 AgentWorker（复用 `run_agent` + `AGENT_TOOLS`），把 `print` 改成事件 | 页面能看到 `executing → verifying → done`；失败可复现 |
| P2（1 天） | 修 §8.1 四元数、用 `info["is_success"]` 替代 MCP、动态任务白名单 | 同一 seed 下成功率与 `sim_eval` 口径一致 |
| P3（可选） | 换 `AsyncStreamingActionManager`（d=1）、帧/推理解耦（`needs_observation`） | 每步耗时从 ~35 ms 降到 ~28 ms，画面无卡顿 |
| P4（可选） | three.js + `robot_state.joints.pos`(7) 驱动 Panda URDF 真 3D | 视觉与真值一致；工作量约翻倍 |

### 9.7 备选拓扑：把推理拆到另一台机器（**不推荐，但要知道代价**）

如果你真要把「GPU 推理」和「仿真」拆到两台机器：

- 每步要传 2×256×256×3 = **384 KB** 裸图（或 25 KB JPEG×2），10 Hz ≈ 0.5 MB/s；<br>加上动作回传（7×8 B）可忽略。
- 协议建议：**二进制帧 + 长度前缀 + JSON 头**，或直接沿用 RoboTwin 的参考实现（`sim_eval/robotwin/.../policy_model_server.py`：裸 TCP、`TCP_NODELAY`、JSON+base64、首次调用 timeout 300 s、**无鉴权无 TLS**）。
- 代价：图像编解码往返 + 状态跨机同步 + 明明同机可以免掉的全部序列化。**同机部署时请坚持进程内调用。**

---

## 10. 附：证据索引

| 主题 | 位置 |
| :--- | :--- |
| Agent 工具定义 | `MiniMax-Agent-VLA-Demo/agent_mode.py:158-186` |
| 模型切换（探测 / 加载 / cfg 覆盖） | `agent_mode.py:79-156`（`detect_policy_type` / `load_policy_and_processors`） |
| 开关定义 | `agent_mode.py:63-76` |
| 初始化（模型 → manager → 环境 → warmup） | `agent_mode.py:228-279` |
| 异步管理器构建 | `agent_mode.py:249-266` |
| warmup（穿冷启动 + CUDA graph） | `agent_mode.py:297-323`（`AsyncStreamingActionManager.warmup`：`flashvla/async_manager.py:252-290`） |
| obs 处理（四元数已修 xyzw） | `agent_mode.py:325-344`、`:399-417` |
| 任务绑定 / 环境重建 | `agent_mode.py:356-397`（`_task_id_for` / `_build_env`） |
| 执行循环（早停 + 冷启动预算） | `agent_mode.py:419-522` |
| MCP 调用 | `agent_mode.py:524-571` |
| 截图 + 校验提示词 | `agent_mode.py:573-631` |
| 工具分发 + 动态白名单 | `agent_mode.py:667-684` |
| LLM 循环 / system prompt | `agent_mode.py:686-788` |
| viewer 生命周期 | `agent_mode.py:828-872` |
| `LiberoEnv` API（reset/step/render/相机映射） | `lerobot/envs/libero.py:224-229, 245-295, 296-321, 322-352`（相机映射 `:138-143`） |
| `preprocess_observation`（只拼键，不做 quat 转换） | `lerobot/envs/utils.py:49-113` |
| 官方 `_quat2axisangle`（xyzw） | `lerobot/processor/env_processor.py:114-152` |
| `make_pre_post_processors` 签名 | `lerobot/policies/factory.py:243-253` |
| robosuite `eef_quat` 为 xyzw | `robosuite/robots/single_arm.py:308-312` |
| FlashVLA 冷启动语义 | `flashvla/policies/pi05/modeling_pi05_flashvla.py:1180-1256` |
| 异步重叠管理器 | `flashvla/async_manager.py:29-260` |
| VLA 接口全量契约（观测/动作/manifest） | `flashvla/docs/vla_interface_manifest.md` |
| LIBERO 评测运行方法 | `flashvla/docs/howtodo.md` |

**本次实测记录（可复现）**：在 `flashvla` 环境用 `MUJOCO_GL=egl` 构造 `LiberoEnv(libero_goal, task_id=0)` 并 reset，打印 `robot_state` 与该帧的两种 quat→axis-angle 结果，用于核对 §8.1；env 构造+reset ≈ 1.4 s，与本仓库文档记录一致。
