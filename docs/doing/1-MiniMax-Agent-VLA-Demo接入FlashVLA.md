# 1 · MiniMax-Agent-VLA-Demo 接入 FlashVLA（工作总结）

> 日期：2026-11-05
> 目标：把 `/home/zjr/StreamingVLA/MiniMax-Agent-VLA-Demo` 这个「LLM Agent + π0.5 + LIBERO」的 demo，
> 改成「LLM Agent + **FlashVLA** + LIBERO」，并为后续「远端跑仿真、用户主机看网页」做准备。
> 所有数字均为本机（8×L40S，仅 GPU 2 有余量）实测；未验证处明确标注。

---

## 0. 一句话状态

- **已完成**：Agent 的 5 个阻断级缺陷已修；`agent_mode.py` 现在支持 **两个 ckpt 一键切换**（π0.5 基线 / FlashVLA）+ **异步重叠 d=1 + torch.compile + fusion**，真机端到端跑通，稳态 **23.6 ms/步**。
- **未完成**：Web 服务化（只有设计，无代码）；`run_robot.py` 未同步；π0.5 分支只做桩验证；改动**未提交**。

---

## 1. 背景

三段现成资产：

| 资产 | 位置 | 说明 |
| :--- | :--- | :--- |
| 模型仓库 | `/home/zjr/StreamingVLA/flashvla` | FlashVLA（π0.5 + 流式动作块），含 `AsyncStreamingActionManager`、LIBERO 评测脚本、ckpt |
| Agent demo | `/home/zjr/StreamingVLA/MiniMax-Agent-VLA-Demo` | MiniMax LLM 工具调用 + π0.5 推理 + LIBERO 仿真 + MCP 视觉校验 |
| 运行环境 | `conda env flashvla` | Python 3.12 / torch 2.9.1+cu128 / lerobot 0.5.1（**demo 实际依赖 site-packages 的 lerobot，仓库内 `lerobot_back/` 是无效残骸**） |

---

## 2. 阶段一：文档化（产出 `docs/minimax_agent_vla_demo.md`）

把 demo 拆成四层并写成接口规格：

```
L1 规划层  MiniMax LLM（Anthropic 兼容 tool-use：execute_task / get_scene_info）
L2 执行层  VLA 策略（7 维 delta-EEF，每步 obs→pre→policy→post→env.step）
L3 仿真层  LiberoEnv（2×256×256 RGB + eef/gripper/joints）
L4 验证层  MCP + 视觉模型（uvx 子进程，秒级、外网依赖）
```

覆盖：输入/输出契约表、用法、配置总表、已知问题 8 类、以及「远端主机 ↔ 用户主机」通信方案（三线程模型 + REST/WS 协议 + 带宽预算 + SSH 隧道 + 骨架代码）。

**关键结论**：它不是一个服务，而是**单进程单会话命令行 Agent**——无 HTTP、无回调、无取消、无结构化结果。做 Web 必须先重构。

---

## 3. 阶段二：故障诊断（用户实测「任务做不成」）

用户跑 `put the wine bottle on the rack` 失败（两轮 280 步、VLM 判失败）。定位到 **5 条叠加原因**：

| # | 问题 | 证据 |
| :--- | :--- | :--- |
| 1 | **`task_id` 硬编码 0**（最致命） | `agent_mode.py` 恒用 `LiberoEnv(task_id=0)`；`task_id` 构造时绑定。0 号任务的 BDDL 目标是 `(Open wooden_cabinet_1_middle_region)` → 任何其他任务的 `is_success` **永远 False**；`env.task_description` 也是错的（日志里 LLM 自己都发现了 "current scene says open the middle drawer"）。两个任务的物体位置只差 1~2 cm，画面看不出差别 |
| 2 | **四元数顺序错** | robosuite `robot0_eef_quat` 是 `(x,y,z,w)`（`convert_quat(..., to="xyzw")`），代码按 `(w,x,y,z)` 解析。实测同一帧：错误实现 `[0.0031,-0.0565,0.0006]` vs 正确 `[3.1397,0.0049,-0.0887]`，ckpt 训练均值 `[2.972,-0.220,-0.126]` → **归一化后 −8.62σ vs +0.49σ** |
| 3 | **跑满 280 步不早停** | `execute_task` 完全忽略 `done/is_success/terminated` → 成功后继续推，把物体顶歪（用户第二次看到「斜卡在架子里、机械臂压在架上」就是过冲形态） |
| 4 | **retry 只重置机械臂** | `_reset_robot_pose()` 只改 qpos，物体保持被扰动的状态 |
| 5 | **白名单硬编码 `libero_goal` 10 条** | `libero_object` / `libero_spatial` 的合法任务会被全拒 |

另有非阻断项：`env.auto_reset=False` 是空操作（lerobot 0.5.1 无此属性）、`make_pre_post_processors(config.type, ...)` 传字符串侥幸能跑、`.env`/README 里有真实 API Key（建议轮换）。

---

## 4. 阶段三：代码修复（`agent_mode.py`）

| 编号 | 修复 | 实现 |
| :--- | :--- | :--- |
| P0 | 指令 → `task_id` → 重建 env | `_task_id_for()` + `_build_env()`；`execute_task` 入口先映射再按需重建（模型不重载，~1.4 s）；`_build_env` 会先关闭 viewer（它与旧 MuJoCo model 绑定） |
| P1 | 用仿真器真值早停 | 逐步读 `info["is_success"]`，成功/`done`/`truncated` 立即 break；返回 `{steps, success, message}`；system prompt 改为「success=true 不必再调 VLM」 |
| P2 | 全场景复位 | 用 `env.reset()` 替代 `_reset_robot_pose`（整段删除），并在 `policy.reset()` 之前执行 |
| P3 | 四元数修正 | `w = quat[..., 3]`，零旋转走 mask 分支，与 lerobot `EnvProcessorPipeline` 对齐 |
| 配套 | 动态白名单 / prompt / viewer 生命周期 | 任务清单从 `task_suite.tasks[*].language` 生成；viewer 改手动管理以支持换任务 |

**验证**：`py_compile` 通过；桩测试覆盖四条分支（跑满预算 / 中途成功 / done 终止 / 不支持指令）；真实 `LiberoEnv` 上 state 归一化从 −8.62σ 回到 **±3σ 内**；任务切换 `task_id 0 → 9` 且 `task_description` 正确。

---

## 5. 阶段四：模型切换与异步加速

### 5.1 三件必须成对切换的东西（漏一个就跑不起来）

| 漏改项 | 实测报错 |
| :--- | :--- |
| 不 import FlashVLA 配置模块 | `DecodingError: Couldn't find a choice class for 'pi05-flashvla'` |
| 换策略类但仍用 `lerobot.policies.factory` | `ImportError: Processor step 'flashvla_pi05_prepare_language_processor_step' not found in registry` |

`agent_mode.py` 里集中在 `load_policy_and_processors()`：`detect_policy_type()` 读 ckpt 的 `config.json.type` → 选策略类 + 处理器工厂 + 套 cfg 覆盖。

### 5.2 新增开关

| 环境变量 | 默认 | 作用 |
| :--- | :--- | :--- |
| `PI05_MODEL_PATH` | `ckpt/flashvla-pi05-libero` | 一键切 ckpt（改回 `ckpt/pi05_libero` 即基线） |
| `FLASHVLA_OVERLAP_STEPS` | `1` | 论文的 `d`；`0`=同步，`1..n_action_steps`=异步 |
| `PI05_COMPILE` | 跟随 `d` | `torch.compile`；**异步不编译等于白开**（与 `eval.sh` 的 `COMPILE_MODEL := d>0` 同逻辑） |
| `PI05_FUSE` | 跟随 compile | qkv / gate-up 融合 |
| `FLASHVLA_N_ACTION_STEPS` | ckpt 的 `10` | 论文 LIBERO 用 5（≤ `chunk_size`） |
| `FLASHVLA_COLD_START` | `zero_delta` | delta-EEF 必须 `zero_delta` |
| `SKIP_STALE_ACTIONS` | `0` | RTC 重对齐（实验中保持关闭） |

### 5.3 异步与冷启动的关键事实

- `d=1`：chunk 还剩 1 步时启动下一次推理 → `manager.act()` 的慢步恰在 **8 / 18 / 28 / 38 / 48 / 58**（可用来验证重叠生效）。
- 冷启动：FlashVLA 前 `num_buffer_slots-1 = 3` 次推理返回 hold 动作（真实动作全 0）→ **前 30 步机器人不动**。
- 预算自动补偿：`budget = max_steps + cold_steps`，被 `env._max_episode_steps`(300) 截断，冷启动不再吃任务步数。
- warmup 必须穿过冷启动（`manager.warmup()`），否则 CUDA graph 捕获会掉在第一个任务第 30 步附近。

---

## 6. 实测数据（L40S / GPU 2 / FlashVLA + compile + fuse + d=1）

| 指标 | 数值 | 备注 |
| :--- | :--- | :--- |
| 启动总耗时 | **211.4 s** | 加载 148.9 s + compile 预热 59.1 s（manifest 记 145.6 / 223） |
| 冷启动 | 前 **30 步动作全 0** | 3 次 hold 推理 |
| 单步预处理（含 tokenizer） | **p50 1.9 ms** | 首次 270 ms |
| `manager.act()` | **p50 0.0 ms** | 每 10 步一次 launch（编译后单次 ~50.7 ms） |
| `env.step` | **p50 21.5 ms** | EGL 离屏，2 视角 256×256 |
| **单步合计** | **p50 23.6 ms** | 论文 LIBERO Time/Step 22.1 ms |
| 端到端 70 步 rollout | **5.9 s** | 含 task 9 的 env 重建 ~1.5 s |
| 不编译对照 | 推理 p50 80.0 ms | 单步 ≈30 ms 且重叠失效 |
| 显存峰值 | ~8.3 GiB | 本机仅 GPU 2 有余量 |

两套 ckpt 的配置差异（`config.json`）：`pi05` → `chunk/n_action_steps=50/50`、无 buffer、无冷启动、`compile=True`；`pi05-flashvla` → `10/10`、`num_buffer_slots=4`、`cold_start_mode=zero_delta`、`compile=False`。**两者共用同一套 state 归一化统计**，所以四元数修复对二者都适用。

---

## 7. 当前状态与下一步

**已验证**
- FlashVLA 分支：真机端到端（加载 → warmup → 冷启动 → 稳态 → 早停/预算）跑通。
- 两套 ckpt 的分支 dispatch / cfg 覆盖 / 处理器链：桩验证通过。
- `py_compile` 通过；IDE 报错均为 Pylance 解释器版本噪声（改动前就存在）。

**未做 / 待决**
1. **Web 服务化**：`minimax_agent_vla_demo.md` §9 有完整设计（三线程 + REST/WS + JPEG 帧 + SSH 隧道 + 骨架），但**没有落地代码**；`fastapi`、`websockets` 均未安装（`uvicorn/starlette/aiohttp` 已在）。
2. **`run_robot.py` 未同步**：仍是旧四元数实现、只支持 π0.5。
3. **π0.5 分支未做完整加载跑测**（只桩验证），需要时可补。
4. **MCP 视觉校验仍在链路里**：现在 `execute_task` 已返回真值 `success`，建议默认走 `is_success`，MCP 降级为可选。
5. **安全**：`.env` / README 中为真实 API Key，建议轮换并改模板变量。
6. **改动未提交**（`git status` 未暂存）。

**建议顺序**：先补 `run_robot.py` 一致性 → 再做 Web 骨架（P0：MJPEG 推流 + `/api/direct`）→ 再接 Agent 事件流。

---

## 8. 复现命令

```bash
conda activate /home/zjr/StreamingVLA/miniconda_ygc/envs/flashvla
cd /home/zjr/StreamingVLA/MiniMax-Agent-VLA-Demo
export CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl; unset DISPLAY

# 默认：FlashVLA + 异步 d=1 + compile + fuse（首启约 3.5 min）
python agent_mode.py

# 基线对照
PI05_MODEL_PATH=/home/zjr/StreamingVLA/flashvla/ckpt/pi05_libero python agent_mode.py

# 快速调试（跳过编译预热，异步收益归零）
PI05_COMPILE=0 python agent_mode.py

# 其他实验旋钮
FLASHVLA_OVERLAP_STEPS=2 SKIP_STALE_ACTIONS=1 python agent_mode.py
FLASHVLA_N_ACTION_STEPS=5 python agent_mode.py
```

---

## 9. 相关文件

| 文件 | 说明 |
| :--- | :--- |
| `MiniMax-Agent-VLA-Demo/agent_mode.py` | 主入口，本次全部改动都在这里（未提交） |
| `flashvla/docs/minimax_agent_vla_demo.md` | demo 详解 + 通信方案 + §7 模型切换与异步加速 + §8 问题清单（含修复状态） |
| `flashvla/docs/vla_interface_manifest.md` | FlashVLA 观测/动作/延迟接口契约（golden sample 与三条断言） |
| `flashvla/async_manager.py` | `AsyncStreamingActionManager`（`act/pop_cached_action/needs_observation/warmup/reset`） |
| `flashvla/policies/pi05/modeling_pi05_flashvla.py` | `PI05FlashVLAPolicy`（`select_action` 内置冷启动、`predict_action_chunk` 返回 `None`） |
| `flashvla/ckpt/{pi05_libero,flashvla-pi05-libero}` | 两套 8 文件 ckpt |
| `MiniMax-Agent-VLA-Demo/lerobot_back/` | ⚠️ 无效残骸，不要加进 `PYTHONPATH` |
