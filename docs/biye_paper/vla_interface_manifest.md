# FlashVLA 接口清单（开工前信息答卷）

> 本文回答「搭 VLA + 仿真 + 前端」之前必须先弄清楚的那一组问题：**输入/输出的契约是什么**。
> 所有结论都来自本仓库代码、本地 checkpoint 元数据，以及在本机（8×L40S）上**实际跑出来的数字**；
> 凡是没有证据的，一律标注 `未验证` 或 `不知道`，不做推测性填空。
>
> 证据基准路径：`/home/zjr/StreamingVLA/flashvla`（下称「仓库」）。
> 实测日期：2026-11-05。环境：`/home/zjr/StreamingVLA/miniconda_ygc/envs/flashvla`。

---

## 0. 一页答复：先回答你最后那个问题

> 「这台远端的 VLA，是你自己能重启和改代码的，还是别人的服务你只能调用？」

**是你自己能重启、能改代码的本地进程，而且目前根本没有服务。**

- 仓库里**没有任何 HTTP/gRPC 推理服务**：全仓库搜不到 `fastapi / flask / uvicorn 应用 / grpc`。
- 唯一存在的「服务器」是 RoboTwin 的那套**裸 TCP + 长度前缀 + JSON(base64 numpy)** 私有协议，且只服务 RoboTwin：
  `sim_eval/robotwin/overlay/script/policy_model_server.py`（`ModelServer`，默认端口 `9999`，无鉴权、无 TLS）。
- 所以你要写的是**进程内适配器（in-process adapter）+ 你自己定义的 HTTP/WS 层**，而不是「HTTP 客户端」。
  即：`sim（同机进程） → 进程内函数调用 → in-proc policy，再由我加一层 HTTP/WS 给浏览器`。
- 你给的 `http://10.0.0.12:8000/predict` **不存在**：本机 IP 是 `172.30.1.26`，`8000` 端口空闲、无监听。

**结论：接口形态可以不猜，直接读代码；我按下面的 manifest 做配置驱动，不会把常量写死在适配层。**

---

## 1. 本机身份与权限边界（「远端权限边界」答案）

| 项目 | 事实 | 证据 / 命令 |
| :--- | :--- | :--- |
| 主机 | 就是当前机器，你有 root 级常规权限（能写仓库、能起长驻进程） | 工作区 `/home/zjr/StreamingVLA` |
| 内网 IP | `172.30.1.26`（另有 docker 网桥 `172.17~172.22.*`） | `hostname -I` |
| SSH | `22` 端口已监听，可隧道 | `ss -ltnp` |
| 占用端口 | `7890/7891`(clash) `8053` `8123` `9090` `5432` `50012/50016` `38012/38013` | `ss -ltnp` |
| 可用端口 | **`8000` 空闲**；`9999`（RoboTwin 服务端默认）也空闲 | 同上 |
| 出网 | 走代理 `http_proxy=127.0.0.1:7890`；HF 有本地缓存，`~/.cache/huggingface` 3.3 GB | `env \| grep proxy` |
| GPU | **8×NVIDIA L40S，每卡 46068 MiB，驱动 580.105.08 / CUDA 13.0** | `nvidia-smi` |
| **显存余量（硬约束）** | 0/1/3/4/5/6/7 号卡被 `VLLM::Worker` **各占 40~43 GB**，仅剩 ~3 GB；**2 号卡被 Isaac Sim 占 5.6 GB，可用 ~40 GB** | `nvidia-smi` |
| 是否与别人共享 | **是**。2 号卡上有别人长驻的 `/isaac-sim/.../open_usd.py`（自 10 月 23 日起，CPU 371%，GPU util 71%） | `ps aux \| grep isaac` |
| 能装包吗 | 能。`conda`/`pip` 可用，`flashvla` 是 `pip install -e` 装的；缺 `fastapi`，有 `uvicorn` | `importlib.util.find_spec` |
| 能跑长驻进程吗 | 能，仓库文档里已有 `nohup ... &` 的用法先例 | `docs/howtodo.md` §5.2 |
| 前端工具链 | **没有 `node` / `npm`** → 前端不能上打包构建，走原生 HTML+JS 静态页 | `node -v` 无输出 |
| 仿真可用性 | LIBERO/robosuite/MuJoCo **已装且实测可无头渲染**；SAPIEN（RoboTwin）**未装** | 见 §7.4 |

> 只有 2 号卡能跑 VLA（8.3 GiB 显存足够），但它同时被 Isaac Sim 占用，**p99 会被别人拉高**。
> 这是本项目唯一的稳定性风险来源，建议：显存独占预留不现实，改为「接受 p99 抖动 + 前端用插值平滑」，或申请一张空卡。

---

## 2. Manifest（可直接做配置驱动）

```yaml
# 由 ckpt/flashvla-pi05-libero/ 的 config.json + policy_preprocessor.json
# + modeling_pi05_flashvla.py 推导，全部字段均有代码证据。

policy:
  family: pi05-flashvla            # config.type
  base: lerobot/pi05_base + FlashVLA streaming decoding
  params: 4.14e9                   # 实测 813 tensors / 4_143_404_816
  dtype: bfloat16                  # 98.5% BF16，其余 F32(1.2e8)
  weights_on_disk: 8.5 GB          # model.safetensors
  peak_vram: 8.3 GiB               # 实测 torch.cuda.max_memory_allocated
  local_ckpt: /home/zjr/StreamingVLA/flashvla/ckpt/flashvla-pi05-libero
  # 仓库内另有一个对照基线 ckpt/pi05_libero（同域，无 FlashVLA 流式）

transport:
  existing: none                   # 仓库无 HTTP 服务
  reference_only: tcp+json+base64  # RoboTwin ModelServer, port 9999, 无鉴权
  recommended: 进程内调用 + 自建 FastAPI/WebSocket(8000)
  auth: none
  network: 本机内网 172.30.1.26，本机进程间可直接 unix socket/127.0.0.1

observation:
  cameras:
    - key: observation.images.image        # LIBERO agentview（第三人称固定机位）
      role: third_person
      source_field: robot_state/pixels["image"]   # libero env 原始名 agentview_image
    - key: observation.images.image2       # robot0_eye_in_hand（腕部）
      role: wrist
      source_field: pixels["image2"]              # 原始名 robot0_eye_in_hand_image
  key_order_matters: true                  # image_features 的插入顺序即图像 token 顺序
  input_size: [256, 256]                   # ckpt input_features = 3x256x256
  model_resize: [224, 224]                 # 内部 resize_with_pad(224,224,pad_value=0)
  color: rgb                               # uint8 HWC, 直接 /255（无 BGR/无通道交换）
  normalization: "x/255 -> resize_with_pad(224) -> x*2-1  => [-1,1]"
  imagenet_mean_std: NOT_USED              # stats 里有 0.485/0.456/0.406，但 VISUAL=IDENTITY
  flip: rot180                             # 送模型前必须上下+左右翻转（LIBERO 相机惯例）
  fov: 未在 ckpt 中声明                     # 由仿真相机决定，见 §8「不知道」
  extrinsics: 未在 ckpt 中声明              # 训练域固定，换机位即域偏移

  proprio:
    key: observation.state
    dim: 8
    order: [eef_pos(3, m), eef_axis_angle(3, rad), gripper_qpos(2, m)]
    frame: world/base
    normalization: MEAN_STD
    mean: [-0.04652, 0.03441, 0.76455, 2.97221, -0.22047, -0.12558, 0.02691, -0.02719]
    std:  [ 0.10494, 0.15177, 0.37852, 0.34427,  0.90695,  0.32539, 0.01418,  0.01406]
    gripper_open_qpos: [-0.0392, +0.0392]   # 实测开合两端 ≈ ±0.04 m

  history:
    frames_per_view: 1
    n_obs_steps: 1
    include_prev_action: false             # 不需要客户端回传上一动作
    stateful_server_side: true             # 模型自带 4 槽 × 10 步 动作 buffer
    reset_required: true                   # 每个 episode 必须 reset_model()

  instruction:
    mode: text
    preprocessing_by_server: true          # 客户端不要自己拼 prompt
    template: "Task: {task}, State: {8 个 256 分箱整数};\nAction: "
    text_cleanup: "strip; '_'->' '; '\\n'->' '"
    lowercase: false                       # 代码不做小写化（LIBERO 语料本身小写）
    tokenizer: google/paligemma-3b-pt-224  # gated 仓库，需 hf auth login
    max_length: 200
    padding: right / max_length

action:
  key: action
  dim: 7
  order: [dx, dy, dz, dax, day, daz, gripper]
  type: delta_eef                          # 只给增量，不含绝对位姿
  rotation_repr: axis_angle                # 3 维，不是 6D、不是四元数
  units: normalized[-1,1] -> 物理       # 见下方 scale
  scale_pos: 0.05 m / step                 # robosuite OSC output_max
  scale_rot: 0.50 rad / step
  frame: world (= robot base)
  compose: left_mul                        # R_goal = R_delta @ R_cur
  position_update: "p_goal = p_cur + delta"
  ref_pose: current_eef                    # 每控制步从当前 EE 重新取参考
  gripper:
    dim: 1
    range: [-1, 1]
    semantics: "+1 张开 / -1 闭合（robosuite 约定）"
    threshold: 0                          # 二值判定用 0；训练数据是双峰的 -1 / +0.92
  chunk:
    chunk_size: 10                         # 每次推理返回的步数
    num_buffer_slots: 4                    # 动作 buffer 槽数
    total_horizon: 40                      # 4*10
    n_action_steps: 10                     # ckpt 默认；论文 LIBERO 用 5
    cold_start_chunks: 3                   # 前 3 次推理返回 None(=hold 动作)
    inference_period_steps: n_action_steps # 稳态每 n_action_steps 个 env step 推理一次
    dt: 0.1                                # 训练数据 10 Hz（见 §4.2）
  dequant:
    mode: MEAN_STD
    formula: "real = norm * std + mean"
    mean: [ 0.06278,  0.08684, -0.09037,  0.00054,  0.00564, -0.00523, -0.04964]
    std:  [ 0.33552,  0.37845,  0.44473,  0.03924,  0.06339,  0.07797,  0.99877]
    note: "由 postprocessor 自动完成；若绕过 policy 自己发请求，必须手写这一步"

latency_ms:            # 本机 L40S 实测，batch=1，2 视角 256x256，不含客户端预处理
  no_compile: { p50: 80.0, p99: 82.3 }
  compile_fuse: { p50: 50.7, p99: 54.1 }
  preprocess_cpu: { p50: 8.2, p99: 150 }
  compile_warmup_s: 223
  model_load_s: 145.6

train_domain:
  benchmark: LIBERO (spatial/object/goal/10)
  dataset: HuggingfaceVLA/libero（273,465 帧 / 1,692 episodes / 40 tasks）
  robot: Franka Panda + robosuite OSC_POSE（7-DoF，非 UR5、非 ALOHA）
  cameras_n: 2
  freq_hz: 10
  sim_only: true                 # 训练即仿真，不存在真机→仿真衰减问题
```

---

## 3. 必须项逐条详解

### 3.1 推理入口：没有服务，只有一份可执行策略

| 选项 | 现状 | 你要写的东西 |
| :--- | :--- | :--- |
| HTTP/gRPC 地址 | **无** | — |
| 鉴权 | 无 | — |
| SSH/内网/公网 | 本机内网 `172.30.1.26`，同机进程直接 `127.0.0.1` | — |
| 现成进程 | 无 | 自己起长驻进程（`uvicorn` 已在，`fastapi` 需装） |
| 唯一参考实现 | RoboTwin 的 `ModelServer`（TCP 9999，JSON+base64，`TCP_NODELAY`，首次调用 timeout 设 300 s 因为要等 `torch.compile`） | 可照抄它的序列化思路，但**协议要换成 HTTP/WS 给浏览器** |

模型调用面只有 4 个方法（`sim_eval/robotwin/policy/pi05_flashvla/pi05_flashvla_model.py`）：

```122:184:sim_eval/robotwin/policy/pi05_flashvla/pi05_flashvla_model.py
    def call(self, func_name: str, obs: Any = None): ...
    def reset_model(self, *_args, **_kwargs) -> None: ...
    def get_action(self, obs_dict: dict) -> np.ndarray: ...
    def needs_image_obs(self) -> bool: ...
    def get_cached_action(self) -> np.ndarray: ...
```

`needs_image_obs()` / `get_cached_action()` 是给你省渲染和带宽的：**不需要推理的那几步，连图都不用渲染**。
在网页场景里这条尤其值钱——前端要的帧可以只在「真要推理时」才全分辨率渲染。

### 3.2 观测约定

- **相机路数 = 2**，名称与顺序**必须**是 `observation.images.image`（第三人称）→ `observation.images.image2`（腕部）。
  顺序敏感：`prepare_images` 按 `config.image_features` 的插入顺序堆叠图像 token。
- **输入尺寸 256×256**（ckpt `input_features` 是 `3×256×256`）；模型内部 `resize_with_pad(224, 224, pad_value=0)` 再 `*2-1`。
  所以送 256×256 最省事（等比缩放到 224 无变形）；送非方形图会被 letterbox 补 0。
- **颜色空间 RGB，uint8，HWC**，客户端只做 `/255` 转 float32 CHW，**不要做 ImageNet 归一化**：

```1295:1298:flashvla/policies/pi05/modeling_pi05_flashvla.py
        for key in present_img_keys:
            img = batch[key]
            img = resize_with_pad(img, *self.config.image_resolution, pad_value=0)
            img = img * 2.0 - 1.0
```

  > 陷阱：ckpt 的 stats 文件里确实存着 `0.485/0.456/0.406`（ImageNet mean/std），但 `normalization_mapping.VISUAL = IDENTITY`，
  > 这两个数**完全不参与计算**。照着你原来模板里的 `norm: imagenet` 实现会得到「机器人动了但方向幅度不对」的经典 bug。
- **必须 180° 翻转（rot180）**：lerobot 的 LIBERO 观测处理器把 H、W 两个维度都 flip 一次。
  这是最容易漏的一条，漏了画面上下颠倒但仍能推理，成功率会莫名其妙地掉。

```57:61:/home/zjr/StreamingVLA/miniconda_ygc/envs/flashvla/lib/python3.12/site-packages/lerobot/processor/env_processor.py
            if key.startswith(f"{OBS_IMAGES}."):
                img = processed_obs[key]
                # Flip both H and W
                img = torch.flip(img, dims=[2, 3])
```

- **本体状态 = 8 维**：`[eef_pos(3, m), eef_axis_angle(3, rad), gripper_qpos(2, m)]`，world/base 坐标系。
  注意**不是** `joint_pos + gripper`，也**不是** UR5-like：

```68:75:/home/zjr/StreamingVLA/miniconda_ygc/envs/flashvla/lib/python3.12/site-packages/lerobot/processor/env_processor.py
            eef_pos = robot_state["eef"]["pos"]  # (B, 3,)
            eef_quat = robot_state["eef"]["quat"]  # (B, 4,)
            gripper_qpos = robot_state["gripper"]["qpos"]  # (B, 2,)
            eef_axisangle = self._quat2axisangle(eef_quat)  # (B, 3)
            state = torch.cat((eef_pos, eef_axisangle, gripper_qpos), dim=-1)
```

  四元数是 `(x, y, z, w)` → axis-angle；夹爪 qpos 是**两个指关节的绝对位移**（张开 ≈ ±0.039，闭合 ≈ 0），不是开合度 0~1。
- **指令预处理：做，但由服务端做，客户端别碰**。模板把 **8 维状态（normalized 后按 [-1,1] 256 分箱）** 塞进 prompt：

```83:90:flashvla/policies/pi05/processor.py
            for i, task in enumerate(tasks):
                cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
                per_sample = discretized_states[i]
                ...
                full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
```

  → 含义：**prompt 里已经隐含了状态**。如果你手动拼 prompt，就会得到 `Task: ..., State: ..., State: ...` 的双份 state，属于静默错误。
  也意味着「改一句指令」会连带改变 token 长度分布——不要在同一条 rollout 里随意改指令。
- **历史帧 = 1**（`n_obs_steps=1`、`num_frames_per_view=1`）。时序信息全部由服务端的动作 buffer 承担，客户端不回传历史，也不回传上一帧动作。

### 3.3 动作约定：这一节最值钱

| 维度 | 结论 | 证据 |
| :--- | :--- | :--- |
| 维度/顺序 | 7 = `[dx,dy,dz,dax,day,daz,gripper]` | ckpt `output_features.action.shape=[7]` |
| 旋转表示 | **axis-angle（轴角）3 维**，不是 6D、不是四元数 | robosuite `axisangle2quat(delta[3:6])` |
| 单位 | 模型输出是**归一化 `[-1,1]`**；物理缩放 = 位置 `×0.05 m`、旋转 `×0.5 rad` | `osc.py: input_max=1, output_max=(0.05,0.05,0.05,0.5,0.5,0.5)` |
| 坐标系 | **world / base**（不是 tool / eef） | `set_goal` 用 `self.ee_pos`、`self.ee_ori_mat` |
| 合成方式 | 平动：`p_goal = p_cur + Δ`；转动：**左乘** `R_goal = R_delta @ R_cur` | `control_utils.py: goal_orientation = np.dot(rotation_mat_error, current_orientation)` |
| 参考位姿 | **当前 EE**（每控制步重新取），不是「上一帧目标」 | `self.update()` 后立即用 `self.ee_pos` |
| 夹爪 | 1 维 `[-1,1]`，`+1` 开 `-1` 合，判定阈值 `0`；训练数据双峰（q01=-1, q99=+0.92） | stats 的 `action.q01/q99` |
| action chunk | `chunk_size=10`，`num_buffer_slots=4`，总视界 40 步 | ckpt `config.json` |
| 执行长度 | `n_action_steps`（ckpt 10，论文 LIBERO 用 5） | `deploy_policy.yml` / `eval.sh` |
| dt | 0.1 s（训练数据 10 Hz） | stats 里 `timestamp.max=50.4` / `frame_index.max=504` |
| 反归一化 | 存在，`MEAN_STD`：`real = norm*std + mean`，均值方差见 manifest | `policy_postprocessor.json` |

**最容易踩的三个坑（会表现出「动了，但方向或幅度不对」）：**

1. **把 7 维当成「米 + 弧度」直接下发**。实际是归一化值，物理位移 = `a * 0.05` m。
   一步 `a=1.0` 才 5 cm；直接把 `0.25` 当 0.25 m 用，机器人会飞出去。
2. **旋转右乘**。你原模板里写的是 `compose: right_mul`，而这里是**左乘（world 轴）**。
   对接近单位姿态的末端差别很小，姿态一大就 90° 级偏差——正是你担心的那种「要跑通才发现差个 90°」。
3. **把 `frame` 当成 tool**。这里是 **world**。腕部视角看着「往前伸」，在 world 里可能是「往下伸」。

**冷启动语义（前端必须体现）**：前 `num_buffer_slots-1 = 3` 次 `predict_action_chunk` 返回 `None`，管理层用 hold 动作补；
LIBERO 场景 ckpt 的 `cold_start_mode = zero_delta`，即**前 3×n_action_steps = 30 步机器人完全不动**。
（RoboTwin 那套是绝对关节位置，必须用 `current_state`，否则机械臂会冲向零位——文档里明确写了会 crash。）

```9:12:sim_eval/robotwin/policy/pi05_flashvla/deploy_policy.yml
# Cold-start action source for the first (num_buffer_slots - 1) chunks:
#   - "current_state": hold current qpos (required for RoboTwin)
#   - "zero_delta":    return zero delta (breaks absolute-qpos envs)
cold_start_mode: current_state
```

### 3.4 golden sample

**先说一个坏消息**：FlashVLA 的推理入口**没有暴露 `noise` / `seed`**（`predict_action_chunk(batch)` 内部自己采高斯噪声做 flow matching），
所以**逐位可复现的「固定输入 → 固定输出」不存在**。别按「存一个数值数组，单测里 assert 相等」的思路做。

正确做法（我按这个思路给的样本）：**用「量级 + 各维分布 + 物理一致性」做断言**。
下面是本机实跑的一组真实数值，可用于校验你的适配层单位与坐标系是否写反。

固定输入（全部可复现）：

```python
# 两路图像 256x256 uint8：cam0 纯红 R=200，cam1 纯绿 G=120（其余通道 0）
# instruction: "pick up the black bowl and place it on the plate"
# state(8) = [-0.05, 0.02, 0.78,  2.9, -0.2, -0.12,  0.026, -0.027]
```

实跑输出（`ckpt/flashvla-pi05-libero`，bf16，无 compile，第 24 次推理，即稳态）：

```
chunk shape = (1, 10, 7)

normalized[0,0] = [-0.08986, -0.16252, 0.77250, -0.04440, 0.17667, -0.10951, 0.63299]
real[0,0]       = [ 0.03263,  0.02533, 0.25318, -0.00120, 0.01684, -0.01377, 0.58257]

real[0..3, :4]（前 4 步的 xyz + dax）:
  [0.03263, 0.02533, 0.25318, -0.00120]
  [0.04518, 0.03198, 0.29346,  0.00076]
  [0.06674, 0.04300, 0.30620,  0.00158]
  [0.08766, 0.05214, 0.26907,  0.00196]
```

怎么用它做单测（三条断言，任一条失败就是适配层错了）：

1. **单位**：位置三维的量级应落在 `±0.3`（归一化域）。若你的实现产出 `0.0026` 这种量级，说明你多除了 100，或者你以为输出是米。
2. **物理换算**：`real[:3] * 0.05` 应得到 `1.6 mm/步` 量级（`0.0326*0.05≈1.6mm`，`0.253*0.05≈12.7mm`）。10 Hz 下 ≈ 1.3 cm/s，是合理的机械臂速度。
3. **坐标系**（关键）：状态里 `eef_z = 0.78`（桌面高度上方），动作 `dz = +0.25` 且为正 → **world 系下向上**。
   如果你的仿真里机器人朝下扎，说明你把 z 反了或当成 tool 系了。

复现脚本（本机实测通过，直接拷走）：

```python
import numpy as np, torch
from lerobot.configs.policies import PreTrainedConfig
from flashvla.policies.factory import make_pre_post_processors
from flashvla.policies.pi05.configuration_pi05 import PI05FlashVLAConfig  # noqa: 注册 pi05-flashvla
from flashvla.policies.pi05.modeling_pi05_flashvla import PI05FlashVLAPolicy

CKPT = "/home/zjr/StreamingVLA/flashvla/ckpt/flashvla-pi05-libero"
cfg = PreTrainedConfig.from_pretrained(CKPT)
cfg.device, cfg.cold_start_mode, cfg.compile_model = "cuda", "zero_delta", False
policy = PI05FlashVLAPolicy.from_pretrained(CKPT, config=cfg).to("cuda").eval()
pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CKPT,
                                    preprocessor_overrides={"device_processor": {"device": "cuda"}})

H = W = 256
a = np.zeros((H, W, 3), np.uint8); a[..., 0] = 200     # cam0
b = np.zeros((H, W, 3), np.uint8); b[..., 1] = 120     # cam1
s = np.array([-0.05, 0.02, 0.78, 2.9, -0.2, -0.12, 0.026, -0.027], np.float32)

cvt = lambda x: torch.from_numpy(np.ascontiguousarray(x)).float().div(255).permute(2, 0, 1)[None].cuda()
batch = pre({"observation.images.image": cvt(a), "observation.images.image2": cvt(b),
             "observation.state": torch.from_numpy(s)[None].cuda(),
             "task": ["pick up the black bowl and place it on the plate"]})

with torch.inference_mode():
    for _ in range(4):                     # 前 3 次返回 None（冷启动）
        out = policy.predict_action_chunk(batch)
print(out.shape)                            # (1, 10, 7)
print(post(out)[0, 0])                      # real[0,0]，量级参考上文
```

> 复现耗时参考：进程内首次加载 **145.6 s**（含 tokenizer + 8.5 GB 权重），之后每步 80 ms。

---

## 4. 「应该拿到」那一组

### 4.1 模型种类与参数量、dtype

- 家族：**π0.5（pi05-flashvla）**，不是 OpenVLA / RDT / ACT / Diffusion Policy。
- 结构：PaliGemma 视觉语言骨干（Gemma-2B LM + SigLIP 视觉）+ **Gemma-300M action expert**，flow matching 解码。
- 实测权重：**813 个张量 / 4,143,404,816 参数（4.14 B）**，其中 BF16 占 4.024 B、F32 占 0.12 B；文件 8.5 GB。
- 实测峰值显存 **8.29 GiB**（batch=1，2 视角）。

### 4.2 训练时的控制频率 = 10 Hz

证据：stats 里 `frame_index.max = 504` 与 `timestamp.max = 50.4` → slope 恰好 10。
所以 `dt = 0.1 s`、`chunk_size = 10` 恰好是「1 秒的动作」。
**但你原模板写的 30 Hz / 16 步 chunk 都不对**（16×20 是 RoboTwin 的设置，那个 ckpt 本地没有）。

注意一个容易搞混的事实：仿真器本身是 **20 Hz**（`control_timestep = 0.05`，robosuite OSC_POSE）。
lerobot 的 LIBERO 评测是「不限速地连续 step」（实测 26.5 ms/步 ≈ 37 Hz），
即**物理推进速度与你 step 的快慢绑定**。做网页演示时若想「看起来自然」，应按 10 Hz 节拍推进并做前端插值。

### 4.3 训练数据域：仿真，不是真机

- 数据集 `HuggingfaceVLA/libero`：273,465 帧 / 1,692 episodes / 40 tasks。
- 机器人：Franka Panda（7-DoF）+ robosuite `OSC_POSE`，`use_delta=True`。
- 相机 2 路，256×256，固定第三人称 + 腕部。
- **完全仿真域** → 不存在「真机模型搬到仿真」的性能衰减问题；反过来，**搬到真机/换机械臂/换相机位姿才会衰减**。
  这条很重要：**不要用 UR5 或 Isaac Sim 里的自定义机器人去接这个 ckpt**，动作语义和视觉域都会错。

### 4.4 实测延迟（本机 L40S，batch=1，2 视角，256×256）

| 配置 | p50 | p95 | p99 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| 不编译（`compile_model=false`） | **80.0 ms** | 82.4 ms | 82.3 ms | 24 次采样，稳态 |
| 编译 + fuse（`max-autotune`） | **50.7 ms** | 53.8 ms | 54.1 ms | 编译预热 **223 s** |
| 客户端预处理（CPU，tokenizer 等） | 8.2 ms | — | 150 ms | 首调用含缓存 miss |
| 进程内模型加载 | — | — | — | **145.6 s** |

对照组（论文，RTX 4090，2 views）：FlashVLA 26.7 ms、π0.5 45.8 ms。
L40S 没做过 CUDA-graph 级别的细致优化，实测比论文慢 1.9×，**以本机实测为准**。

> 结论：**一次推理 50~80 ms，远小于你预算里的「260/410 ms」**。
> 你的主延迟大头会变成「HTTP 往返 + 图像编码 + 前端渲染」，不是模型本身。

### 4.5 30 秒录制轨迹

**本地没有**：`HuggingFaceVLA/libero` 数据集没下载（`benchmarks/` 那套脚本会自动拉几十 GB）。
替代方案（二选一，都不需要下数据集）：

- 用 §3.4 的 golden sample + 仿真闭环自己录：`sim_eval/libero` 跑 1 个 episode，把 `(obs, action)` 按步 dump 成 npz 即可离线回放。
- 直接跑 `bash sim_eval/libero/eval.sh`，`EVAL_MAX_RENDERED>0` 时会把 mp4 写到 `$OUTPUT_ROOT/.../videos/`。

### 4.6 manifest 的交付方式

已在上文 §2 给出，字段名按你原模板的键名对齐，可直接当 YAML 配置读。

---

## 5. 与你给的 endpoint 模板逐字段 diff

| 你的字段 | 你给的值 | 实际情况 | 严重度 |
| :--- | :--- | :--- | :--- |
| `endpoint` | `http://10.0.0.12:8000/predict` | **不存在**。本机 `172.30.1.26`，无 HTTP 服务，需自建 | 阻断 |
| `image.keys` | `cam_high, cam_wrist` | `observation.images.image`、`observation.images.image2`，**顺序敏感** | 高 |
| `image.size` | `224` | 输入 **256**，内部再 resize 到 224 | 中 |
| `image.norm` | `imagenet` | **不是 ImageNet**：`/255 → resize → *2-1`（`[-1,1]`）；stats 里的 ImageNet 数不生效 | **高（静默错）** |
| `image.color` | `rgb` | ✅ 正确 | — |
| （缺） | — | **送模型前必须 rot180**，模板里没有这一项 | **高** |
| `proprio.fields/dim/order` | `joint_pos, gripper` / 8 / `ur5_like` | `eef_pos(3)+axis_angle(3)+gripper_qpos(2)`；Panda/LIBERO，不是 UR5 | **高** |
| `instruction.lowercase/prefix` | `true` / `""` | 不做小写化；**模板由服务端生成**（内含 8 维 state 的分箱串） | 中 |
| `history.frames` | `1` | ✅ 正确（`n_obs_steps=1`） | — |
| `history.include_prev_action` | `true` | 不需要；时序在服务端 4 槽 buffer 里 | 中 |
| （缺） | — | **必须每 episode `reset_model()`**，且调用必须**按步严格顺序** | **高** |
| `action.type / dim` | `delta_eef` / `7` | ✅ 正确 | — |
| `action.rot` | `6d` | **axis-angle（3 维）** | **高** |
| `action.frame` | `tool` | **world / base** | **高** |
| `action.compose` | `right_mul` | **left_mul**（`R_goal = R_delta @ R_cur`） | **高** |
| `action.ref` | `current` | ✅ 正确（当前 EE） | — |
| `action.units` | `[m, m, m, rad...]` | 输出是**归一化 `[-1,1]`**，物理量 = `×0.05 m` / `×0.5 rad` | **高** |
| `action.chunk` | `16` | `chunk_size=10`，`num_buffer_slots=4`，`n_action_steps` 10（论文 5） | 中 |
| `action.dt` | `0.1` | ✅ 正确（10 Hz） | — |
| `action.dequant` | `mean/std` | ✅ 存在，MEAN_STD，数值见 §2 | — |
| `latency_ms.p50/p99` | `260 / 410` | 实测 **80.0/82.3**（不编译）、**50.7/54.1**（编译） | 低（好消息） |
| `train_domain.robot` | `ur5e` | **Franka Panda（LIBERO）** | **高** |
| `train_domain.freq_hz` | `10` | ✅ 正确 | — |
| `train_domain.cameras` | `2` | ✅ 正确 | — |

**一句话总结 diff**：`type/dim/dt/cameras/freq` 这几项你猜对了；
**坐标系（tool→world）、旋转表示（6D→axis-angle）、合成（right→left）、单位（m→归一化）、proprio 组成（joint→eef 位姿）、图像归一化（imagenet→`[-1,1]`）这六项全错**，
而这六项正好是「机器人动了但方向幅度不对」的全部来源。

---

## 6. 你问「哪几项确实不知道」——这一节

| 项 | 状态 | 兜底方案 |
| :--- | :--- | :--- |
| 相机 **FOV** | **不知道**。ckpt 里没有记录，只有 256×256 分辨率 | 用 LIBERO/robosuite 默认相机即可（训练即该配置）。换 FOV = 域偏移，别换 |
| 相机**安装外参** | **不知道**。ckpt 不存外参 | 直接用 LIBERO 的 `agentview` / `robot0_eye_in_hand`，不要自定义机位 |
| 单次推理 **p99 的稳定性** | **半知**。实测 p99 54~82 ms，但 2 号卡共享给 Isaac Sim（71% util），长跑 p99 会被别人拉高 | 前端按 10 Hz 节拍插值 + 允许 1~2 帧抖动；或申请独占卡 |
| **逐位可复现**的 golden sample | **做不到**（推理未暴露 seed） | 用 §3.4 的量级断言；如需强复现，得给 `sample_actions` 加 `noise` 参数（改代码，可做） |
| 30 s 真轨迹 | **本地没有** | 自己跑 1 个 episode dump npz（§4.5） |
| 训练超参细节（50k steps / lr 1e-4） | 有 `train_config.json`，但不是部署必需 | 忽略 |
| RoboTwin 那套 ckpt（ALOHA/14 维） | **本地没有**，只有 LIBERO ckpt（`ckpt/` 下 2 个都是 LIBERO 域） | 若你要 ALOHA/双臂，需要另外下 `z-lab/flashvla-pi05-robotwin` 并装 SAPIEN |

---

## 7. 实测环境数据（决定骨架形状）

### 7.1 推理

见 §4.4。补充：`needs_image_obs()` 为 `False` 的步可以完全跳过渲染与传输，
一个 episode 里「需要渲染的步数 : 总步数」≈ `1 : n_action_steps`（n_action_steps=10 时约 10%）。

### 7.2 显存与并发

- 单实例 **8.29 GiB**。2 号卡余量 ~40 GB → 理论上可跑 4 个实例，但**同一张卡上多实例会互相抢 SM**，
  对本项目（单用户）**只跑 1 个实例**，用 batch=1。

### 7.3 端口与前端

- `fastapi` 未装（`pip install fastapi` 走代理即可），`uvicorn` 已有。
- **无 `node`/`npm`** → 前端只能是原生静态页（`index.html` + `app.js`），
  用 `WebSocket` 推 `obs 状态 + 渲染帧`，或干脆 MJPEG（`multipart/x-mixed-replace`）最省事。
- 浏览器要连 `172.30.1.26:8000`，确认没有防火墙拦截（本次未验证防火墙规则 → **未验证**）。

### 7.4 仿真侧实测（LIBERO 直接可跑）

在本机 `flashvla` 环境里实测（`MUJOCO_GL=egl`，无头）：

| 指标 | 实测值 |
| :--- | :--- |
| `LiberoEnv` 构造 + reset | **1449 ms** |
| `env.step(7 维动作)`（含 2×256×256 渲染） | **26.5 ms（p50 26.7）** |
| 控制频率 | 20 Hz（`control_timestep=0.05`，`model_timestep=0.002`） |
| 动作空间 | `Box(7, -1, 1)`，`OSC_POSE`，`use_delta=True` |
| episode 上限 | 280 步（`libero_spatial`） |
| 观测 | `pixels.image` / `pixels.image2` = 256×256 uint8 RGB；`robot_state.{eef,gripper,joints}` |
| 资源 | `~/.cache/libero/assets` 已就位（586 文件已下载），无需再拉 |
| RoboTwin / SAPIEN | **未安装**，不可用 |

> 单步预算核算：`sim 26.5 ms + 推理 50~80 ms`，若每 10 步推理一次，则平均每步 ≈ `26.5 + 8 = 35 ms` → 可稳定 10~20 Hz 出帧给前端。

---

## 8. 这些结论对骨架的影响（3 条）

1. **不要写 HTTP 客户端，写进程内适配器 + 自建服务端。** 模型在本机、可重启、可改代码；
   进程内调用省掉序列化（单帧 256×256×3 = 192 KB 裸数据，JSON+base64 后 ≈ 256 KB，每帧多 33% 且要过一遍 Python 的 CPU 编解码）。
   建议：仿真线程 → `queue` → 推理线程（持 policy 单例）→ WebSocket 广播给浏览器。

2. **适配层只需要 3 个函数，但要 3 个都写对：`build_obs()` / `postprocess_action()` / `reset()`。**
   `build_obs` 里那两行（rot180、`/255→*2-1`）和 `postprocess_action` 里的 `×0.05 / ×0.5` 就是全部数学。
   `reset()` 必须在每 episode 开始调用，否则 buffer 会带着上一个任务的 40 步动作尾巴。

3. **机器人选型不要自由发挥。** 这个 ckpt 是 LIBERO/Panda 域：7 维 delta-EEF + world 系 + 2 固定相机 + 10 Hz。
   换成 UR5 + tool 系 + 6D 旋转，不是「适配一下」的问题，是**换模型**的问题。
   本机现成可跑的仿真只有 LIBERO（SAPIEN/RoboTwin 未装、Isaac Sim 被占用且机器人模型不匹配）。

---

## 附：还没被验证、需要你拍板的两件事

1. **前面板展示什么**：(a) 只推 LIBERO 渲染帧（最稳，26.5 ms/帧，视觉与模型输入天然一致）；
   (b) 在 three.js 里用 `robot_state.joints.pos`(7) 驱动一个 Panda URDF 做真 3D（好看，但要引入 URDF + 运动学，工作量翻倍）。
   我建议先 (a)，把 (b) 作为第二阶段。
2. **是否需要多个 episode 并发**：单实例 8.3 GiB，2 号卡余量够 4 个，但共享 SM 会拉高 p99。
   若是单用户演示 → 1 实例；若是多用户 → 建议加请求队列而不是加实例。
