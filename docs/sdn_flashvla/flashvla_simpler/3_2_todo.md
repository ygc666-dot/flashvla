# P2 数据工程：准备清单与执行方案（TODO）

> 定位：`3.flashvla_simpler.md` §4「P2 数据工程」的**执行文档**。
> 上游：方案 §4（要做什么）、`3_1.md`（P1 实测交付了什么）。本文回答：**现在能开始吗、口径定了没有、
> 每一步的产出与验收、哪些坑已经踩过**。
>
> 代码引用格式 `起始行:结束行:文件路径`，路径相对 `flashvla/` 仓库根。
> 标注约定：**［已核实］**= 本次实测 / 实读代码；**［待核实］**= 需 P2 落地后一测即知。

---

## 0. TL;DR

| 项 | 状态 |
| :--- | :--- |
| **能否开始** | **能**。P1 输入已冻结（2300 集 / 4.0 GB），P2 全程不占 GPU；**4 个口径已拍板（§3）** |
| **唯一的新代码** | 1 个转换器 `sim_eval/simpler/convert_simpler_to_lerobot.py`（~170 行，模板 `convert_robotwin_to_lerobot.py:1-135`） |
| **交付物** | 18 个 LeRobot 根：`{fractal,bridge} × {success,failure} × 任务`，共 **202,275 帧 / ≈1.8 GB** |
| **新的未知数** | 已全部在 §2 实测排除（LeRobot 写盘、AV1 编码、stats、离线、`finalize`） |
| **工作量** | 编码 2.8 h（后台）+ 工程 1.5 天；比「只转成功」多 1.8 h 编码、多 1.2 GB |
| **必须并行起的一件事** | G3 的 π0\* 基线（§4 T0，3 h，占 GPU4，不占 P2 的人手） |

---

## 1. 输入清单（P1 已交付，视为冻结）

### 1.1 数据位置

```
third_party/SDN/eval_logs/p1_dump/vanilla/seeds=<1000..1499>/<task>/
├── 000_episodes.json / 000_episode_stats.json / 000_success_<rate>.log
└── dump/episode_XXXXX.{npz, mp4}      # 每个 episode 一对
```

| 组 | 任务数 | 集数 | 帧数 | fps | 图像 | 指令 |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| **fractal**（google_robot） | 5 | 1550 | 209,200 | 3 | 512×640 `overhead_camera` | 1–26 条/任务 |
| **bridge**（widowx） | 4 | 750 | 54,000 | 5 | 480×640 `3rd_view_camera` | 1 条/任务 |
| 合计 | 9 | 2300 | 263,200 | — | — | — |

### 1.2 npz 字段（P2 的输入契约，详见 `3_1.md` §1.2）

`proprio[T,8]`(f32) / `action[T,7]`(f64) / `success[T]`(bool) / `instruction_id[T]`(i32) /
`instruction_vocab[V]`(U192) / `meta`(json，含 `steps` / `first_success` / `fps` / `image_shape`)，
图像在同名 `.mp4`（h264，`n_frames == steps` 已逐集核对）。

> 转换器只需要这 5 个字段 + mp4；`raw_chunk` / `query_*` 是给 P3 调试与 SDN 复用的，本阶段不用。

---

## 2. 已核实的环境事实（省掉重复调研）

全部在 `flashvla` 环境（py3.12 / lerobot 0.5.1）用**真实 apple 数据**跑过一遍［已核实］：

| 项 | 结论 | 依据 |
| :--- | :--- | :--- |
| 写盘 API | `LeRobotDataset.create(root=..., use_videos=True)` → `add_frame` → `save_episode` → **`finalize()`** | 最小闭环 + 真实 398 帧转换均通过 |
| **`finalize()` 必须调用** | 不调 → `meta/info.json` 不完整 → 重新打开时**回落到 Hub 检查** → 离线环境 `ConnectError` | 实测：不调 finalize 必然失败；调了则正常 |
| 离线开关 | 必须 `HF_HUB_OFFLINE=1`；`create` 里的 hub 检查在纯断网时会 `httpx.ConnectError` | 实测 |
| 视频编码 | 默认 `vcodec='libsvtav1'`（SVT-AV1 preset 10, CRF 30, yuv420p）**开箱可用**，不需要系统 ffmpeg；`vcodec='h264'` 亦可 | 实测两者均成功 |
| 编码体积 | AV1 **8.8 KB/帧**（源 h264 是 18 KB/帧） | apple 2 集 398 帧实测 |
| 转换吞吐 | **50 ms/帧**（单进程，含 mp4 解码 + AV1 编码） | 398 帧 / 20.1 s |
| 读回 | `observation.images.*` → `(3,H,W) float32 ∈ [0,1]`；`observation.state (8,)`、`action (7,)` | 实测 |
| stats | `meta.stats` 自动含 `mean/std/q01/q10/q50/q90/q99`，逐 feature；多根时 `aggregate_stats` 池化 | 实测 + `flashvla_dataset.py:230` |
| `len(dataset)` | `== num_frames`（不是 /16）；`action_delta_indices = range(N·C)` | `lerobot_dataset.py` + `configuration_pi05.py:248-249` |

---

## 3. 决策记录（已拍板，2026-11-21）

| # | 决策项 | **决定** | 直接后果 |
| :- | :--- | :--- | :--- |
| 1 | 截断 `k` | **k = 2** | 成功轨迹保留 `first_success + 3` 帧 |
| 2 | 失败轨迹 | **成功、失败都转** | 集数 2300（全量）；帧数 68,627 → **202,275**；编码 1.0 h → **2.8 h**；体积 0.6 GB → **≈1.8 GB** |
| 3 | 图像键名 | **组内统一**：全部用 `observation.images.top` | 跨组同名 → P3 的 fractal/bridge 两份 config 只差 `roots`/`fps`；原相机名写入 provenance 不丢 |
| 4 | vcodec | **默认 AV1**（`libsvtav1`） | 8.8 KB/帧；若 G3 掉点再退 `h264`/PNG |

### 3.1 由决策 2 推导出的实现口径（需你确认一次，改一行即可）

决策只说了「都转」，没说**怎么组织**。默认按 **outcome 分根**，理由与备选：

| 方案 | 效果 | 采不采 |
| :--- | :--- | :--- |
| **A. `{group}/<task>/{success,failure}/` 两个根** | G3 主实验只挂 `success` 根（与方案 §5.1 口径一致）；§5.3 的增广实验再挂 `failure` 根 | **采用** |
| B. 一个根混装 | 失败帧占 **66%**（133,648 / 202,275），会静默改写 G3 基线；且与方案 §5.3「混 10–20%」不符 | 不采（`--outcome-split false` 可切） |

**为什么不能只是「都塞进去」**：失败轨迹的价值是「漂移后能救回来」，方案要求混 10–20%；
若混装成 66%，G3 的「不掉点」判据就变成了「混合配比」的判据，基线不可比。分根后
「挂哪些根」= 训练时的一个配置项，**一次转换支持两种用法，不用重转**。

### 3.2 决策 3 的语义说明

「组内统一」在本方案里落成 **全部数据集同名 `observation.images.top`**（既满足组内，也顺带跨组一致）：
P3 只需维护一个 config 模板，fractal/bridge 只改 `roots` 与 `fps`；将来若把两族 pool 起来
（需重采样 fps）也不需要改 feature 名。
源相机名（`overhead_camera` / `3rd_view_camera`）写进 `meta/p1_provenance.json`，信息不丢。
> 若你原意是「按机器人族命名」（`overhead_camera`/`3rd_view_camera`），改 §5.3 的 `cam` 一行即可，
> 其余设计不变。

---

## 4. 任务分解

### T0（并行，先起）：G3 的 π0\* 基线（3 h，GPU4）

**为什么现在做**：P3 的判据是「M=1 在留出 seeds 上 macro ≥ π0\* − 2 pp」，而这个 π0\* 数**还不存在**
（采数占用了 1000-1499，见 `3_1.md` §6.4）。它不阻塞 P2，但阻塞 P3，且 P2 不占卡，正好并行。

```bash
cd third_party/SDN
for t in google_robot_close_drawer google_robot_move_near google_robot_open_drawer \
         google_robot_pick_coke_can google_robot_place_apple_in_closed_top_drawer \
         widowx_carrot_on_plate widowx_put_eggplant_in_basket widowx_spoon_on_towel widowx_stack_cube; do
  python -m policy.pizero.evaluate --method vanilla --task "$t" \
      --n-trajs 50 --start-seed 2000 --result-root ./eval_logs/p3_baseline --no-video
done
```

- **产出**：`eval_logs/p3_baseline/vanilla/seeds=2000-2049/<task>/000_success_<rate>.log`（9 任务 macro）
- **验收**：9 个任务都有 `000_success_*.log`，macro 落在 55–70%（与 `3_1.md` §4.3 的 67.2% 同量级）
- **成本**：44,600 env step × 0.24 s ≈ 3 h（含 9 次冷启动）
- **注意**：seeds 2000-2049 与训练集 1000-1499、R1 评测集 0-49 **三方不相交**

### T1：转换器骨架（0.5 天）

| 项 | 内容 |
| :--- | :--- |
| 输入 | `--dump-root <p1_dump 的 vanilla 目录>`（递归发现所有 `seeds=*/<task>/dump/`）、`--tasks ...`、`--out-root`、`--group`、`--truncate-k 2`、`--outcome-split/--no-outcome-split`、`--vcodec`、`--limit-per-outcome`（调试用，`--limit` 为别名） |
| 产出 | `sim_eval/simpler/convert_simpler_to_lerobot.py` |
| 验收 | `--limit-per-outcome 1` 转 1 成 1 败：两个 outcome 目录都生成、`finalize` 后都能重新打开、`num_frames == 各自 keep 的帧数和` |
| 依据 | §5 的设计 + 模板 `sim_eval/robotwin/robotwin_pipeline/scripts/convert_robotwin_to_lerobot.py:59-114` |

> **实现口径（P2 回写，2026-11-21）**：P1 把 9 个任务分散在 4 个 `seeds=*` 目录
> （`1000-1149` / `1000-1199` / `1000-1299` / `1000-1499`，各任务的 seed 区间不相交），
> 所以 `--dump-root` 必须指向 **`vanilla/`** 并递归发现；只给某个 `seeds=*` 会
> `FileNotFoundError`（不会产出错数据）。同一 episode 经两条根被看到时会**直接报错**，
> 不会静默双写。`--limit-per-outcome N` = 每个 (task, outcome) 最多 N 集。
>
> 另外：本 IDE 终端里跑会先打印一条 safe-delete 告警 —— 该钩子把删除转成「移入
> `<mount>/.Trash-$uid`」，`/data` 无可写回收站会让 `save_episode` 收尾时崩溃。此时加
> `CODEBUDDY_SAFE_DELETE_ENABLED=0`；普通终端/CI 无此钩子，可忽略。

### T2：转 fractal 组（2.4 h 计算，后台）

```bash
export HF_HUB_OFFLINE=1 CODEBUDDY_SAFE_DELETE_ENABLED=0   # 本 IDE 终端必需
python sim_eval/simpler/convert_simpler_to_lerobot.py \
  --dump-root third_party/SDN/eval_logs/p1_dump/vanilla \
  --out-root /data/zjr/simpler_lerobot/fractal --group fractal --truncate-k 2
# 想并行就一个任务一个进程（自动跳过 bridge 任务）：
# --tasks <one-task> --image-writer-threads 4
```

- **产出**：`/data/zjr/simpler_lerobot/fractal/<task>/{success,failure}/`（5×2 = 10 个 LeRobot 根）
  注：`/data` 根不可写（root:root 755），实际落在 `zjr` 子目录；两份 P3 config 里已写死该路径
- **验收**：`success` 根合计 **50,843 帧**、`failure` 根合计 **119,248 帧**，`fps==3`，
  `observation.images.top` 的 shape 全为 `(512,640,3)`
- **耗时**：170,091 帧 × 50 ms ≈ 2.4 h（单进程；用 `--tasks` 拆 5 个进程可压到 ~40 min）

### T3：转 bridge 组（0.5 h）

同上，`--group bridge --out-root /data/zjr/simpler_lerobot/bridge`，4 个 widowx 任务 → 8 个根，
`success` **17,784** / `failure` **14,400** 帧，`fps=5`，shape `(480,640,3)`，≈27 min。
（bridge 的 4 个任务分散在 `seeds=1000-1149` / `1000-1299` 两个目录，更说明 `--dump-root` 要给 `vanilla/`。）

### T4：G2 验收（0.5 天）

见 §6 清单。核心四条：`[64,7]` 形状、窗口不跨 episode、stats 对拍 <1e-3、**两个 outcome 根的帧数与任务串**。

### T5：抽检（0.5 h）

随机 10 条 episode（成功/失败各 5）导出 gif + action 曲线。因为 P2 用的是**已经编码的 LeRobot 视频**回读，
这一步同时验证「二次编码（h264→AV1 CRF30）没有把画面/对齐搞坏」。

### T6：交付 P3 接口（0.2 天）

按 §7 给两份 config（fractal / bridge）+ 一条实测：`len(dataset)`、`steps/epoch`。

---

## 5. 转换器设计（最关键的一节）

### 5.1 字段映射

| npz / mp4 | LeRobot feature | 处理 |
| :--- | :--- | :--- |
| `proprio[t]` (8) | `observation.state` f32 (8,) | 直传 raw；`names.motors = [x,y,z,qw,qx,qy,qz,gripper]`（见下方说明） |
| `action[t]` (7) f64 | `action` f32 (7,) | **转 f32**（f64 只为 P1 的逐位断言）；语义 `[Δxyz, rot_axangle, gripper]` |
| `mp4[t]` | `observation.images.top` video (H,W,3) | `imageio.get_reader` → uint8 HWC 直传（**统一键名**，决策 3） |
| `instruction_vocab[instruction_id[t]]` | `task` (str) | **必须逐帧取**，见 §5.2① |
| `meta` | 不写入 | 另存 `<root>/meta/p1_provenance.json`：episode→seed、`source_camera`、`outcome`、`truncate_k` |

**关于 `observation.state` 要不要跟 π0\* 一致**：π0\* 在两侧各有一套改造 —— fractal 把 quat 从 wxyz 滚成
xyzw、gripper 取「闭合度」（`simpler_adapter.py:216-237`），bridge 把四元数换成相对 top-down 的欧拉角、
proprio 变 **7 维**（`simpler_adapter.py:179-192`）。P2 **一律沿用 P1 dump 的 raw 8 维**：

1. 学生是 FlashVLA，不是 π0\*，**不需要与 π0\* 的输入编码对齐**（跨架构这一层在 G0 已判定走数据自举）；
2. 任何确定性重参数化对网络都可学，换来的是「数据自描述、可复现、少一处静默错位」；
3. 代价：bridge 的 state 维度是 8 而非 π0\* 的 7 —— 只影响「拿 π0\* 权重直接对比输入」这类场景，P3 不用。

### 5.2 三条必须做对的地方

**① `task` 要逐帧取，不能取首帧。**［已核实］`place_apple_in_closed_top_drawer` 的 **500/500 集**
都含 2 条指令（"open top drawer" → "place apple into top drawer"，平均切换 1.0 次）；
其余 8 个任务每集内恒定。取首帧会让 apple 后半段的语言标签全错 —— 训练不报错，
只表现为「语言条件学不到东西」。

**② 不要照抄 robotwin 的 `state=ja[:-1] / action=ja[1:]` 平移**（`convert_robotwin_to_lerobot.py:48-53`）。
robotwin 的 h5 里 action 是「下一步的关节目标」，必须平移；我们的 `action[t]` 已经是**第 t 帧观测对应的
执行动作**（P1 逐位断言保证），再平移会整体错位一拍。

**③ `finalize()` 一定要调**（§2）。转换器跑完 2.4 h 才在 T4 读回时炸掉，是最贵的一类坑。

### 5.3 features 常量（一份即可，因为图像键名已统一）

```python
STATE_NAMES = ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]
ACTION_NAMES = ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]

GROUPS = {   # 差异只剩 fps / shape / robot_type
    "fractal": dict(fps=3, shape=(512, 640, 3), robot_type="google_robot", source_camera="overhead_camera"),
    "bridge":  dict(fps=5, shape=(480, 640, 3), robot_type="widowx",       source_camera="3rd_view_camera"),
}
IMAGE_KEY = "observation.images.top"          # 决策 3：组内 + 跨组统一

def build_features(group):
    g = GROUPS[group]
    return {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": {"motors": STATE_NAMES}},
        "action":            {"dtype": "float32", "shape": (7,), "names": {"motors": ACTION_NAMES}},
        IMAGE_KEY:           {"dtype": "video", "shape": g["shape"], "names": ["height", "width", "rgb"]},
    }
```

> **同组内 feature keys 必须逐字一致**：`_MultiDatasetMeta.__init__` 用 `set(features.keys())` 判定，
> 不一致直接 `raise`（`flashvla/datasets/flashvla_dataset.py:152-157`）——而且是在**训练启动时**才炸。
> 统一键名 + 常量 dict 让这条从「易错」变成「不可能错」。

### 5.4 主循环骨架（含 outcome 分根）

```python
import imageio, json, numpy as np, torch
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

for task in tasks:
    writers = {}
    for outcome in ("success", "failure"):
        root = out_root / task / outcome
        writers[outcome] = LeRobotDataset.create(
            repo_id=f"simpler/{group}/{task}/{outcome}", fps=g["fps"], features=build_features(group),
            root=root, robot_type=g["robot_type"], use_videos=True, vcodec=vcodec)

    for npz_path in sorted((dump_root / task / "dump").glob("episode_*.npz")):
        z = np.load(npz_path, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        success = bool(m["success"])
        if success:
            keep = min(m["steps"], int(m["first_success"]) + 1 + truncate_k)   # 决策 1：k=2
        else:
            keep = m["steps"]                                                   # 决策 2：失败全留
        outcome = "success" if success else "failure"
        ds = writers[outcome]
        frames = np.stack(list(imageio.get_reader(str(npz_path).replace(".npz", ".mp4"))))  # [T,H,W,3]
        for t in range(keep):
            ds.add_frame({
                "observation.state": torch.from_numpy(z["proprio"][t].astype(np.float32)),
                "action":            torch.from_numpy(z["action"][t].astype(np.float32)),
                IMAGE_KEY:           torch.from_numpy(frames[t]),
                "task": str(z["instruction_vocab"][z["instruction_id"][t]]),   # 逐帧（§5.2①）
            })
        ds.save_episode()
        provenance.append(dict(episode=m["episode"], outcome=outcome, kept=keep, steps=m["steps"],
                               first_success=m["first_success"], source_camera=g["source_camera"]))
    for ds in writers.values():
        ds.finalize()                      # §5.2③：必须
        write_json(ds.root / "meta" / "p1_provenance.json", provenance)
```

> `--no-outcome-split` 时 `outcome` 恒为 `"all"`，其余逻辑不变（备选方案 B）。

### 5.5 输出目录（可直接被 P3 的 `root` + `config_subdirs` 自动发现）

```
/data/zjr/simpler_lerobot/              # 48 T 的盘（/data 根不可写），别放 3.2 T 的系统盘
├── fractal/<task>/success/             # 5 个根，fps=3，合计 50,843 帧
├── fractal/<task>/failure/             # 5 个根，fps=3，合计 119,248 帧
├── bridge/<task>/success/              # 4 个根，fps=5，合计 17,784 帧
└── bridge/<task>/failure/              # 4 个根，fps=5，合计 14,400 帧
```

**为什么这个层级有用**：`make_robotwin_multitask_flashvla_dataset` 支持
`root` + `config_subdirs` 的自动发现模式 —— 它扫描 `<root>/<task>/<subdir>/meta/info.json`
（`flashvla_dataset.py:259-283`）。所以 P3 只要写 `root=/data/zjr/simpler_lerobot/fractal` 和
`config_subdirs: [success]`（或 `[success, failure]`），**不用硬编码 18 条路径**。

**逐任务帧数**（转换后的验收基线，已实测推导）：

| 任务 | 集 | 成功 | 失败 | success 根 | failure 根 | 合计 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `place_apple_in_closed_top_drawer` | 500 | 117 | 383 | 20,273 | 76,600 | 96,873 |
| `open_drawer` | 500 | 223 | 277 | 14,793 | 32,132 | 46,925 |
| `close_drawer` | 200 | 159 | 41 | 8,010 | 4,756 | 12,766 |
| `move_near` | 200 | 144 | 56 | 3,984 | 4,480 | 8,464 |
| `pick_coke_can` | 150 | 134 | 16 | 3,783 | 1,280 | 5,063 |
| **fractal 小计** | 1550 | 777 | 773 | **50,843** | **119,248** | **170,091** |
| `carrot_on_plate` | 300 | 171 | 129 | 6,007 | 7,740 | 13,747 |
| `put_eggplant_in_basket` | 150 | 131 | 19 | 4,560 | 2,280 | 6,840 |
| `spoon_on_towel` | 150 | 119 | 31 | 3,564 | 1,860 | 5,424 |
| `stack_cube` | 150 | 108 | 42 | 3,653 | 2,520 | 6,173 |
| **bridge 小计** | 750 | 529 | 221 | **17,784** | **14,400** | **32,184** |
| **合计** | **2300** | **1306** | **994** | **68,627** | **133,648** | **202,275** |

体积 ≈ **1.8 GB**（202,275 × 8.8 KB；失败帧更静止，实际会更小），源 dump 4.0 GB 保留不动。

---

## 6. G2 验收清单（可测，逐条给命令）

```python
export HF_HUB_OFFLINE=1
cfg.dataset.repo_id = "simpler/fractal"
cfg.robotwin_multitask.enable = True
cfg.robotwin_multitask.root = "/data/zjr/simpler_lerobot/fractal"
cfg.robotwin_multitask.config_subdirs = ["success"]        # 混装实验改 ["success","failure"]
ds = make_robotwin_multitask_flashvla_dataset(cfg)
```

| # | 检查 | 判据 | 怎么测 |
| :- | :--- | :--- | :--- |
| G2-0 | 根的数量与帧数 | 4 组共 18 个根；`success`/`failure` 帧数与 §5.5 表逐行相符 | 遍历 `meta/info.json` 的 `total_frames` 求和 |
| G2-1 | 单样本形状 | `action.shape == [N·N·C, 7] = [64, 7]`；`action_is_pad.shape == [64]`；`observation.state == [8]`；图像 `[3, H, W]` | `ds[0]` 打印 shape |
| G2-2 | pad 结构 | 第 k 块（k=1..N）的前 `k·C` 行**原样继承**基础样本的 `action_is_pad[:k·C]`，其余 `H_cfg−k·C` 行恒为 True；即真实槽位随 k 递增、pad 只在尾部（`flashvla_dataset.py:112-119`） | 逐块检查 |
| G2-3 | 窗口不跨 episode | episode 末尾样本的 `action_is_pad` 在边界处为 True（不串到下一条） | 取各 episode 最后一个 idx |
| G2-4 | stats 复算 | 用 `ds.meta.stats` 归一化后 `action`/`state` 均值 ≈0、方差 ≈1，相对误差 <1e-3；**按实际参与训练的根池化后的统计量** | 全量扫一遍 |
| G2-5 | 动作连续性 | `|a_{t+1} − a_t|` 的 p99 无孤立尖峰（截断边界除外） | 逐 episode 算 diff |
| G2-6 | 图像-动作对齐 | 成功/失败各 5 条导出 gif + action 曲线，人眼确认 | T5 |
| G2-7 | 语言标签 | `task` 集合：fractal 的 apple 必须**同时出现**两条指令；`close_drawer`/`open_drawer` 各 3 条；`move_near` 26 条 | `meta.tasks` 计数 |
| G2-8 | 跨子集一致性 | 同组所有叶子的 `features.keys()` 与 `fps` 完全相同 | 构造成功即通过（`_MultiDatasetMeta` 强制） |

> ⚠ **方案 §4.3 的判据要修正**：原文写「`action.shape == [N·H, 7] = [16, 7]`」。
> 按 `flashvla_dataset.py:103-122`，`__getitem__` 返回的是 **N 个 config 块拼接**，每块长
> `H_cfg = N·C = 16`，所以实际是 **`[4×16, 7] = [64, 7]`**（`action_is_pad` 同理 `[64]`）。
> `[16, 7]` 是**单块**长度（也是 LeRobot 原始样本的 `action` 长度）。验收按 `[64, 7]`，
> 否则会把正确实现判成不合格。

---

## 7. 与 P3 的接口（P2 交付时一并给出）

```yaml
# train/configs/pi05/simpler/pi05_flashvla_fractal.yaml
policy:
  type: pi05-flashvla
  pretrained_path: lerobot/pi05_base
  num_buffer_slots: 4
  chunk_size: 4
  n_action_steps: 4          # 必须 == chunk_size
  state_cond: true
  empty_cameras: 2           # 单相机补位（configuration_pi05.py:111,166）
  normalization_mapping:
    VISUAL: IDENTITY
    STATE: MEAN_STD
    ACTION: MEAN_STD
dataset:
  repo_id: simpler/fractal    # 只是 roots/repo_id 的前缀
  video_backend: torchcodec
robotwin_multitask:           # 键名沿用现有实现，行为是「显式多根拼接 / 自动发现」
  enable: true
  root: /data/zjr/simpler_lerobot/fractal
  config_subdirs: [success]   # G3 主实验：只用成功轨迹
  # 混装实验（§5.3 抗漂移增广）改成 [success, failure]
batch_size: 8
grad_accum_steps: 4
steps: 12000
fsdp:
  enable: true                # 模板 train/configs/pi05/robotwin/pi05_flashvla.yaml:40-47
```

**epoch 长度（bs 32，`len(dataset) == num_frames`）**：

| 训练集 | 帧数 | step/epoch | `steps: 12000` ≈ |
| :--- | ---: | ---: | ---: |
| fractal `[success]`（G3 主实验） | 50,843 | 1,589 | **7.6 epoch** |
| fractal `[success, failure]`（增广） | 170,091 | 5,316 | 2.3 epoch |
| bridge `[success]` | 17,784 | 556 | 21.6 epoch |
| bridge `[success, failure]` | 32,184 | 1,006 | 11.9 epoch |

→ **bridge 组的 12000 step 明显过长**，建议 bridge 从 4,000 step 起（≈7 epoch）；fractal 主实验 12,000 合理。
［待核实：`len(dataset)` 落地后一测即知，误差只在 episode 尾部 pad 的几帧］

---

## 8. 时间表（半天粒度）

| 时段 | 内容 | 并行 |
| :--- | :--- | :--- |
| D1 上午 | T0 起基线（后台 3 h）；T1 写转换器 | GPU4 跑基线 |
| D1 下午 | T1 调通 `--limit-per-outcome 1`（1 成 1 败）；T2 起 fractal 转换（后台 2.4 h） | 转换进程 |
| D1 晚 | G2-0/1/2/3；T3 转 bridge（27 min） | — |
| D2 上午 | G2-4/5/6/7 全量验收 + 抽检；T6 交付 config | — |
| D2 下午 | 写 `3_3.md`（P2 实现 + G2 报告），进 P3 准备（4 卡排期） | — |

**串行依赖**：T1 → T2/T3 → T4 → T5 → T6。T0 与全部无依赖。

---

## 9. 风险与止损

| # | 风险 | 概率 | 止损 |
| :- | :--- | :--- | :--- |
| 1 | 二次编码（h264→AV1 CRF30）损害精细任务 | 中 | T5 对比 dump 原始帧与回读帧的 PSNR；<30 dB 改 `vcodec='h264'` 或 `use_videos=False` |
| 2 | 失败轨迹混进 G3 主实验（配比失控） | **中→高（已用分根规避）** | 主实验只挂 `success`；混装只在增广臂显式开 `[success, failure]` |
| 3 | 组内 feature keys 不一致 → 训练启动才 raise | 低（已统一键名 + 常量 dict） | §5.3 |
| 4 | `task` 取错（apple 的 2 条指令） | **高（不写就会错）** | 逐帧取；G2-7 专门验 |
| 5 | 忘记 `finalize()` | 中 | §5.4 骨架已含；T4 读回会暴露 |
| 6 | 磁盘写满系统盘 | 低 | 一律写 `/data`（48 T） |
| 7 | 18 个根的 config 写错 | 低 | 用 `root` + `config_subdirs` 自动发现（§5.5），不硬编码路径 |
| 8 | 失败帧导致 action 分布长尾（flailing） | 中 | G2-4 按根分别看 stats；P3 的归一化用池化 stats 自动吸收 |
| 9 | fps 混用（3 vs 5） | 低 | 两个顶层根，`_MultiDatasetMeta` 会 raise |

---

## 10. 决策落点一览（供回溯）

| 决策 | 落在本文哪一节 | 落在代码哪里 |
| :--- | :--- | :--- |
| k=2 | §3、§5.4 `keep = first_success + 1 + k` | `--truncate-k 2` |
| 成功失败都转 | §3.1、§5.4 outcome 分流、§5.5 帧数表 | `--outcome-split`（默认开） |
| 命名组内统一 | §3.2、§5.3 `IMAGE_KEY` | features 常量 dict |
| 默认 AV1 | §3 | `LeRobotDataset.create(vcodec="libsvtav1")` |

---

## 11. 附：本文所有实测数字的来源

| 数字 | 来源 |
| :--- | :--- |
| 2300 集 / 263,200 帧 / 1306 成功 / 4.0 GB | `3_1.md` §4.1、§4.5 + `scripts/p1/check_raw.py --json` |
| 每任务 usable@k=2（apple 20,273 等） | `3_1.md` §4.1 表 |
| 每任务失败帧数（= 失败集数 × horizon） | 同上（horizon 与成功率均来自 P1 统计） |
| AV1 8.8 KB/帧、50 ms/帧、读回 dtype、stats 键 | 本次真实数据转换（apple 2 集 398 帧，`/tmp/lr_real`） |
| `finalize` / 离线两个坑 | 本次最小闭环实测 |
| `task` 逐帧必要性（apple 500/500 集双指令） | 本次全量 npz 扫描 |
| `[64,7]` 形状 | `flashvla/datasets/flashvla_dataset.py:103-122` 实读 |
| `root` + `config_subdirs` 自动发现 | `flashvla/datasets/flashvla_dataset.py:259-283` 实读 |
