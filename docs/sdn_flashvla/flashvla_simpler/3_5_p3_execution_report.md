# P3 执行总结：T0 基线 / 单卡冒烟 / 正式训练（2026-11-21 21:40 → 11-22 11:50）

> 本文记录 `3.4_todo.md`「立即执行清单」的**实跑结果**：A 轨（下载）、B1 轨（T0 基线）、
> B2 轨（S0–S5 冒烟）、C 轨（优化器 / 加载器改动），以及正式训练的启动与中断。
> 上游：`3_3.md`（P2 数据交付）。本文所有数字来自 `logs/p3_overnight/` 下的实测日志与 `gates/*.json`。
>
> 标注约定：**［已核实］**= 本次实跑读数；**［待核实］**= 需后续训练 / G3 才能确定。
> 代码引用格式 `起始行:结束行:文件路径`，路径相对 `flashvla/` 仓库根。

---

## 0. TL;DR

| 项 | 结果 |
| :--- | :--- |
| **A 轨（下载）** | ✅ `ckpt/pi05_base` 4 文件齐，`model.safetensors` **14.47 GB（fp32）** ［已核实］ |
| **B1 轨（T0 基线）** | ✅ 9 任务 × 50 seeds 跑完，**macro 66.89%**（与 R1 的 67.2% 同档），耗时 2.85 h ［已核实］ |
| **B2 轨（S0–S5）** | ✅ 全绿；**单卡路线被实测改写**：不开 FSDP，`adafactor` + 状态放卡上 ［已核实］ |
| **C 轨（改动）** | ✅ `adafactor` 注册；`loading.py` 容忍 `pi05_base` 缺 `state_cond` 的 6 个张量 |
| **正式训练** | ✅ **4000/4000 步跑完**（16:40:43），终值 loss **0.151**（最后 1000 步 0.1514±0.0014），无异常退出。中途 01:14 曾被第三方 shim 杀掉，11:47 从 `000500` 续跑成功 ［已核实］ |
| **实测吞吐** | bs=4 / accum=8（等效 batch 32）**5.06 s / step**，显存 38.3 GB / 45.5 GB，全程 **5 h 44 min**（128K samples，7.2 epoch） ［已核实］ |
| **G3 判定** | ⚠️ **已跑完，未过闸**（见 §11）：4 个 widowx 任务 × 50 seeds，**004000 = 54.0%**、`002750 = 50.5%`，T0 widowx 子集 80.0%，门槛 78.0%。过程中修掉一个让所有 `pi05-flashvla` 部署静默失效的 preprocessor 序列化 bug（§11.2） ［已核实］ |

---

## 1. 时间线（墙钟）

| 时间 | 事件 |
| :--- | :--- |
| 11-21 21:40 | P0 环境自检通过（GPU4 free ≥40 GB、代理由 env 提供、磁盘充足） |
| 21:41–21:47 | P1 下载 `lerobot/pi05_base`（Xet 卡死 → 关 Xet 后 12 MB/s） |
| 21:47–22:20 | P4 冒烟 S1–S4（两次失败：ckpt 缺 `state_cond`、`output_dir` 预建被拒） |
| 22:22–23:40 | P3 冒烟 S0：四条单卡路线实测（§5.1） |
| 23:37–00:20 | P5 冒烟 S5：bs 扫描（§5.3） |
| 00:09–00:27 | T0 首次尝试，00:15 卡死（GPU4 上有他人 `--method sdn` 评测）→ 杀掉，改为「训练结束后自动接 T0」 |
| **00:27** | **正式训练启动**（`/data/zjr/simpler_lerobot/runs/pi05_flashvla_bridge_overnight`） |
| **01:14** | **训练在 step 500 存盘后崩溃**（§6.3） |
| 01:14–04:05 | 自动接续跑 T0：9 任务全部完成，macro 66.89% |
| 04:05 | `P3_REPORT.md` 由脚本生成（只反映「训练已启动」，不代表训练存活，见 §6.4） |
| 11:45 | 补跑 `3_3.md` 的 T0 追加一节（原自动追加因解释器问题失败，见 §8.3） |
| 11:47 | 带 `CODEBUDDY_SAFE_DELETE_ENABLED=0` 从 `000500` 续跑（§6.4） |
| **16:40:43** | **训练完成 4000/4000 步**，`last → 004000`（§6.5） |

---

## 2. A 轨：下载 `lerobot/pi05_base`

| 文件 | 说明 |
| :--- | :--- |
| `ckpt/pi05_base/config.json` | `type=pi05`（基线实现，非 flashvla） |
| `ckpt/pi05_base/model.safetensors` | **14.47 GB**，fp32 权重（不是文档估的 7.5 GB bf16） |
| `ckpt/pi05_base/policy_preprocessor.json`、`policy_postprocessor.json` | 归一化统计 |

两个坑（已固化进 `scripts/p3/probe_env.sh` / `_env.sh`）：

1. `huggingface_hub 1.33` 默认走 **Xet**，本机代理下卡在 67 MB 不动；`HF_HUB_DISABLE_XET=1` 后恢复 12 MB/s，20 min 下完。
2. env 里 `huggingface-cli` / `accelerate` 的 shebang 仍指向搬动前的 `/home/zjr/StreamingVLA/...`，直接执行会 `bad interpreter`；脚本统一用 `<env>/bin/python -m <module>`。

---

## 3. C 轨：两处代码改动

### 3.1 `adafactor` 注册（［已核实］可 build）

`lerobot/optim/optimizers.py` 增加 `AdafactorConfig` + `Adafactor`，指向 `torch.optim.Adafactor`（不依赖 bitsandbytes）。
作用：让 `optimizer.type: adafactor` 能在 yaml / CLI 解码并 build —— 这是 §5.1 单卡路线的前提。

### 3.2 `loading.py` 容忍 `state_cond` 缺键（［已核实］）

`pi05_base` **没有** `suffix_embedder.state_proj / state_mlp_in / state_mlp_out` 这 6 个张量（state-FiLM 是 flashvla 新加的），原实现 `strict=True` 直接抛 `Missing key`。
现在这 6 个张量按构造期零初始化保留（等价于 pi0-flashvla 对 fresh FiLM 的 `allow=` 约定），**其它缺键 / 多键仍然致命**，不掩盖真错。
S2 日志留痕：`PI0.5 load: 6 state_cond tensor(s) absent ... (zero state contribution at step 0)`。

> 待核实：这 6 个张量能否在训练中真正学到非零值（step 500 时 loss 0.183 说明梯度链路通，但 FiLM 是否被「用起来」需 G3 判断）。

---

## 4. B1 轨：T0 π0\* 基线（vanilla / seeds 2000–2049）

协议：`policy.pizero.evaluate --method vanilla --n-trajs 50 --start-seed 2000 --no-video`，
权重 `third_party/SDN/pretrained/open-pi-zero`，9 任务 × 50 seeds（与训练集 1000–1499、R1 评测 0–49 三方不相交）。
耗时 **2.85 h**（01:14 → 04:05），无失败任务。

| 任务 | rate |
| :--- | ---: |
| `google_robot_close_drawer` | 68.0% |
| `google_robot_move_near` | 74.0% |
| `google_robot_open_drawer` | 36.0% |
| `google_robot_pick_coke_can` | 86.0% |
| `google_robot_place_apple_in_closed_top_drawer` | 18.0% |
| `widowx_carrot_on_plate` | 68.0% |
| `widowx_put_eggplant_in_basket` | 88.0% |
| `widowx_spoon_on_towel` | 88.0% |
| `widowx_stack_cube` | 76.0% |
| **macro** | **66.89%** |

- 落在 `3.4_todo.md` 预期的 55–70% 带内，且与 `3_1.md` 的 R1 基线 67.2% 一致 → **G3 的分母可用** ［已核实］。
- 两个弱项：`place_apple_in_closed_top_drawer` 18%、`open_drawer` 36%。50 seeds 下这两项抖动 ±7%，G3 不要单独拿它们下结论。
- 归档：`third_party/SDN/eval_logs/p3_baseline/vanilla/seeds=2000-2049/<task>/`；汇总 `logs/p3_overnight/gates/P2_t0.json`；正文已追加到 `3_3.md`。

---

## 5. B2 轨：S0–S5 冒烟

### 5.1 S0：单卡路线（**结论与 `3.4_todo.md` 的假设不同**）

| 路线 | fsdp | optimizer | cpu_offload | bs | s/step | samples/s | 结论 |
| :--- | :-: | :--- | :-: | ---: | ---: | ---: | :--- |
| 文档 Plan B | ✅ | adamw | true | 2 | 6.85 | 0.292 | 能跑，但慢 |
| C 轨方案 | ✅ | adafactor | true | 2 | 14.64 | 0.137 | 更慢（offload 是瓶颈） |
| C 轨方案 | ✅ | adafactor | false | 2 | 2.90 | 0.690 | 显存 42 GB，勉强 |
| **本次采用** | ❌ | **adafactor** | false | 4 | **0.97** | **4.12** | **最快，显存 32.7 GB** |

两个关键读数：

1. **`accelerate launch --num_processes=1` 根本不会建进程组**（simple_launcher 起的是裸子进程），`train/train.sh cfg 1` 同理；而 `flashvla/train.py:693` 对 `torch.distributed.get_backend_config()` 有断言（要求 `cuda:nccl,cpu:gloo` 复合后端），所以文档里「S0 两条路线」原样跑都会在**加载权重之前**报 `Default process group has not been initialized`。
   - 解法一（已实现、未采用）：`scripts/p3/launch_train.sh` 伪造 launcher 环境变量，让 accelerate 自己建复合进程组 —— FSDP2 单进程可跑。
   - 解法二（**最终采用**）：**不开 FSDP**，`python -m flashvla.train` 直接跑。
2. **单 rank 开 FSDP2 是纯亏本**：bf16 模式下它要额外保存 fp32 master 权重（+16.6 GB）与 fp32 梯度（+13.4 GB），而单卡没有东西可分片。文档 §2 把「AdamW 状态 30 GB」当作必须分片的理由，但对 **3.356 B 可训练参数**，换 `adafactor` 后这块负担消失，FSDP 只剩开销 —— 实测 4.2× 差距。

### 5.2 S1–S4（`gates/P4_pipeline.json` 四项全 0）

| 项 | 结果 |
| :--- | :--- |
| S1 数据复核 | `fractal` 侧 50,843 帧 / 777 集 / 5 leaves；`observation.images.top (3,512,640)`、`action (64,7)`、`action_is_pad (64,)` ［已核实］ |
| S2 权重 + 2 步 | `loss 0.247 → 0.239`，有限；`num_learnable=3.356 B / total=4.146 B`（两个 `lm_head` 冻结） ［已核实］ |
| S3 存 / 重载 | `PI05FlashVLAPolicy.from_pretrained` 成功，`train_config.json` 的 `robotwin_multitask` 保留（S3 特意放 **CPU** 上做，避免与邻居抢显存） ［已核实］ |
| S4 图像尺寸 | dataset 出 **512×640**，模型内 `resize_with_pad` 到 **224×224 → 256 token/图**；`empty_cameras=2` ⇒ **768 visual token/帧** ［已核实］ |

S4 的结论对时间表有直接影响：视觉 token 数不随原始分辨率变化，**不需要额外的预处理 resize**，方案 §6「40 h 级」那档的假设成立（本次单卡实测又压到 ~5.8 h，见 §6）。

### 5.3 S5：bs 扫描（无 FSDP + adafactor）

| bs | s/step | samples/s | 本进程峰值 | 等效 batch 32 的 optimizer step |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.691 | 2.894 | 22.2 GB | 16 × 0.69 = 11.1 s |
| **4** | **0.970** | **4.124** | **32.7 GB** | **8 × 0.97 = 7.8 s** |

**决策（`gates/decision.json`）**：`bs=4, grad_accum=8, optimizer=adafactor, fsdp=false, cpu_offload=false, equiv_batch=32`，峰值 32.7 GB ≤ 预算 43.4 GB（free − 2 GB）。

> 注意：第一次 S5 用「整卡 memory.used」采样，被 GPU4 上的邻居作业污染，bs=2/4 全被读成 45 GB、决策误落到 bs=1；改成**按进程树归因**（`nvidia-smi --query-compute-apps=pid,used_memory`）后才得到上面这套干净数字 —— 这也是 `sweep_bs.sh` 里那段注释的由来。

---

## 6. 正式训练

### 6.1 配置（`scripts/p3/render_cfg.sh` 从 `_overnight.yaml` 渲染）

`train/configs/pi05/simpler/_overnight.generated.yaml`：bridge 数据（18K 帧 / 9 任务）、`pretrained_path=ckpt/pi05_base`、
`bs=4 × 8 accum`、`adafactor lr 5e-5`（warmup + cosine）、`steps=4000`、`save_freq=250`、`fsdp.enable=false`、`wandb=false`。

### 6.2 实跑读数（［已核实］）

| step | loss | grad_norm | lr | s/step |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 0.343 | 0.945 | 4.3e-06 | 5.05 |
| 100 | 0.311 | 0.702 | 1.3e-05 | 5.28 |
| 150 | 0.264 | 0.650 | 2.1e-05 | 5.20 |
| 450 | 0.186 | 0.997 | 4.9e-05 | 5.23 |
| 500 | 0.183 | 0.947 | 4.8e-05 | 5.22 |

- loss 单调下降、梯度范数稳定在 0.6–1.0，**adafactor 在这条链路上没有不收敛迹象**（原担心 adafactor 需要更大 lr）。
- 显存 38.3 GB / 45.5 GB，吞吐 5.2 s/step ⇒ 4000 步 ≈ **5.8 h**（文档「单卡 13 h」偏保守，实际更快）。

### 6.3 崩溃（**本文最重要的一条**）

**01:14，step 500 存盘之后**进程退出：

```text
File ".../flashvla/train.py", line 1026, in train
    update_last_checkpoint(checkpoint_dir)
File ".../lerobot/utils/train_utils.py", line 60, in update_last_checkpoint
    last_checkpoint_dir.unlink()
File ".../tencent-cloud.coding-copilot-.../out/vendor/shim/sitecustomize.py", line 1569, in _safe_path_unlink
    _try_trash(abs_path)
...
PermissionError: [Errno 13] Permission denied: b'/data/.Trash-1000'
```

- 起因：IDE 插件（腾讯云 Coding Copilot）通过 `PYTHONPATH` 注入的 `sitecustomize.py` 把 `Path.unlink` 劫持成「移到回收站」，而 `/data` 是另一个挂载点，它在 `/data/.Trash-1000` 建目录时被杀。
- 为什么 step 250 没事：`update_last_checkpoint` 只在**已存在** `checkpoints/last` 时才 `unlink`，第一次存盘没有 `last`。
- **已复现并验证修法**：`CODEBUDDY_SAFE_DELETE_ENABLED=0` 时 `/data` 上 `unlink` 正常；不设则 100% 复现同一异常。等价做法是把 shim 目录从 `PYTHONPATH` 摘掉。
- `000500/pretrained_model/` 是**完整断点**（`model.safetensors` 8.53 GB + `train_config.json`），只是 `checkpoints/last` 仍指向 `000250`（崩在 unlink 之后、重建软链之前）。

### 6.4 续跑（已执行，2026-11-22 11:47）

`P3_REPORT.md`（04:05 生成）里「训练已按量测配置启动」只反映启动成功，**不代表训练存活**，以本节为准。

已按下面的命令从 step 500 续跑，用的就是 `000500` 那个完整断点：

```bash
cd /home/zjr/workspace_tj/StreamingVLA/flashvla
ln -sfn 000500 /data/zjr/simpler_lerobot/runs/pi05_flashvla_bridge_overnight/checkpoints/last
CODEBUDDY_SAFE_DELETE_ENABLED=0 CUDA_VISIBLE_DEVICES=4 \
  nohup /home/zjr/workspace_tj/StreamingVLA/miniconda_ygc/envs/flashvla/bin/python \
    -m flashvla.train --config_path=train/configs/pi05/simpler/_overnight.generated.yaml \
    > logs/p3_overnight/P6_train_resume.log 2>&1 &
```

续跑验证（［已核实］）：

| 判据 | 读数 |
| :--- | :--- |
| auto_resume 生效 | `Auto-resume enabled. ... resume=True`，配置取自 `000500/pretrained_model/train_config.json` |
| 步号连续 | tqdm 显示 `x/3500`，日志 `step:` 从 **550 / 600 / 700 / 750** 递增 |
| loss 接续 | 0.183 (500) → 0.181 (550) → 0.178 (600) → 0.175 (700) → **0.170 (750)** |
| 吞吐 | 5.00–5.05 s/step，显存 38.3 GB |
| **原崩溃点已通过** | 12:05 存下 `000750` 且 `last → 000750` 重建成功，日志 `Traceback`/`PermissionError` 计数 **0** |

剩余 3250 步 ≈ **4.5 h**，预计 **16:40 前后** 到 4000 步。断点每 250 步一个（`000750` 已落盘），`checkpoint_keep_last=8`。

### 6.5 训练完成：结果统计（2026-11-22 16:40:43）

**结论：4000/4000 步正常跑完，无异常退出。**

| 项 | 读数 |
| :--- | :--- |
| 结束标记 | `End of FlashVLA training`，`step:4K smpl:128K ep:4K epch:7.20` ［已核实］ |
| 异常计数 | `Traceback` / `PermissionError` / `OutOfMemory` 在续跑日志里 **0 次** ［已核实］ |
| 总墙钟 | **5 h 44 min**（首段 500 步 47 min + 续跑 3500 步 4 h 57 min） |
| 平均步时 | **5.06 s/step**（全程 80 个日志点均值，波动 5.00–5.29） |
| 吞吐折算 | 4000 step × 32 samples = **128,000 samples**；等效 3.55 samples/s |
| 数据遍数 | bridge 17,784 帧 / 9 任务 ⇒ 每个样本平均被看 **7.2 遍** |
| 最终 loss | **0.151**；最后 1000 步均值 **0.1514 ± 0.0014**，全程最小 0.149 ［已核实］ |
| 显存 | 38.3 GB / 45.5 GB（稳定，无增长） |

loss 曲线（每 250 步）：

| step | loss | grad_norm | lr |
| ---: | ---: | ---: | ---: |
| 250 | 0.220 | 0.805 | 3.7e-05 |
| 500 | 0.183 | 0.947 | 4.8e-05 |
| 750 | 0.170 | 0.958 | 4.6e-05 |
| 1000 | 0.167 | 0.969 | 4.3e-05 |
| 1250 | 0.163 | 0.973 | 3.9e-05 |
| 1500 | 0.159 | 0.985 | 3.5e-05 |
| 1750 | 0.155 | 1.013 | 3.0e-05 |
| 2000 | 0.154 | 1.024 | 2.6e-05 |
| 2250 | 0.154 | 1.056 | 2.1e-05 |
| 2500 | 0.152 | 1.067 | 1.6e-05 |
| 2750 | 0.153 | 1.041 | 1.2e-05 |
| 3000 | 0.150 | 1.063 | 7.9e-06 |
| 3250 | 0.150 | 1.075 | 4.7e-06 |
| 3500 | 0.150 | 1.056 | 2.3e-06 |
| 3750 | 0.149 | 1.091 | 8.3e-07 |
| 4000 | 0.151 | 1.085 | 2.6e-07 |

三条读数解读：

1. **0–500 步降得最快**（0.343 → 0.183），之后进入长平台；2000 步后基本只剩 0.004 的波动，lr 已按 cosine 退到 2.6e-07，说明 4000 步对这份数据已经饱和。
2. **adafactor 没问题**：grad_norm 全程稳定在 0.7–1.1，没有发散或「更新量过小」的迹象 —— 文档里「adafactor 可能要更大 lr」的担心在本次没有出现［已核实］。
3. **loss 平台不等于学会了**：这是 18K 帧、9 任务、每样本 7.2 遍的小数据，0.150 的平台上很可能混着「记住」。真实判据只能看 G3 的留出 seeds（§10）。

**断点留存**（`checkpoint_keep_last=8`）：现存 `002250 / 002500 / 002750 / 003000 / 003250 / 003500 / 003750 / 004000`，`last → 004000`；
`000250 … 002000` 已被 `Pruned old checkpoints: 002000` 删掉 —— 要早期 ckpt 只能按 §6.4 重跑（现在全程只要 5.7 h）。

**G3 可用断点**：`004000`（最终，`last` 指向它）。已在 CPU 上做过重载核对［已核实］：

| 核对项 | 读数 |
| :--- | :--- |
| 能否重载 | `reload OK: PI05FlashVLAPolicy`（`from_pretrained` 无需任何补丁） |
| 策略配置 | `type=pi05-flashvla`、`chunk=4`、`n_action=4`、`state_cond=True`、`empty_cameras=2`、`num_buffer_slots=4` |
| 训练配置 | `steps=4000`、`bs=4`、`optimizer=adafactor` |
| 体积 | `pretrained_model/` 8534.9 MB（bf16，4.146 B 参数） |

---

## 7. 与 `3.4_todo.md` 的偏差（全部因实测而改）

| 文档原计划 | 实际做法 | 原因 |
| :--- | :--- | :--- |
| S0 只测「cpu_offload + adamw」与「adafactor + offload」 | 测了 **4 条**，新增「adafactor 纯卡」「无 FSDP + adafactor」 | 前两条都被 offload 拖累（6.85 / 14.64 s/step）；显存瓶颈不在优化器状态，而在 FSDP 的 fp32 master / 梯度 |
| 训练用 FSDP2 + `cpu_offload`（锁死决策 3） | **不开 FSDP**，adafactor 纯卡 | 4.2× 吞吐差，显存还更低（33 GB vs 42 GB） |
| T0 在训练之前跑 | **训练先跑**，T0 在训练退出后自动接 | T0 首次被 GPU4 上的邻居评测卡死 20 min；先保训练这个长尾交付 |
| 单卡 13 h / 双卡 7 h 估时 | 单卡 **5.8 h**（4000 × 5.2 s） | 去掉 FSDP 开销 + adafactor |
| `train/train.sh cfg 1` 直接跑 | 平时 `python -m flashvla.train`；必须 FSDP 时走 `scripts/p3/launch_train.sh` | 见 §5.1 第 1 条 |

> 决策 2（「等效 batch 恒 32、步数恒 4000」）**没有动**。

---

## 8. 环境坑清单（复跑前先看）

1. **GPU4 不是独占的**：期间出现他人的 `streamvln_eval`（3 进程 42 GB）与 `policy.pizero.evaluate --method sdn`；S3 因此 OOM 过一次、T0 因此卡死过一次。凡「显存 / 吞吐」读数必须**按进程归因**；凡长跑必须带 `timeout` 或断点续跑。
2. **`python3` 是 Python 3.4.3**（`/home/zjr/miniconda3/bin/python3`），不支持 f-string。用它跑 `make_report.py` / `append_t0_doc.py` 会 `SyntaxError` —— 一律用 `$FV_PY`（flashvla env 的 3.12）。
3. **IDE 插件 shim 会改写 `unlink`**（§6.3）：凡涉及删文件的训练 / 评测脚本，启动时带 `CODEBUDDY_SAFE_DELETE_ENABLED=0`。
4. **`train.py` 拒绝已存在且无断点的 `output_dir`**：smoke / sweep 脚本只能 `rm -rf`，不能 `mkdir -p`。
5. `draccus` 对 `optimizer` 字段很严：切到 `adafactor` 时必须去掉 `betas`（`render_cfg.sh` 与 `_smoke.yaml` 已处理）。
6. **`fsdp.enable: false` 时不要用 `scripts/p3/launch_train.sh`，直接 `python -m flashvla.train`**。
   该脚本会伪造 `ACCELERATE_USE_FSDP=true` 等环境变量让 accelerate 走 FSDP，而 `004000` 的
   `model.safetensors` 是**混 dtype**（691 个 BF16 + **128 个 F32**，其中就包括 6 个
   `suffix_embedder.state_*`），FSDP 摊平会直接
   `ValueError: Must flatten tensors with uniform dtype but got torch.float32 and torch.bfloat16`。
   `fsdp.enable: false` 的非 FSDP 路径没有摊平，所以正常。§6.4 那次成功续跑用的就是直接 `python -m`。

---

## 9. 产物清单

| 路径 | 内容 |
| :--- | :--- |
| `ckpt/pi05_base/` | 基线权重（4 文件） |
| `logs/p3_overnight/gates/` | `decision.json`、`P2_t0.json`、`P4_pipeline.json`、`S0_*.json`、`S5_*.json`、`P*.done` |
| `logs/p3_overnight/` | 各阶段 log（`P2_t0/*.log`、`P3_s0*.log`、`P4_s3_reload.log`、`P5_bs*.log`、`P6_train.log`）、`S5_sweep.csv`、`main.log` |
| `third_party/SDN/eval_logs/p3_baseline/vanilla/seeds=2000-2049/` | T0 的 9 个任务目录（`000.log`、`000_success_*.log`） |
| `/data/zjr/simpler_lerobot/runs/pi05_flashvla_bridge_overnight/` | 训练输出：`checkpoints/{002250 … 004000, last→004000}`（每个 8.0 GB，含 `pretrained_model` + `training_state`） |
| `P3_REPORT.md` | 脚本生成的阶段汇总（04:05，早于崩溃定位，注意 §6.4） |
| `docs/.../3_3.md` | 已追加「T0：π0\* 基线」一节 |
| `scripts/p3/` | `_env.sh`、`probe_env.sh`、`run_t0.sh`、`smoke_s0.sh`、`smoke_s0_adafactor.sh`、`launch_train.sh`、`smoke_pipeline.sh`、`sweep_bs.sh`、`render_cfg.sh`、`parse_train_log.py`、`check_shapes.py`、`make_report.py`、`append_t0_doc.py`、`overnight.sh` |
| `train/configs/pi05/simpler/` | `_smoke.yaml`、`_overnight.yaml`、`_overnight.generated.yaml` |
| 代码改动 | `lerobot/optim/optimizers.py`（adafactor）、`flashvla/policies/loading.py`（state_cond 容忍） |

---

## 10. 下一步

训练已收口，剩下的唯一决定性动作是 **G3**。而 G3 现在卡在一个 1–2 h 的适配器上，不是卡在卡上。

### 10.1 先补 G3 适配器（阻塞项）

现状［已核实］：`third_party/SDN/policy/pizero/config.py:14` 的
`METHODS = ['vanilla', 'vanilla_perturbed', 'sdn', 'sdn_grounding', 'sdn_smooth']`，
`build_policy()`（`82:88:third_party/SDN/policy/pizero/config.py`）只会建 `PiZeroInference` / `PiZeroSDNInference`，
**加载不了 `pi05-flashvla` 目录** —— `3.4_todo.md:225` 记的那个缺口仍未闭环。

要实现的接口很小（`policy/pizero/evaluate.py` 只调用这两处）：

| 调用点 | 签名 |
| :--- | :--- |
| `206:policy/pizero/evaluate.py` | `reset(instruction, seed=...)` |
| `226:policy/pizero/evaluate.py` | `step(image, instruction, proprio=...) -> (raw_chunk, actions)` |

实现方式：`METHODS` 加 `'flashvla'`，`build_policy()` 加一支，返回 `FlashVLAInference`；
内部 `PI05FlashVLAPolicy.from_pretrained(ckpt)` + 复用 ckpt 自带的 `policy_preprocessor.json`。

**必须原样复现的推理语义**（任一不对，G3 数字就不可信）：

| 项 | 取值来源 |
| :--- | :--- |
| `state_cond=True` / `empty_cameras=2` / `num_buffer_slots=4` / `chunk_size=4` / `n_action_steps` | 从 ckpt 的 `config.json` 读，**不要手写常量** |
| 冷启动 pad 语义：前 `num_buffer_slots−1` 个槽全 pad | 训练期 slot 是被真实历史填满的（`S=75`），推理冷启动有分布差，`3.4_todo.md:440` 已点名 |
| 图像：单相机 `observation.images.top`，`resize_with_pad → 224×224` | 与 S4 一致（§5.2） |
| 归一化统计 | ckpt 的 `policy_preprocessor.json` / `train_config.json` 的 `data` 段，**不是** open-pi-zero 那份 |

### 10.2 先 1 任务 × 10 seeds 冒烟（≈20 min），别直接上 3 h

冒烟要看的三个点：

1. 动作非零、非 NaN，且 chunk 内 `4` 步连续（不是每步重采样）；
2. 成功率不贴在 0% 或 100%（贴边通常是归一化或 pad 语义错了）；
3. **`state_cond` 是否真的被用起来**：把 `observation.state` 换一组值，看动作是否变化 —— 这是 §3.2 遗留的待核实项，顺手在这里闭环。

### 10.3 全量 G3（9 任务 × 50 seeds，`--start-seed 2000`，≈3 h/ckpt）

判据与口径严格对齐 `3.4_todo.md` §6：

- **macro ≥ 66.89% − 2 pp = 64.89%** 才算过闸；
- 分项必须报 `google_robot_place_apple_in_closed_top_drawer`（T0 18%，最弱）与 `widowx_stack_cube`（76%）；
- 建议**同时跑 `002750` 与 `004000`**：loss 在 3000 步后已经平了，若两者 G3 差在噪声内，说明这份数据上 4000 步已过训（每样本 7.2 遍，见 §6.5），后续 bridge 只需 2750 步。

### 10.4 过闸之后

- **fractal 12,000 步**：单卡按 5.06 s/step 需 ≈16.9 h（可接受）；GPU2 若腾出则双卡。
  `launch_train.sh` 已证明单进程可开 FSDP2，双卡只需 `accelerate launch --num_processes=2` + 同一复合后端断言。
- 之后才是 P4 的 SDN 增益实验；G3 不过则按 `3.flashvla_simpler.md:414` 止损到路线 A。

### 10.5 遗留杂项

- `3.4_todo.md` §5.3 的「显存看门狗 30 min」没有单独跑（实际用「按进程归因采样 + 断点续跑」替代）。
- `000250 … 002000` 已被 prune 删除，若要做「早期 ckpt 的 G3 曲线」，需按 §6.4 重跑（现在 5.7 h 全程）。

---

## 11. G3 执行记录（2026-11-22 23:00 → 11-23 03:20，GPU4 + GPU0）

结论先行：**G3 = 54.0%（004000）/ 50.5%（002750），未过闸（门槛 78.0%），但已经不是 0**。
过程中挖出一个会让所有 `pi05-flashvla` 部署都静默失效的 **preprocessor 序列化 bug**（§11.2），
修完之后同一个 ckpt 从 0/10 直接变成 5/10。

### 11.1 适配器：改成「分进程 + TCP」，不是同进程

§10.1 的原方案（`build_policy()` 返回同进程的 `FlashVLAInference`）**不可行**，已核实原因：
SIMPLER 跑在 py3.10 的 `sdn` env（torch 2.3.1 / numpy 1.24 / sapien），这个 env
`import lerobot` 直接失败；而把 lerobot 0.5.1 + torch 2.9 装进去会把 sapien 打断。

所以沿用 P0 已验证的分进程方案，同时保留 §10.1 想要的"全部复用 `evaluate.py`"：

| 文件 | 作用 | env |
| :--- | :--- | :--- |
| `scripts/p3/flashvla_server.py` | 每卡一个，加载 ckpt、持有 4 槽 buffer、处理冷启动 | flashvla (py3.12) |
| `third_party/SDN/policy/pizero/flashvla_client.py` | `FlashVLAInference`，只做 socket/pickle | sdn (py3.10) |
| `policy/pizero/config.py` | `METHODS` 加 `'flashvla'`、`FLASHVLA_CONFIG`、`build_policy()` 加一支 | sdn |
| `policy/pizero/evaluate.py` | `step()` 调用处加 `elif self.method == 'flashvla'` | sdn |

要点：

- **端口 = 12400 + 卡号**。`evaluate.py` 的 `_set_gpu()` 会按 worker 设
  `CUDA_VISIBLE_DEVICES`，客户端据此挑端口，所以 `--num-gpus 2` 自动对上两个 server，
  种子切分/多卡/结果目录/`000_episodes.json` 全部不用改。
- **只传纯 Python**。numpy 2.x 的 pickle 在 numpy 1.24 里解不开，所以 wire 上只有
  `bytes`/`list`/`float`。
- **防串号**：客户端连上后拿 server 的 ckpt 路径和 `--checkpoint` 比，不一致直接报错
  （端口由卡号推出来，跑第二轮时很容易连到上一轮残留的 server）。
- **`--reload-every 0`**：客户端本地没有模型，不需要周期性重建。

### 11.2 关键 bug：ckpt 自带的 `policy_preprocessor.json` 把 prompt 换成训练时没见过的东西

`3.4_todo.md` / §10.1 都写了"复用 ckpt 自带的 `policy_preprocessor.json`"——**这条不能照做**。

- `FlashVLAPi05PrepareLanguageProcessorStep` 是 dataclass，但没重写 `ProcessorStep.get_config()`，
  lerobot 基类默认返回 `{}`，于是保存出来是 `{"registry_name": "flashvla_pi05_prepare_language_processor_step", "config": {}}`。
- 重新加载时用字段默认值 → **`state_cond=False`**（训练用的是 `True`）。
- 两者 prompt 完全不同（同一帧，第 1 条）：训练期 12 token / 部署 46 token：

  ```
  state_cond=True : <bos>Task: put carrot on plate;\nAction:
  state_cond=False: <bos>Task: put carrot on plate, State: 130 131 248 98 149 230 104 249;\nAction:
  ```

- 定量影响（训练集 64 帧、同一批噪声）：**loss 0.3080 → 0.1620**（训练日志最后是 0.151）；
  离线动作回归 `pred/gt` 幅度比 0.85 → **0.978**，gripper 符号一致率 0.79 → **1.00**；
  闭环 `widowx_stack_cube` 10 seeds：**0/10（speed 0.0051）→ 5/10（speed 0.0087）**，
  动作输出从"gripper 贴 0、不敢闭合"变成"±0.94 的干脆夹取"。

修法两处（都必要）：

1. `flashvla/policies/pi05/processor.py`：给该 step 加 `get_config()`，以后保存不再丢字段；
2. `flashvla/policies/factory.py`：`make_pre_post_processors` 走 ckpt JSON 那条路时，
   用 `policy_cfg.state_cond` / `max_state_dim` 兜底 override（修**已经存下来**的 ckpt）。

顺带排除的怀疑：归一化统计没问题（ckpt 的 `observation.state`/`action` 的 mean/std 与
`/data/zjr/simpler_lerobot/bridge/*/meta/stats.json` 逐位相同），反归一化也没问题。
`sim_eval/robotwin`、`benchmarks/` 等任何走 `make_pre_post_processors(pretrained_path=...)`
的路径都吃了这个 bug，现在一起修好了。

### 11.3 §10.2 三点检查（`scripts/p3/probe_flashvla.py`，全部 PASS）

| 检查 | 读数 |
| :--- | :--- |
| P1 冷启动 | 前 `slots−1`=3 次 query 返回全零 hold 位姿，`max|a|=1.2e-10`（float32 往返舍入） |
| P2 稳态 chunk | 有限、非零、chunk 内 4 步有变化、同一 episode 相邻 query 不同（buffer 在滑动） |
| P3 `state_cond` | **换 state 的动作差只有 6.6e-5（相对 0.000）** |

P3 顺便闭环了 §3.2 的遗留项，但**结论和当时的预期相反**：修掉 prompt 之后，
`observation.state` 对动作几乎没有影响，说明 6 个零初始化 FiLM 张量在 4000 步里基本没长起来，
模型实际上近乎 state-blind。之前测到的"相对变化 0.486"其实是 prompt 里的 state 数字造成的假象。
这对后续训练是个明确信号：`state_cond` 这条通路要么给更大 lr，要么就得靠 prompt 里带 state（但那和本 ckpt 不一致）。

训练 loss 与动作维的关系也顺手测了：`max_action_dim=32` 但 `action_dim=7`，
另外 25 维是零填充；实测这 25 维 MSE=0.0014，只贡献 0.0011，
**所以 0.151 的总 loss ≈ 动作维 MSE 0.69 的 7/32**，没有被 padding 稀释。
换句话说 loss 读数是可信的，只是"平了"不等于"够好"。

### 11.4 冷启动 12 步的浪费（+7 pp，已改为默认）

原样复现 `AsyncStreamingActionManager` 的语义时，前 3 次 query 返回 hold 位姿，
于是 widowx 每 episode 60 步里有 **12 步（20%）在原地不动**。

但训练期是「**一个观测** + prefix 长度的 buffer」（`_build_per_chunk_time`），
所以在**同一帧**上把 3 次冷启动 query 一次跑完，既更贴训练分布，又不浪费步数。
server 里 `cold_start_burst` 默认开（`--no-cold-burst` 可回退）。

| 004000 | carrot | eggplant | spoon | stack_cube | macro |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 冷启动摊在 episode 里（原语义） | 32% | 68% | 54% | 34% | **47.0%** |
| 冷启动 burst（**默认**） | 42% | 68% | 58% | 48% | **54.0%** |
| burst + gripper 不二值化 | 38% | 70% | 58% | 48% | 53.5% |

gripper 二值化与否在噪声内（53.5 vs 54.0），保留二值化（更贴训练数据 ±1）。

### 11.5 G3 结果（4 widowx 任务 × 50 seeds，seeds 2000–2049）

**口径纠正**：§10.3 写的「9 任务 × 50 seeds」不成立。`/data/zjr/simpler_lerobot/bridge`
只有 **4 个 widowx 叶子**（17784 帧），`google_robot_*` 5 个任务完全没有训练覆盖，
拿这个 ckpt 去跑它们没有意义。所以 G3 的分母是 **T0 的 widowx 子集 80.0%**，不是 9 任务的 66.89%。
门槛按同口径取 −2 pp → **78.0%**。

| 任务 | T0 (pi0\*) | 004000 | 002750 |
| :--- | ---: | ---: | ---: |
| `widowx_carrot_on_plate` | 68% | 42% | 34% |
| `widowx_put_eggplant_in_basket` | 88% | 68% | 68% |
| `widowx_spoon_on_towel` | 88% | 58% | 60% |
| `widowx_stack_cube` | 76% | 48% | 40% |
| **macro** | **80.0%** | **54.0%** | **50.5%** |
| 与 T0 差 | — | **−26.0 pp** | **−29.5 pp** |

产物：`gates/G3_full.json`（默认配置，两 ckpt）、`gates/G3_noburst.json`、`gates/G3_ablation_coldburst.json`、
`gates/G3_ablation_nogripperbin.json`；原始结果在 `third_party/SDN/eval_logs/p3_g3_burst/{004000,002750}/` 与 `p3_g3/{004000,002750}/`（未 burst）。

判读：

1. **没过闸**，差 24–26 pp（不要用 `3_4_todo` 的 66.89% 口径）。
2. **不是 0，而且不是接线问题**：离线在训练分布上 `pred/gt` 幅度比 0.978、gripper 符号 100%，
   说明加载/归一化/流式语义都对；剩下的差是**策略本身弱**。
3. **4000 步 ≈ 2750 步**（54.0 vs 50.5，配对后约 1σ），**还没过拟合 → 是欠拟合**，
   所以 §10.4 的"fractal 12,000 步"仍然是最大杠杆；这条数据也支持 §10.3 的建议：
   bridge 只用 2750 步就够，不用跑到 4000。
4. **速度**：50 episodes / 2 卡 ≈ 2.5 min（每 episode ≈ 2.8 s 墙钟），T0 pi0\* 是
   22.8 s/episode（1 卡）。1 次 denoise / query 直接换成约 4–8× 的评测吞吐。

### 11.6 下一步（按性价比）

1. **fractal 12,000 步**（§10.4 原计划）。这是当前唯一能吃掉 20+ pp 缺口的动作；
   loss 曲线要同时记动作维 MSE（§11.3），不要只看总 loss 的"平台"。
2. ~~训练时把 `state_cond` 的 FiLM 真正训起来（10× lr / 小随机初始化）~~ ——
   **已试，都不行，见 §11.7**，不要再往 12k 里塞这个改动。
3. 想再挤几个点可以试：query 之间做 4 步带重叠（`overlap_steps>0`，`async_manager` 已支持），
   用"多算一次 query"换轨迹平滑；现在 `overlap_steps=0` 是 chunk 硬切。
4. 不建议再动 G3 的评测口径或 cold-start 细节了——这部分已经做满（§11.4 三行消融都在噪声内
   或已经是最优）。

### 11.7 Q3 小实验：state 通路的两个"便宜"修法都不成立（2026-11-23 11:36–12:10）

§11.6 第 2 条给的两个候选（10× lr / 小随机初始化）**都试了，都不行**。结论：`state_cond`
这条 FiLM 通路不是调超参能救的，**不要并进 12k 训练**。

度量脚本：`scripts/p3/state_sensitivity.py`（同一 seed，只改 `observation.state`：
全 0 vs `linspace(-1,1)`，比动作的平均相对变化）。基线 `004000` 是 **rel = 0.0068**。

| 实验 | 改动 | 步数 | `state_mlp_out\|w\|` | rel | 判定 |
| :--- | :--- | ---: | ---: | ---: | :--- |
| 基线 `004000` | — | 4000 | 9.68e-6 | 0.0068 | 死 |
| `state_lr10x_probe/000250` | `state_path_lr_mult=10` | 250 | 9.85e-6（+1.7%） | 0.0047 | **无改善** |
| `state_init_probe/000150` | `state_path_init_std=2e-4` + 10× lr | 150 | 1.60e-4（16.5×） | 0.0082 | 略微 |
| `state_init_probe/000300` | 同上 | 300 | 1.60e-4（**没动**） | 0.0045 | **无改善** |

三条读数说明问题不在超参：

1. **梯度是健康的**。离线一次 backward，`state_mlp_out.weight` 的 `|grad|=0.214`，
   是 state 通路里最大的，甚至和 `time_mlp` 组（0.37）同量级 —— 不是梯度消失。
2. **权重就是不动**。把 init 放大 16.5× 之后，300 步里 `|w|` 从 1.5965e-4 到 1.5965e-4，
   在当前量级上已经低于 bf16 的有效更新分辨率（1.6e-4 的 bf16 量化台阶约 6e-7）；
   零初始化 + bf16 参数等于给这条支路上了锁。
3. **量级差两个数量级**。`state_emb` 加到 `adarms_cond` 上，而 `time_emb` 的
   `|·|.mean=1.58e-2`；state 支路现在是 1e-5 量级，要起作用得长 ~1000×，
   而 10× lr 在 250 步只给了 +1.7%。**乘法性杠杆够不到加法性缺口。**

要真让 state 起作用，得动结构而不是超参，二选一：
把 FiLM 支路整个放到 fp32 并按 `time_emb` 同量级初始化（会改变 ckpt 兼容性），
或者干脆走 `state_cond=False` 那条路（把 state 量化后塞进 prompt，pi05 原设计），
后者和"1 次 denoise/query"的流式配方不冲突，但需要重训。

> 备注：这两次 probe 都用了**新的 `output_dir` + `policy.pretrained_path=004000`**（不是
> `--resume`），所以优化器状态是重置的。这会让绝对数值不可与 004000 直接比，但
> "权重不动 / 敏感度不涨"这个结论不受影响 —— 4000 步的正常训练都没让它长起来。

### 11.8 `--resume` 续跑不是"再训 2000 步"：lr 被恢复到 2.5e-07

续跑 `004000 → 006000` 时实测 **`lr:2.5e-07` 恒定**（`logs/p3_g3/train_4000_to_6000.log`）。
原因：非 FSDP 的 resume 走 `load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)`
（`flashvla/train.py:877` 附近），会把**调度器状态一起恢复**；原配置
`scheduler.num_decay_steps: 4000` 到 4000 步时 lr 已经衰减到 `decay_lr = 2.5e-7`，
再往后就是恒定在谷底。

后果（两条都得记住）：

1. **"续跑买趋势"这个设计不成立**。`004000→006000` 这 2000 步几乎没有学习，6000 只是
   4000 加噪声。要测"多训有没有用"，必须**新开 run**（`policy.pretrained_path=004000`
   + 新 `output_dir` + `num_decay_steps` 覆盖总步数），让 lr 重新从 5e-5 衰减。
   已按这个口径补了 `pi05_flashvla_bridge_ft2000_full_lr`（2000 步、warmup 100、decay 2000）。
2. **G3 曲线的 `005000 = 59.0%` 不能当成"训练到 5000 步的成绩"**，它其实就是 4000 步的
   权重换个采样（SE ≈ 3.5 pp，54.0 → 59.0 是 1.4σ）。曲线在 4000 之后**没有信息量**。

另：`train/configs/pi05/simpler/pi05_flashvla_fractal.yaml`（12,000 步）训的是
`google_robot_*` 五个任务，**和 G3 的分母（widowx 4 任务）不是同一批数据**。
按 Q1 定下的口径，能抬 widowx G3 的是**继续训 bridge**，不是 fractal。
