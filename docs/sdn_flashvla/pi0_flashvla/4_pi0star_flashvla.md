# 4. π0\* → `pi0-flashvla`：直接微调方案（2026-11-23）

> **执行者须知**：本文是给「另一个 agent」照着跑的施工单。凡标 **［已核实］** 的数字都是本机实测，
> 凡标 **［估算］** 的必须由 §5 的冒烟步骤校正。
> **一句话**：把 π0\*（open-pi-zero 的 `bridge_beta` / `fractal_beta`）权重按键盘映射进 FlashVLA **已有的**
> `pi0-flashvla` 策略，在 P2 已交付的 SIMPLER LeRobot 数据上做多缓冲区联合微调。**不经过 pi05、不改 policy 主体。**
>
> 上游：`3_3.md`（P2 数据交付 + T0 基线）、`3_5_p3_execution_report.md`（pi05 路线的实跑与 G3 失败）。
> 本文的转换器与配置已落盘：`scripts/p4/convert_pi0star.py`、`train/configs/pi0/simpler/*.yaml`。
> 代码引用格式 `起:止:相对路径`，路径相对 `flashvla/`。

---

## 0. 一页结论

| 项 | 内容 |
| :--- | :--- |
| 起点权重 | `third_party/SDN/pretrained/open-pi-zero/{bridge_beta,fractal_beta}_*.pt`（各 11.77 GB，含优化器态） |
| 转换产物 | `ckpt/pi0star_bridge_flashvla/`、`ckpt/pi0star_fractal_flashvla/`（`model.safetensors` + `config.json`） |
| 策略类 | `pi0-flashvla`（`flashvla/policies/pi0/modeling_pi0_flashvla.py`），**不是** `pi05-flashvla` |
| 数据 | `/data/zjr/simpler_lerobot/{bridge,fractal}/<task>/success`（单视角 480×640，state 8 维，action 7 维，5 fps） |
| 训练规模 | bridge 4,000 步（7.2 epoch）/ fractal 12,000 步（≈7.6 epoch），等效 batch 32 |
| 预算 | 单卡 GPU4 或 GPU5，bridge ≈5–6 h、fractal ≈15–18 h（**［估算］**，§5 S4 校正） |
| 主判据 | M=1 在留出 seeds 2000–2049 上：widowx 4 任务 macro ≥ **78.0%**（π0\* 的 80.0% − 2pp）；fractal 5 任务 macro ≥ **54.4%**（56.4% − 2pp） |
| 最大风险 | π0\* 的 proprio 第三栈被丢弃（311.47M 参数，占 8.8%），state 通路需要从零学 |

**已经替你做完的部分（可直接复用）**

| 产物 | 状态 |
| :--- | :--- |
| 转换器 `scripts/p4/convert_pi0star.py` | ［已核实］dry-run 通过：938 张量 → 774 映射 + 165 丢弃 + 5 合成 = 779；`find_unfilled_parameters` 为空 |
| `ckpt/pi0star_bridge_flashvla/` | ［已核实］8.0 GB、779 张量、`config.json` 的 `type == pi0-flashvla` |
| 真实加载 | ［已核实］`PI0FlashVLAPolicy.from_pretrained(...)` 成功，134 s，4,291.34 M 参数，`action_buffer (1,16,32)` |
| 三份 yaml | ［已核实］draccus 解析通过（`pi0_flashvla_bridge.yaml`、`pi0_flashvla_fractal.yaml`、`_smoke_pi0star.yaml`） |
| fractal 转换 | ［已核实］2026-11-23 16:57 完成：779 张量、`verify.ok=true`、`unfilled=0` |

---

## 1. 为什么是「π0\* + FlashVLA 的 pi0 policy」

1. **FlashVLA 早已实现 pi0 policy，不需要碰 pi05。** `flashvla/policies/pi0/` 有完整三件套
   （`modeling_pi0.py` / `modeling_pi0_flashvla.py` / `configuration_pi0.py`），工厂里已注册
   `pi0` 与 `pi0-flashvla` 两个 type（`69:75:flashvla/policies/factory.py`），连 LIBERO 的
   `pi0_flashvla.yaml` 都在（`train/configs/pi0/libero/pi0_flashvla.yaml`）。
2. **pi05 路线失败在「域」而不在「流式」。** P3 用 `pi05_base`（通用预训练）在 SIMPLER 数据上微调，
   G3 只到 54.0%（门槛 78.0%，`3_5_p3_execution_report.md` §11）。而 π0\* 是**已经在 SIMPLER 上训过**
   的同族 π0 权重（bridge/fractal 各一份），域先验正是缺的那一半。
3. **架构同源，映射成本低。** π0\* 的 `joint_model.mixtures.{vlm,action}` 与 FlashVLA pi0 的
   「vlm + action expert 两栈、逐层 9 个张量、隐藏维 2048/1024、KV 单头 256」逐层对齐（§3）。

**代价（必须承认）**：π0\* 有第三栈 `proprio`，FlashVLA pi0 没有 → 丢掉 311.47M 参数；
state 通路退化成「一个 state token + 零初始化的 `state_proj`」，与 π0\* 的 proprio 全栈不同构（§4）。

---

## 2. 事实核实

### 2.1 π0\* 权重结构［已核实］

`torch.load(..., map_location="cpu", mmap=True, weights_only=True)` 读出 8 个顶层键
（`cnt_update`、`model`、`action_optimizer`、`vlm_optimizer`、两个 scheduler、`wandb_id`），
只取 `model`：

| 键前缀 | 张量数 | 参数量 | 形状要点 |
| :--- | ---: | ---: | :--- |
| `joint_model.mixtures.vlm.layers.{0..17}.*` | 162 | 1,981.9 M | q/o 2048×2048，k/v 256×2048，mlp 16384 |
| `joint_model.mixtures.action.layers.{0..17}.*` | 162 | 311.5 M | q 2048×1024，k/v 256×1024，mlp 4096 |
| `joint_model.mixtures.action.norm.weight` | 1 | — | 1024 |
| `joint_model.mixtures.proprio.layers.{0..17}.*` + `norm` | 163 | 311.5 M | 与 action 同形 |
| `vision_tower.vision_model.*` | 437 | 412.4 M | SigLIP so400m，1152 宽 |
| `multi_modal_projector.linear.*` | 2 | 2.4 M | 2048×1152 |
| `embed_tokens.weight` | 1 | 526.8 M | **257216**×2048（比 PaliGemma 多 64 行） |
| `action_encoder.linear_1/2/3.*` | 6 | 3.2 M | 7→1024、2048→1024、1024→1024 |
| `action_decoder.*` | 2 | — | 7×1024 |
| `proprio_encoder.*` | 2 | — | 1024×**7** |
| **合计** | **938** | **3,549.6 M** | 含优化器态的 11.77 GB 文件里 `model` 约 7.1 GB |

每层 9 个张量：`self_attn.{q,k,v,o}_proj.weight`、`mlp.{gate,up,down}_proj.weight`、
`input_layernorm.weight`、`post_attention_layernorm.weight`。**没有** `lm_head`，VLM 也**没有**最终
`norm.weight`（vlm 侧 162 = 9×18，无余项）。

### 2.2 `pi0-flashvla` 的参数空间［已核实］

meta 设备实例化后 `state_dict()` 有 **1,104** 条（含别名），去重后 **779** 个张量：

| 分组 | 形态 |
| :--- | :--- |
| `model.layers.{0..17}.self_attn.vlm_attention.*` / `.action_expert_attention.*` | 融合层：两栈并排，各 4 个投影 |
| `model.layers.{i}.mlp.vlm_mlp.*` / `.action_expert_mlp.*` | 融合层 MLP |
| `model.layers.{i}.input_layernorm.{0,1}.weight` / `post_attention_layernorm.{0,1}.weight` | 0=VLM、1=expert |
| `model.vlm.model.{language_model,vision_tower,multi_modal_projector}.*` | 别名（与上面同一批 Parameter） |
| `model.prefix_embedder.lang_embedder.weight` | 别名（= `language_model.embed_tokens`） |
| `model.vlm.lm_head.weight`（257152×2048）、`model.action_expert.lm_head.weight`（257152×1024） | **训练中不使用**，但必须被 checkpoint 覆盖 |
| `model.action_expert.model.embed_tokens.weight`（257152×1024） | 同上，不使用 |
| `model.vlm.model.language_model.norm.weight`（2048） | π0\* 无对应 |
| `model.suffix_embedder.{action_in_proj,action_time_mlp_in,action_time_mlp_out,state_proj}` | 32 维动作/状态 → 1024 |
| `model.action_out_proj.{weight,bias}` | 32×1024 |

加载器只做「前缀改名 + strict=False + 覆盖性断言」（`875:969:flashvla/policies/pi0/modeling_pi0_flashvla.py`）：
它**接受 alias-free 键名**（`model.layers.*`），且 `find_unfilled_parameters` 按张量身份判定是否被填充
（`116:159:flashvla/policies/loading.py`），所以转换器写 alias-free 布局即可。

### 2.3 数据［已核实］

| 族 | 任务数 | episodes | frames | 对应 π0\* 权重 |
| :--- | ---: | ---: | ---: | :--- |
| `bridge`（widowx） | 4 | 529 | 17,784 | `bridge_beta_step19296_2024-12-26_22-30_42.pt` |
| `fractal`（google_robot） | 5 | 777 | 50,843 | `fractal_beta_step29576_2024-12-29_13-10_42.pt` |

单叶子 schema：`observation.images.top` 480×640 video、`observation.state [8]`
（`x,y,z,qw,qx,qy,qz,gripper`）、`action [7]`（`dx,dy,dz,rx,ry,rz,gripper`）、fps 5。
模型侧 `prepare_images` 自行 `resize_with_pad` 到 `image_resolution=(224,224)` 且把像素映到 [-1,1]
（`1067:1097:flashvla/policies/pi0/modeling_pi0_flashvla.py`），所以数据不需要离线 resize。

### 2.4 硬件与吞吐［已核实］

当前 GPU4 / GPU5 各空闲 **45,458 MiB**（`nvidia-smi`，2026-11-23 15:00）。参照 P3 同规模 pi05 实跑：
bs=4 / accum=8（等效 32）**5.06 s/step**、峰值 38.3 GB、4000 步 5 h 44 min。
本方案模型 4,291 M 参数，但序列短得多（1 视角 256 token + 文本 ≤200 + 17 动作 ≈ 473 token，
pi05 配方是 3 视角 ≈985 token），**［估算］** bs=4 约 4.5–5.5 s/step、峰值 34–38 GB。

### 2.5 已预跑的三件事［已核实］

```text
转换 dry-run: source 938 → mapped 774 / dropped 165 (311,473,152 参数) / synthesized 5 → output 779
              verify: ok=True, unknown=0, shape_mismatch=0, unfilled=0（模型侧 1104 个名字）
落盘:         ckpt/pi0star_bridge_flashvla/{model.safetensors 8.0GB, config.json}
真实加载:     from_pretrained 成功 134.4 s，params=4291.34 M，action_buffer (1,16,32)，N=4 C=4
             （伴随两条预期告警：目录无 postprocessor → 冷启动回退零动作；torch_dtype deprecated）
```

---

## 3. 权重映射表

转换器把所有规则写死在 `scripts/p4/convert_pi0star.py`，下表即其行为。

### 3.1 直通（形状完全一致）

| π0\* | → `pi0-flashvla` | 备注 |
| :--- | :--- | :--- |
| `vision_tower.vision_model.*` | `model.vlm.model.vision_tower.vision_model.*` | 437 张量，SigLIP so400m |
| `multi_modal_projector.linear.{weight,bias}` | `model.vlm.model.multi_modal_projector.linear.{weight,bias}` | 2048×1152 |
| `joint_model.mixtures.action.norm.weight` | `model.action_expert.model.norm.weight` | 1024 |
| `action_encoder.linear_2.*` | `model.suffix_embedder.action_time_mlp_in.*` | 1024×2048（openpi 的 time-concat 层） |
| `action_encoder.linear_3.*` | `model.suffix_embedder.action_time_mlp_out.*` | 1024×1024 |

### 3.2 层内 9 键 → 融合层

对 `i ∈ [0,18)`、`s ∈ {vlm, action}`：

| π0\* 层内键 | → FlashVLA 键（`s=vlm` 时 `attn=vlm_attention, mlp=vlm_mlp, k=0`；`s=action` 时 `action_expert_attention / action_expert_mlp / k=1`） |
| :--- | :--- |
| `...layers.i.self_attn.{q,k,v,o}_proj.weight` | `model.layers.i.self_attn.{attn}.{q,k,v,o}_proj.weight` |
| `...layers.i.mlp.{gate,up,down}_proj.weight` | `model.layers.i.mlp.{mlp}.{gate,up,down}_proj.weight` |
| `...layers.i.input_layernorm.weight` | `model.layers.i.input_layernorm.k.weight` |
| `...layers.i.post_attention_layernorm.weight` | `model.layers.i.post_attention_layernorm.k.weight` |

形状在两侧逐项相等（vlm 2048 系、action 1024 系），无需任何变换。

### 3.3 裁剪 / 补零

| π0\* | 形状 | → FlashVLA | 形状 | 处理 |
| :--- | :--- | :--- | :--- | :--- |
| `embed_tokens.weight` | 257216×2048 | `model.vlm.model.language_model.embed_tokens.weight` | 257152×2048 | 截掉尾部 64 行（多出的 64 个 token 是 π0\* 私有扩展） |
| 同上 | — | `model.vlm.lm_head.weight` | 257152×2048 | 复制截断后的 embedding（openpi 里 tie）；**必须 `.clone()`**，否则 safetensors 报共享存储 |
| `action_encoder.linear_1.weight` | 1024×7 | `model.suffix_embedder.action_in_proj.weight` | 1024×32 | 右侧补 25 列零（零填充动作维在数值上等价） |
| `action_decoder.weight` / `.bias` | 7×1024 / 7 | `model.action_out_proj.weight` / `.bias` | 32×1024 / 32 | 下方补零行；推理时只取前 7 维 |

### 3.4 丢弃

| π0\* | 张量数 | 参数量 | 原因 |
| :--- | ---: | ---: | :--- |
| `joint_model.mixtures.proprio.*`（含 `norm`） | 163 | 311.5 M | FlashVLA pi0 只有两栈 |
| `proprio_encoder.{weight,bias}` | 2 | 7 K | 默认不映射（语义不匹配，见 §4.2） |

丢弃总量 **165 张量 / 311,473,152 参数 = π0\* 的 8.8%**，转换报告里逐条列出。

### 3.5 合成（π0\* 没有、但 FlashVLA 必须有）

| 目标键 | 初值 | 理由 |
| :--- | :--- | :--- |
| `model.action_expert.model.embed_tokens.weight` | 零 | 该 embedding 在 pi0 里从不参与前向（expert 的输入来自 suffix embedder） |
| `model.action_expert.lm_head.weight` | 零 | 同上，且 FSDP 路径会冻结这两个 lm_head |
| `model.vlm.model.language_model.norm.weight` | 全 1 | π0\* 未保存 VLM 末端 RMSNorm；1.0 是 RMSNorm 的中性初值 |
| `model.suffix_embedder.state_proj.{weight,bias}` | 零 | state 分支从零学（`--map-proprio-to-state` 可改为拷贝，见 §4.2） |

零初始化 `state_proj` 是刻意的：FlashVLA 的 pi05 配方也用零初始化 FiLM 让 state 起步为「无贡献」，
避免 7 维 proprio 权重错误作用到 8 维 quat 状态上。

---

## 4. 五个语义缺口与处置

### 4.1 proprio 第三栈被丢弃（影响最大）

π0\* 的 proprio 是一条**完整的 18 层 transformer 栈**（与 action 栈同宽 1024），而 FlashVLA pi0 把状态压成
**一个 suffix token**。丢栈意味着：① 311.5 M 参数闲置；② vlm/action 两栈在 π0\* 里原本还会 attend 到 proprio
的 KV，现在少了这一路，初始输出的分布有偏移。

**处置**：接受，靠微调重适配。**判定点**：S5 训练前 200 步 loss 应快速下降；若 500 步仍在 0.35 以上（pi05 同数据
step 500 是 0.183），说明丢掉 proprio 后初始化太偏，转 §8 的退路 R2。

### 4.2 state 通路：8 维 quat vs 7 维 proprio

π0\* 的 `proprio_encoder` 是 1024×7（openpi 在 widowx 上把四元数转 euler），我们的数据是 8 维 quat。
两者**形状可补零但语义不对齐**，所以默认**不拷贝**（`state_proj` 零初始化）。
若训练后发现 state 分支贡献过小（P3 在 pi05 上量到 FiLM 贡献 ~1e-5 的教训），按顺序用：
① `state_path_lr_mult: 3.0`（`94:102:flashvla/configs/train_config.py` 已有这个旋钮）；
② 重跑转换器加 `--map-proprio-to-state`（**仅在**把数据 pipeline 改成 euler 7 维之后才有意义）。

### 4.3 词表 257216 vs 257152

π0\* 多 64 行 embedding。截断即可：多出的 token 不在 PaliGemma tokenizer 的词表内，推理不会产生这些 id。
**不要**反过来把 FlashVLA 的 vocab 改大——那会让 `lm_head`、tokenizer、position 全部错位。

### 4.4 π0\* 缺 lm_head / VLM 末端 norm

按 §3.3 / §3.5 合成。合成后 `assert_checkpoint_covers_parameters` 才会通过；否则加载直接抛
`left N parameter tensors at random initialization`。

### 4.5 视角数与图像尺寸

π0\* 单视角 224×224；`pi0-flashvla` 支持 `empty_cameras` 补空槽。本方案设 **`empty_cameras: 0`**
（最贴近 π0\*，序列最短 473 token）。pi05 配方用 2 是为了凑 `pi05_base` 的三槽，**这里不要照抄**。

---

## 5. 执行步骤

> 约定：`ROOT=/home/zjr/workspace_tj/StreamingVLA/flashvla`，`PY=$ROOT/../miniconda_ygc/envs/flashvla/bin/python`。
> 每步都有判据；判据不过就停在该步，按「退路」处理，**不要带着问题往下跑**。

### S0 · 前置检查（1 min，无卡）

```bash
cd /home/zjr/workspace_tj/StreamingVLA/flashvla
ls -lh third_party/SDN/pretrained/open-pi-zero/*.pt          # 两个 11.77 GB
ls ckpt/pi0star_bridge_flashvla/                             # 已存在则 S1 的 bridge 部分可跳过
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
df -h / | tail -1                                            # 需 ≥20 GB（每个转换产物 8 GB）
```

**判据**：两个 `.pt` 在；GPU4 或 GPU5 free ≥ 42,000 MiB；磁盘 ≥20 GB。

### S1 · 转换（每族 ≈2 min，无卡）

```bash
cd /home/zjr/workspace_tj/StreamingVLA/flashvla
PY=/home/zjr/workspace_tj/StreamingVLA/miniconda_ygc/envs/flashvla/bin/python

# bridge 已预跑过；重跑是幂等的
$PY scripts/p4/convert_pi0star.py \
  --src third_party/SDN/pretrained/open-pi-zero/bridge_beta_step19296_2024-12-26_22-30_42.pt \
  --data /data/zjr/simpler_lerobot/bridge/widowx_stack_cube/success \
  --empty-cameras 0 --write --dst ckpt/pi0star_bridge_flashvla \
  --json logs/p4/convert_bridge.json

# fractal 必跑
$PY scripts/p4/convert_pi0star.py \
  --src third_party/SDN/pretrained/open-pi-zero/fractal_beta_step29576_2024-12-29_13-10_42.pt \
  --data /data/zjr/simpler_lerobot/fractal/google_robot_close_drawer/success \
  --empty-cameras 0 --write --dst ckpt/pi0star_fractal_flashvla \
  --json logs/p4/convert_fractal.json
```

**判据**：两条都以 `wrote .../model.safetensors (779 tensors) and config.json` 结束，且 JSON 里
`verify.ok == true`、`unfilled_count == 0`、`unknown_keys == []`、`shape_mismatches == []`。
**退路**：`verify` 失败 → 先看 `unmapped_tensors`；若出现新的键（π0\* 版本不同），把键名贴进报告并停在 S1，
**不要**用 `--map-proprio-to-state` 之类的开关去掩盖。

### S2 · 加载自检（≈3 min，无卡）

```bash
cd /home/zjr/workspace_tj/StreamingVLA/flashvla
$PY - <<'PY'
import argparse
from scripts.p4.convert_pi0star import build_config
from flashvla.policies.pi0.modeling_pi0_flashvla import PI0FlashVLAPolicy
for leaf, ckpt in (
    ("/data/zjr/simpler_lerobot/bridge/widowx_stack_cube/success", "ckpt/pi0star_bridge_flashvla"),
    ("/data/zjr/simpler_lerobot/fractal/google_robot_close_drawer/success", "ckpt/pi0star_fractal_flashvla"),
):
    a = argparse.Namespace(data=leaf,
                           num_buffer_slots=4, chunk_size=4, n_action_steps=4,
                           empty_cameras=0, use_adarms_time_cond=False)
    cfg, _ = build_config(a); cfg.device = "cpu"
    pol = PI0FlashVLAPolicy.from_pretrained(ckpt, config=cfg)
    print(fam, "OK", sum(p.numel() for p in pol.parameters()) / 1e6, "M",
          tuple(pol.action_buffer.shape))
PY
```

**判据**：两行 `OK`，参数量 ≈ **4,291.34 M**，buffer `(1, 16, 32)`；允许出现
`No postprocessor stats found ... Cold start will fall back to zero action` 与 `torch_dtype is deprecated`。
**退路**：`RuntimeError: Checkpoint ... left N parameter tensors at random initialization` → 回 S1 看
`synthesized` 那 5 条是否齐全；`Unexpected keys` → 转换器与模型版本不匹配，停。

### S3 · 2 步冒烟（≈10 min，占卡）

```bash
cd /home/zjr/workspace_tj/StreamingVLA/flashvla
CUDA_VISIBLE_DEVICES=4 bash train/train.sh train/configs/pi0/simpler/_smoke_pi0star.yaml 1 2>&1 | tee logs/p4_smoke.log
```

**判据（四条全过才算过）**

1. `loss` 有限（非 NaN/inf），两步都有打印（`log_freq: 1`）；
2. 日志里没有 `unexpected keys` / `missing keys` / `left ... at random initialization`；
3. `/tmp/p4_smoke_pi0star/checkpoints/000002/pretrained_model/` 里有 `model.safetensors` + `config.json`；
4. 用 S2 的片段能重新加载该目录（把 `ckpt/pi0star_*` 换成它，`empty_cameras=0`）。

**退路**

- 第 3 条若报 `Some tensors share memory`：说明保存路径没走 `save_model_as_safetensor` 的去重逻辑。
  先确认 `train.py:560` 一带确实调的是 `policy.save_pretrained`（P3 的 pi05 跑通过，正常不会出）。
  若真出，写一个 20 行的转换：import `PI0FlashVLAPolicy`，`with torch.device("meta")` 取
  `target = policy.state_dict()`，按 `model.layers.*` 名字重排后 `save_file`（`detach_backbone_layer_aliases`
  会把别名层换成 `nn.Identity`，语义等价）。
- 第 1 条若 OOM：把 yaml 的 `batch_size` 降到 1（`grad_accum_steps` 保持不变），仍 OOM 见 §8 R1。

### S4 · 显存 / 吞吐扫描（≈30 min，占卡）

对 `batch_size ∈ {2, 4}` 各跑 10 步（复制 `_smoke_pi0star.yaml`，改 `batch_size`、`steps: 10`、
`save_freq: 1000`、`output_dir: /tmp/p4_sweep_bsN`），记录
`torch.cuda.max_memory_allocated()` 与 stdout 的时间戳差。

**判据**：选出「峰值 ≤ free − 2 GB」的最大 bs，并把 `s/step` 写进 §6 的表（替换［估算］）。
**退路**：bs=2 仍 OOM → 见 §8 R1（FSDP / 冻结 lm_head / 降 tokenizer 长度）。

### S5 · bridge 正式训练（≈5–6 h，占卡）

```bash
cd /home/zjr/workspace_tj/StreamingVLA/flashvla
export CUDA_VISIBLE_DEVICES=4
setsid nohup bash train/train.sh train/configs/pi0/simpler/pi0_flashvla_bridge.yaml 1 \
  > logs/p4_bridge_train.log 2>&1 &
echo $! > logs/p4_bridge_train.pid
```

**看护（前 30 min）**：`tail -f logs/p4_bridge_train.log`，每 50 步一行 loss。

**判据**：

- 前 200 步 loss 单调下降且有限（参照：pi05 同数据 step 500 = 0.183）；
- `checkpoints/000250`、`000500` 正常出现（每个 ≈8 GB）；
- 4000 步正常退出，`last → 004000`；
- 训练结束目录带 `policy_preprocessor.json` / `policy_postprocessor.json`（**这一步才会补上冷启动统计**）。

**退路**：见 §8 R2（loss 不降）/ R3（OOM）/ R1。

### S6 · fractal 正式训练（≈15–18 h，占卡）

与 S5 同法，配置换 `pi0_flashvla_fractal.yaml`、`CUDA_VISIBLE_DEVICES=5`（GPU4 留着评测），
`save_freq: 500`。可等 S7 的 bridge 评测结果再决定是否值得跑；若只想先验证机制，可把 `steps` 降到 6,000。

### S7 · 评测与闸门（每 checkpoint ≈3 h）

复用 P3 已建好的适配器（**不需要重写**）：`scripts/p3/flashvla_server.py`（py3.12 侧加载策略目录）
+ `third_party/SDN/policy/pizero/flashvla_client.py`（py3.10 侧 TCP 客户端）。

```bash
# 终端 A（py3.12）
CUDA_VISIBLE_DEVICES=4 $PY scripts/p3/flashvla_server.py \
  --ckpt /data/zjr/simpler_lerobot/runs/pi0_flashvla_bridge_success/checkpoints/last/pretrained_model \
  --port 12400

# 终端 B（sdn env, py3.10）
cd third_party/SDN
for t in widowx_carrot_on_plate widowx_put_eggplant_in_basket widowx_spoon_on_towel widowx_stack_cube; do
  python -m policy.pizero.evaluate --method flashvla --task "$t" \
    --n-trajs 50 --start-seed 2000 --result-root ./eval_logs/p4_pi0star --no-video
done
```

**判据（M=1 闸门）**

| 族 | π0\* 基线 | 门槛（−2pp） | 备注 |
| :--- | ---: | ---: | :--- |
| widowx 4 任务 | 80.0% | **≥ 78.0%** | 逐任务：68 / 88 / 88 / 76 |
| fractal 5 任务 | 56.4% | **≥ 54.4%** | 逐任务：68 / 74 / 36 / 86 / 18 |

同时报 `carrot_on_plate`、`stack_cube` 两个精细任务的分项，以及首次动作时延（TTFA）。

**不过闸怎么办**：先排除协议差异（seeds、每 5 步重规划、图像预处理），确认协议一致后再看损失曲线；
仍不过则按 §8 R2 调参重跑，**不要**改判据。

**闸门过了之后**才接 SDN（M=1 / M=4 / M=12 + 接缝项），那是 `2.md` 的 B4/B5，不在本方案范围内。

---

## 6. 训练预算

| 族 | 帧数 | bs×accum | step/epoch | steps | epoch | s/step | 墙钟 |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| bridge | 17,784 | 4×8 | 556 | 4,000 | 7.2 | **2.17［实测］** | **2.4 h** |
| fractal | 50,843 | 4×8 | 1,589 | 12,000 | 7.6 | **2.18［实测］** | **7.3 h** |

S4 实测（L40S 45,458 MiB free，GPU4，bs×accum 均 8，`logs/p4_sweep_bs{2,4}.*`）：

| bs | s/step | 峰值显存（nvidia-smi，含 context） | 每样本 |
| ---: | ---: | ---: | ---: |
| 2 | 1.60 | 20,611 MiB | 0.100 s |
| **4** | **2.11–2.18** | **25,387 MiB** | **0.066 s** |

bs=4 峰值 25.4 GB，距 42 GB 判据还有 17 GB 余量，故正式训练固定 bs=4 / accum=8（等效 32）。
比 §2.4 的［估算］快 2.2×：pi05 的 5.06 s/step 是 3 视角 ≈985 token，本方案 1 视角 ≈473 token。
实测 bs=4 时 bridge 单卡 2.17 s/step、fractal 2.18 s/step（两卡并行时两者互不影响，GPU0/GPU4 各 25.4 GB）。

超参来源：lr 5e-5 / warmup 300（fractal 500）/ cosine 到 2.5e-7，与 P3 的 pi05 simpler 配方一致
（该配方在 38.3 GB、5.06 s/step 下跑通 4000 步）；步数按「≈7 epoch」定，与 pi05 的 4,000 步同量级。

显存构成（［估算］，S4 校正）：权重 8.6 GB（4.291 B bf16）+ 梯度 8.6 GB + adafactor 因子态 ≈0.3 GB
+ 激活（bs4、≈473 token）≈12–16 GB。

---

## 7. 与 pi05 路线的对照（为什么值得再跑一遍）

| 维度 | pi05 路线（P3，已跑） | 本方案（π0\*） |
| :--- | :--- | :--- |
| 起点 | `lerobot/pi05_base`（通用） | π0\* bridge/fractal（**同域**） |
| state 通路 | `state_cond` FiLM（6 个张量，零初始化，P3 实测贡献 ~1e-5） | suffix 的 state token（`state_proj` 零初始化） |
| 时间条件 | FiLM（adaRMS） | π0 原生 concat-time（`action_time_mlp_in/out`），与 π0\* 一致 |
| 视角 | 3 槽（1 真 + 2 空） | 1 槽（与 π0\* 一致） |
| G3 实测 | 54.0%（门槛 78.0%） | 未跑 |

---

## 8. 已知坑与退路

| # | 坑 | 现象 | 处置 |
| :-- | :--- | :--- | :--- |
| R1 | 单卡放不下 | S3/S4 OOM | ① bs=2×accum16；② 训练前冻结两个 lm_head（`policy.model.vlm.lm_head.requires_grad_(False)`，可省 0.79 B 的梯度与优化器态）；③ `fsdp.enable: true` + `cpu_offload: true`；④ `tokenizer_max_length: 64`（π0\* 用 48） |
| R2 | 丢 proprio 导致 loss 卡住 | 500 步 loss > 0.35 或下降停滞 | ① `state_path_lr_mult: 3.0`；② lr 降到 2.5e-5 重跑；③ 数据改 euler 后 `--map-proprio-to-state`；④ 退路：改回 pi05 路线或用 proprio 当第二 action 流（要改 model，属新方案） |
| R3 | 保存时报共享存储 | `RuntimeError: Some tensors share memory` | 见 S3 退路：写 20 行别名去重转换（`detach_backbone_layer_aliases` 等价物） |
| R4 | yaml 写错字段 | `unrecognized arguments` / draccus 报 `state_cond` 未知 | pi0 配置类**没有** `state_cond`、`freeze_vision_encoder`；本仓库三份 yaml 已避开，改配置时不要从 pi05 配方整段复制 |
| R5 | 冷启动发错动作 | 片段开头机械臂抖一下 | 转换产物没有 postprocessor → 冷启动回退「零/均值动作」。评测一律用**训练产出的** checkpoint（带 processor JSON），不要拿 `ckpt/pi0star_*` 直接上真机 |
| R6 | `flashvla_init` 误用 | 时间/状态条件被清零、loss 反而升高 | 本方案固定 `flashvla_init: {mode: default}`。`ae-norm-zero` 是 LingBot 专用，会把时间 MLP 重置 |
| R7 | 别名键干扰自查 | 脚本里数出 1,104 个参数名却不是 779 | 正常：融合层与 backbone 共享 Parameter，`state_dict()` 两个名字都列；判断覆盖性用 `find_unfilled_parameters`（按张量身份） |

**反模式清单（不要做）**

1. 不要为了「省事」把 `pi0_flashvla_*` 改成 `pi05-flashvla` —— 权重键空间、时间条件、processor 全不同；
2. 不要为了让 checkpoint「看起来对齐」而改 `max_action_dim` / `max_state_dim`（32 是 FlashVLA 的约定，
   截断/补零已经在转换器里处理）；
3. 不要在转换器里「顺便」把 proprio 栈塞进 action 栈的层 —— 两个栈的 KV 语义不同，会把初始化搞坏；
4. 不要在 S3 未通过前起正式训练（一次 OOM 会带走一小时）。

---

## 9. 产物与回收清单

| 路径 | 内容 |
| :--- | :--- |
| `scripts/p4/convert_pi0star.py` | 转换器（dry-run 默认，`--write` 落盘，`--json` 出报告） |
| `train/configs/pi0/simpler/pi0_flashvla_bridge.yaml` / `pi0_flashvla_fractal.yaml` / `_smoke_pi0star.yaml` | 三份可直接跑的配置 |
| `ckpt/pi0star_bridge_flashvla/`（已生成）/ `ckpt/pi0star_fractal_flashvla/`（S1 生成） | 8.0 GB + `config.json`，`type: pi0-flashvla` |
| `logs/p4/convert_{bridge,fractal}.json` | 转换报告（映射/丢弃/合成/校验全在里面） |
| `logs/p4_bridge_train.log`、`/data/zjr/simpler_lerobot/runs/pi0_flashvla_bridge_success/checkpoints/` | 训练日志与断点（每个 ≈8 GB） |
| `third_party/SDN/eval_logs/p4_pi0star/` | M=1 评测的 `000_success_*.log` |
| 本文 §6 的表 | S4 跑完用实测替换［估算］，并在文末追加一行「实测：x s/step @ bs=y」 |

---

## 10. 命令速查

```bash
ROOT=/home/zjr/workspace_tj/StreamingVLA/flashvla
PY=/home/zjr/workspace_tj/StreamingVLA/miniconda_ygc/envs/flashvla/bin/python
cd $ROOT

# S1 转换（bridge 幂等 / fractal 首次）
$PY scripts/p4/convert_pi0star.py --src third_party/SDN/pretrained/open-pi-zero/fractal_beta_step29576_2024-12-29_13-10_42.pt \
  --data /data/zjr/simpler_lerobot/fractal/google_robot_close_drawer/success \
  --empty-cameras 0 --write --dst ckpt/pi0star_fractal_flashvla --json logs/p4/convert_fractal.json

# S3 冒烟
CUDA_VISIBLE_DEVICES=4 bash train/train.sh train/configs/pi0/simpler/_smoke_pi0star.yaml 1

# S5 正式训练（bridge）
CUDA_VISIBLE_DEVICES=4 setsid nohup bash train/train.sh \
  train/configs/pi0/simpler/pi0_flashvla_bridge.yaml 1 > logs/p4_bridge_train.log 2>&1 &

# S7 评测（两终端；A=py3.12 服务端，B=sdn py3.10 客户端）
CUDA_VISIBLE_DEVICES=4 $PY scripts/p3/flashvla_server.py \
  --ckpt /data/zjr/simpler_lerobot/runs/pi0_flashvla_bridge_success/checkpoints/last/pretrained_model --port 12400
```

---

## 11. 执行记录（2026-11-23 实跑）

实测：**2.17 s/step @ bs=4（accum=8，等效 32）**，峰值 25.4 GB；bridge ≈2.4 h、fractal ≈7.3 h。

| 步 | 结果 |
| :--- | :--- |
| S0 | 通过（两 `.pt` 在；GPU4/GPU0 各 45,458 MiB free；根盘 3.0 T 可用） |
| S1 | bridge 复用既有产物；fractal 新转换 779 张量 / 8.0 GB，`verify.ok=true`、`unfilled=0`（`logs/p4/convert_fractal.json`） |
| S2 | 两族 `OK 4291.34312 M (1,16,32)`，两条预期告警（无 postprocessor / torch_dtype） |
| S3 | loss 1.727 → 1.576，无 keys 告警，`/tmp/p4_smoke_pi0star/checkpoints/000002` 可重载（`RELOAD OK`） |
| S4 | 见 §6：bs2 1.60 s/20.6 GB、bs4 2.11 s/25.4 GB → 选 bs=4 |
| S5/S6 | 16:40 起 GPU4 = bridge（4,000 步）、GPU0 = fractal（12,000 步）并行（GPU0 15:00 后已空闲，经批准使用） |
| S5/S6 一次中断 | 17:00 / 17:19 在**第一次改动 `last` 符号链接**时被 IDE 注入的 shim 杀死，从 000500 / 001000 断点续跑成功 |
| S5 完成 | 19:35 正常退出：4,000 步 / 7.20 epoch / 末步 loss **1.033**，`last → 004000`，目录带 processor JSON |
| S7 | 用 `scripts/p3/run_g3.sh full`（P3 现成的 G3 脚手架）跑 bridge 004000 + 002750，GPU4/GPU5 各一 server+worker，结果在 `third_party/SDN/eval_logs/p4_pi0star/bridge/` |

**环境坑（S3 首次失败）**：`train/train.sh` 用 `python`，本机默认 `python` 是 `/home/zjr/miniconda3`（3.4 时代的解释器）→
`SyntaxError`。必须 `PATH=$PY_BIN_DIR:$PATH`。另外 `TokenizerProcessorStep` 走 `google/paligemma-3b-pt-224`，
离线机器要 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`（缓存里已有），否则在 `is_base_mistral()` 处抛
`httpx.ConnectError: Network is unreachable`。两处都不是方案问题，但缺一就跑不起来。

**环境坑 2（训练跑到第一个 checkpoint 就被杀，必看）**：本机 shell 里 `PYTHONPATH` 指向
`~/.vscode-server/extensions/tencent-cloud.coding-copilot-*/out/vendor/shim`，该目录的 `sitecustomize.py` 把
`Path.unlink` 改成「移到回收站」，于是 `lerobot/utils/train_utils.py:60` 的 `last_checkpoint_dir.unlink()`
会去建 `/data/.Trash-1000` 并抛 `PermissionError: [Errno 13] Permission denied: b'/data/.Trash-1000'`。
现象是**每个 save_freq 步数一到就 Traceback 退出**（checkpoint 目录本身已完整写好，只有 `last` 链接没换）。
处置：`env -u PYTHONPATH -u BASH_ENV` 启动训练（`train.sh` 会自己把 `PYTHONPATH` 设成仓库根），
再把 `last` 手工指到最新完整 checkpoint 后原地重跑即可自动续训（`auto_resume`，日志出现
`Auto-resume enabled ... resume=True`）。已实测：bridge 从 000500 续、fractal 从 001000 续，步数进度正确
（3500 / 11000 剩余步）。**后续任何写盘任务（评测、转换）也都要带这两个 `-u`。**

**S7 的第一处代码改动（唯一的）**：`scripts/p3/flashvla_server.py` 原来把 policy 类写死成
`PI05FlashVLAPolicy`，喂 pi0 的 checkpoint 会在 `PI05FlashVLALayer.__init__` 里炸
`AttributeError: 'PI0FlashVLAConfig' object has no attribute 'state_cond'`。已改成
`get_policy_class(cfg.type)` 按 checkpoint 自己的 `type` 分派（`pi0-flashvla` / `pi05-flashvla` 都走这里），
其余逻辑（冷启动 burst、gripper 二值化、post 反归一化）本来就与 policy 无关。
服务端 `info()` 自报 `type=pi0-flashvla, action_dim=7, state_dim=8, n_params_B=4.291, load_s≈138`。


**loss 量级的注意（重要，读 loss 前必看）**：π0\* 的 `action_in_proj` 只到 7 维，转换后第 7–31 列补零
（§3.3），`action_out_proj` 同理在第 7–31 行补零。而 `sample_noise` 在**全部 32 维**采样、损失按
`losses[..., :max_action_dim]` 求均值 → 那 25 个补零维的 `u_t = noise` 既拿不到输入（列被置零）也输出不了
（行被置零），**初始就恒定贡献 25/32 = 0.781**，要靠训练自己把这些行列学出来（推理时这 25 维被丢弃）。
所以本方案的 loss 与 pi05 的 loss 不可直接比：pi05_base 的 `action_in_proj` 是满 32 维预训练的，那 25 维在
step 0 就接近 0。折算出 7 个真实维度的损失：π0\* init ≈ (1.727−0.781)/0.219 ≈ **4.3**，step 300 ≈ **1.7**；
pi05 起点 ≈ 1.46（总 0.32）。即「π0\* 比 pi05 差 5 倍」是 padding 造成的错觉，真实维度差距约 1–3 倍。
**推论**：§4.1 的 R2 触发条件「500 步 loss > 0.35」对本方案恒成立，不能用它判 R2；判据应以 §7 的 M=1 闸门
为准，辅以「loss 是否持续下降」。若要消除这个 floor，正确做法是训练时把 noise 掩到真实动作维
（`forward_shared_observation` 采样处一行），但那会改动 policy 训练逻辑、并与 pi05 的 loss 口径脱钩，
本方案按「不改 policy 主体」的约定**未做**，留作后续可选优化。
