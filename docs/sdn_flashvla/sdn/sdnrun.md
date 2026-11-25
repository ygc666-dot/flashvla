# R3 实施方案：在 π0.5 + LIBERO 上跑通 SDN（2026-11-19 已实测跑通）

> 目标：**不碰 SDN 仓的推理栈、不进 py3.10 环境、不装 SIMPLER**，用 flashvla 已有的
> `ckpt/pi05_libero` 与 `sim_eval/libero`，把 SDN 的「多噪声采样 + 选择」接到非流式 π0.5 上。
>
> 关联：`sdn.md`（融合方案与代价分析）、`code_analysis.md`（FlashVLA 实现分析）、
> `third_party/SDN`（submodule @ `d8d575b`，只读参考）、`sdn_r1_env.md` / `sdn_r1_result.md`（R1 环境与结果）。
> 代码引用格式 `起始行:结束行:文件路径`，路径相对 `flashvla/` 仓库根。

---

## 0. 结论先行

**现在就能跑，且已跑通。** 2026-11-19 16:40–17:35 在 GPU4 上与 R1 的 9×50 评测**并存**完成：

| 检查 | 结果 |
| :--- | :--- |
| 离线闸门（G1 逐位等价 / attach / 选择一致性） | **全部 PASS**，峰值 9.6 GiB，2 分钟 |
| 端到端冒烟 基线（libero_object 10 ep） | 80.0%，17.1 s/ep |
| 端到端冒烟 SDN N=8（同上） | 100.0%，41.0 s/ep（**2.40×**） |
| 对 R1 的影响 | 无中断，R1 正常从 `move_near` 推进到 `widowx_spoon_on_towel` |

但有 **3 个环境坑**和 **1 个代码阻塞点**必须先修，否则一启动就失败（全部已修，见 §3、§5）：

1. `CUDA_VISIBLE_DEVICES` 对 `eval.sh` **无效**（脚本内部按 `GPUS` 逐 job 重设），必须传 `GPUS=4`；
2. `~/.libero/config.yaml` 里写死的是旧前缀 `/home/zjr/StreamingVLA`（缺 `workspace_tj`），
   `bddl_files`/`init_states` 全 missing，**任何 LIBERO 评测（含基线）都起不来**；
3. tokenizer 要联网拉 `google/paligemma-3b-pt-224`，本机 shell 无网络 → 必须 `HF_HUB_OFFLINE=1`；
4. 原包装层返回 `nn.Module`，而 `eval.py` 有 `isinstance(policy, PreTrainedPolicy)` 硬校验 → `ValueError`。

---

## 1. 与 R1 的关系：能借什么、不能借什么

| 项 | 能否复用 | 说明 |
| :--- | :--- | :--- |
| **解释器 / conda 环境** | ❌ | `sdn`(py3.10, torch2.3.1+cu118) 与 `flashvla`(py3.12, torch2.9.1+cu128) 依赖树冲突，R3 一律用 `flashvla` |
| **`third_party/SDN/sdn/selection.py`** | ✅ | 纯 torch，backbone-agnostic；submodule 已初始化，直接 `sys.path` 引用 |
| **GroundingDINO 权重**（1.8 GiB） | ✅ | `pretrained/grounding-dino-base/` 是**标准 transformers 格式**（`model_type: grounding-dino`）；flashvla 的 transformers 5.3.0 直接 `GroundingDinoForObjectDetection` 可加载（已实测 import OK）→ M2 不需要跨环境 RPC |
| **SAM2 权重**（857 MiB） | ✅（暂不用） | `zeros_bbox` 模式不需要掩码 |
| **调好的超参** | ✅ | `N=12 / k∈{6,10} / M∈{3,5} / long_horizon=4 / negative_mode=zeros_bbox`（`run_sdn.sh`），R3 起步操作点可直接照抄 |
| **踩坑记录** | ✅ | 无 ffmpeg（要出视频先装）、GPU 分配变更、conda sed 打坏 ELF 的教训 |
| **`pretrained/paligemma-3b-pt-224/`** | ✅（备选） | 含 `tokenizer.model`，可作为 `HF_HUB_OFFLINE` 失败时的本地 tokenizer 兜底 |

一句话：**R1 给 R3 的是「资产 + 一个调好的操作点」，不是环境**。

---

## 2. 方案边界

**做**：同一观测采 N 个独立噪声 → N 个候选 chunk → 打分择优 → 执行选中 chunk。

**不做**（属 `sdn.md` 方案 1/2）：
- 不改 `modeling_pi05_flashvla.py` 的流式 buffer；
- 不做重噪候选（重噪是流式下的多样性折中，非流式直接用独立噪声，多样性满格）；
- 不引入接缝项与滞回（无跨 chunk 选择，不存在接缝）。

M0 只做 **Stage 2**（平滑度择优），Stage 1（kNN grounding）需要负观测，见 §8。

---

## 3. 环境：四处必须先对齐（全部为 2026-11-19 实测踩到）

### 3.1 绝对路径解释器（不要 `conda activate`）

```bash
cd /home/zjr/workspace_tj/StreamingVLA/flashvla
export PY=/home/zjr/workspace_tj/StreamingVLA/miniconda_ygc/envs/flashvla/bin/python
export PATH=/home/zjr/workspace_tj/StreamingVLA/miniconda_ygc/envs/flashvla/bin:$PATH
```

`PATH` 是必须的：`eval.sh` 内部直接调 `python` / `python3`（`sim_eval/libero/eval.sh:210,230`），
不 export 会落到系统解释器。**不要写进 `~/.bashrc`。**

### 3.2 坑 1（阻塞）：`CUDA_VISIBLE_DEVICES` 对 `eval.sh` 无效，必须用 `GPUS`

```127:129:sim_eval/libero/eval.sh
GPU_ARR=($GPUS)
N_GPUS=${#GPU_ARR[@]}
```

```210:212:sim_eval/libero/eval.sh
    CUDA_VISIBLE_DEVICES=$gpu python sim_eval/libero/eval.py \
```

`GPUS` 默认 `1`（`eval.sh:25`），每个 job 启动时**重新赋值** `CUDA_VISIBLE_DEVICES=$gpu`。
所以外面 `CUDA_VISIBLE_DEVICES=4 bash eval.sh ...` 会被覆盖成物理卡 1 —— 实测直接
`torch.AcceleratorError: CUDA error: out of memory`（卡 1 被 VLLM 占 41 GB，只剩 3.6 GB，
装不下 7 GB 权重）。

**正确写法**：`GPUS=4 ... bash sim_eval/libero/eval.sh 0 <seed>`。
（历史 `results_pi05_baseline` 不受影响，那时卡 1 是空的。）

### 3.3 坑 2（阻塞）：`~/.libero/config.yaml` 指向已不存在的旧前缀

症状（preflight 直接 exit 1）：

```
[preflight] ERROR: bddl_files dir missing: /home/zjr/StreamingVLA/miniconda_ygc/.../libero/libero/bddl_files
                  init_states dir missing: .../libero/libero/init_files
```

`eval.sh:67` 只在 config.yaml **不存在**时才引导生成（`eval.sh:69-88`），已存在就沿用坏路径。
修复（已执行，备份 `config.yaml.bak`）：

```bash
sed -i 's|/home/zjr/StreamingVLA/miniconda_ygc|/home/zjr/workspace_tj/StreamingVLA/miniconda_ygc|g' \
  ~/.libero/config.yaml
```

修后 `bddl_files` / `init_states` 均 exist=True。`assets` 仍不存在，但 `~/.cache/libero/assets`
（407 MB）已存在，`eval.sh:115` 的 preflight 只警告不报错，env 构建走缓存 —— 与基线跑通时行为一致。
若想彻底摆脱缓存依赖，可把 `assets` 指向 `SVLA/third_party/libero/libero/libero/assets`（405 MB）。

### 3.4 坑 3（阻塞）：tokenizer 联网

```
ValueError: Failed to instantiate processor step 'tokenizer_processor' ... 
Error: [Errno 101] Network is unreachable
```

`make_pre_post_processors` 会解析 `tokenizer_name: google/paligemma-3b-pt-224`。
本机 shell 无代理，但 HF 缓存已在（`~/.cache/huggingface/hub/models--google--paligemma-3b-pt-224`，
含 `tokenizer.json`），**加两个环境变量即可离线跑**：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

### 3.5 `compile_model`：ckpt 里是 `true`，R3 必须显式关掉

`ckpt/pi05_libero/config.json` 的 `compile_model=True` + `compile_mode=max-autotune`，
而 `modeling_pi05.py:581` 在 `__init__` 里就 `torch.compile` 了 `sample_actions`。
后果：**每出现一个新 batch size（=N×B）就重新 autotune 5–10 分钟**（实测 N=4 一个 batch size
把自检拖过 20 分钟超时）。

`eval.sh` 在 `OVERLAP_STEPS=0` 时自动 `COMPILE_MODEL=false`（`eval.sh:34-40`），所以
**R3 一律用 `bash sim_eval/libero/eval.sh 0 <seed>`**，不要用 overlap>0。
自检脚本也显式 `compile_model=False`（§6.1）。

### 3.6 三个依赖边界（别混）

| 组件 | 环境 | 说明 |
| :--- | :--- | :--- |
| flashvla + lerobot + π0.5 | `flashvla`（py3.12） | **R3 全部在这里跑** |
| `third_party/SDN/sdn/selection.py` | 无额外依赖 | 只 `import torch`，py3.12 直接 import |
| SDN 的 SIMPLER 评测栈 | py3.10（R1 在用） | R3 **完全不用** |
| GroundingDINO 权重 | 任一 | 权重格式通用，**推理放 py3.12**（transformers 5.3.0 原生支持） |

---

## 4. 落点：就地挂载，不替换 policy 对象

调度器对所有 policy 统一调用 `predict_action_chunk`：

```187:191:flashvla/async_manager.py
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(processed_obs)
```

π0.5 的 `config.type == "pi05"` 不在 `_is_streaming` 四类型里（`async_manager.py:85-90`），
走非流式分支，恰好是 SDN 需要的语义。

但 **不能返回包装对象**：

```287:290:sim_eval/libero/eval.py
    if not isinstance(policy, PreTrainedPolicy):
        raise ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )
```

`PreTrainedPolicy` 还带 5 个 `abc.abstractmethod` 和 `config_class`/`name` 元类约束
（`lerobot/policies/pretrained.py:78-90`），为包装去实现它们只会换来更多耦合。

**结论：用实例属性覆盖 `predict_action_chunk`** —— 类型、`config`、`parameters()`、
调度器路径全部不变，一行注入。

---

## 5. 代码现状（三处修复已完成）

| 文件 | 状态 |
| :--- | :--- |
| `flashvla/policies/pi05/sdn_selector.py` | 167 行。`SDNChunkSelector`（可独立用于离线测试）+ `attach_sdn()`（eval 入口用） |
| `sim_eval/libero/eval.py:535-547` | 改为调 `attach_sdn(...)`，仍由 `SDN_NUM_SAMPLES` 开关 |
| `scripts/sdn/verify_m0_offline.py` | 离线自检，**已跑通**，约 2 分钟 |

### 5.1 `attach_sdn`（`sdn_selector.py:149-167`）

```python
def attach_sdn(policy, **kwargs):
    selector = SDNChunkSelector(policy, **kwargs)

    def _predict_action_chunk(batch, noise=None, profile=False):
        return selector.predict_action_chunk(batch, noise=noise, profile=profile)

    policy.predict_action_chunk = _predict_action_chunk  # 实例属性优先于类方法
    return policy
```

### 5.2 本次修掉的三个 bug

1. **`isinstance` 阻塞**：原实现 `policy = SDNChunkSelector(policy, ...)`，
   `eval_policy` 立刻 `ValueError`。→ 改 `attach_sdn`。
2. **N=1 自递归**：`attach_sdn` 后 `SDNChunkSelector.predict_action_chunk` 里的
   `self.policy.predict_action_chunk` 会命中被替换的版本。→ `__init__` 里预先存
   `self._base_predict = policy.predict_action_chunk`（`sdn_selector.py:66-68`），N≤1 走它透传。
3. **`noise` 参数在 N>1 分支被忽略**：导致 G2（"N 个候选 == 同一噪声序列下独立跑 N 次"）
   无法校验，自检里 `selected == cand[argmin]` 恒为 False。→ 现在 `noise` 形状为
   `[N*B, C, D]` 时直接用，为 `[B, C, D]` 时复制 N 份（`sdn_selector.py:120-126`）。

### 5.3 eval 入口（`eval.py:535-547`）

```python
    if os.environ.get("SDN_NUM_SAMPLES"):
        # 就地挂载，不替换 policy 对象：eval_policy 有 isinstance(policy, PreTrainedPolicy) 校验
        from flashvla.policies.pi05.sdn_selector import attach_sdn

        attach_sdn(
            policy,
            num_samples=int(os.environ["SDN_NUM_SAMPLES"]),
            smoothness=os.environ.get("SDN_SMOOTHNESS", "acceleration"),
            long_horizon=(...),
        )
```

环境变量而非 config 字段：ckpt 的 `config.json` 里没这些键，走 dataclass 需要额外向后兼容；
不设 `SDN_NUM_SAMPLES` 时**行为完全不变**。

---

## 6. 实测数据（GPU4，与 R1 并存）

### 6.1 离线闸门（`scripts/sdn/verify_m0_offline.py`，compile=False，2026-11-19 17:0x）

```
G1 base(1, 5, 7) bitwise_equal=True max|diff|=0.000e+00
attach_sdn isinstance(policy, PreTrainedPolicy)=True
attach_sdn N=1 透传 out(1, 5, 7) finite=True same_as_base=True
N=1 baseline latency=338.9 ms/chunk
N= 4 latency= 431.7 ms  x1.27  peak_alloc=8.62 GiB
N= 8 latency= 617.6 ms  x1.82  peak_alloc=9.12 GiB
N=12 latency= 875.6 ms  x2.58  peak_alloc=9.60 GiB
candidate scores=[2.2532, 3.1375, 3.1259, 2.271]
argmin=0 candidate_std=0.11347
selected chunk == cand[argmin]: True
ALL CHECKS PASSED
```

三个结论：

1. **G1 成立**（`SDN_NUM_SAMPLES=1` 与基线逐位相同），后续数字可信；
2. **候选确实有分歧**（`candidate_std=0.113`，分数 2.25–3.14 不等），不是 N 份同样的东西；
3. **成本是次线性的**：N=12 只有 2.58×，因为 bsz=1 时 GPU 远未饱和，
   N 份前缀在同一 batch 里摊薄了。§7.3 旧版按 10–12× 估算是**高估**。

### 6.2 成本剖面（L40S 46 GB，batch_size=1）

| N | 单 chunk 推理 | 相对基线 | 显存峰值（含 7 GB 权重） |
| :--- | ---: | ---: | ---: |
| 1 | 339 ms | 1.00× | 8.6 GiB |
| 4 | 432 ms | 1.27× | 9.1 GiB |
| 8 | 618 ms | 1.82× | 9.1 GiB |
| 12 | 876 ms | 2.58× | 9.6 GiB |

端到端（含仿真/渲染，与 R1 并存时绝对值被拉高，**只有比值可比**）：

| 运行 | 成功率 | 单 episode |
| :--- | ---: | ---: |
| 基线（libero_object, 10 ep） | 80.0% | 17.1 s |
| SDN N=8（同上） | 100.0% | 41.0 s（**2.40×**） |

历史无争用基线（`results_pi05_baseline/ov0`，500 ep/seed）：`libero_object` 7.2–7.8 s/ep。
说明**与 R1 并存会让绝对值约 2.2×**，写论文数字必须等 R1 跑完再重跑。

### 6.3 现成可复用的基线锚点（省掉 N=1 那一臂）

`results_pi05_baseline/ov0/seed{1000..5000}/{suite}/eval_info.json`，每格 500 episode：

| suite | 500-ep 成功率（5 seed） | s/ep（无争用） |
| :--- | :--- | ---: |
| `libero_object` | 97.4 / 98.4 / 98.2 / 98.0 / 98.2 | 7.2–7.8 |
| `libero_spatial` | 98.0 / 96.6 / 97.2 / 97.2 / 97.4 | 7.3–8.2 |
| `libero_goal` | 96.2 / 94.8 / 95.0 / 96.0 / 95.6 | 6.3–6.8 |
| `libero_10` | 90.4 / 92.2 / 91.6 / 93.6 / 91.8 | 11.9–12.6 |

**R3 的 N=1 臂不需要重跑**，直接与这张表对照（同 `OVERLAP_STEPS=0`、同 `n_action_steps=5`、同 seed）。

---

## 7. 正式评测设计：先认天花板，再谈信号

### 7.1 天花板效应（必须先承认）

基线在 `libero_object`/`spatial` 已经 97–98%。SDN 的理论收益上限只有 2–3 个百分点，
而 50 episode 的二项 95% CI 半宽在 p=0.98 时约 ±1.8pp、在 p=0.92 时约 ±3.4pp。
**换句话说：在 object/spatial 上做 50 episode，结构上不可能看出差别。**

因此 `sdn.md` §7.1 那套「N 增大时成功率单调上升（G4）」在 R3 上要改成：

| 优先级 | 设计 | 理由 |
| :--- | :--- | :--- |
| **P0** | 管线 + 成本（§6.1/6.2），已完成 | 与成功率无关，先锁死正确性 |
| **P1** | `libero_10` × 50 ep × 3 seed（1000/2000/3000），N∈{1,4,8,12} | 唯一有空间的 suite（90–93.6%），CI 半宽 ±3.4pp，配 N 单调性看趋势 |
| **P2** | 失败救援：只看基线失败的 task | `eval_info.json` 有 `per_task`，可先挑 `pc_success` 最低的 3–4 个任务，用 `--env.task` 单跑，n 小但检出力最高 |
| **P3** | 对照臂（随机选 / 恒选第 0 个） | 证明「选择」而非「多采样」带来收益；随机选臂只需把 `smoothness` 换成一个返回 `randint` 的桩 |

### 7.2 P1 的命令（R1 跑完后执行，约 3.5 h）

```bash
cd /home/zjr/workspace_tj/StreamingVLA/flashvla
export PATH=/home/zjr/workspace_tj/StreamingVLA/miniconda_ygc/envs/flashvla/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

for N in 4 8 12; do
  GPUS=4 SDN_NUM_SAMPLES=$N SDN_LONG_HORIZON=4 SDN_SMOOTHNESS=acceleration \
  POLICY_PATH=ckpt/pi05_libero SUITES="libero_10" N_EPISODES=50 \
  OUTPUT_ROOT=results_sdn/p1_n$N \
  bash sim_eval/libero/eval.sh 0 1000
done
```

- 单格耗时 ≈ 50 ep × 12.2 s/ep × 2.4（N=8）≈ 25 min；三档 × 一个 seed ≈ 1.5 h，
  加两个 seed 共约 4.5 h。
- `SDN_LONG_HORIZON=4` 对齐 R1 调好的操作点（`run_sdn.sh` 的 `long_horizon=4`）；
  不设则用满 `chunk_size=50`，两者是不同假设，别混着比。
- `libero_10` 的 `n_action_steps` 仍是 5（`eval.sh:24` 默认），不要用 ckpt 里的 50。

### 7.3 与 R1 并行的资源约定（重要）

2026-11-19 实测快照（8×L40S，各 46 GB）：

| GPU | 已用 | 说明 |
| :--- | ---: | :--- |
| 0 | 17 MiB | 无人占用（**但归属未知，用前先确认**） |
| 1 / 3 / 6 / 7 | 41.8–42.5 GB | VLLM / 他人 |
| 2 | 15.2 GB | isaac-sim |
| **4** | **7.4–14.5 GB（R1，随任务阶段波动）** | **R1 正在跑 9×50，余量 31–38 GB** |
| 5 | 28.7 GB（util 100%） | streamvln-kb 他人 |

- R3 走 GPU4 与 R1 并存是**安全**的：N=12 峰值 9.6 GiB + 仿真/渲染 ~3 GiB ≈ 13 GiB，
  14.5（R1 上界）+ 13 = 27.5 GB < 46 GB，实测两次冒烟 R1 无异常（进程 2111755 持续推进）。
- **N=8 是并行期的推荐上限**（余量 > 20 GB）；N=12 建议等 R1 结束（约 3–5 h）再上，
  否则两边的 allocator 峰值叠加时容错变小。
- 并行期**不要用 R3 的 `eval_s` 当论文数字**（被 R1 拉高 2.2×，见 §6.2），只用来算比值。
- 每次启动前先确认 R1 还在：
  `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv`，
  GPU4 上应只有一个 ~14 GB 的 `python` 进程。

---

## 8. Stage 1（负观测）：R1 已经把料备好了

SDN 的 Stage 1 需要「物体被遮挡」的观测。R1 的下载让 M2 的成本大幅下降：

| 方案 | 做法 | 现在还剩什么成本 |
| :--- | :--- | :--- |
| **GroundingDINO + bbox 置零**（推荐） | `AutoModelForZeroShotObjectDetection.from_pretrained("third_party/SDN/pretrained/grounding-dino-base")`，在 py3.12 单进程内跑，把 bbox 区域置零得到 `neg_image`，再走一次 `sample_actions` 得 `neg_actions`，最后 `sdn_select(cand, neg_cand, knn_k, top_m, smoothness_horizon=long_horizon)` | **权重已下（1.8 GiB）、transformers 5.3.0 原生支持（已实测 import OK）、无需跨环境**。只剩接线 |
| GT 掩码 | 从 robosuite sim 取 instance segmentation | 需 hack lerobot 的 libero env 暴露 `env.sim` |
| 不做 Stage 1 | 只用 `select_by_smoothness`（M0 现状） | 0，但放弃 SDN 论文里 +13.33% 的主要来源 |

代价提醒：Stage 1 每次决策要**多跑一遍 N 候选的前向**（`neg_actions` 也是 N 份），
即成本再翻一倍（N=12 从 2.58× 变成约 5×），并且 DINO 每步几十 ms（无 CUDA 扩展，纯 PyTorch fallback）。
开工前先用 `sdn.md` §7.1 的 G4 判据做快速断言：**「看得到物体的那一路 ratio 才大」**。

---

## 9. 已知坑（累计）

1. **`GPUS` 而不是 `CUDA_VISIBLE_DEVICES`**（§3.2）——最容易静默踩到，落到 VLLM 卡上直接 OOM。
2. **别开 `compile` 和异步 overlap**：`eval.sh` 在 `OVERLAP_STEPS>0` 时默认 `COMPILE_MODEL=true`，
   而 N>1 每换一个 batch size 就 autotune 5–10 分钟（§3.5）。R3 用 `bash eval.sh 0 <seed>`。
3. **`n_action_steps` 必须显式是 5**：ckpt 里是 50，`eval.sh` 默认 5，忘了覆盖会让「执行 50 条」。
4. **N 份必须一次前向**（§5）：KV cache 是模块属性（`modeling_pi05.py:985-986` 每层 `reset_cache()`），
   循环调用会互相串味。
5. **`ignore_gripper=True` 依赖动作布局**：`smoothness_scores` 用 `actions[..., :-1]` 丢最后一维。
   LIBERO 7 维动作末位确实是夹爪（0/1），成立；换 benchmark 要重新确认。
6. **`smoothness` 的归一化是跨候选的**（`selection.py:54-80` 用 `std(dim=(0,1))`），
   N=1 时该分数恒为常数 —— 这就是 G1 用「透传」而不是「N=1 走打分」的原因。
7. **`jerk` 需要 `T>=4`、`acceleration` 需要 `T>=3`**（`selection.py:83-124`）。
   `SDN_LONG_HORIZON=4` 对两者都合法；调到 <4 会让 `jerk` 抛异常。
8. **CUDA Graph 会覆盖上一次输出**：若哪天开了 compile，任何「先算 A 再算 B 再比较 A」的测试
   都必须 `.clone()`，否则报 `accessing tensor output of CUDAGraphs that has been overwritten`。
9. **动作是 per-step delta**：LIBERO/π0.5 的 7 维动作是增量，`acceleration`（二阶差分）
   实际对应位置的三阶量。与 R1（pi0 + SIMPLER，同样是 delta）一致，故沿用；换 absolute 动作的
   benchmark 时要改用 `jerk(delta=False)` 或降阶。
10. **逐 episode 记录没落盘**：`eval_info.json` 只有 `per_task` 聚合，没有每个 episode 的
    seed/success。想跑 McNemar（G5）得先在 `eval.py` 落盘逐 episode 结果，否则只能比较均值 ± CI。

---

## 10. 回退

- 删掉 `eval.py` 里那段 `if os.environ.get("SDN_NUM_SAMPLES")` 即可完全回到基线；
  不设该变量时根本不挂载。
- `sdn_selector.py` 是新增文件，不影响任何现有 import。
- 基线锚点：`results_pi05_baseline/ov0`（5 seed × 4 suite × 500 ep），任何时候都能对照。

---

## 11. 一页速查

```bash
cd /home/zjr/workspace_tj/StreamingVLA/flashvla
export PATH=/home/zjr/workspace_tj/StreamingVLA/miniconda_ygc/envs/flashvla/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# ① 离线闸门（2 min，峰值 9.6 GiB）
CUDA_VISIBLE_DEVICES=4 python scripts/sdn/verify_m0_offline.py

# ② 端到端冒烟（各 2–6 min）
GPUS=4 POLICY_PATH=ckpt/pi05_libero SUITES="libero_object" N_EPISODES=1 \
  OUTPUT_ROOT=results_sdn/r3_smoke_base bash sim_eval/libero/eval.sh 0 1000

GPUS=4 SDN_NUM_SAMPLES=8 SDN_LONG_HORIZON=4 POLICY_PATH=ckpt/pi05_libero \
  SUITES="libero_object" N_EPISODES=1 \
  OUTPUT_ROOT=results_sdn/r3_smoke_n8 bash sim_eval/libero/eval.sh 0 1000

# ③ 正式对比（R1 结束后，§7.2）
```

---

## 12. R1（SDN + SIMPLER）相关内容

见 **`sdn_r1_env.md`**（环境方案）与 **`sdn_r1_result.md`**（最终版本清单、5 处必须偏离、
权重清单、GPU4 上的 2×2 冒烟结果与 12.8 GB 峰值）。`conda list` 快照见 `env_sdn.txt`。
