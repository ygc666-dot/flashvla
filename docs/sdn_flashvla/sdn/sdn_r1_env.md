# R1 环境配置方案：SDN 原样复现（pi0 + SIMPLER）

> 目标：按 `third_party/SDN/README.md` 的原样流程，把 SDN 论文数字（π0\* + SDN on SIMPLER）跑出来，
> 作为融合工作的「可信基线」。
>
> 本阶段只做 **CPU + 网络** 部分：conda 环境、依赖、权重下载、静态验证。
> GPU 推理留到资源放开之后（只准用 GPU5，见 `sdnrun.md` §2.5）。

---

## 0. 与 R3 的隔离关系（别搞混）

| | R1（本文） | R3（`sdnrun.md`） |
| :--- | :--- | :--- |
| backbone | π0\*（open-pi-zero 重实现） | π0.5（lerobot） |
| 基准 | SIMPLER（Bridge / Fractal） | LIBERO |
| 环境 | **py3.10 + torch 2.3.1+cu118** | py3.12 + torch 2.9.1+cu128（已就绪） |
| 代码位置 | `third_party/SDN/`（submodule） | `flashvla/flashvla/policies/pi05/` |
| 复用 | 不复用 flashvla 的 conda 环境 | 不复用 SDN 的环境 |

**两个环境完全隔离**，不要试图合并：`transformers 4.44.2` vs lerobot 0.5.1 的依赖树冲突，
且 torch 2.3.1 与 2.9.1 的 CUDA 运行时不同（cu118 vs cu128）。

---

## 1. 资源盘点（实测）

| 项 | 数值 | 说明 |
| :--- | :--- | :--- |
| 磁盘可用 | 3.3 TB | `/home/zjr` 所在卷，充足 |
| 最小权重集 | **38.1 GB** | beta 权重 ×2 + PaliGemma + DINO + SAM2 |
| 全量权重集 | 62.0 GB | 再加 uniform 权重 ×2 与 LaMa |
| conda 环境 | ~10 GB | torch cu118 + tensorflow + 仿真栈 |
| HF 带宽（走代理） | **5.2 MB/s** | 实测 `huggingface.co` 大文件 |
| hf-mirror 带宽 | 4.5 MB/s | 直连，无优势，且 gated 模型仍需官方 token |
| 下载耗时估算 | 最小集 **≈2.1 h**，全量 **≈3.4 h** | 按 5.2 MB/s |

结论：**用 HF 官方 + 代理**（`export https_proxy=http://127.0.0.1:7890`），不要用镜像
——gated 模型（PaliGemma）在镜像上认证会绕路，且速度没更快。

---

## 2. Step 0：修 conda（必须先做）

当前 `conda` 的 shebang 指向已不存在的 `/home/zjr/StreamingVLA/miniconda_ygc/bin/python`
（缺 `workspace_tj`，68 个 `bin/` 脚本 + `conda.sh` 4 处受影响）。**不修就无法 `conda create`**。

```bash
M=/home/zjr/workspace_tj/StreamingVLA/miniconda_ygc

grep -rl "/home/zjr/StreamingVLA/miniconda_ygc" "$M/bin/" \
  | xargs sed -i 's|/home/zjr/StreamingVLA/miniconda_ygc|/home/zjr/workspace_tj/StreamingVLA/miniconda_ygc|g'
sed -i 's|/home/zjr/StreamingVLA/miniconda_ygc|/home/zjr/workspace_tj/StreamingVLA/miniconda_ygc|g' \
  "$M/etc/profile.d/conda.sh"

source "$M/etc/profile.d/conda.sh"
conda --version          # 期望输出 conda 版本号
```

这一步只改 shebang 里的路径字符串，不改任何 Python 包，**不影响已可用的 `flashvla` 环境**。
若不想动 conda，也可以直接用 `$M/bin/conda` 的绝对路径，但后续 `activate` 仍不可用。

---

## 3. Step 1：创建独立环境

```bash
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890

conda create -n sdn python=3.10 -y
conda activate sdn
python -V          # 期望 Python 3.10.x
which python       # 期望 .../envs/sdn/bin/python
```

环境名固定 `sdn`，与 `flashvla`（py3.12）并存，互不干扰。

---

## 4. Step 2：依赖安装

原始脚本在 `third_party/SDN/scripts/install_dependencies.sh`，按原样执行即可：

```bash
cd /home/zjr/workspace_tj/StreamingVLA/flashvla/third_party/SDN
bash scripts/install_dependencies.sh
```

但有 **4 个坑**必须先知道：

### 坑 1：不要安装 CUDA toolkit（本机没有 nvcc，这是好事）

```78:104:third_party/SDN/third_party/grounded_sam_2/grounding_dino/setup.py
    if CUDA_HOME is not None and (torch.cuda.is_available() or "TORCH_CUDA_ARCH_LIST" in os.environ):
        extension = CUDAExtension
        ...
            "-gencode=arch=compute_86,code=sm_86",
    else:
        define_macros += [("WITH_HIP", None)]
        extra_compile_args["nvcc"] = []
        return None
```

两点判断：

1. `CUDA_HOME is None`（本机无 `nvcc`、无 `/usr/local/cuda`）时直接 `return None`，
   **不编译任何扩展**，GroundingDINO 走纯 PyTorch 的 deformable-attention fallback，功能完整。
2. 即使装上 nvcc，它只编 `sm_70/75/80/86`，而本机是 **L40S（sm_89）**——编了也用不上。
   `sam2/setup.py` 同理（同样有 `CUDA_HOME is None` 的跳过分支）。

所以 **不要** `conda install cuda-toolkit`，让两个包走 CPU 分支，省 3 GB 且避免兼容问题。

### 坑 2：TensorFlow 是硬依赖，不是可选项

`install_dependencies.sh` 会装 `tensorflow[and-cuda]==2.15.0`（SIMPLER 的图像预处理用到）。
它会拉一套 cu11 的 `nvidia-*` pip 包，与 torch 2.3.1+cu118 的依赖版本恰好一致，通常能共存。
若出现 `nvidia-cudnn-cu11` 版本冲突，先装 torch 再装 tensorflow（脚本顺序已如此）。

### 坑 3：`ManiSkill2_real2sim` 是纯 Python，但资产在仓库里

`third_party/ManiSkill2_real2sim/setup.py` 只有 `setup(...)`、无 `ext_modules`，不需要编译。
它的 325 MB 3D 资产（`data/hab2_bench_assets`、`data/custom/models`）**已随 submodule 拉下来**，
不需要额外下载——这是当初 462 MB 仓库体积的来源。

### 坑 4：LaMa 只有 `negative_mode inpaint` 才需要

`pip install -r third_party/inpaint_anything/lama/requirements.txt` 会装一套 inpainting 依赖。
SDN 默认 `negative_mode='zeros_bbox'`（`policy/pizero/config.py`），**第一轮可跳过**这步以省时间，
真要做 inpaint 消融时再补。

### 验收（装完立刻查）

```bash
python -c "
import torch, tensorflow as tf, sapien, groundingdino, sam2
print('torch', torch.__version__)
print('tf   ', tf.__version__)
"
python -c "import mani_skill2_real2sim; print('maniskill ok')"
```

无 GPU 也能跑这三行（只 import，不占显存）。`sapien` 是 ManiSkill2 的仿真后端。

---

## 5. Step 3：权重下载（38 GB 起）

### 5.1 先看清单：哪些必需

`pizero_model.py` 按 `(policy_setup, flow_sampling)` 从目录里挑文件：

```14:19:third_party/SDN/simpler_env/policies/pizero/pizero_model.py
    ('widowx_bridge', 'beta'): 'bridge_beta_step19296_2024-12-26_22-30_42.pt',
    ('widowx_bridge', 'uniform'): 'bridge_uniform_step19296_2024-12-26_22-31_42.pt',
    ('google_robot', 'beta'): 'fractal_beta_step29576_2024-12-29_13-10_42.pt',
    ('google_robot', 'uniform'): 'fractal_uniform_step29576_2024-12-31_22-26_42.pt',
```

`config.py` 里 `flow_sampling='beta'` 是默认值。`sdn.sh` 的 9 个任务同时覆盖 Bridge 与 Fractal，
所以**两个域的 beta 权重都要**：

| 文件 | 大小 | 必需 | 用途 |
| :--- | ---: | :--- | :--- |
| `bridge_beta_...pt` | 11.77 GB | ✅ | widowx_* 任务 |
| `fractal_beta_...pt` | 11.77 GB | ✅ | google_robot_* 任务 |
| `bridge_uniform_...pt` | 11.77 GB | ⬜ | uniform 消融 |
| `fractal_uniform_...pt` | 11.77 GB | ⬜ | uniform 消融 |
| `paligemma-3b-pt-224` | 11.72 GB | ✅ | 视觉塔/embedding 初始化（`vla/pizero.py:172` 会 glob 其 `*.safetensors`） |
| `grounding-dino-base` | 1.87 GB | ✅ | 负样本 bbox |
| `sam2.1_hiera_large.pt` | 0.94 GB | ✅ | 负样本掩码 |
| `big-lama` | ~0.2 GB | ⬜ | `negative_mode=inpaint` |

**最小集 38.1 GB**（前两行 + PaliGemma + DINO + SAM2）。

### 5.2 下载命令（最小集）

```bash
cd /home/zjr/workspace_tj/StreamingVLA/flashvla/third_party/SDN
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890
conda activate sdn          # huggingface-cli 由 requirements 里的 huggingface_hub 提供
mkdir -p pretrained

# ① pi0 权重：只取两个 beta（跳过 uniform，省 23.5 GB）
huggingface-cli download allenzren/open-pi-zero \
  bridge_beta_step19296_2024-12-26_22-30_42.pt \
  fractal_beta_step29576_2024-12-29_13-10_42.pt \
  --local-dir pretrained/open-pi-zero

# ② PaliGemma（gated，token 已验证可访问，HTTP 200）
huggingface-cli download google/paligemma-3b-pt-224 \
  --local-dir pretrained/paligemma-3b-pt-224

# ③ Grounding-DINO
huggingface-cli download IDEA-Research/grounding-dino-base \
  --local-dir pretrained/grounding-dino-base

# ④ SAM 2.1（不在 HF 上，走 fbaipublicfiles）
wget -P pretrained https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

`huggingface-cli` 支持断点续传（`.incomplete` 文件），中断后重跑同一条命令即可。

### 5.3 已验证的访问状态

| 仓库 | 状态 |
| :--- | :--- |
| HF 登录身份 | `GrankYu`（`~/.cache/huggingface/token` 存在） |
| `google/paligemma-3b-pt-224` | **200**，已接受 gated 协议，无需再点同意 |
| `IDEA-Research/grounding-dino-base` | 200 |
| `allenzren/open-pi-zero` | 无 `config.json`（404 属正常，是纯权重仓），文件列表已确认 4 个 `.pt` |

### 5.4 目录与 git 安全

`pretrained/` 已被 SDN 仓的 `.gitignore` 覆盖（`third_party/SDN/.gitignore` 首行 `pretrained/`），
62 GB 权重**不会**被误提交。`eval_logs/`、`results*/` 同样已忽略。

---

## 6. Step 4：无 GPU 也能做的验证

GPU5 现在只剩 2.7 GB（`sdnrun.md` §2.5），以下全部只跑 CPU/IO，不占显存：

```bash
cd /home/zjr/workspace_tj/StreamingVLA/flashvla/third_party/SDN

# 1) 依赖完整性
python -c "import torch, tensorflow, sapien, groundingdino, sam2, transformers, timm; print('deps ok')"

# 2) 权重文件齐全（大小对不上就是没下完）
du -sh pretrained/*

# 3) 权重可被 torch 读取（只读 header，不加载权重到显存）
python - <<'PY'
import torch
for f in ["pretrained/open-pi-zero/bridge_beta_step19296_2024-12-26_22-30_42.pt",
          "pretrained/open-pi-zero/fractal_beta_step29576_2024-12-29_13-10_42.pt"]:
    ckpt = torch.load(f, map_location="cpu", weights_only=False)
    print(f.split("/")[-1], "keys:", len(ckpt), "top:", list(ckpt)[:3])
PY

# 4) 评测脚本可解析参数（不构造 policy，不占显存）
python -m policy.pizero.evaluate --help | head -20

# 5) SIMPLER 资产在位
ls third_party/ManiSkill2_real2sim/data/custom/models | wc -l
```

第 3 步最关键：它能提前暴露「权重没下完 / 文件损坏 / torch 版本不兼容（pickle 协议）」三类问题，
而不需要等 GPU 空出来才发现。

---

## 7. 风险与对策

| # | 风险 | 影响 | 对策 |
| :--- | :--- | :--- | :--- |
| 1 | GPU5 显存不足（当前仅 2.7 GB） | 无法启动 π0\* 推理 | 先查占用进程；π0\* 3B + N=12 候选估计需 20–30 GB |
| 2 | `tensorflow[and-cuda]` 与 torch 的 `nvidia-*` 包冲突 | import 阶段报错 | 按脚本顺序（先 torch 后 tf）；冲突时 `pip install nvidia-cudnn-cu11==8.9.2.26` 对齐 |
| 3 | GroundingDINO 无 CUDA 扩展，检测变慢 | 每步多几十 ms | 可接受；不要为它装 toolkit（见坑 1） |
| 4 | 下载中断 | 38 GB 白下 | 用 `huggingface-cli`（支持续传），别用 `wget` 下 HF 大文件 |
| 5 | `conda create` 需要外网 | 建环境失败 | 先 `export https_proxy`，再确认 `conda config` 没有残留的旧代理设置 |
| 6 | SDN submodule 内做实验导致主仓 gitlink 漂移 | 主仓 `git status` 频繁出现 `modified: third_party/SDN` | R1 阶段只读不改 SDN 源码；若必须改，按 `docs/gitdocs/11_19.md` 的双层提交流程先推子仓 |

---

## 8. 执行顺序（可直接照抄）

```bash
# ---------- 0. 修 conda ----------
M=/home/zjr/workspace_tj/StreamingVLA/miniconda_ygc
grep -rl "/home/zjr/StreamingVLA/miniconda_ygc" "$M/bin/" | xargs sed -i \
  's|/home/zjr/StreamingVLA/miniconda_ygc|/home/zjr/workspace_tj/StreamingVLA/miniconda_ygc|g'
sed -i 's|/home/zjr/StreamingVLA/miniconda_ygc|/home/zjr/workspace_tj/StreamingVLA/miniconda_ygc|g' \
  "$M/etc/profile.d/conda.sh"
source "$M/etc/profile.d/conda.sh"

# ---------- 1. 建环境 ----------
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890
conda create -n sdn python=3.10 -y && conda activate sdn

# ---------- 2. 依赖 ----------
cd /home/zjr/workspace_tj/StreamingVLA/flashvla/third_party/SDN
bash scripts/install_dependencies.sh

# ---------- 3. 权重（最小集 38 GB）----------
mkdir -p pretrained
huggingface-cli download allenzren/open-pi-zero \
  bridge_beta_step19296_2024-12-26_22-30_42.pt \
  fractal_beta_step29576_2024-12-29_13-10_42.pt \
  --local-dir pretrained/open-pi-zero
huggingface-cli download google/paligemma-3b-pt-224 --local-dir pretrained/paligemma-3b-pt-224
huggingface-cli download IDEA-Research/grounding-dino-base --local-dir pretrained/grounding-dino-base
wget -P pretrained https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# ---------- 4. 验证 ----------
du -sh pretrained/* && python -m policy.pizero.evaluate --help | head -5
```

第 2 步之后建议 `conda list > env_sdn.txt` 存档，便于复现。

---

## 9. GPU 空出来之后的第一步

不要直接跑 `sdn.sh`（默认 4 GPU / 50 trajs）。先做单卡冒烟：

```bash
conda activate sdn
export CUDA_VISIBLE_DEVICES=5
cd /home/zjr/workspace_tj/StreamingVLA/flashvla/third_party/SDN
python -m policy.pizero.evaluate \
  --method vanilla --task widowx_carrot_on_plate \
  --num-gpus 1 --n-trajs 2 \
  --checkpoint pretrained/open-pi-zero \
  --result-root ./eval_logs/smoke
```

先用 `--method vanilla`（不算 SDN 的 2N 候选），确认模型/环境/渲染都通，再上 `--method sdn`。
`--num-gpus` 只是 episode 分片数，单卡完全够用（见 `sdnrun.md` 记录的代码依据）。

---

## 10. 实际执行记录与方案修正（2026-11-19 实测）

环境已按上面流程装完（`sdn` = py3.10.22 / torch 2.3.1+cu118 / tf 2.15.0 / sapien 2.2.2）。
执行中发现 **5 处必须偏离原文**，后续复现请直接照本节，不要照 §8 的原始命令：

### 修正 1（重要）：§2 的 `grep -rl | xargs sed -i` 会破坏 conda 二进制

`$M/bin` 下有 **27 个 ELF 文件**（`python3.14`、`xz`、`xmllint`、`dbus-*`、`gettext-*`、
`x86_64-conda-linux-gnu-ld` 等）内嵌了旧 prefix 字符串。路径变长 10 字节后 sed 会重写整个字节流，
**ELF 段表偏移全部错位 → 段错误**（实测 `$M/bin/python -V` 直接 SIGSEGV）。

- 抢救：按字节把新串换回旧串（此时该文件**正好**恢复原长，如 `ld` 从 2946309 回到 2941304，
  与 `pkgs/ld_impl_linux-64-2.44-*/bin/` 一致），27 个文件全部恢复，`conda --version` 正常。
- 结论：**只对文本文件做 sed**，二进制一律不动。想改路径更稳的做法是建软链：
  `ln -s /home/zjr/workspace_tj/StreamingVLA /home/zjr/StreamingVLA`（旧 prefix 立刻有效，零文件改动）。

### 修正 2：`tensorflow[and-cuda]==2.15.0` 已装不上

`tensorrt-libs==8.6.1` 已从 PyPI 下架（现存只有 9.x），`[and-cuda]` extra 解析必然失败。
改用 `pip install tensorflow==2.15.0`（CPU wheel）。SDN 里 TF 只用在
`simpler_adapter.py:41` 的 `tf.io.gfile.GFile` 读数据集统计，**不需要 GPU**。
torch 自带的那套 `nvidia-*-cu11` 也因此不会与 TF 抢版本（§7 风险 2 实际不发生）。

### 修正 3：`bitsandbytes` 未钉版本会把 torch 顶到 2.14.1 + CUDA 13

直接 `pip install -r requirements.txt` 时，`bitsandbytes` 最新版拉起 `cuda-toolkit-13.0.3.0`，
并把 torch 升级成 `2.14.1`（cu13），cu118 栈被整体替换。必须：

```bash
pip install "bitsandbytes==0.44.1"      # 与 torch 2.3.1 匹配
pip uninstall -y cuda-toolkit cuda-bindings cuda-pathfinder   # 以及 nvidia-*cu13*
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
pip install --force-reinstall --no-deps nvidia-nccl-cu11==2.20.5   # 被 cu13 卸载清空，必须补回
```

最后一行不做会得到 `ImportError: libnccl.so.2: cannot open shared object file`。

### 修正 4：SAPIEN 缺 `libvulkan.so.1`

系统 `/usr/lib64` 无 vulkan loader，`import sapien` 直接失败。装进本环境即可：

```bash
conda install -n sdn -c conda-forge vulkan-loader -y
```

### 修正 5：pip 26 不再自带 `pkg_resources`

`sapien/core/renderer_config.py` 里 `import pkg_resources`，而新 pip 不带 setuptools：

```bash
pip install "setuptools==75.8.0"   # <81 才有 pkg_resources
```

### 建议的实际安装顺序

```bash
# 1) conda（见修正 1：只改文本，或直接建软链）
# 2) torch cu118
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
# 3) TF CPU
pip install tensorflow==2.15.0
# 4) 仿真栈
pip install ruckig --only-binary=:all:
pip install -e third_party/ManiSkill2_real2sim
# 5) 负样本检测
pip install -e third_party/grounded_sam_2
pip install --no-build-isolation -e third_party/grounded_sam_2/grounding_dino   # 无 nvcc → 纯 PyTorch fallback
# 6) 主依赖 + 版本纠正
pip install -r requirements.txt
pip install "bitsandbytes==0.44.1"
pip uninstall -y cuda-toolkit cuda-bindings cuda-pathfinder
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
pip install --force-reinstall --no-deps nvidia-nccl-cu11==2.20.5
# 7) 系统级缺失
conda install -n sdn -c conda-forge vulkan-loader -y
pip install "setuptools==75.8.0"
```

### 验收结果（全部通过）

```
torch 2.3.1+cu118   cuda_avail True
tf    2.15.0 (CPU)  sapien 2.2.2
groundingdino ok    sam2 ok    mani_skill2_real2sim ok
transformers 4.44.2 timm 0.9.10  numpy 1.24.4  opencv 4.11.0.86
python -m policy.pizero.evaluate --help   # 参数解析正常
```

### 权重与环境存档（实测）

| 路径 | 实际大小 |
| :--- | ---: |
| `pretrained/open-pi-zero/`（bridge_beta + fractal_beta） | 22 GiB（各 11773024888 / 11774183772 B） |
| `pretrained/paligemma-3b-pt-224/` | 11 GiB |
| `pretrained/grounding-dino-base/` | 1.8 GiB |
| `pretrained/sam2.1_hiera_large.pt` | 857 MiB |
| 合计 | ≈ 36 GiB（对应清单 38.1 GB） |

两个 pi0 权重均通过 `torch.load(..., weights_only=False)` 校验：`keys=8`，
顶层键 `['cnt_update','cnt_batch','model', ...]`，无损坏、无 pickle 版本问题。

`conda list` 快照放在 `docs/sdn_flashvla/env_sdn.txt`（**不要**写进 `third_party/SDN/`，
否则子模块多出未跟踪文件，触发 §7 风险 6 的 gitlink 漂移）。
