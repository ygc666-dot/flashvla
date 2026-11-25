# R1 执行结果：SDN 原样复现的环境与单卡验证（2026-11-19 实测）

> 本文从 `sdnrun.md` 的 R1 附录摘出单独成文，内容是 `sdn_r1_env.md` 那套方案的**执行结果摘要**，
> 供原样复现参考。
>
> 关联：`sdn_r1_env.md`（R1 环境配置方案）、`sdnrun.md`（R3：π0.5 + LIBERO，与本文环境完全隔离）、
> `env_sdn.txt`（`conda list` 快照）、`third_party/SDN`（submodule，本文阶段只读不改）。

R1 用 `sdn` 环境（py3.10 + torch 2.3.1+cu118 + SIMPLER），R3 用 flashvla 环境（py3.12 + torch 2.9.1），
两套**完全隔离**：`transformers 4.44.2` 与 lerobot 的依赖树冲突，且 cu118 / cu128 运行时不同。

---

## 1. 最终版本清单

| 组件 | 版本 | 备注 |
| :--- | :--- | :--- |
| conda env | `sdn`，Python 3.10.22 | 位于 `miniconda_ygc/envs/sdn`，体积 8.4 GB |
| torch / torchvision | 2.3.1+cu118 / 0.18.1+cu118 | `torch.cuda.is_available()=True` |
| tensorflow | 2.15.0（CPU wheel） | 仅用于 `simpler_adapter.py:41` 的 `tf.io.gfile` |
| sapien | 2.2.2 | 需 `libvulkan.so.1`（conda-forge `vulkan-loader`） |
| mani_skill2_real2sim | 0.5.3（`-e`，纯 Python） | 资产随 submodule，无需下载 |
| groundingdino / SAM-2 | 0.1.0（`-e`，无 CUDA 扩展）/ 1.0 | 无 nvcc → 纯 PyTorch deformable-attention fallback |
| transformers / timm | 4.44.2 / 0.9.10 | `tokenizers 0.19.1`、`huggingface_hub 0.26.3` |
| numpy / opencv | 1.24.4 / 4.11.0.86 | 按 `requirements.txt` 钉死 |
| bitsandbytes | 0.44.1 | **必须钉版本**，见 §2 |

---

## 2. 必须偏离原方案的 5 处（踩坑记录）

1. **`sdn_r1_env.md` §2 的 `grep -rl bin/ | xargs sed -i` 会打坏 conda**：`bin/` 下 27 个 ELF
   （`python3.14`、`ld`、`xmllint`、`dbus-*`、`gettext-*`…）内嵌旧 prefix，替换后长度 +10
   导致段表偏移错位 → `bin/python -V` SIGSEGV。已按字节回退恢复；更安全的做法是建软链
   `ln -s /home/zjr/workspace_tj/StreamingVLA /home/zjr/StreamingVLA`，零文件改动。
2. **`tensorflow[and-cuda]==2.15.0` 已装不上**：`tensorrt-libs==8.6.1` 从 PyPI 下架，
   extra 解析必然失败。改装 `tensorflow==2.15.0`（CPU）；SIMPLER 只用 TF 读文件，不需 GPU。
3. **`bitsandbytes` 未钉版本会把 torch 顶到 2.14.1 + CUDA 13**：`pip install -r requirements.txt`
   时最新版 `bitsandbytes` 拉起 `cuda-toolkit-13.0.3.0`，cu118 栈被整体替换。修法：
   `pip install bitsandbytes==0.44.1` → 卸载 `cuda-toolkit/cuda-bindings/cuda-pathfinder` 与
   `nvidia-*cu13*` → 重装 `torch==2.3.1 torchvision==0.18.1`（cu118 index）→
   `pip install --force-reinstall --no-deps nvidia-nccl-cu11==2.20.5`（被 cu13 卸载清空，
   不补则 `ImportError: libnccl.so.2`）。
4. **SAPIEN 缺 `libvulkan.so.1`**：系统无 vulkan loader → `conda install -n sdn -c conda-forge vulkan-loader -y`。
   ICD 用系统自带的 `/etc/vulkan/icd.d/nvidia_icd.json`（driver 580.105.08）。
5. **pip 26 不再自带 `pkg_resources`**：`sapien/core/renderer_config.py` 会 import 它 →
   `pip install setuptools==75.8.0`（<81 才保留 `pkg_resources`）。

LaMa（`negative_mode=inpaint` 才需要）按原方案跳过，`negative_mode='zeros_bbox'` 用不到。

---

## 3. 权重（最小集，共 ≈36 GiB / 38.1 GB）

```
pretrained/open-pi-zero/bridge_beta_step19296_2024-12-26_22-30_42.pt    11773024888 B
pretrained/open-pi-zero/fractal_beta_step29576_2024-12-29_13-10_42.pt   11774183772 B
pretrained/paligemma-3b-pt-224/                                         11 GiB
pretrained/grounding-dino-base/                                        1.8 GiB
pretrained/sam2.1_hiera_large.pt                                       857 MiB
```

`uniform` 两个权重（各 11.77 GB）与 `big-lama` 未下，做消融时再补。
下载走 HF 官方 + 代理 `http://127.0.0.1:7890`，实测 ≈6.8 MB/s，总耗时约 1h50m；
`huggingface-cli` 支持断点续传。`pretrained/` 被 `third_party/SDN/.gitignore` 覆盖。

---

## 4. 验收（全部通过）

- `import torch, tensorflow, sapien, groundingdino, sam2, transformers, timm, mani_skill2_real2sim` 全 ok
- 两个 pi0 `torch.load(..., weights_only=False)`：`keys=8`，顶层 `['cnt_update','cnt_batch','model', …]`
- `python -m policy.pizero.evaluate --help` 参数解析正常
- `third_party/ManiSkill2_real2sim/data/custom/models` 53 项资产生效
- 环境快照：`docs/sdn_flashvla/sdn/env_sdn.txt`（**不要**写进 `third_party/SDN/`，避免 gitlink 漂移）

---

## 5. 单卡验证的执行约定（GPU 分配变更）

原方案约定只在 **GPU5** 上跑推理（`sdnrun.md` §2.5）。实测 GPU5 被其他进程占用
（14 GB 已用，且不稳定），**改为固定使用 GPU4**：

```bash
conda activate sdn
export CUDA_VISIBLE_DEVICES=4          # 原为 5，其余不变
cd /home/zjr/workspace_tj/StreamingVLA/flashvla/third_party/SDN
python -m policy.pizero.evaluate \
  --method vanilla --task widowx_carrot_on_plate \
  --num-gpus 1 --n-trajs 2 \
  --checkpoint pretrained/open-pi-zero \
  --result-root ./eval_logs/smoke
```

`--num-gpus` 只是 episode 分片数，单卡传 1 即可；先用 `--method vanilla` 打通
「模型加载 → 环境 reset → 渲染 → 推理 → 视频落盘」，再上 `--method sdn`。

---

## 6. 单卡验证结果（GPU4，2026-11-19 13:10–13:14，已通过）

两次冒烟都在 **GPU4**（`nvidia-smi` 编号 4，L40S 46068 MiB）上跑，`n-trajs=2`
（seed 0/1），任务 `widowx_carrot_on_plate`，均 **2/2 成功**、每 episode 60 步。

| 步骤 | 命令要点 | 结果 |
| :--- | :--- | :--- |
| ① vanilla | `--method vanilla --num-gpus 1 --n-trajs 2 --no-video` | 13:10:39 Run Info（`gpus: 4`）→ 13:11:24 与 13:11:32 两条 episode 全 True，`000_success_1.0.log`；约 54 s |
| ② sdn | `--method sdn`（其余同上） | 12:36 起模型/负样本构建 → 13:13:05、13:13:30 两条 episode 全 True；约 55 s |

命中的配置（从日志读回，说明负样本与打分链真的在跑，不是退化成 vanilla）：

```
negative_mode: zeros_bbox
num_samples: 12   knn_k: 6   top_m: 3
```

日志里的警告都在预期内，不是故障：

- `sam2` 的 `cannot import name '_C'` + "Skipping the post-processing step"
  —— 无 nvcc、走纯 PyTorch fallback（对应 §2 修正 1 / 原方案坑 1），不影响结果；
- `Memory efficient kernel not used because ... dtype: float` —— SAM2 用 fp32 推理的 SDPA 提示；
- `TF-TRT Warning: Could not find TensorRT` —— TF 是 CPU wheel。

资源实测：**显存峰值约 12.8 GB**（采样值 12841 MiB），比 `sdn_r1_env.md` §7 风险 1 估的
20–30 GB 低得多；单 episode（60 步、N=12）walltime 约 30 s，说明 N=12 的一次前向
批量采样确实被 `num_repeats` 摊平了，**不需要为了显存而降 N**。

缺口与遗留：

- 正式基线（9 任务 × 50 episode，450 episode）已于同日 13:29–20:20 跑完，完整统计见 **`results.md`**（整体 67.3%）。

- 机器上没有 `ffmpeg`，两条冒烟都用 `--no-video`（GIF 落盘未验证）。要出视频先
  `conda install -n sdn -c conda-forge ffmpeg -y`。
- 本次只跑 2 条 episode × 1 个任务，**不能当作论文数字**；正式基线仍是
  `policy/pizero/scripts/run_sdn.sh` 的 9 任务 × 50 trajs（多卡分片），
  按 §5 把 `CUDA_VISIBLE_DEVICES` 改为 4 即可。
- 产物在 `third_party/SDN/eval_logs/`（已 gitignore，子模块 `git status` 仍干净）：
  `smoke/vanilla/seeds=0-1/widowx_carrot_on_plate/`、
  `smoke_sdn/sdn/seeds=0-1/widowx_carrot_on_plate/`，各含 `000_episodes.json`、
  `000_episode_stats.json`、`000_success_1.0.log`。
