# P3 执行报告（2026-11-22 04:05）

日志目录：`/home/zjr/workspace_tj/StreamingVLA/flashvla/logs/p3_overnight`　GPU：`P3_GPU` 见各阶段日志　决策文件：`gates/decision.json`

## 结论

- T0 在带宽内、S0–S5 全部通过、训练已按量测配置启动。

## 1. T0（π0* 基线，vanilla / seeds 2000–2049）

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

## 2. S0（单卡路线：1 进程 FSDP2）

| 变体 | optimizer | cpu_offload | bs | 成功 | samples/s | s/step | loss |
| :--- | :--- | :-: | ---: | :-: | ---: | ---: | ---: |
| `adafactor` | adafactor | true | 2 | ✅ | 0.137 | 14.64 | 0.322 |
| `adafactor_no_offload` | adafactor | false | 2 | ✅ | 0.690 | 2.90 | 0.321 |
| `cpu_offload_adamw` | adamw | true | 2 | ✅ | 0.292 | 6.85 | 0.308 |

## 3. S1–S4（数据 / 单步 / 存重载 / 图像尺寸）

- 返回码：S1=0，S2=0，S3=0，S4=0（0 = 通过）

```
[cfg] policy=pi05-flashvla pretrained_path=/home/zjr/workspace_tj/StreamingVLA/flashvla/ckpt/pi05_base
[cfg] empty_cameras=2 num_buffer_slots=4 chunk_size=4 state_cond=True
[data] frames=50843 episodes=777 leaves=5
[dataset] observation.images.top: (3, 512, 640) torch.float32
[dataset] observation.state: (8,) torch.float32
[dataset] episode_index: () torch.int64
[dataset] action: (64, 7) torch.float32
[dataset] action_is_pad: (64,) torch.bool
[preprocessed] action: (1, 64, 7) torch.float32
[preprocessed] action_is_pad: (1, 64) torch.bool
[preprocessed] episode_index: (1,) torch.int64
[preprocessed] observation.images.top: (1, 3, 512, 640) torch.float32
[preprocessed] observation.state: (1, 8) torch.float32
[preprocessed] observation.language.tokens: (1, 200) torch.int64
[preprocessed] observation.language.attention_mask: (1, 200) torch.bool
[policy] image_resolution=(224, 224) -> 256 tokens/image, 768 visual tokens/frame
```

## 4. S5（bs 扫描）

| bs | optimizer | cpu_offload | 本进程峰值 MiB | 整卡峰值 MiB | samples/s | s/step | loss | ok |
| ---: | :--- | :-: | ---: | ---: | ---: | ---: | ---: | :-: |
| 2 | adafactor | false | 22166 | 22175 | 2.8944 | 0.6910 | 0.335 | True |
| 4 | adafactor | false | 32688 | 32697 | 4.1237 | 0.9700 | 0.317 | True |

```json
{
  "trainable": true,
  "bs": 4,
  "grad_accum_steps": 8,
  "optimizer": "adafactor",
  "cpu_offload": "false",
  "fsdp": "false",
  "throughput": 4.1237,
  "peak_mib": 32688,
  "budget_mib": 43410,
  "equiv_batch": 32,
  "runs": [
    {
      "batch_size": 2,
      "ok": true,
      "peak_mib": 22166,
      "samples_per_s": 2.894356005788712,
      "s_per_step": 0.6910000000000001,
      "loss": 0.335
    },
    {
      "batch_size": 4,
      "ok": true,
      "peak_mib": 32688,
      "samples_per_s": 4.123711340206186,
      "s_per_step": 0.97,
      "loss": 0.317
    }
  ]
}
```

## 5. 正式训练（P6）

- pid `1728007`，已退出
- loss 行数 10；前 3 行 / 后 3 行：
```
  step    50  loss 0.3430  updt_s 5.05
  step   100  loss 0.3110  updt_s 5.28
  step   150  loss 0.2640  updt_s 5.20
  ...
  step   300  loss 0.2100  updt_s 5.29
  step   350  loss 0.2000  updt_s 5.22
  step   400  loss 0.1910  updt_s 5.21
```

## 6. 阶段流水（main.log）

```
[11-21 21:51] == all requested stages finished
[11-21 22:03] == P4 S1-S4 冒烟
[11-21 22:08]    P4 }
[11-21 22:08] == all requested stages finished
[11-21 22:12] == P4 S1-S4 冒烟
[11-21 22:18]    P4 }
[11-21 22:18] == P3 S0：单进程 FSDP2 三条路线（cpu_offload+adamw / adafactor+offload / adafactor 纯卡）
[11-21 22:19]    S0 cpu_offload+adamw 失败
[11-21 22:19]    S0 adafactor(+offload) 失败
[11-21 22:19]    S0 adafactor 纯卡 失败
[11-21 22:22] == P3 S0：单进程 FSDP2 三条路线（cpu_offload+adamw / adafactor+offload / adafactor 纯卡）
[11-21 22:22]    S0 cpu_offload+adamw 失败
[11-21 22:22]    S0 adafactor(+offload) 失败
[11-21 22:23]    S0 adafactor 纯卡 失败
[11-21 22:23] == all requested stages finished
[11-21 22:32] == P3 S0：单进程 FSDP2 三条路线（cpu_offload+adamw / adafactor+offload / adafactor 纯卡）
[11-21 22:37]    S0 cpu_offload+adamw OK
[11-21 22:37]    S0 adafactor(+offload) 失败
[11-21 22:37]    S0 adafactor 纯卡 失败
[11-21 22:37] == all requested stages finished
[11-21 23:37] == P5 S5 bs 扫描（bs ∈ 1 2 4)
[11-21 23:46] == all requested stages finished
[11-21 23:48] == P5 S5 bs 扫描（bs ∈ 1 2 4)
[11-21 23:58] == all requested stages finished
[11-22 00:09] == P2 T0 基线（9 任务 x 50 seeds，幂等跳过已完成任务）
[11-22 00:27] == P6 起正式训练（gate 未过就不起）
rendered /home/zjr/workspace_tj/StreamingVLA/flashvla/train/configs/pi05/simpler/_overnight.generated.yaml
  batch_size       = 4
  grad_accum_steps = 8  (equiv batch 32)
  optimizer.type   = adafactor
  fsdp.cpu_offload = False
  pretrained_path  = /home/zjr/workspace_tj/StreamingVLA/flashvla/ckpt/pi05_base
  steps            = 4000  save_freq=250
[11-22 00:27] == P6 已启动 pid=1728007，配置 /home/zjr/workspace_tj/StreamingVLA/flashvla/train/configs/pi05/simpler/_overnight.generated.yaml
[11-22 00:27] == all requested stages finished
[2026-11-22 01:14:30] training pid 1728007 exited, starting T0
[11-22 01:14] == P2 T0 基线（9 任务 x 50 seeds，幂等跳过已完成任务）
macro 0.6689 failed []
[11-22 04:05] == all requested stages finished
[11-22 04:05] == P7 报告（已完成 7 个阶段）
```
