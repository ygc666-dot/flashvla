# FlashVLA 核心算法实现分析：多缓冲区联合微调 与 缓冲区流式推理

> 分析范围：`flashvla/flashvla/` 下 FlashVLA 算法本身（`datasets/`、`policies/pi05/modeling_pi05_flashvla.py`、`policies/pi05/utils.py`、`policies/pi05/patches.py`、`layers/attention.py`、`async_manager.py`、`configs/`）。不讨论基线 `modeling_pi05.py` 的训练细节、`sim_eval/` 的仿真适配与部署细节。
>
> 本文所有代码引用格式为 `起始行:结束行:文件路径`，路径相对仓库根 `/home/zjr/StreamingVLA`。运行示例采用 LIBERO 训练配置 `num_buffer_slots=4, chunk_size=10`（`flashvla/train/configs/pi05/libero/pi05_flashvla.yaml`）。

---

## 0. 一句话概括

FlashVLA 把 π0.5 从"一个观测 → 迭代 10 步 ODE 去噪 → 一整段动作 chunk"改造成 **"一个观测 → 只做 1 步欧拉积分 → 输出 1 段已完全去噪的 chunk"**：

- 维护 **N 个错位的动作缓冲槽**（`num_buffer_slots=N`），每槽 C 条动作（`chunk_size=C`），N 个槽分别停在流匹配时间网格 `t = 1/N, 2/N, …, 1` 上；
- 每次前向把 **N 个槽一次性全部推进一步**（每槽 `Δt = 1/N`），最"干净"的槽 0 刚好被推到 `t=0`，作为这次推理的输出；
- 随后整体左移一格（`slot i ← slot i+1`），最后一个槽重新灌纯噪声 —— 于是每个新噪声样本恰好经历 N 次前向、每次 `1/N` 步，最终到达 `t=0`。
- 训练时为了让模型见过"缓冲未填满"的所有情形，**一条观测同时展开 N 个缓冲配置（1 个真实槽、2 个真实槽、…、N 个真实槽）联合优化**，共用同一个 prefix（图像+语言），这即"多缓冲区联合微调"。

---

## 1. 缓冲区的时间语义与数据布局

### 1.1 t 的方向：0 = 动作，1 = 噪声

与 π0.5 完全一致（`flashvla/flashvla/policies/pi05/modeling_pi05.py:695-699` 的注释）：`x_t = t·noise + (1−t)·action`，`u_t = noise − action`。因此 **t 越大越像噪声**，欧拉更新 `x ← x − Δt·v` 是沿噪声→动作方向前进。

FlashVLA 的推理侧沿用同一约定：

```1009:1022:flashvla/flashvla/policies/pi05/modeling_pi05_flashvla.py
        time_per_slot = torch.arange(1, N + 1, device=device, dtype=torch.float32) / N
        time = time_per_slot.unsqueeze(0).unsqueeze(2).expand(bsz, -1, C).reshape(bsz, buf_len)

        v_t = self.denoise_step(
            prefix_pad_masks, prefix_att_masks, state,
            buffer, time, padding_mask=None,
        )

        dt = 1.0 / N
        buffer = buffer - dt * v_t
        actions_to_execute = buffer[:, :C, :self.config.max_action_dim]
        new_buffer = self.sample_noise((bsz, buf_len, self.config.max_action_dim), device)
        new_buffer[:, :-C, :] = buffer[:, C:, :]
        buffer = new_buffer
```

`slot 0` 的时间是 `1/N`（最干净），`slot N−1` 是 `N/N = 1`（纯噪声）。

### 1.2 三个维度的关系

| 符号                     | 含义                                    | 代码位置                                                      | LIBERO 取值 |
| :----------------------- | :-------------------------------------- | :------------------------------------------------------------ | :---------: |
| `N = num_buffer_slots` | 缓冲槽数，也等于流水线级数              | `flashvla/flashvla/policies/pi05/configuration_pi05.py:227` |      4      |
| `C = chunk_size`       | 每槽动作条数，也是每次输出的 chunk 长度 | `configuration_pi05.py:226`                                 |     10     |
| `H = N·C`             | 单配置的动作 token 数                   | `configuration_pi05.py:232-235`                             |     40     |
| `N·H = N²C`          | 训练时一个样本的 suffix token 总数      | `modeling_pi05_flashvla.py:738-739`                         |     160     |

注意 **训练时 suffix 长度是 `N·H = N²C`，不是 `H`**：因为有 N 个配置，每个配置自身占 H 个 token。

```737:739:flashvla/flashvla/policies/pi05/modeling_pi05_flashvla.py
        batch_size = states.shape[0]
        N, C = self.N, self.C
        num_offsets = N
        suffix_length = H = N * C
```

---

## 2. 多缓冲区联合微调

核心动机：流式推理会经历 **N 个不同的缓冲填满程度**（第 0 次推理只有 1 个真实槽，第 1 次有 2 个，…）。若只用"缓冲全满"训练，模型在这些中间状态上分布外。FlashVLA 的解法是**把 N 个填满程度做成同一 batch 内的 N 路并行分支**，一次前向全部监督。

### 2.1 数据侧：一条观测展开 N 个配置

`FlashVLADataset` 在 `LeRobotDataset.__getitem__` 的基础上把动作目标复制 N 份，第 k 份（1-based）只保留前 `k·C` 条真实动作，其余置零并标记为 pad：

```103:122:flashvla/flashvla/datasets/flashvla_dataset.py
        H_cfg = N * C

        actions_list = []
        action_is_pad_list = []

        for k_idx in range(N):
            k = k_idx + 1

            action_config = torch.zeros_like(full_actions)
            action_is_pad_config = torch.ones(H_cfg, dtype=torch.bool)

            real_len = min(k * C, full_actions.shape[0])
            action_config[:real_len] = full_actions[:real_len]
            action_is_pad_config[:real_len] = full_action_is_pad[:real_len]

            actions_list.append(action_config)
            action_is_pad_list.append(action_is_pad_config)

        result["action"] = torch.cat(actions_list, dim=0)
        result["action_is_pad"] = torch.cat(action_is_pad_list, dim=0)
```

要点：

1. **观测/语言/状态在 `result` 里只存一份**（`flashvla_dataset.py:96-98` 显式只拷贝 `observation.images.*`、`task`、`observation.state`、`episode_index`）。N 个配置共享同一观测，这是"联合"的前提。
2. `real_len = min(k*C, len)` 的 `min` 同时吸收了 **episode 末尾截断**：`full_action_is_pad` 由 LeRobot 在越界处标 True，和缓冲级 pad 合并到同一个 mask 里，模型只需处理一个 mask。
3. 因为 `PI05FlashVLAConfig.action_delta_indices = range(N·C)`（`configuration_pi05.py:247-249`），`full_actions` 天然就是 `N·C = 40` 步的动作窗口，所以 N 个配置的"真实前缀"是严格嵌套的：配置 k 的 40 个位置里前 `kC` 个是真动作。这与推理侧"噪声逐槽灌入、越靠前越干净"的物理过程对应（见 §3.2）。

### 2.2 模型侧：共享 prefix + N 路互不干扰的 suffix

`forward_shared_observation` 把 N 个配置拼进同一个 suffix 序列，长度 `num_offsets × H = N·H`，并且**只编码一次 prefix**：

```755:779:flashvla/flashvla/policies/pi05/modeling_pi05_flashvla.py
        real_mask = ~action_is_pad if action_is_pad is not None else None
        suffix_embs, suffix_pad_masks, suffix_att_masks, suffix_adarms_cond = self.suffix_embedder(
            states, x_t, time_per_token,
            padding_mask=real_mask,
        )
        hidden_dim = suffix_embs.shape[-1]

        suffix_pad_masks_per_offset = suffix_pad_masks.view(batch_size, N, H)

        suffix_att_masks = suffix_att_masks[:, :H]

        suffix_adarms_cond = suffix_adarms_cond.view(batch_size, N, H, -1)

        backbone_dtype = self.backbone_dtype()
        prefix_embs = prefix_embs.to(dtype=backbone_dtype)
        suffix_embs = suffix_embs.to(dtype=backbone_dtype)

        attention_mask, position_ids = self._build_shared_obs_mask(
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_masks=prefix_att_masks,
            suffix_pad_masks_per_offset=suffix_pad_masks_per_offset,
            suffix_att_masks=suffix_att_masks,
            num_offsets=num_offsets,
            dtype=prefix_embs.dtype,
        )
```

注意力拓扑由 `_build_shared_obs_mask` 决定，三件事同时完成：

```686:705:flashvla/flashvla/policies/pi05/modeling_pi05_flashvla.py
        cumsum = torch.cumsum(full_att_masks, dim=1)
        att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]

        pad_2d_masks = full_pad_masks[:, None, :] & full_pad_masks[:, :, None]
        att_2d_masks = att_2d_masks & pad_2d_masks

        suffix_positions = torch.arange(num_offsets * suffix_length, device=device)
        offset_ids = suffix_positions // suffix_length
        cross_offset_mask = offset_ids.unsqueeze(1) == offset_ids.unsqueeze(0)
        suffix_start = prefix_length
        att_2d_masks[:, suffix_start:, suffix_start:] = att_2d_masks[:, suffix_start:, suffix_start:] & cross_offset_mask

        position_ids = torch.zeros(batch_size, total_length, dtype=torch.long, device=device)
        prefix_pos = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1
        position_ids[:, :prefix_length] = prefix_pos
        last_prefix_pos = prefix_pos[:, -1]
```

1. **块因果（block-causal）**：`cumsum[j] <= cumsum[i]`，prefix 是 block 0，suffix 每 `C` 个 token 开一个新 block（由 `PI05SuffixEmbedder` 生成的 `att_masks` 决定，见下）。因此 prefix 只能看 prefix，suffix 的 block 只能看 prefix 与自身及以前的 block。
2. **跨配置屏蔽（`cross_offset_mask`）**：suffix 内的注意力只在同一配置内成立，N 个配置彼此不可见 —— 于是"一个 batch 里的 N 路分支"在数学上等价于 N 个独立样本，只是共享了 prefix 的 K/V。
3. **位置编码按配置重置**：每个配置的 suffix 位置 = `last_prefix_pos + 1 …`，由各自的 pad mask 累加得到，因此第 k 个配置的 token 起点与第 1 个完全相同。

`att_masks` 的块结构来自 suffix embedder，每 `C` 个 token 一个块首：

```262:265:flashvla/flashvla/policies/pi05/modeling_pi05_flashvla.py
        C = self.C
        bsz, L = noisy_actions.shape[:2]
        num_slots = L // C
        device = noisy_actions.device
```

```298:300:flashvla/flashvla/policies/pi05/modeling_pi05_flashvla.py
        att_row = torch.zeros(num_slots, C, dtype=suffix_embs.dtype, device=device)
        att_row[:, 0] = 1
        att_masks = att_row.flatten(0, 1).unsqueeze(0).expand(bsz, -1)
```

> 这里 `num_slots = L // C = N·N`，即"N 个配置 × 每配置 N 个槽"，块粒度正好是"一个缓冲槽"。

### 2.3 每槽独立采样流匹配时间（分段 + Beta）

这是联合微调最关键的一处设计：**同一样本里 N 个配置、N 个槽，每个槽有自己独立的 t，并且 t 的取值被约束在与推理栅格一致的区间内**。

```621:642:flashvla/flashvla/policies/pi05/modeling_pi05_flashvla.py
        seg_start = torch.zeros(N, num_slots_per_config, device=device, dtype=torch.float32)
        is_real = torch.zeros(N, num_slots_per_config, device=device, dtype=torch.bool)
        for k_idx in range(N):
            num_real = k_idx + 1
            for i in range(num_real):
                seg_start[k_idx, i] = (N - num_real + i) / N
                is_real[k_idx, i] = True

        time_per_slot = seg_start[None, :, :] + samples / N
        time_per_slot = (
            time_per_slot * self.config.time_sampling_scale
            + self.config.time_sampling_offset
        )
        time_per_slot = torch.where(
            is_real[None, :, :], time_per_slot, torch.ones_like(time_per_slot),
        )
        time_per_token = (
            time_per_slot.unsqueeze(-1)
            .expand(-1, -1, -1, C)
            .reshape(batch_size, -1)
        )
```

其中

- 配置 `k_idx`（0-based）有 `num_real = k_idx+1` 个真实槽；槽 `i` 的时间区间是 `[(N−num_real+i)/N, (N−num_real+i+1)/N]`，区间长度恰为 `1/N`；
- `samples ~ Beta(α=1.5, β=1.0)`，`time = seg_start + samples/N` → **在区间内随机取点**（不用固定端点，见后文原因）；
- 非真实槽 `t = 1.0`（纯噪声），随后 `x_t = noise`，再由 pad mask 屏蔽，不参与 loss；
- `scale/offset = 0.999/0.001` 避免恰好取到 0 或 1（与基线 `modeling_pi05.py:689` 一致）；
- 同一 `(config, slot)` 内的 C 个 token 共享同一个 t（`expand(-1,-1,-1,C)`），符合"一段 chunk 用同一个 t 去噪"的流匹配约束；不同 `(config, slot)` 之间互相独立，即使它们指向同一段 ground-truth 动作。

**与推理栅格的对齐关系**（这是理解全篇的钥匙）：

|                                         | 训练                           | 推理                           |
| :-------------------------------------- | :----------------------------- | :----------------------------- |
| 配置 `num_real = r` 时槽 `i` 的区间 | `[(N−r+i)/N, (N−r+i+1)/N]` | 取上端点 `(N−r+i+1)/N`      |
| 一次前向后槽 `i` 的状态               | x_t 落在区间内任意处           | 恰好落在下端点 `(N−r+i)/N`  |
| 全满（r=N）时槽 `i`                   | `[i/N, (i+1)/N]`             | 进入 `(i+1)/N`，离开 `i/N` |

也就是说：**推理是栅格上确定性的 1 步欧拉；训练把整条区间都监督到**，让模型对"偏离栅格的 x_t"也稳健（真实系统里 t=0 附近累积误差、以及 `n_action_steps < C` 时的时间重标定都会造成偏离）。这与 π0.5 训练"t ~ Beta 全程随机"的思路一致，只是把随机范围从 `[0,1]` 收窄到该槽所属的 `1/N` 区间。

### 2.4 逐 token 的 adaRMS 条件（架构改动）

π0.5 的 adaRMS 条件是 **每个样本一个向量** `[B, D]`；FlashVLA 需要 **每个 token 一个条件** `[B, L, D]`（因为同一序列里 40~160 个 token 分属不同 t）。

suffix embedder 把 time MLP 输出直接当作 per-token cond（`state_cond` 时再加状态项，广播到所有 token）：

```279:296:flashvla/flashvla/policies/pi05/modeling_pi05_flashvla.py
        time_emb = self.time_mlp_in(time_emb)
        time_emb = F.silu(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = F.silu(time_emb)

        adarms_cond = time_emb

        suffix_embs = self.action_in_proj(noisy_actions)

        if self.config.state_cond:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)
            state_emb = self.state_proj(state)
            state_emb = self.state_mlp_in(state_emb)
            state_emb = F.silu(state_emb)
            state_emb = self.state_mlp_out(state_emb)
            state_emb = F.silu(state_emb)
            adarms_cond = adarms_cond + state_emb.unsqueeze(1)
```

对应的 RMSNorm 必须支持 3D cond（per-token）。`FlashVLARMSNorm` 继承 `PiGemmaRMSNorm`，只在 `cond` 为 3D 时把调制apply到每 token：

```23:41:flashvla/flashvla/policies/pi05/patches.py
    def forward(self, x, cond=None):
        dtype = x.dtype
        normed_inputs = self._norm(x)

        if cond is None or self.dense is None:
            normed_inputs = normed_inputs * (1.0 + self.weight.float())
            return normed_inputs.to(dtype), None

        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dimension {self.cond_dim}, got {cond.shape[-1]}")

        modulation = self.dense(cond.to(self.dense.weight.dtype))
        if len(x.shape) == 3 and modulation.ndim == 2:
            modulation = modulation.unsqueeze(1)

        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        normed_inputs = normed_inputs * (1 + scale.to(torch.float32)) + shift.to(torch.float32)

        return normed_inputs.to(dtype), gate.to(dtype)
```

由于 N 个配置的 cond 不同、且 RMSNorm 是逐 token 的，`FlashVLAPI05ModelLayer` 在共享观测路径上把 suffix **折叠 batch 维**为 `[B·N, H, D]` 再逐 token 归一化，然后摊回 `[B, N·H, D]` 参与联合注意力：

```402:414:flashvla/flashvla/policies/pi05/modeling_pi05_flashvla.py
        suffix = hidden_states[1]
        hidden_dim = suffix.shape[-1]
        suffix_flat = suffix.view(batch_size * num_offsets, suffix_length, hidden_dim)

        cond_flat = suffix_adarms_conds.view(batch_size * num_offsets, suffix_length, -1) if suffix_adarms_conds is not None else None

        suffix_normed_flat, suffix_gate_flat = self.input_layernorm[1](suffix_flat, cond=cond_flat)

        suffix_normed = suffix_normed_flat.view(batch_size, num_offsets * suffix_length, hidden_dim)
        hidden_states[1] = suffix_normed

        suffix_gates = suffix_gate_flat.view(batch_size, num_offsets * suffix_length, -1)
        gates.append(suffix_gates)
```

### 2.5 损失与掩码

```783:816:flashvla/flashvla/policies/pi05/modeling_pi05_flashvla.py
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask,
                position_ids,
                suffix_adarms_conds=suffix_adarms_cond,
                num_offsets=num_offsets,
                suffix_length=suffix_length,
            )
        ...
        suffix_out = hidden_states[1]
        suffix_out = suffix_out.view(batch_size * N, H, hidden_dim)
        cond = suffix_adarms_cond.view(batch_size * N, H, -1)
        suffix_out, _ = norms[1](suffix_out, cond=cond)
        suffix_out = suffix_out.view(batch_size, N * H, hidden_dim)

        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        v_t = self.action_out_proj(suffix_out)

        losses = F.mse_loss(u_t, v_t, reduction="none")

        if action_is_pad is not None:
            loss_mask = ~action_is_pad
        else:
            loss_mask = torch.ones(batch_size, N * H, dtype=torch.bool, device=device)

        losses = losses[loss_mask]
```

- 只有 `action_is_pad=False` 的 token 进入 loss → **配置 k 只监督前 k·C 条动作**，pad 槽（t=1 的纯噪声输入）完全不贡献梯度；
- 由于 emb / cond 在 suffix embedder 里也被 `padding_mask` 清零（`modeling_pi05_flashvla.py:304-307`），pad token 既不进 attention 也不进 loss，是彻底"空"的；
- 损失是 **N 个配置的逐元素 MSE 混在一起取 mean**（`modeling_pi05_flashvla.py:1277`），等价于按"真实 token 总数"加权：填得越满的配置贡献越多，但每个 token 权重相同。

### 2.6 为什么要"联合"而不是"分 N 次微调"

1. **算力**：N 路配置共享一次 prefix 编码（视觉塔 + projector + 语言 embed），suffix 是唯一被复制 N 份的部分。分 N 次微调要多付 N−1 次视觉编码。
2. **一致性**：prefix 的 K/V 对 N 个配置是同一份，避免了"同一观测在不同缓冲阶段被编码成不同表征"的分布撕裂。
3. **梯度**：一次 `optimizer.step()` 同时收到 r=1…N 全部阶段的损失，不需要为不同阶段设计采样权重；`torch.where` + mask 保证各阶段互不干扰（跨配置注意力被 `cross_offset_mask` 切断）。

---

## 3. 缓冲区流式推理

### 3.1 状态定义与复位

策略持有**一个** buffer 与一个步数计数器：

```1132:1138:flashvla/flashvla/policies/pi05/modeling_pi05_flashvla.py
        buf_len = config.total_buffer_length
        self.register_buffer(
            "action_buffer",
            torch.zeros(1, buf_len, config.max_action_dim),
            persistent=False,
        )
        self._step_counter = 0
```

复位时"槽 0 = 噪声，其余 = 0（padding）"，与训练侧配置 `r=1` 的形状严格一致：

```1180:1190:flashvla/flashvla/policies/pi05/modeling_pi05_flashvla.py
    def reset(self):
        """Reset buffer and step counter. Call when environment resets.

        Buffer layout [noise, P, P, P, P]: slot 0 is noise, rest zeros (padding).
        """
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        C = self.config.chunk_size
        self.action_buffer.zero_()
        self.action_buffer[:, :C, :].normal_()
        self._step_counter = 0
        self._cold_start_action = None
```

每次 `predict_action_chunk` 只做"取 buffer → 前向 → 写回 buffer → 计数器 +1"：

```1207:1220:flashvla/flashvla/policies/pi05/modeling_pi05_flashvla.py
        actions, buffer, new_step, profile_results = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state,
            buffer=self.action_buffer,
            step_counter=self._step_counter,
            profile=profile,
        )

        self.action_buffer.copy_(buffer)
        self._step_counter = new_step

        if actions is None:
            if profile:
                return None, profile_results
            return None
```

### 3.2 冷启动：`_cold_start`

`step_counter < N−1` 时走冷启动：真实槽数 `num_real = step_counter+1` 逐步增长，时间取该配置对应区间的**上端点**，一次性把整个 buffer 推进 `1/N`：

```921:950:flashvla/flashvla/policies/pi05/modeling_pi05_flashvla.py
        num_real = step_counter + 1

        slot_idx = torch.arange(buf_slots, device=device)
        slot_idx_buf = torch.arange(buf_len, device=device) // C

        is_real = slot_idx < num_real
        time_slot = (N - num_real + 1 + slot_idx).to(torch.float32) / N
        time_slot = torch.where(is_real, time_slot, torch.ones_like(time_slot))

        padding_mask = is_real.unsqueeze(0).unsqueeze(2).expand(bsz, -1, C).reshape(bsz, buf_len)
        time = time_slot.unsqueeze(0).unsqueeze(2).expand(bsz, -1, C).reshape(bsz, buf_len)

        v_t = self.denoise_step(
            prefix_pad_masks, prefix_att_masks, state,
            buffer, time, padding_mask=padding_mask,
        )

        dt = 1.0 / N
        new_noise = self.sample_noise((bsz, buf_len, self.config.max_action_dim), device)
        updated_buffer = buffer - dt * v_t
        keep_mask = slot_idx_buf < num_real
        noise_mask = slot_idx_buf == num_real

        keep_mask = keep_mask.view(1, buf_len, 1)
        noise_mask = noise_mask.view(1, buf_len, 1)
        buffer = torch.where(
            keep_mask,
            updated_buffer,
            torch.where(noise_mask, new_noise, torch.zeros_like(updated_buffer)),
        )
```

用 `torch.where` 一次性完成三分类写入（**保留并更新** / **灌新噪声** / **清零**），因此每一次调用都只做一次前向、一次写回，没有 Python 分支导致的形状变化。注意 `num_real = s+1` 与 §2.3 的区间定义完全对应：第 s 次调用对应"配置 r = s+1"，槽 0 的进入时间是 `(N−s)/N`。

冷启动阶段**不产出可执行动作**（`sample_actions` 返回 `actions=None`），由策略层给出"保持不动"的兜底动作：

```1239:1257:flashvla/flashvla/policies/pi05/modeling_pi05_flashvla.py
        mode = getattr(self.config, "cold_start_mode", "zero_delta")
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, profile=False)
            if actions is None:
                cold_action = compute_cold_start_action(
                    mode=mode,
                    stats=self._cold_start_stats,
                    action_dim=self.config.action_feature.shape[0],
                    device=self.action_buffer.device,
                    state_normalized=batch[OBS_STATE] if mode == "current_state" else None,
                    cache=self._cold_start_action if mode == "zero_delta" else None,
                )
                if mode == "zero_delta":
                    self._cold_start_action = cold_action
                for _ in range(self.config.n_action_steps):
                    self._action_queue.append(cold_action)
            else:
                self._action_queue.extend(actions.transpose(0, 1)[:self.config.n_action_steps])
```

两种兜底语义（`flashvla/flashvla/policies/pi05/utils.py:200-321`）：

- `zero_delta`：反归一化后落在 **真实动作 0**（相对动作环境，等于"不动"），可以缓存；
- `current_state`：反归一化后落在 **当前真实 qpos**（绝对位置环境，如 RoboTwin，发 0 会撞零位），需要每步从 state 现算。

### 3.3 稳态：`_steady_streaming`（流水线式 1 步/次）

缓冲填满后，每次调用固定做一次"全槽推进 + 左移 + 尾部灌噪声"：

```1017:1022:flashvla/flashvla/policies/pi05/modeling_pi05_flashvla.py
        dt = 1.0 / N
        buffer = buffer - dt * v_t
        actions_to_execute = buffer[:, :C, :self.config.max_action_dim]
        new_buffer = self.sample_noise((bsz, buf_len, self.config.max_action_dim), device)
        new_buffer[:, :-C, :] = buffer[:, C:, :]
        buffer = new_buffer
```

把槽位时间记账写清楚（`N=4, C=10`，稳态每次调用）：

| 时刻               | slot 0             | slot 1             | slot 2             | slot 3        |
| :----------------- | :----------------- | :----------------- | :----------------- | :------------ |
| 本次调用输入 t     | 0.25               | 0.50               | 0.75               | 1.00          |
| 本次调用输出 t     | **0.00**     | 0.25               | 0.50               | 0.75          |
| 左移后新 slot 内容 | ← 旧 slot1 (0.25) | ← 旧 slot2 (0.50) | ← 旧 slot3 (0.75) | 新噪声 (1.00) |

**下一次调用输入恰好等于本次输出左移一格** → 表格的 "输出" 行与下一轮 "输入" 行完全相同，因此这个不变量在所有调用间都成立。

由此得到流水线的本质：每个噪声样本在尾部进入，被左移 N−1 次、累计接受 N 次 `Δt = 1/N` 的欧拉积分，最后一次走到 `t=0` 时正好位于 slot 0 被读出执行。**"10 步 ODE" 没有消失，它被摊平成了 4 次调用 × 1 步**，每次只付 1 步的代价。这也是插件式加速（不改 chunk 长度、不改重规划频率）的原因。

### 3.4 单步去噪：KV cache 复用

一次调用内部，prefix 只算一次并缓存 K/V，suffix 只算一次：

```841:869:flashvla/flashvla/policies/pi05/modeling_pi05_flashvla.py
        suffix_embs, suffix_pad_masks, suffix_att_masks, suffix_adarms_cond = self.suffix_embedder(
            state, x_t, timestep, padding_mask=padding_mask
        )

        backbone_dtype = self.backbone_dtype()
        suffix_embs = suffix_embs.to(dtype=backbone_dtype)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        full_attention_mask, full_position_ids = build_attention_mask_and_position_ids(
            pad_masks, att_masks, suffix_embs.dtype,
        )

        L_suf = suffix_embs.shape[1]
        attention_mask = full_attention_mask[:, :, -L_suf:, :]
        position_ids = full_position_ids[:, -L_suf:]

        hidden_states = [None, suffix_embs]
        conds = [None, suffix_adarms_cond]

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids, conds, use_cache=True)

        suffix_hidden = hidden_states[1]
        suffix_hidden, _ = self.action_expert.model.norm(suffix_hidden, cond=suffix_adarms_cond)

        suffix_out = suffix_hidden.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out)
```

配套的缓存机制非常轻量 —— 每个 `Attention` 模块自带两个张量，prefill 时 `clone` 保存 prefix 的 K/V，suffix 前向时 `cat` 上去：

```71:84:flashvla/flashvla/layers/attention.py
        if use_cache:
            if self.k_cache is None:
                self.k_cache = k.clone()
                self.v_cache = v.clone()
                k_full = k
                v_full = v
            else:
                k_full = torch.cat([self.k_cache, k], dim=2)
                v_full = torch.cat([self.v_cache, v], dim=2)
        else:
            k_full = k
            v_full = v
```

每次调用开头都会 `reset_cache()` 再 prefill（`modeling_pi05_flashvla.py:904-905`、`:993-994`），因为观测变了、prefix 必须重编码。**注意 FlashVLA 并不省 prefix 编码**，它省的是"同一 prefix 上的重复 suffix 前向"：

|                    | π0.5（每 chunk 一次推理）                                               | FlashVLA（每 chunk 一次推理）   |
| :----------------- | :----------------------------------------------------------------------- | :------------------------------ |
| prefix 编码/预填充 | 1 次（18 层）                                                            | 1 次（18 层）                   |
| suffix 前向        | **10 次**（各 10 token，串行，10 次 kernel 启动 + 10 次图/流边界） | **1 次**（160 token）     |
| 输出               | 最后一个 Euler step 的 x                                                 | slot 0 的 x（本轮刚被推到 t=0） |

suffix 的 FLOPs 量级接近（`10×C` vs `N²C`），但**串行步数从 10 降到 1**，在 batch=1 的延迟受限场景下这是决定性的（kernel launch / CUDA Graph replay / 同步次数都降一个数量级）。

### 3.5 调度分流与端到端接口

```1065:1086:flashvla/flashvla/policies/pi05/modeling_pi05_flashvla.py
        if buffer is None:
            buffer = self.sample_noise((bsz, buf_len, self.config.max_action_dim), device)

        if step_counter < N - 1:
            self._cold_step_t.fill_(step_counter)
            buffer, breakdown = self._cold_start(
                images, img_masks, tokens, masks, state, buffer, self._cold_step_t,
                profile=profile,
            )
            actions_to_execute = None
        else:
            actions_to_execute, buffer, breakdown = self._steady_streaming(
                images, img_masks, tokens, masks, state, buffer,
                profile=profile,
            )
```

`step_counter` 用 **0 维 int64 buffer** `_cold_step_t` 传入，而不是 Python int：这样 dynamo 只编译一张图、CUDA Graph 只需捕获一次就能跨所有 `step_counter` 值回放（注释见 `modeling_pi05_flashvla.py:875-880`）。`reset()` 里 `_step_counter=0` 时对应的 `_cold_step_t` 每步 `fill_`，是流式部署能稳定拿到低延迟的关键工程约束之一。

---

## 4. 异步重叠执行（`AsyncStreamingActionManager`）

FlashVLA 的"1 步/次"已经把单次推理压到很小，但要拿到论文的端到端加速，还必须把这次推理**藏进 `env.step` 的时间里**。这一层由 `flashvla/flashvla/async_manager.py` 实现，与模型解耦（对任何暴露 `predict_action_chunk` 的策略都能用）。

### 4.1 触发点：chunk 结束前 `overlap_steps` 步启动下一次

```113:119:flashvla/flashvla/async_manager.py
    def _should_launch_next(self) -> bool:
        """True when this call should start the next inference asynchronously."""
        return (
            self.overlap_steps > 0
            and self.chunk_index
            == self._skip_offset + self.n_action_steps - self.overlap_steps
        )
```

`overlap_steps=d` 即论文里的 `d`：当前 chunk 还剩 d 条动作时，用**当前观测**提前发起下一次推理。`d=0` 退化为同步。

### 4.2 非阻塞发射：GPU 张量留在显存里，CPU 只在切换时同步

```176:191:flashvla/flashvla/async_manager.py
    def _launch_inference(self, processed_obs: dict) -> torch.Tensor:
        """Run policy + handle cold-start fallback.

        Returns a GPU tensor ``[B, n_action_steps, action_dim]``. NOT synced
        to CPU here — caller decides when to ``.cpu().numpy()``.

        Under torch.compile + CUDA graphs, this call is non-blocking: the
        compiled ``_steady_streaming`` is dispatched to the GPU and the
        Python returns immediately with a tensor reference. The GPU work
        completes in the background; subsequent ``.cpu().numpy()`` syncs.
        """
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(processed_obs)
```

`act()` 中只有"提升下一段为当前段"或"冷启动首段"才会 `.detach().cpu().numpy()`（真同步），而预发射路径只保存 GPU 张量引用：

```219:250:flashvla/flashvla/async_manager.py
        if not self.is_running():
            self.current_chunk = self._launch_inference(processed_obs).detach().cpu().numpy()
        elif self.chunk_index == 0:
            if self.next_chunk is not None:
                self._promote_next_chunk()
            elif self.overlap_steps == 0:
                self.current_chunk = self._launch_inference(processed_obs).detach().cpu().numpy()
            else:
                raise RuntimeError(
                    "Async overlap is enabled but next_chunk is None at "
                    "transition. Did _launch_inference fail silently?"
                )

        if self._should_launch_next():
            self.next_chunk = self._launch_inference(processed_obs)

        # Keep execution on CPU: moving the tiny action back to the policy GPU
        # would synchronize it behind an in-flight inference on the default
        # stream and defeat overlap.
        return self._pop_current_action()
```

最后那段注释是实践中的关键坑：**动作必须留在 CPU**。若把单步动作搬回 GPU 取用，就会排在后台推理之后同步，重叠收益全部丢失。

### 4.3 RTC 式重对齐：`skip_stale_actions`

提前 d 步算出的新 chunk，其前 d 条动作对应的是"已经被旧 chunk 执行过"的控制步。`skip_stale_actions` 在提升时跳过这 d 条：

```145:162:flashvla/flashvla/async_manager.py
    def _promote_next_chunk(self) -> None:
        """Make ``next_chunk`` current. Under skip_stale_actions, realign by
        resuming at index ``overlap_steps``: the chunk was launched that many
        steps early, so its first ``overlap_steps`` actions target control
        steps the outgoing chunk already executed. The consumption window
        [overlap_steps, overlap_steps + n_action_steps) draws on the policy's
        full chunk, which predict_action_chunk returns unsliced."""
        self.current_chunk = self.next_chunk.detach().cpu().numpy()
        self.next_chunk = None
        if self.skip_stale_actions and self.overlap_steps > 0:
            if self.current_chunk.shape[1] < self.overlap_steps + self.n_action_steps:
                raise ValueError(
                    "skip_stale_actions needs overlap_steps + n_action_steps <= "
                    f"chunk length; got chunk length {self.current_chunk.shape[1]} "
                    f"< {self.overlap_steps} + {self.n_action_steps}"
                )
            self.chunk_index = self.overlap_steps
            self._skip_offset = self.overlap_steps
```

约束 `overlap_steps + n_action_steps <= chunk_size`（`async_manager.py:73-81`）保证跳过后仍有 `n_action_steps` 条可用。评测里由环境变量 `SKIP_STALE_ACTIONS=1` 打开（`sim_eval/libero/eval.py:137`）。

### 4.4 warmup：把编译/CUDA Graph 捕获挪出 rollout

第一次进入编译后的 `_steady_streaming` 大约发生在第 `(N−1)·n_action_steps` 步，若不预热，会阻塞 10–30 s：

```276:278:flashvla/flashvla/async_manager.py
        if num_steps is None:
            n_buf = getattr(self.policy.config, "num_buffer_slots", None)
            num_steps = (n_buf + 1) if n_buf is not None else 3
```

```285:292:flashvla/flashvla/async_manager.py
        for _ in range(num_steps):
            with torch.inference_mode():
                _ = self.policy.predict_action_chunk(processed_obs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        self.reset()
```

预热后立刻 `reset()`，把 buffer 和计数器打回初始态，正式 rollout 仍从冷启动开始，但图已捕获完毕。

### 4.5 时序图

```
控制步        0    1    2    3    4    5    6    7    8    9   10
             |---- chunk A (n_action_steps=5) ----|---- chunk B ----|
推理           [launch A]                [launch B]         [launch C]
                                        (d=1: 在 A 剩 1 步时发射)
GPU 时间线       ▓▓▓A▓▓▓ (与 env.step 重叠)  ▓▓▓B▓▓▓ ...
CPU 取动作       A0..A4 从 numpy 缓存取         B0..B4
```

在 FlashVLA 的模型侧，`[launch B]` 同时也把 buffer 推进了一步并把新噪声灌入尾部 —— 也就是说**流式解码的时钟与 chunk 切换时钟是同一个时钟**。

---

## 5. 训练/推理一致性检查表

| 环节          | 训练（`forward_shared_observation`）                             | 推理（`_cold_start` / `_steady_streaming`） |     是否一致     |
| :------------ | :----------------------------------------------------------------- | :---------------------------------------------- | :---------------: |
| 观测共享      | 1 个 prefix，N 个 suffix 分支                                      | 1 个 prefix，1 个 suffix（N 槽拼接）            |        ✅        |
| 槽内 token 数 | C                                                                  | C                                               |        ✅        |
| 每槽时间      | `seg_start + U(0,1)/N`（区间内随机）                             | 区间上端点（栅格）                              | 训练覆盖推理点 ✅ |
| 空槽语义      | `action_is_pad=True`（emb/cond 清零、attention 屏蔽、loss 屏蔽） | `padding_mask` 同语义；冷启动额外把未用槽填 0 |        ✅        |
| 因果结构      | 块因果（每 C 个 token 一块）+ 跨配置隔离                           | 块因果（每槽一块），无跨配置概念                |        ✅        |
| 时间条件注入  | per-token adaRMS（`[B, L, D]`）                                  | per-token adaRMS（`[B, buf_len, D]`）         |        ✅        |
| 初始状态      | 配置 r=1：1 个真实槽 + 噪声                                        | `reset()`：slot0 噪声、其余 0                 |        ✅        |

一致性最强的证据是 §2.3 的区间推导：**推理的进入时间恰好是训练区间的右端点，离开时间恰好是左端点**，而训练在该区间内随机采样，因此推理路径上出现的每一个 `x_t` 都在训练分布内（边缘处还有些许 `scale/offset` 收缩）。

---

## 6. 关键设计取舍与易踩的坑

1. **ODE 时钟 = 推理调用时钟，而不是控制步时钟。**
   `_steady_streaming` 每次调用只推进 `Δt=1/N`，而调度层每 `n_action_steps` 步才调用一次（`async_manager.py:139-143`）。当 `n_action_steps < chunk_size` 时（LIBERO 评测为 `C=10, n_action_steps=5`），一次调用产出的 C 条动作只被消费前 `n_action_steps` 条，剩余部分丢弃，同时 buffer 已经左移。也就是说**时间轴是按"推理次数"而非"控制步数"计量的**；若把 `n_action_steps` 调到远小于 `C`，等价于把流匹配的时间精度稀释（每次推进仍 1/N，但两槽之间实际间隔了更多控制步）。要严格保持"每控制步一步欧拉"，需要 `n_action_steps == chunk_size`（配置默认值正是 `n_action_steps=10=C`）。
2. **冷启动有 `N−1` 次"空转"。** 前 `N−1` 次调用返回 `None`，机器人执行 hold 动作。episode 短或频繁 reset 时，这段开销占比会明显上升（`reset()` 每 episode 至少一次）。这是"padding 冷启动"相较于"完整 ODE 冷启动"的固有代价，但把代价从"逐步 ODE"降到了"逐步灌噪声"。
3. **`flashvla_init.mode=ae-norm-zero` 只用于从零训练的场景**（`configs/train_config.py:36-50`、`train.py:218-302`），会重置 time MLP 与 action-expert adaRMS 的调制投影。微调/续训时必须保持 `default`，否则会破坏已有 checkpoint 的时间条件通路。
4. **跨配置共享同一份 noise。** `noise = self.sample_noise(actions.shape)` 只采一次（`modeling_pi05_flashvla.py:742-743`），N 个配置在相同 token 位置用同一噪声。因为注意力被 `cross_offset_mask` 切断，这只造成训练信号的轻微相关，不影响正确性；但如果要做"每个配置独立噪声"的消融，这里是改动点。
5. **`_build_shared_obs_mask` 与 `utils.build_shared_obs_attention_mask_and_position_ids` 不是同一份实现。** 后者（`utils.py:566-651`）在 pi05 FlashVLA 路径里并未被调用，实际生效的是模型内部的 `_build_shared_obs_mask`（多了 per-config pad mask 的 reshape 与位置铺排）。阅读时不要被 utils 里那份"看起来很像"的函数误导。
6. **KV cache 是模块属性而非 registered buffer**（`layers/attention.py:42-43`），所以不会被 `.to(device)`/`state_dict` 管理，也不参与 FSDP 分片；必须每次调用前 `reset_cache()`，否则会跨观测串味。
7. **`profile=True` 的分段计时在 CUDA Graph 回放下无意义**：`_steady_streaming` 的 docstring 明确说明此时应由外层测量墙钟（`modeling_pi05_flashvla.py:968-974`）。benchmark 时别把图内 event 当作真实分段耗时。

---

## 7. 与 π0.5 的逐项对照

| 维度                | π0.5（`modeling_pi05.py`）               | FlashVLA（`modeling_pi05_flashvla.py`）                             |
| :------------------ | :------------------------------------------ | :-------------------------------------------------------------------- |
| 每次推理的 ODE 步数 | 10（`dt = -1/num_steps` 从 t=1 积分到 0） | **1**（每槽 `Δt = 1/N`）                                     |
| 时间条件            | 每样本标量 → `[B, D]` adaRMS             | **每 token** → `[B, L, D]` adaRMS                            |
| suffix 长度         | `C`                                       | `N·C`（推理）/ `N²·C`（训练）                                  |
| 训练目标            | 单段动作 + 随机 t                           | **N 个缓冲配置 × 每配置 N 槽的独立 t**                         |
| 冷启动              | 无（10 步 ODE 本身就是冷启动）              | `N−1` 次 hold 动作 + 逐槽灌噪声                                    |
| 输出的 chunk        | 最后一个 Euler step 的结果                  | slot 0 被推到 `t=0` 的结果                                          |
| 额外状态            | 无                                          | `action_buffer`、`_step_counter`、KV cache、`_cold_start_stats` |
| 优化手段            | QKV/gate-up 融合、CUDA Graph                | 同上 + 静态形状 + tensor 化 step_counter 的单图捕获                   |

---

## 8. 代码地图

| 关注点                                   | 位置                                                                           |
| :--------------------------------------- | :----------------------------------------------------------------------------- |
| 超参定义（N、C、H、`cold_start_mode`） | `flashvla/flashvla/policies/pi05/configuration_pi05.py:218-249`              |
| 训练数据：N 个配置的构造                 | `flashvla/flashvla/datasets/flashvla_dataset.py:75-123`                      |
| 训练数据：多任务拼接                     | `flashvla/flashvla/datasets/flashvla_dataset.py:208-321`                     |
| per-token 时间采样                       | `flashvla/flashvla/policies/pi05/modeling_pi05_flashvla.py:592-642`          |
| 共享观测注意力 mask                      | `modeling_pi05_flashvla.py:644-714`                                          |
| 联合微调前向 + loss                      | `modeling_pi05_flashvla.py:716-816`                                          |
| per-token adaRMS 层                      | `modeling_pi05_flashvla.py:364-453` + `policies/pi05/patches.py:20-47`     |
| 单步去噪（KV cache）                     | `modeling_pi05_flashvla.py:819-869` + `flashvla/layers/attention.py:21-84` |
| 冷启动                                   | `modeling_pi05_flashvla.py:872-965`                                          |
| 稳态流式解码                             | `modeling_pi05_flashvla.py:967-1037`                                         |
| 策略状态/复位/兜底动作                   | `modeling_pi05_flashvla.py:1120-1257`                                        |
| 异步重叠调度                             | `flashvla/flashvla/async_manager.py:30-297`                                  |
| 冷启动动作反归一化                       | `flashvla/flashvla/policies/pi05/utils.py:200-321`                           |
| 训练入口（数据集选择、N 的注入）         | `flashvla/flashvla/train.py:334-391`、`:864-874`                           |

> 其余三个骨干（`pi0`、`smolvla`、`lingbot` 的 `modeling_*_flashvla.py`）共用同一套 `_cold_start` / `_steady_streaming` / `sample_actions` / `reset` 结构，差异仅在 prefix 编码器与 suffix 拼接方式；本文以 `pi05` 版本为准。
