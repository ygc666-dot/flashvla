# Git 上传记录（2026-11-18）

## 一、以后修改目录后的上传方式

日常只需四条命令，`.gitignore` 已经挡掉了全部大文件，`git add -A` 是安全的：

```bash
cd /home/zjr/workspace_tj/StreamingVLA/flashvla

git status                    # 1. 看改了什么
git add -A                    # 2. 暂存（ckpt/results/logs 已被忽略）
git status --short            # 3. 复核：确认列表里没有大文件
git commit -m "改动说明"       # 4. 提交
git push                      # 5. 推送到 origin/main
```

### 提交前必做的两步自查

```bash
# 确认待提交文件里没有 ckpt / results / logs
git diff --cached --name-only | grep -E "^(ckpt|results|results_|logs)/"

# 确认没有混入权重类二进制
git diff --cached --name-only | grep -E "\.(safetensors|pt|pth|bin|ckpt|onnx|h5|pkl)$"
```

两条命令都输出「无，干净」才 commit。GitHub 单文件上限 100MB，`ckpt/` 里的
`model.safetensors` 是 8.0G / 7.0G，误提交后 push 一定失败且清理历史极麻烦。

### 常用补救

```bash
# 误把大文件加进暂存区（还没 commit）
git restore --staged ckpt/

# 检查某路径为何被/没被忽略
git check-ignore -v ckpt/pi05_libero/model.safetensors

# 确认远端最新提交
git ls-remote origin
git log --oneline -5
```

### 必须记住的三条约束

1. **远端主机名是 `github-ygc`，不是 `github.com`**。这是 `~/.ssh/config` 里的别名，
   指向 `ygc666-dot` 账号。任何时候手工拼 URL 都要写 `git@github-ygc:ygc666-dot/xxx.git`。
2. **不要用 `git add -f` 强行加 `ckpt/`**。需要分享权重时走 GitHub Release 附件
   （单个 2GB 上限）或 Git LFS（免费额度 1GB，7G+8G 必超额）。
3. **`upstream` 指向原项目 `z-lab/flashvla`**，仅作留档参考；本仓库历史已与上游切断，
   不要 `git pull upstream`，否则会把上游历史重新混进来。

---

## 二、仓库现状

| 项         | 值                                                           |
| ---------- | ------------------------------------------------------------ |
| 本地路径   | `/home/zjr/workspace_tj/StreamingVLA/flashvla`             |
| 远端仓库   | https://github.com/ygc666-dot/flashvla                       |
| 当前分支   | `main` → `origin/main`                                  |
| 初始提交   | `5176df8` Initial commit: FlashVLA（单条提交，无上游历史） |
| 远端文件数 | 110 个，约 2.3 MB                                            |
| origin     | `git@github-ygc:ygc666-dot/flashvla.git`                   |
| upstream   | `https://github.com/z-lab/flashvla`（留档，不再使用）      |

本地保留但**未上传**的内容（共约 15.8G）：

- `ckpt/` —— 15G，含 `flashvla-pi05-libero/model.safetensors`（8.0G）、
  `pi05_libero/model.safetensors`（7.0G）
- `results/`、`results_pi05_baseline/`、`results_source_free/` —— 约 768M，
  绝大部分是 `*.log`，少数 `eval_info.json`
- `logs/` —— 全部 `*.log`

---

## 三、本次操作回顾

### 1. 现状盘点

该目录**原本就是一个 git 仓库**（不是普通目录），`origin` 指向上游官方仓库
`https://github.com/z-lab/flashvla`，`main` 与 `origin/main` 同步，94 个跟踪文件，
`.git` 仅 1.3M。未提交改动为 `sim_eval/libero/eval.sh`（+63 行），未跟踪目录为
`ckpt/`、`results*/`、`docs/`、`logs/`。

### 2. 扩充 `.gitignore`

原文件只忽略了 `checkpoints/`（带 s），**没有 `ckpt/`**，直接 `git add -A` 会把 15G
权重加进去。现按「模型检查点 / 运行结果 / 日志」三类补全：

```gitignore
# Python
__pycache__/
*.pyc
*.egg-info/
build/
dist/

# Model checkpoints / weights
ckpt/
checkpoints/
outputs/
weights/
wandb/
*.safetensors
*.bin
*.pt
*.pth
*.ckpt
*.onnx
*.h5
*.pkl

# Evaluation results
results/
results_*/
eval_results/

# Logs
logs/
*.log
*.log.*
nohup.out

# Editors / misc
*.swp
*.swo
.DS_Store
```

`results_*/` 一条即覆盖 `results_pi05_baseline/` 与 `results_source_free/`；
`*.log` 兜住散落各处的日志。

### 3. SSH 身份问题

`ssh -T git@github.com` 返回的是 `Hi xing05188!`，不是目标账号 `ygc666-dot`。
原因是原有 `~/.ssh/id_ed25519`（`2351289@tongji.edu.cn`）绑定在 `xing05188` 账号上，
而 GitHub 同一把公钥只能绑一个账号（重复添加会报 `Key is already in use`）。

处理方式：**新生成一把专用 key + 配置 Host 别名**，两个账号并存互不影响。

```bash
ssh-keygen -t ed25519 -C "ygc666-dot@github" -f ~/.ssh/id_ed25519_ygc -N ""
# 公钥粘贴到 https://github.com/settings/ssh/new
# 下面追加到 ~/.ssh/config（原文件已备份为 config.bak.*）
```

```sshconfig
Host github-ygc
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_ygc
    IdentitiesOnly yes
    ProxyCommand python3 ~/.ssh/http_proxy_cmd.py %h %p
    ServerAliveInterval 30
```

新公钥指纹：`SHA256:MZ+DnCjfDxAeyjWug/mDvGP3mwkLU5GNrrsM5iIyoYg`

**排查过程记录**：绑定前 `ssh -T git@github-ygc` 仍返回 `xing05188`，`ssh -vT`
显示 ssh 把两把 key 都 offer 了出去——OpenSSH 用 `HostName github.com` 做二次配置
解析时又匹配到了上面的 `Host github.com` 块，把旧 key 一并带上。GitHub 接受了旧
key，所以身份错。**新公钥加到 ygc666-dot 后，先被 offer 的新 key 被接受，身份恢复
正确**（返回 `Hi ygc666-dot!`）。

> 本机网络必须走 Clash 代理 `127.0.0.1:7890`，SSH 靠 `~/.ssh/http_proxy_cmd.py`
> 做 CONNECT 隧道；而 `git config http.proxy` **未配置**，所以 HTTPS 方式还需额外
> 补代理，SSH 方式无需。

### 4. 采用方案 B：孤儿提交，切断上游历史

目标是不让新仓库带上 `z-lab/flashvla` 的 94 条历史提交。执行序列：

```bash
git remote rename origin upstream            # 上游留档为 upstream
git checkout --orphan newroot                # 建无父提交分支
git rm -rf --cached . -q                     # 清空索引，工作区不动
git add -A                                   # 按 .gitignore 重新暂存
git commit -m "Initial commit: FlashVLA"     # 得到 110 文件 / 2.3M 的单条提交
git branch -M main                           # 用新历史覆盖本地 main
git remote add origin git@github-ygc:ygc666-dot/flashvla.git
git push -u origin main
```

关键点：

- `git rm -rf --cached .` 只清索引，**工作区文件完全不受影响**，15G 的 `ckpt/` 仍在原位。
- `--orphan` 之后原 `main` 的引用被 `-M` 覆盖，但旧历史可通过 `git reflog`
  或 `git checkout -b old-main @{1}` 找回。
- 推送前用 `git diff --cached --name-only | xargs du -h | sort -rh | head` 复核体积，
  最大文件仅 `assets/demo-first-frame.png` 344K。

### 5. 验证结果

```
远端 ref      5176df8  refs/heads/main
本地 HEAD     5176df8
远端文件数    110
远端是否含    ckpt/ results/ results_*/ logs/  →  无，干净
本地工作区    16G 完好，ckpt/ 下两个 safetensors 均在
```

---

## 四、遗留事项

1. `upstream` 的 URL 是 HTTPS 且 `http.proxy` 未配置，`git fetch upstream` 会失败；
   若需要跟上上游更新，先执行：
   ```bash
   git config --global http.proxy http://127.0.0.1:7890
   git config --global https.proxy http://127.0.0.1:7890
   ```
2. 提交作者身份为 `xing05188 <2351289@tongji.edu.cn>`（来自 `git config --global`），
   与 `ygc666-dot` 账号不一致。如需统一：
   ```bash
   git config --local user.name "ygc666-dot"
   git config --local user.email "<GitHub 邮箱>"
   ```
3. 权重与实验日志目前只存在于本机，无任何远端备份，建议定期另行归档。
