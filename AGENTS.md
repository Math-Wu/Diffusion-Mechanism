
---

## 【项目配置】

```
PROJECT_NAME   = Belief-CL-Diff-PDE
LOCAL_ROOT     = /Users/wudalong/Desktop/Diffusion Mechanism
GITHUB_REPO    = git@github.com:Math-Wu/Diffusion-Mechanism.git
SERVER_ROOT    = ~/Diffusion-Mechanism
SERVER_ALIAS   = ssh-A100以及ssh-V100
MAIN_BRANCH    = main
```

---

## 服务器环境

- **服务器别名**：ssh-A100（A100 云端 GPU 服务器）ssh-V100(V100 云端GPU 服务器)
- **Python**：`~/miniconda3/bin/python`，已配置 torch 2.7.1+cu118，CUDA 可用
- **禁止**更新服务器上的 torch / CUDA 环境，已锁定版本
- **数据目录**：`data/` 不纳入 git 管理，禁止通过 git 或 MCP 同步数据文件
- **代码写入**：禁止通过 SSH MCP 直接写入代码文件到服务器，所有代码变更统一走 git

---

## 代码同步流程（严格按顺序执行）

### Step 1 — 本地提交并推送到 GitHub

```bash
# 1a. 清理可能存在的 lock 文件（每次 commit 前必执行，不管有没有）
rm -f .git/index.lock .git/refs/heads/*.lock .git/packed-refs.lock

# 1b. 暂存并提交
git add -A
git commit -m "<message>"

# 1c. 推送
git push origin <MAIN_BRANCH>
```

> **规则**：`rm -f` 步骤是强制前置，不可跳过，不可改为"检查后再删"。

---

### Step 2 — A100 / V100拉取代码（强制推进工作树）

通过 ssh-A100 MCP 或者 ssh-V100 MCP 在服务器上执行：

```bash
cd <SERVER_ROOT>
git fetch origin && git reset --hard origin/<MAIN_BRANCH>
```

> **规则**：
> - **禁止单独使用 `git pull`**，在有本地改动时会失败
> - **禁止单独使用 `git fetch`**，不推进工作树
> - 统一使用 `fetch + reset --hard`，强制对齐远端，幂等安全

---

### Step 3 — 验证同步结果

```bash
git log --oneline -3   # 确认 HEAD 已对齐最新 commit
git status             # 确认工作树干净
```

---

## GPU 选择策略

在 A100 / V100 服务器上运行任何训练或调试命令前，自动选择显存占用最低的 GPU：

```bash
# 查询显存占用并选择最空闲的 GPU
GPU_ID=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | sort -t',' -k2 -n | head -1 | cut -d',' -f1)
echo "Using GPU: $GPU_ID"
export CUDA_VISIBLE_DEVICES=$GPU_ID
```

> 对于多卡并行实验（如 DDP），手动指定 `CUDA_VISIBLE_DEVICES`，不使用自动选择。

---

## 训练任务管理

### 启动后台训练（tmux）

```bash
# 创建具名 session，便于后续 attach
tmux new-session -d -s train_<RUN_NAME> \
  "cd <SERVER_ROOT> && CUDA_VISIBLE_DEVICES=$GPU_ID \
   ~/miniconda3/bin/python train.py <args> 2>&1 | tee train_<RUN_NAME>.log"
```

### 检查训练进度

每隔 10 分钟执行一次：

```bash
tail -n 30 train_<RUN_NAME>.log
```

确认以下信息：
- loss 在正常下降范围内
- 无 CUDA OOM / NaN 报错
- step/epoch 按预期推进

### 终止 / 查看 session

```bash
tmux ls                             # 列出所有 session
tmux attach -t train_<RUN_NAME>    # 进入查看
tmux kill-session -t train_<RUN_NAME>  # 终止
```

---

## 实验管理约定

- **实验结果**保存在 `outputs/<run_name>/`，包含 checkpoint、log、config 快照
- **config 快照**：每次启动训练时，自动将当前 config 文件复制到 `outputs/<run_name>/config.yaml`，确保可复现
- **run_name 命名规则**：`<日期>_<方法简称>_<关键超参>`，例如 `20260408_rebel_lr1e4_depth3`
- **并行实验**：不同 run 使用不同 tmux session 名和不同 log 文件，禁止覆盖彼此的 log

---

## 调试工作流

快速验证代码可运行（不启动完整训练）：

```bash
# 用极小规模跑通一个 forward pass / 几个 step
CUDA_VISIBLE_DEVICES=$GPU_ID ~/miniconda3/bin/python train.py \
  --debug --max_steps 5 --batch_size 2
```

调试通过后再用 tmux 启动正式训练。

---

## 工具路由规则

| 场景 | 使用工具 |
|------|----------|
| 消息中提到"A100" | ssh-A100 MCP |
| 消息中提到"V100" | ssh-V100 MCP |
| 未提到服务器 | 默认本地操作 |
| 查看/修改代码文件 | 本地直接编辑 |
| 运行训练 / 调试命令 | ssh-A100 MCP or ssh-V100 MCP|
| 同步数据文件 | ❌ 不操作，data/ 不在 git 中 |

---

## 远端命令执行规范（SSH MCP 专用）

SSH MCP 的命令包装层会对 heredoc（`<<EOF`）做变量展开，导致 Python 收到残缺字符串，
产生 `NameError: name 'PY' is not defined` 等报错。

**核心规则：通过 SSH MCP 在服务器上执行 Python 逻辑时，禁止使用 heredoc。**

### 方案 A：python -c 单行（逻辑简单时用）

```bash
~/miniconda3/bin/python -c "
import os
gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
print(f'Using GPU: {gpu}')
"
```

> 用双引号包裹整个 `-c` 字符串，内部用单引号或 f-string，避免引号嵌套冲突。

### 方案 B：先写临时脚本文件，再执行（逻辑复杂时用）

```bash
# Step 1：写入临时 py 文件（单引号包裹，shell 不展开内部变量）
cat << 'PYEOF' > /tmp/run_task.py
import torch
import os

gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
print(f"Using GPU: {gpu_id}")
# ... 其余 Python 逻辑 ...
PYEOF

# Step 2：执行
CUDA_VISIBLE_DEVICES=$GPU_ID ~/miniconda3/bin/python /tmp/run_task.py
```

> 关键：heredoc 的结束符用 **单引号** `'PYEOF'` 而不是裸 `PYEOF`，
> 单引号告诉 shell 不对内部内容做任何变量展开，Python 变量名得以保留。

### 禁止的写法

```bash
# ❌ 裸 heredoc，shell 会展开 $PY/$GPU_ID 等变量，Python 收到残缺字符串
python << EOF
gpu = $GPU_ID      # 被 shell 展开，Python 看到的是数字而非变量名
PY = "$PY"         # $PY 若未定义则变成空字符串，NameError
EOF
```

---

## 常见错误自动处理

| 错误 | 处理方式 |
|------|----------|
| `.git/index.lock: File exists` | `rm -f .git/index.lock`，然后重新执行 git 命令 |
| A100 上 `git pull` 失败（本地有改动） | 改用 `git fetch origin && git reset --hard origin/<MAIN_BRANCH>` |
| A100 上 `git fetch` 后工作树未变 | 补执行 `git reset --hard origin/<MAIN_BRANCH>` |
| CUDA OOM | 降低 batch_size 或换更空闲 GPU，不重装环境 |
| tmux session 已存在同名 | `tmux kill-session -t <name>` 后重建，或换新 run_name |
| `NameError: name 'PY' is not defined`（SSH heredoc 变量被展开） | 改用 `python -c "..."`  或 `cat << 'PYEOF' > /tmp/x.py && python /tmp/x.py`，见【远端命令执行规范】 |