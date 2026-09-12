

<div align="center">
<h1>Less Harness, More Signal: Efficient In-Harness RL for Autonomous Agents</h1>

[![英文 README](https://img.shields.io/badge/README-English-4d8cd8?style=for-the-badge)](README.md)
[![论文](https://img.shields.io/badge/论文-PDF-5f16a8?style=for-the-badge)](paper/paper.pdf)
[![数据集](https://img.shields.io/badge/数据集-6%2C970%20任务-ffd21e?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/clawLooop/clawloop-data)
[![集成版 VERL](https://img.shields.io/badge/代码-集成版%20VERL%20%2B%20AAM-63cad3?style=for-the-badge)](verl/)
</div>

## 1. 项目概览

ClawLoop 是一个面向长程、工具使用型智能体的轻量化可验证强化学习框架。项目对应论文《Less Harness, More Signal: Efficient In-Harness RL for Autonomous Agents》，公开了完整修改版 VERL、6,970 条任务数据、论文图表和 9B/27B 训练脚本。

```text
任务描述 → 独立工作区 → 原子工具操作
    ↑                         ↓
终态验证器 ← 多轮思考、行动与观察
```

智能体根据自己最终创建的工作区状态获得奖励，而不是被要求复现一条固定的工具调用轨迹。每条 rollout 拥有独立工作区、受限的工具面和终态验证器；不改善策略信号的产品级服务不在关键路径上。

项目同时将 **Asymmetric Advantage Masking（AAM，非对称优势掩码）** 集成到了 VERL：无效交互失去其*正*优势的策略梯度贡献，但保留负优势信号。这针对论文指出的两个瓶颈——产品级 harness 的 CPU/IO 开销让 GPU 空转，以及 GRPO 把单一轨迹优势广播给有效和无效 token。

数据集可直接从 Hugging Face 加载：**[clawLooop/clawloop-data](https://huggingface.co/datasets/clawLooop/clawloop-data)**。

## 2. 目录导航

| 部分 | 内容 |
| --- | --- |
| [项目概览](#1-项目概览) | 研究目标与核心思想 |
| [目录导航](#2-目录导航) | 文档索引 |
| [项目结构](#3-项目结构) | GitHub 仓库文件说明 |
| [主要内容](#4-主要内容) | ClawLoop 框架和 AAM 方法 |
| [训练过程与图片](#5-训练过程与图片) | rollout 生命周期和论文图示 |
| [复现训练](#复现训练) | 数据准备、安装和训练命令 |

## 3. 项目结构

```text
clawloop/
├── README.md
├── README_zh.md
├── LICENSE
├── .gitattributes
├── .gitignore
├── requirements-data.txt
├── data/
│   ├── tasks.jsonl                   # 6,970 条任务记录（Git LFS）
│   ├── metadata.json
│   ├── SCHEMA.md                     # JSONL 字段说明
│   ├── tasks.part1.rar
│   └── tasks.part2.rar
├── paper/
│   ├── paper.pdf                     # 论文全文 PDF
│   ├── clawAgent_main.pdf            # 架构图与 AAM 图示
│   ├── grpo_three_figures_combined.pdf
│   ├── fig_cost.pdf                  # 环境开销图
│   ├── fig_train_eff.pdf             # 训练效率图
│   ├── fig_token_sr.pdf              # 推理 token 效率图
│   └── previews/                     # GitHub 可直接渲染的 PNG 预览
├── scripts/
│   ├── validate_release.py           # 发布数据校验
│   ├── restore_hf_dataset.py         # JSONL 恢复为任务目录
│   └── prepare_hf_dataset.py         # 任务目录导出为 JSONL
└── verl/                             # 完整修改版 VERL
    ├── verl/
    ├── nanoclaw_recipe/              # ClawLoop runtime、工具、AAM 和启动脚本
    ├── tests/recipe/nanoclaw/        # 集成回归测试
    ├── examples/、docs/
    ├── nanoclaw_recipe/train_9b.sh   # Qwen3.5-9B 训练脚本
    └── nanoclaw_recipe/train_27b.sh  # Qwen3.5-27B 训练脚本
```

## 4. 主要内容

#### 图 1：ClawLoop 架构与 AAM

![ClawLoop 与 AAM Agent Loop](paper/previews/clawAgent_main.png)

图中保留任务描述、独立工作区、原子工具、多轮观察和终态验证器，移除 session 管理、插件发现、长期记忆和外部服务编排。

### 4.1 ClawLoop 框架

一次 rollout 由任务记录、环境构建器和终态验证器共同定义：

| 组件 | 作用 | 代码位置 |
| --- | --- | --- |
| 任务 bundle | 解析 prompt、builder、verifier、manifest 及不同任务布局 | [`common.py`](verl/nanoclaw_recipe/common.py) |
| 工作区初始化 | 创建独立目录并在隔离子进程中运行 `env_builder.py` | [`nanoclaw.py`](verl/nanoclaw_recipe/nanoclaw.py) |
| 原子工具 | 列目录、读写文件、搜索、编辑、创建目录和受限 Shell | [`runtime/tools.py`](verl/nanoclaw_recipe/runtime/tools.py) |
| 多轮 agent loop | 组织模型输出、工具调用、观察和 token mask | [`nanoclaw_support.py`](verl/verl/experimental/agent_loop/nanoclaw_support.py) |
| 终态验证器 | 回合结束后检查工作区并输出分数 | 每个任务 bundle 内的 `workplace_verifier.py` |
| VERL Trainer | 计算 GRPO 优势、应用 actor mask 并更新策略 | [`ray_trainer.py`](verl/verl/trainer/ppo/ray_trainer.py) |

工具安全边界如下：

| 工具类别 | 操作 | 约束 |
| --- | --- | --- |
| 检查 | `list_dir`、`read_file`、`grep`、`find` | 仅允许相对路径，限制输出长度 |
| 修改 | `write_file`、`edit_file`、`apply_patch`、`mkdir` | 所有写入都必须位于当前工作区，拒绝 `..` 路径穿越 |
| 计算 | 受限 `bash`（`python`、`grep`、`awk`、`sed`、`sort` 等） | 禁止后台任务、命令替换、危险重定向和高风险 Python 模式 |
| 判分 | `workplace_verifier.py` | 独立子进程、隔离 `HOME/TMPDIR`、可配置超时 |

环境构建器默认超时 120 秒，verifier 默认超时 300 秒。builder 和 verifier 的输出会被捕获，分数文件会在预期位置检查；失败或缺失的 verifier 会产生明确的 fallback 状态。

### 4.2 AAM 方法

标准 GRPO 为一条轨迹中的所有策略 token 使用同一个 group advantage。因此，一条成功轨迹中的冗余读取、重复工具结果、错误调用或截断输出，也可能与真正有用的编辑一起被强化。

AAM 在 rollout 阶段记录无效交互区间，在 GRPO 计算优势之后再应用掩码。当前实现检测四类模式：内部循环、重复工具结果、工具错误，以及达到 response budget 后被截断的最后 assistant turn。掩码公式和消融结果见论文。

## 5. 训练过程与图片

### 5.1 训练流程

任务 JSONL 被解析为每条 rollout 的独立工作区，模型通过受限原子工具进行多轮交互，回合结束后由终态验证器对工作区打分，VERL 计算 GRPO 优势并应用 AAM mask 后更新策略。分阶段的详细说明见论文。

### 5.2 论文图片记录

以下图片直接取自论文。

#### 图 2：GRPO 训练动态与信用分配

![GRPO training dynamics](paper/previews/grpo_three_figures_combined.png)

#### 图 3：训练效率

![Training efficiency](paper/previews/fig_train_eff.png)

## 复现训练

### 数据准备

```bash
git lfs install
python scripts/validate_release.py data/tasks.jsonl
python scripts/restore_hf_dataset.py \
  data/tasks.jsonl \
  --output-dir /tmp/clawloop_tasks
```

也可以直接从 Hugging Face 加载：

```python
from datasets import load_dataset

tasks = load_dataset("clawLooop/clawloop-data", split="train")
```

### 安装并训练

```bash
cd /path/to/clawloop/verl
pip install -e .
```

9B 训练：

```bash
BASE_TASKS=/tmp/clawloop_tasks \
MODEL_PATH=/path/to/Qwen3.5-9B \
bash nanoclaw_recipe/train_9b.sh
```

27B 训练：

```bash
BASE_TASKS=/tmp/clawloop_tasks \
MODEL_PATH=/path/to/Qwen3.5-27B \
bash nanoclaw_recipe/train_27b.sh
```

完整修改版 VERL 已经位于 `verl/`，不需要手动应用 patch、安装外部 recipe 或设置 `VERL_ROOT`。参考配置包含 8,192 prompt tokens、22,768 response tokens、最多 35 turns、FSDP2、异步 vLLM rollout、GRPO 和 AAM。

