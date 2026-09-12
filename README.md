

<div align="center">
<h1>Less Harness, More Signal: Efficient In-Harness RL for Autonomous Agents</h1>

[![Paper](https://img.shields.io/badge/Paper-Manuscript-5f16a8?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](paper/paper.pdf)
[![中文 README](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-ef9a9a?style=for-the-badge)](README_zh.md)
[![Dataset](https://img.shields.io/badge/Dataset-6%2C970%20Tasks-4d8cd8?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/clawLooop/clawloop-data)
[![Integrated VERL](https://img.shields.io/badge/Code-Integrated%20VERL%20%2B%20AAM-63cad3?style=for-the-badge&logo=pytorch&logoColor=white)](verl/)
[![License](https://img.shields.io/badge/License-MIT-2ea44f?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
</div>

<br>

## 1. Overview

ClawLoop is a lightweight, verifiable RL framework for long-horizon tool-using agents. It accompanies the paper *Less Harness, More Signal: Efficient In-Harness RL for Autonomous Agents* and releases the modified VERL tree, the 6,970-task corpus, the paper artifacts, and the training launchers in one repository.

```text
task specification → isolated workspace → atomic actions
        ↑                                      ↓
terminal verifier ← multi-turn observations and actions
```

The policy is rewarded for the terminal workspace it creates, not for reproducing a prescribed tool trajectory. Each rollout gets its own workspace, a small guarded tool surface, and a terminal verifier; product-layer services that do not improve the policy signal are outside the critical path.

The release also integrates **Asymmetric Advantage Masking (AAM)** into VERL: ineffective turns lose their *positive* policy-gradient contribution but keep their negative signal. That targets the paper's two bottlenecks — CPU/IO-bound product harness work idling the accelerator, and GRPO broadcasting a single trajectory advantage to useful and ineffective tokens alike.

The dataset is on the Hugging Face Hub: **[clawLooop/clawloop-data](https://huggingface.co/datasets/clawLooop/clawloop-data)**. The repository mirror carries the same JSONL release through Git LFS.

## 2. Contents

| Section | What to find here |
| --- | --- |
| [1. Overview](#1-overview) | Research goal and central idea. |
| [2. Contents](#2-contents) | This repository guide. |
| [3. Project structure](#3-project-structure) | The GitHub tree and the role of each directory. |
| [4. Main components](#4-main-components) | The ClawLoop framework and the AAM method. |
| [5. Training process and figure record](#5-training-process-and-figure-record) | Rollout/training lifecycle and all directly rendered paper figures. |
| [Reproduction](#reproduction) | Dataset preparation, installation, and 9B/27B launch commands. |

## 3. Project structure

```text
clawloop/
├── README.md
├── LICENSE
├── .gitattributes
├── .gitignore
├── requirements-data.txt
├── data/
│   ├── tasks.jsonl                   # 6,970 validated task records (Git LFS)
│   ├── metadata.json
│   ├── SCHEMA.md                     # Record schema and restoration contract
│   ├── tasks.part1.rar
│   └── tasks.part2.rar
├── paper/
│   ├── paper.pdf                     # Full manuscript PDF
│   ├── clawAgent_main.pdf            # Architecture and AAM diagram
│   ├── grpo_three_figures_combined.pdf
│   ├── fig_cost.pdf                  # Environment/training cost figure
│   ├── fig_train_eff.pdf             # Training efficiency figure
│   ├── fig_token_sr.pdf              # Success/token-efficiency figure
│   └── previews/                     # PNG previews for GitHub rendering
├── scripts/
│   ├── validate_release.py           # Static release validation
│   ├── restore_hf_dataset.py         # JSONL → runnable task layout
│   └── prepare_hf_dataset.py         # Task layout → JSONL export
└── verl/                             # Complete modified VERL source tree
    ├── verl/
    ├── nanoclaw_recipe/              # ClawLoop runtime, tools, AAM, launchers
    ├── tests/recipe/nanoclaw/        # Regression tests for the integration
    ├── examples/ and docs/
    ├── nanoclaw_recipe/train_9b.sh   # Qwen3.5-9B reference launch
    └── nanoclaw_recipe/train_27b.sh  # Qwen3.5-27B reference launch
```

## 4. Main components

#### Figure 1 — ClawLoop architecture

![ClawLoop architecture and AAM](paper/previews/clawAgent_main.png)

The architecture keeps the task specification, isolated mutable workspace, atomic tools, multi-turn observations, and terminal verifier inside the policy-gradient loop. Session management, plugin discovery, long-term memory, and external service orchestration are removed from the learning-critical path.

### 4.1 ClawLoop framework

ClawLoop is a training-oriented workspace harness. A task record supplies a prompt, an environment builder, and a verifier. The builder creates the initial files in a disposable per-rollout workspace; the agent then explores and edits that workspace through guarded tools; the verifier inspects the final state and emits the reward.

| Component | Role in a rollout | Source anchor |
| --- | --- | --- |
| Task bundle | Resolves prompts, builders, verifiers, manifests, and flat/legacy layouts. | [`common.py`](verl/nanoclaw_recipe/common.py) |
| Workspace initializer | Creates a unique workspace and runs the builder in an isolated subprocess. | [`nanoclaw.py`](verl/nanoclaw_recipe/nanoclaw.py) |
| Atomic tools | Lists, reads, searches, writes, edits, creates directories, and runs restricted shell commands. | [`runtime/tools.py`](verl/nanoclaw_recipe/runtime/tools.py) |
| Multi-turn agent loop | Interleaves model reasoning, tool calls, observations, and response masks. | [`nanoclaw_support.py`](verl/verl/experimental/agent_loop/nanoclaw_support.py) |
| Terminal verifier | Scores the resulting workspace after generation; it is not exposed as an agent action. | `workplace_verifier.py` in each task bundle |
| VERL trainer path | Batches rollouts, computes GRPO advantages, applies the actor loss mask, and updates the policy. | [`ray_trainer.py`](verl/verl/trainer/ppo/ray_trainer.py) |

The safety boundary is explicit:

| Tool group | Operations | Runtime guard |
| --- | --- | --- |
| Inspect | `list_dir`, `read_file`, `grep`, `find` | Relative paths and bounded output. |
| Modify | `write_file`, `edit_file`, `apply_patch`, `mkdir` | Workspace-only resolution; parent traversal is rejected. |
| Compute | Restricted `bash` with allowlisted utilities | No background jobs, command substitution, unsupported redirection, or dangerous Python patterns. |
| Judge | `workplace_verifier.py` | Separate subprocess with isolated `HOME`/`TMPDIR` and a configurable timeout. |

The environment builder has a 120-second default timeout; verifier execution has a 300-second default timeout. Builder and verifier output is captured, score files are checked in the expected locations, and failed or missing verifiers receive an explicit fallback status. Memory operations and product-runtime state are not part of the local learning loop.

### 4.2 Asymmetric Advantage Masking (AAM)

In standard GRPO, all policy-generated tokens in a sampled trajectory inherit the same standardized group advantage. A successful episode can therefore reinforce a useful edit together with redundant reads, repeated tool results, error calls, internal loops, or a truncated final response.

AAM records candidate ineffective spans during rollout and applies the advantage condition after GRPO computes the trajectory advantage. The current detector covers four deterministic patterns: looping responses, duplicate tool-result turns, error tool results, and a final assistant turn cut off by the response budget. The mask formulation and ablation are in the paper.

## 5. Training process and figure record

### 5.1 End-to-end training lifecycle

A task JSONL is resolved into a per-rollout workspace, the policy acts through guarded atomic tools, the terminal verifier scores the resulting workspace, and VERL applies the GRPO advantage with the AAM actor mask before updating the policy. The paper gives the stage-by-stage breakdown.

### 5.2 Training figures

The figures below are taken directly from the paper.

#### Figure 2 — GRPO training dynamics and credit misassignment

![GRPO training dynamics](paper/previews/grpo_three_figures_combined.png)

#### Figure 3 — Environment and training efficiency

![Training efficiency](paper/previews/fig_train_eff.png)

## Reproduction

### Dataset

The release contains 6,970 valid task records from a 7,056-record source export. Each record includes `task_id`, `prompt`, `task_yaml`, `env_builder`, `verifier`, a validation manifest, and canonical source-file fields. All released records pass the strict export checks and environment-builder smoke tests.

For Hub-native loading:

```python
from datasets import load_dataset

tasks = load_dataset("clawLooop/clawloop-data", split="train")
```

For local restoration:

```bash
git lfs install
python scripts/validate_release.py data/tasks.jsonl
python scripts/restore_hf_dataset.py \
  data/tasks.jsonl \
  --output-dir /tmp/clawloop_tasks
```

The restore script writes files only; it does not import or execute task builders or verifiers. To regenerate JSONL from a restored task tree:

```bash
python scripts/prepare_hf_dataset.py \
  /path/to/restored_data_all \
  data/tasks.jsonl
```

### Install the integrated VERL tree

```bash
cd /path/to/clawloop/verl
pip install -e .
```

No patch application, external recipe checkout, or `VERL_ROOT` variable is required. The complete framework and ClawLoop integration are already present under `verl/`.

### Train with the reference 9B and 27B profiles

```bash
cd /path/to/clawloop/verl
BASE_TASKS=/tmp/clawloop_tasks \
MODEL_PATH=/path/to/Qwen3.5-9B \
bash nanoclaw_recipe/train_9b.sh
```

```bash
cd /path/to/clawloop/verl
BASE_TASKS=/tmp/clawloop_tasks \
MODEL_PATH=/path/to/Qwen3.5-27B \
bash nanoclaw_recipe/train_27b.sh
```

The reference profiles use the paper-aligned multi-turn setup: 8,192 prompt tokens, 22,768 response tokens, 16,384 assistant tokens, 8,192 tool-observation tokens, 35 turns, FSDP2, asynchronous vLLM rollout, Qwen3-Coder formatting, GRPO, eight responses per prompt, and AAM enabled. Hardware-specific paths and Hydra overrides can be supplied through environment variables or command-line arguments.

Regression tests for masking and final-answer behavior are in [`verl/tests/recipe/nanoclaw/`](verl/tests/recipe/nanoclaw/).
