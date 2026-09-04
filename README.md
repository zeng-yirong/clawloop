---
license: mit
language:
  - en
  - zh
task_categories:
  - reinforcement-learning
  - text-generation
tags:
  - autonomous-agents
  - tool-use
  - agentic-rl
  - grpo
  - react
  - verifiable-rewards
  - clawloop
size_categories:
  - 10K<n<100K
pretty_name: ClawLoop Verifiable Agent RL Tasks
---

<div align="center">
<h1>ClawLoop: Less Harness, More Signal</h1>
<h2>Verifiable Reinforcement Learning for Long-Horizon Tool-Using Agents</h2>

[![Paper](https://img.shields.io/badge/Paper-Manuscript-5f16a8?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](paper/clawAgent_main.pdf)
[![Dataset](https://img.shields.io/badge/Dataset-6%2C970%20Tasks-4d8cd8?style=for-the-badge&logo=huggingface&logoColor=white)](data/tasks.jsonl)
[![Recipe](https://img.shields.io/badge/Recipe-VERL%20%2B%20AAM-63cad3?style=for-the-badge&logo=pytorch&logoColor=white)](recipe/README.md)
[![License](https://img.shields.io/badge/License-MIT-2ea44f?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
</div>

<br>

**News!!!**

- [2026/09] We release the ClawLoop task corpus, ClawLoop recipe, AAM integration, paper artifacts, and portable Qwen3.5-9B/27B launch profiles.
- [2026/09] The JSONL release contains 6,970 tasks that passed strict export validation and environment-builder smoke tests.
- [2026/09] The README includes directly rendered PNG previews, with the full-resolution PDF figures preserved beside them.

## Less Harness, More Signal

ClawLoop is a research release for **verifiable reinforcement learning of long-horizon, tool-using agents**. It accompanies the paper *Less Harness, More Signal: Efficient In-Harness RL for Autonomous Agents* and packages the task corpus, execution recipe, masking implementation, and reproducibility materials in one Hub-ready project.

The central premise is simple: an agent should be trained against the **state it creates**, not against a single prescribed tool trajectory. Every rollout therefore runs in an isolated workspace, uses a small set of general file and shell tools, and receives reward from a verifier that inspects the terminal workspace. Search order, edits, retries, and recovery remain open to the policy.

![ClawLoop execution loop](paper/harness_draft.png)

*ClawLoop keeps the task, isolated workspace, atomic tools, multi-turn interaction, and terminal verifier in the policy-gradient loop while removing product-layer state such as session management, plugin registries, and long-term memory.*

## What the paper studies

In-harness RL exposes two coupled failure modes:

1. **Environment overhead.** In a controlled comparison on the same task, a full product harness spends 54.1 seconds per episode in environment execution, versus 2.5 seconds for a tool-only baseline. The resulting 57.5-second episode is 9.7x slower, and mean GPU utilization falls from 81% to 14% while the system waits on CPU/IO-bound interactions.
2. **Credit misassignment.** GRPO broadcasts a trajectory-level advantage to every generated token. A successful rollout can therefore reinforce redundant reads, repeated tool results, error calls, or a truncated final turn together with the useful edit that actually solved the task. The paper measures the positive-gradient mass on ineffective interactions increasing from 0.04 early in training to 0.20 after the success peak, followed by training collapse.

![In-harness cost](paper/fig_cost2.png)

*The released figure reports the paper's wall-clock and utilization comparison: environment execution dominates the full harness, while the accelerator is underutilized.*

**Artifact consistency note.** The included `fig_cost2.png` labels the tool-only bar as 81% GPU utilization, while an earlier paragraph in `paper/main.tex` states 61%. The source materials should be reconciled before using either number in a camera-ready release; the 14% in-harness value is consistent across the figure and manuscript.

## The ClawLoop approach

### ClawLoop

ClawLoop is a training-oriented, white-box harness reconstructed around five explicit elements:

```text
task specification -> isolated workspace -> atomic tools
        ^                                  |
        |                                  v
 terminal verifier <- multi-turn observations and actions
```

The runtime exposes structured rollout metadata and token spans directly to VERL. Product concerns that do not contribute to policy learning are outside the critical path. In the paper's controlled measurements, this design reduces environment overhead by 8.8x and raises GPU utilization from 14% to 49%.

### Asymmetric Advantage Masking (AAM)

Let `m_base` be the ordinary response mask, `B` the token spans belonging to deterministic bad-turn rules, and `A_t` the GRPO advantage. AAM uses:

```text
m_t = m_base_t * [1 - 1(t in B and A_t > 0)]
```

This asymmetric condition is the key design choice:

| Situation | Update behavior |
| --- | --- |
| Ineffective turn in a positive-advantage rollout | Remove its positive policy-gradient contribution. |
| Ineffective turn in a negative-advantage rollout | Keep the token active so the policy learns to avoid it. |
| Verifier reward and rollout context | Leave unchanged; only the actor loss mask is modified. |

The four candidate detectors are looping responses, duplicate tool-result turns, error tool results, and the final assistant turn cut off by the response budget. Candidate spans are recorded during rollout; the positive-advantage decision is applied after GRPO computes advantages.

## Results

Across five benchmarks (PinchBench, ClawEval, ClawBenchPro, BFCL-v3, and `tau^2`-bench) and model scales from 2B to 27B, the paper reports:

| Metric | Result in the controlled experiments |
| --- | --- |
| ClawLoop environment overhead | 8.8x reduction |
| GPU utilization | 14% -> 49% |
| Qwen3.5-9B AAM vs. standard GRPO | +5.1 success-rate points |
| Qwen3.5-9B AAM vs. base model | +7.8 success-rate points |
| Inference token consumption | 32% lower with competitive accuracy |
| 27B in-domain performance | Matches the reported frontier proprietary reference |

These numbers depend on the model checkpoints, hardware, seeds, verifier setup, and benchmark splits used in the manuscript. The task release makes the environment and scoring boundary inspectable; it is not a claim that the JSONL file alone reproduces every table.

## Datasets

`data/tasks.jsonl` contains **6,970 validated tasks** in portable JSONL form. Each record stores:

```text
task_id       stable export identifier
prompt        natural-language task instruction
task_yaml     runtime metadata
env_builder   source for constructing the initial workspace
verifier      source for terminal-state scoring
manifest      exporter status, smoke-test result, warnings, and fixes
source_files  canonical filenames used during restoration
```

The source export contained 7,056 records. A strict exporter retained 6,970 and rejected 86 because of dangerous path literals, smoke-test failures, or Python syntax errors. Every released record is marked `valid: true` and has a passing environment-builder smoke test. Prompts are multilingual: 4,365 English/other and 2,605 Chinese or mixed-language records.

The dataset contains synthetic workplace scenarios covering data cleaning, coding, documents, finance, operations, and structured file editing. It does not contain model checkpoints, rollout conversations, or private API credentials. See [data/SCHEMA.md](data/SCHEMA.md) and [data/metadata.json](data/metadata.json) for the exact schema and export statistics.

## Model Use

### Environment Setup

Validate the release without executing any task code:

```bash
python scripts/validate_release.py data/tasks.jsonl
```

Restore the JSONL into the directory layout expected by the ClawLoop runtime:

```bash
python scripts/restore_hf_dataset.py \
  data/tasks.jsonl \
  --output-dir /tmp/clawloop_tasks
```

The restore operation writes files only. It never imports or executes `env_builder.py` or a verifier. The resulting tree contains both the canonical `tasks/data_*` and `scripts/data_*/verify_workplace.py` layout and the manifest-relative files needed by flat-layout discovery.

To regenerate the JSONL from the original restored task folders:

```bash
python scripts/prepare_hf_dataset.py \
  /path/to/restored_data_all \
  data/tasks.jsonl
```

### Training

The recipe is an adapter for a compatible [VERL](https://github.com/volcengine/verl) checkout; the full upstream framework is intentionally not vendored here. The public launchers were distilled from the validated Qwen3.5-9B and Qwen3.5-27B experiment scripts:

```bash
VERL_ROOT=/path/to/verl \
BASE_TASKS=/tmp/clawloop_tasks \
MODEL_PATH=/path/to/Qwen3.5-9B \
bash recipe/nanoclaw_recipe/train_9b.sh
```

For the 27B profile:

```bash
VERL_ROOT=/path/to/verl \
BASE_TASKS=/tmp/clawloop_tasks \
MODEL_PATH=/path/to/Qwen3.5-27B \
bash recipe/nanoclaw_recipe/train_27b.sh
```

Both profiles preserve the paper-aligned defaults: 8,192 prompt tokens, 22,768 response tokens, 16,384 assistant tokens, 8,192 tool-observation tokens, 35 turns, FSDP2, async vLLM rollout, Qwen3-Coder multi-turn formatting, GRPO with fixed low-variance KL, eight responses per prompt, and AAM enabled. Override hardware and experiment settings through environment variables or extra Hydra arguments. `train_half_turn.cluster.sh` is the historical NPU/ModelArts launcher and is retained only for provenance; it contains cluster-specific installation and path assumptions.

The implementation and integration details are documented in [recipe/README.md](recipe/README.md), the [ClawLoop recipe runbook](recipe/nanoclaw_recipe/README.md), and [patches/README.md](patches/README.md). CPU regression tests cover delayed positive-advantage masking and final-answer bonus behavior.

**Compatibility note.** The Python package and import paths are still named `nanoclaw_recipe` in the code snapshot so existing VERL integrations remain loadable. `ClawLoop` is the formal public project name; these are internal compatibility identifiers only.

### Inference

The runtime-only path is implemented in [ClawLoop inference](recipe/nanoclaw_recipe/inference.py). It reuses the same task discovery, multi-turn tool loop, workspace isolation, and Qwen3-Coder formatting as training, but does not call the task verifier during rollout. Evaluation can therefore be performed independently after trajectories are collected.

## Paper and Figure Gallery

The manuscript source and figures are in [paper/](paper/). The compiled manuscript is [paper/clawAgent_main.pdf](paper/clawAgent_main.pdf). The design notes [paper/harness设计.md](paper/harness设计.md) and [paper/mask.md](paper/mask.md) provide a longer explanation of the harness boundary and the masking rationale.

GitHub and Hugging Face do not rasterize linked PDFs inside a README. The PDF files below are nevertheless the paper's actual plots and diagrams; each one is paired with a checked-in PNG preview so the figure is visible directly on the project page, while the original PDF remains available for full-resolution download and printing.

### Architecture overview

![ClawLoop architecture](paper/harness_draft.png)

ClawLoop / VERL rollout, isolated workspace, atomic tools, verifier, and policy update. [Original PDF](paper/figure1.pdf)

### Training dynamics

![GRPO and AAM training dynamics](paper/previews/grpo_three_figures_combined.png)

GRPO versus AAM success rate, ineffective-turn rate, and positive-gradient misassignment. [Original PDF](paper/grpo_three_figures_combined.pdf) · [Alternate export](paper/grpo_three_figures_combined-Copy1.pdf)

![GRPO and AAM training dynamics (alternate export)](paper/previews/grpo_three_figures_combined-Copy1.png)

### Environment cost

![In-harness environment cost](paper/previews/fig_cost.png)

Per-episode wall-clock decomposition and GPU utilization. [Original PDF](paper/fig_cost.pdf) · [Earlier PNG export](paper/fig_cost2.png)

### Training efficiency

![Training efficiency](paper/previews/fig_train_eff.png)

Accuracy and efficiency curves used in the controlled training comparison. [Original PDF](paper/fig_train_eff.pdf)

### Token efficiency

![Token efficiency](paper/previews/fig_token_sr.png)

Success rate and inference-time token consumption. [Original PDF](paper/fig_token_sr.pdf)

The manuscript source is [main.tex](paper/main.tex), with bibliography and AAAI style files included alongside it. The compiled paper is [clawAgent_main.pdf](paper/clawAgent_main.pdf).

## Acknowledgement

We thank the [VERL](https://github.com/volcengine/verl) project for providing the distributed reinforcement-learning infrastructure on which the ClawLoop adapter is built.

## Uploading to the Hub

This directory is the intended upload root. Do not upload the parent workspace, which contains unrelated source exports and experiment files.

```bash
cd /home/hyx/hf_up/nanoclaw_hf
git lfs install
hf repo create <namespace>/clawloop-tasks --repo-type dataset
hf upload <namespace>/clawloop-tasks . . --repo-type dataset
```

The 142MB JSONL file is configured for Git LFS through `.gitattributes`. A Hub Dataset repository is recommended because the primary artifact is task data; the same directory can also be mirrored as a code repository for the recipe and paper materials.

## Safety and licensing

Verifiers and environment builders are arbitrary benchmark code. Run them only in a container with network isolation, resource limits, and a disposable filesystem. The validator reports eight absolute-path warnings inherited from synthetic task content; these are example paths inside task fixtures, not paths used by the release tooling.

ClawLoop-specific code and documentation are MIT licensed. VERL-derived integration files remain subject to the upstream Apache-2.0 license. The AAAI author-kit files and the source benchmark export may have additional redistribution terms; review [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before making the Hub repository public.

## Citation

```bibtex
@article{clawloop2026,
  title = {Less Harness, More Signal: Efficient In-Harness RL for Autonomous Agents},
  year  = {2026},
  note  = {ClawLoop release; manuscript source and task dataset included}
}
```
