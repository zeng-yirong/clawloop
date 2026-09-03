---
license: mit
language:
  - en
  - zh
task_categories:
  - reinforcement-learning
  - text-generation
tags:
  - agent
  - tool-use
  - rl
  - grpo
  - react
  - nanoclaw
size_categories:
  - 10K<n<100K
pretty_name: Nanoclaw Verifiable Agent RL Tasks
---

# Nanoclaw: Verifiable Tool-Use RL

This repository packages the artifacts for **Less Harness, More Signal: Efficient In-Harness RL for Autonomous Agents**. The project combines:

* **ClawLoop**, a lightweight, white-box execution loop for multi-turn tool-use rollouts;
* **Asymmetric Advantage Masking (AAM)**, which removes only positive-advantage tokens from deterministic bad-turn spans while preserving negative learning signals; and
* **6,970 validated workplace tasks** with prompts, environment builders, verifiers, and export manifests.

The task collection is intended for research on verifiable agent reinforcement learning. Each task creates an isolated workspace, exposes a small set of file/search/shell tools, and scores the final workspace with a task-specific verifier. A task can have many valid action sequences; only the final state is graded.

## Repository layout

```text
nanoclaw_hf/
  data/tasks.jsonl                 # portable one-record-per-task release
  scripts/prepare_hf_dataset.py   # export source task folders to JSONL
  scripts/restore_hf_dataset.py   # restore JSONL to runtime directory layout
  scripts/validate_release.py      # structural and safety checks
  recipe/                          # Nanoclaw recipe and tool schema
  paper/                           # paper source, figures, and design notes
  THIRD_PARTY_NOTICES.md           # upstream VERL and asset provenance
```

The JSONL record schema is documented in `data/SCHEMA.md`. Code fields contain the original UTF-8 source text, so the release is self-contained and does not rely on local absolute paths.

## Quick start: inspect the dataset

```bash
python scripts/validate_release.py data/tasks.jsonl
python - <<'PY'
import json
from pathlib import Path
for line in Path("data/tasks.jsonl").open(encoding="utf-8"):
    row = json.loads(line)
    print(row["task_id"], row["prompt"][:100].replace("\n", " "))
    break
PY
```

To restore a runtime-compatible task tree:

```bash
python scripts/restore_hf_dataset.py data/tasks.jsonl --output-dir /tmp/nanoclaw_tasks
```

The restored tree has `tasks/data_*` directories and `scripts/data_*/verify_workplace.py`, matching the Nanoclaw recipe's discovery rules. Restoring executes no task code.

## Upload to the Hub

Create a **Dataset** repository and upload the contents of this directory (the JSONL file is tracked with Git LFS):

```bash
git lfs install
hf repo create <namespace>/nanoclaw-tasks --repo-type dataset
hf upload <namespace>/nanoclaw-tasks . . --repo-type dataset
```

Alternatively, clone the empty repository, copy this directory's contents into it, and run `git add`, `git commit`, and `git push`. Do not upload the parent workspace wholesale: it contains source exports and unrelated experiment files outside this release directory.

## Training

The recipe is designed to run inside a compatible VERL checkout (VERL is an upstream dependency, not vendored in full here). Copy `recipe/nanoclaw_recipe` into the VERL checkout or add it to `PYTHONPATH`, then follow `recipe/README.md`. `recipe/nanoclaw_recipe/train_half_turn.sh` is a portable launcher; the `.cluster.sh` file is a historical hardware-specific reference. A typical run supplies:

```text
data.train_files=/path/to/restored_tasks
data.val_files=/path/to/restored_tasks
data.custom_cls.path=pkg://nanoclaw_recipe.nanoclaw
reward.custom_reward_function.path=pkg://nanoclaw_recipe.nanoclaw
actor_rollout_ref.rollout.multi_turn.enable=True
```

The training implementation supports the following environment flags: `NANOCLAW_MASK_LOOPING_RESPONSES`, `NANOCLAW_MASK_BUDGET_EXHAUSTED_LAST_TURN`, `NANOCLAW_MASK_DUPLICATE_TOOL_RESULT_TURNS`, `NANOCLAW_MASK_ERROR_TOOL_RESULT_TURNS`, and `NANOCLAW_MASK_ONLY_POSITIVE_ADVANTAGE`. See `recipe/README.md` for the full command and lifecycle.

## Dataset construction and quality

The source export contained 7,056 task records. The strict exporter retained 6,970 records and skipped 86 records because of dangerous path literals, smoke-test failures, or Python syntax errors. Every released record has `valid: true`, a four-block source export, and a passing environment-builder smoke test. Some manifests retain non-fatal warnings (for example, a verifier that does not mention a prompt filename); these are surfaced rather than silently discarded.

The tasks are synthetic workplace scenarios spanning data cleaning, coding, documents, finance, operations, and other file-based workflows. They may contain culturally specific or fictional names and should be treated as benchmark content, not as factual personal data.

## Limitations and safety

* Verifiers are arbitrary task code. Run them only in an isolated container with resource and network restrictions.
* `env_builder.py` files are untrusted input from a benchmark export. The restore script writes files but never imports or executes them.
* The release does not include model checkpoints, rollout conversations, or private API credentials. The original verifier defaults to a local mock endpoint; configure a sandboxed endpoint before evaluation.
* Results in the paper are research claims tied to the stated model, hardware, seeds, and benchmark splits. The released task set alone does not reproduce those numbers.

## Citation

```bibtex
@article{nanoclaw2026,
  title   = {Less Harness, More Signal: Efficient In-Harness RL for Autonomous Agents},
  year    = {2026},
  note    = {Nanoclaw release; manuscript source included in this repository}
}
```

## License

The Nanoclaw-specific code and documentation are MIT licensed. The training framework remains subject to the Apache-2.0 license of upstream VERL; see `THIRD_PARTY_NOTICES.md`. Before mirroring task content to a public Hub repository, review the provenance and licensing terms of the source benchmark export.
