# Running the recipe

`nanoclaw_recipe/` is the self-contained Nanoclaw task adapter. It expects a compatible VERL checkout because the agent loop, `DataProto`, and distributed workers are provided by VERL. The portable `train_9b.sh` and `train_27b.sh` profiles are distilled from the original experiment launchers at the workspace root; they retain the validated 31K context, FSDP2, GRPO, vLLM multi-turn, and AAM defaults without the original cluster's package installation or private paths.

1. Restore `data/tasks.jsonl` with `scripts/restore_hf_dataset.py`.
2. Add `recipe/` to the VERL checkout's `PYTHONPATH` (or copy `recipe/nanoclaw_recipe` into its source tree).
3. Set `data.train_files` and `data.val_files` to the restored directory.
4. Run `nanoclaw_recipe/train_9b.sh` or `nanoclaw_recipe/train_27b.sh` after reviewing model, GPU, and verifier settings. For a custom model, use `train_half_turn.sh` directly.

The detailed Chinese runbook copied from the experiment workspace is in `nanoclaw_recipe/README.md`. It documents task discovery, workspace isolation, verifier scoring, rollout-only inference, and all AAM environment flags. `train_half_turn.sh` is a portable launcher; the historical `train_half_turn.cluster.sh` contains the original NPU/ModelArts settings and is provided for provenance only.

The minimal portable integration points are:

```text
nanoclaw_recipe/common.py       task discovery and safe bundle copying
nanoclaw_recipe/nanoclaw.py     RLHF dataset, workspace tools, and reward
nanoclaw_recipe/runtime/        standalone rollout/runtime helpers
nanoclaw_recipe/nanoclaw_tool_config.yaml
```

Run the CPU regression tests from a matching VERL checkout:

```bash
pytest -q tests/recipe/nanoclaw/test_behavior_masks_on_cpu.py \
          tests/recipe/nanoclaw/test_final_answer_bonus_on_cpu.py
```
