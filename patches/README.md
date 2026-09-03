# VERL integration patch set

The files in this directory are the Nanoclaw-specific integration points copied from the working VERL checkout used for the experiments:

* `nanoclaw_support.py` records rollout metadata, detects bad turns, and persists trajectories;
* `tool_agent_loop.py` wires Nanoclaw workspace events into the multi-turn agent loop;
* `ray_trainer.py` applies the delayed positive-advantage mask after GRPO advantages are available;
* the two `test_*` files are CPU-only regression tests for the masking and final-answer bonus behavior.

These files are snapshots of files that also contain upstream VERL code. Apply them against the matching VERL revision rather than importing this directory as a standalone Python package. The upstream revision and license are recorded in `THIRD_PARTY_NOTICES.md`.

