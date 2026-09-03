# Task record schema

`data/tasks.jsonl` is UTF-8 JSON Lines. Each line is one validated task and contains:

| Field | Type | Meaning |
| --- | --- | --- |
| `task_id` | string | Stable export identifier such as `data_006837`. |
| `prompt` | string | User-facing task instruction. |
| `task_yaml` | string | Runtime/task metadata from `task.yaml`. |
| `env_builder` | string | Source of `env_builder.py`, used to create the initial workspace. |
| `verifier` | string | Source of `workplace_verifier.py`, used to score the final workspace. |
| `manifest` | object | Original export manifest and validation metadata. |
| `source_files` | object | Relative source filenames included in the record. |

No record contains an already-created workspace. Consumers should run the environment builder inside a sandbox for each rollout and execute the verifier only after the agent stops.

