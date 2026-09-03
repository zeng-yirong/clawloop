#!/usr/bin/env python3
"""Restore portable Nanoclaw JSONL records to the directory runtime layout."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TASK_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.jsonl.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if args.limit is not None and count >= args.limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = str(row["task_id"])
            if not TASK_ID.fullmatch(task_id):
                raise SystemExit(f"line {line_no}: unsafe task_id {task_id!r}")
            task_dir = args.output_dir / "tasks" / task_id
            write_text(task_dir / "task.yaml", str(row["task_yaml"]))
            write_text(task_dir / "env_builder.py", str(row["env_builder"]))
            # Keep the flat export's manifest-relative filenames as well as
            # the canonical legacy layout used by older Nanoclaw runners.
            write_text(task_dir / "prompts.md", str(row["prompt"]))
            write_text(task_dir / "workplace_verifier.py", str(row["verifier"]))
            write_text(args.output_dir / "tasks" / "prompts" / f"{task_id}.md", str(row["prompt"]))
            write_text(args.output_dir / "scripts" / task_id / "verify_workplace.py", str(row["verifier"]))
            manifest = row.get("manifest")
            if isinstance(manifest, dict):
                write_text(task_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            count += 1
    print(f"restored {count} records from {args.jsonl} to {args.output_dir}")


if __name__ == "__main__":
    main()
