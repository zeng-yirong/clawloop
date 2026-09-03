#!/usr/bin/env python3
"""Validate JSONL records without executing benchmark code."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REQUIRED = ("task_id", "prompt", "task_yaml", "env_builder", "verifier", "manifest", "source_files")
ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_])/(?:home|root|mnt|opt|tmp)/")
TASK_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    args = parser.parse_args()
    seen: set[str] = set()
    total = 0
    warnings = 0
    with args.jsonl.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [key for key in REQUIRED if key not in row]
            if missing:
                raise SystemExit(f"line {line_no}: missing {missing}")
            task_id = str(row["task_id"])
            if not TASK_ID.fullmatch(task_id):
                raise SystemExit(f"line {line_no}: unsafe task_id {task_id!r}")
            if task_id in seen:
                raise SystemExit(f"line {line_no}: duplicate task_id {task_id}")
            seen.add(task_id)
            manifest = row["manifest"]
            if not isinstance(manifest, dict) or manifest.get("valid") is not True:
                raise SystemExit(f"line {line_no}: task {task_id} is not marked valid")
            if any(not isinstance(row[key], str) or not row[key].strip() for key in ("prompt", "task_yaml", "env_builder", "verifier")):
                raise SystemExit(f"line {line_no}: task {task_id} has empty source text")
            for key in ("env_builder", "verifier"):
                if ABSOLUTE_PATH.search(row[key]):
                    warnings += 1
            total += 1
    print(f"validated {total} records; {warnings} absolute-path warnings")


if __name__ == "__main__":
    main()
