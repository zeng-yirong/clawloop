#!/usr/bin/env python3
"""Convert restored Nanoclaw task folders into portable JSONL records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def build_record(task_dir: Path) -> dict:
    manifest_path = task_dir / "manifest.json"
    manifest = json.loads(read_text(manifest_path)) if manifest_path.is_file() else {}
    required = ("task.yaml", "prompts.md", "env_builder.py", "workplace_verifier.py")
    missing = [name for name in required if not (task_dir / name).is_file()]
    if missing:
        raise ValueError(f"{task_dir}: missing {', '.join(missing)}")
    task_id = str(manifest.get("export_id") or task_dir.name)
    return {
        "task_id": task_id,
        "prompt": read_text(task_dir / "prompts.md"),
        "task_yaml": read_text(task_dir / "task.yaml"),
        "env_builder": read_text(task_dir / "env_builder.py"),
        "verifier": read_text(task_dir / "workplace_verifier.py"),
        "manifest": manifest,
        "source_files": {name: name for name in required},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path, help="Directory containing data_* task folders")
    parser.add_argument("output", type=Path, help="Output JSONL path")
    args = parser.parse_args()
    task_dirs = sorted(path for path in args.source_dir.glob("data_*") if path.is_dir())
    if not task_dirs:
        raise SystemExit(f"no data_* directories found in {args.source_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for task_dir in task_dirs:
            try:
                record = build_record(task_dir)
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
                raise SystemExit(str(exc)) from exc
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
    print(f"wrote {written} records to {args.output}")


if __name__ == "__main__":
    main()

