#!/usr/bin/env python3
"""Apply a fixed per-task source selection to labeled or unlabeled predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from esg_score_postprocess import TASKS, normalize_label


def parse_source(value: str) -> tuple[str, Path, str | None]:
    parts = value.split("=", 2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("source must be name=path or name=path=pred_key")
    return parts[0], Path(parts[1]), parts[2] if len(parts) == 3 else None


def parse_task_source(value: str) -> tuple[str, str]:
    parts = value.split("=", 1)
    if len(parts) != 2 or parts[0] not in TASKS:
        raise argparse.ArgumentTypeError(f"task-source must be one of {TASKS}=source_name")
    return parts[0], parts[1]


def row_prediction(row: dict[str, Any], pred_key: str | None) -> dict[str, Any]:
    if pred_key:
        return row[pred_key]
    return row.get("pred_taskwise") or row.get("pred_thresholded") or row.get("pred")


def load_source(path: Path, pred_key: str | None) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in rows:
        pred = row_prediction(row, pred_key)
        out[str(row["id"])] = {"row": row, "pred": pred}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply fixed taskwise ESG prediction sources.")
    parser.add_argument("--source", action="append", required=True, type=parse_source, help="name=path or name=path=pred_key")
    parser.add_argument("--task-source", action="append", required=True, type=parse_task_source, help="task=source_name")
    parser.add_argument("--output", required=True)
    parser.add_argument("--submission-output", help="Optional compact JSON list with id and task labels only.")
    args = parser.parse_args()

    sources = {name: load_source(path, pred_key) for name, path, pred_key in args.source}
    task_sources = dict(args.task_source)
    missing_tasks = [task for task in TASKS if task not in task_sources]
    if missing_tasks:
        raise SystemExit(f"Missing task-source for: {missing_tasks}")
    missing_sources = sorted({source for source in task_sources.values() if source not in sources})
    if missing_sources:
        raise SystemExit(f"Unknown task sources: {missing_sources}")

    common_ids = set.intersection(*(set(rows) for rows in sources.values()))
    if not common_ids:
        raise SystemExit("No common ids across sources.")

    first_source = sources[next(iter(sources))]
    output_rows = []
    compact_rows = []
    for row_id in sorted(common_ids, key=lambda value: int(value) if value.isdigit() else value):
        base = first_source[row_id]["row"]
        pred = {}
        for task in TASKS:
            source = task_sources[task]
            pred[task] = normalize_label(task, sources[source][row_id]["pred"].get(task))
        item = {"id": base["id"], "pred": pred, "source_by_task": task_sources}
        if "gold" in base:
            item["gold"] = base["gold"]
        output_rows.append(item)
        compact_rows.append({"id": base["id"], **pred})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    if args.submission_output:
        Path(args.submission_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.submission_output).write_text(json.dumps(compact_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.submission_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
