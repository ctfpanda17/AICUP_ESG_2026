#!/usr/bin/env python3
"""Create a submission by taking selected tasks from JSON prediction sources."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from esg_score_postprocess import TASKS, normalize_label


def load_json_preds(path: Path, pred_key: str) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["id"]): row[pred_key] for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-csv", required=True, type=Path)
    parser.add_argument("--json-source", required=True, type=Path)
    parser.add_argument("--pred-key", default="pred_thresholded")
    parser.add_argument("--replace-task", action="append", required=True, choices=TASKS)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = load_json_preds(args.json_source, args.pred_key)
    replace_tasks = set(args.replace_task)
    out_rows = []
    with args.base_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            row_id = str(row["id"])
            pred = dict(row)
            for task in replace_tasks:
                pred[task] = normalize_label(task, source[row_id][task])
                if task == "verification_timeline" and pred[task] == "longer_than_5_years":
                    pred[task] = "more_than_5_years"
            out_rows.append(pred)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", *TASKS], lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
