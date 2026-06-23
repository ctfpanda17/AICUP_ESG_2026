#!/usr/bin/env python3
"""Apply selected task label transitions from JSON predictions to a CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--pred-json", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--transition", action="append", required=True, help="OLD->NEW")
    parser.add_argument("--require-base", action="append", default=[], help="FIELD=VALUE condition on the base CSV row")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    transitions = set()
    for item in args.transition:
        old, new = item.split("->", 1)
        transitions.add((old, new))
    requirements = []
    for item in args.require_base:
        field, value = item.split("=", 1)
        requirements.append((field, value))

    with args.base_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    pred_rows = json.loads(args.pred_json.read_text(encoding="utf-8"))
    pred_by_id = {str(row["id"]): row for row in pred_rows}

    changed = 0
    for row in rows:
        old = row[args.task]
        new = pred_by_id[str(row["id"])][args.task]
        if (old, new) in transitions and all(row.get(field) == value for field, value in requirements):
            row[args.task] = new
            changed += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "promise_status", "verification_timeline", "evidence_status", "evidence_quality"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "changed": changed, "transitions": sorted(transitions)}, ensure_ascii=False))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
