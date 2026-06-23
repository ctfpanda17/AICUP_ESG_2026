#!/usr/bin/env python3
"""Evaluate a selective transition rule on validation predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from esg_score_postprocess import LABELS, TASKS, WEIGHTS, macro_f1


def score(rows: list[tuple[dict, dict]]) -> dict[str, float]:
    scores = {}
    for task in TASKS:
        scores[task] = macro_f1([gold[task] for gold, _ in rows], [pred[task] for _, pred in rows], LABELS[task])
    scores["weighted_macro_f1"] = sum(scores[task] * WEIGHTS[task] for task in TASKS)
    return scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-val", type=Path, required=True)
    parser.add_argument("--cand-val", type=Path, required=True)
    parser.add_argument("--base-key", default="pred_taskwise")
    parser.add_argument("--cand-key", default="pred_oof")
    parser.add_argument("--task", required=True)
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--require-base", action="append", default=[])
    args = parser.parse_args()

    base_rows = json.loads(args.base_val.read_text(encoding="utf-8"))
    cand_by_id = {str(row["id"]): row for row in json.loads(args.cand_val.read_text(encoding="utf-8"))}
    requirements = [tuple(item.split("=", 1)) for item in args.require_base]

    before = [(row["gold"], dict(row[args.base_key])) for row in base_rows]
    after = []
    changed = 0
    for row in base_rows:
        pred = dict(row[args.base_key])
        cand = cand_by_id[str(row["id"])][args.cand_key][args.task]
        if pred[args.task] == args.old and cand == args.new and all(pred.get(field) == value for field, value in requirements):
            pred[args.task] = args.new
            changed += 1
        after.append((row["gold"], pred))

    print(json.dumps({"changed": changed, "before": score(before), "after": score(after)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
