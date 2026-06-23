#!/usr/bin/env python3
"""
Score and post-process ESG competition predictions.

Input format is compatible with output/week9_*/week9_predictions.json:
[
  {"id": ..., "gold": {"promise_status": ...}, "pred": {"promise_status": ...}},
  ...
]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


TASKS = [
    "promise_status",
    "verification_timeline",
    "evidence_status",
    "evidence_quality",
]

LABELS = {
    "promise_status": ["Yes", "No"],
    "verification_timeline": ["already", "within_2_years", "between_2_and_5_years", "longer_than_5_years", "N/A"],
    "evidence_status": ["Yes", "No", "N/A"],
    "evidence_quality": ["Clear", "Not Clear", "Misleading", "N/A"],
}

WEIGHTS = {
    "promise_status": 0.20,
    "verification_timeline": 0.15,
    "evidence_status": 0.30,
    "evidence_quality": 0.35,
}

ALIASES = {
    "NA": "N/A",
    "N\\A": "N/A",
    "None": "N/A",
    "yes": "Yes",
    "no": "No",
    "clear": "Clear",
        "not_clear": "Not Clear",
        "not clear": "Not Clear",
        "NotClear": "Not Clear",
        "misleading": "Misleading",
        "more_than_5_years": "longer_than_5_years",
        "more than 5 years": "longer_than_5_years",
    }


def normalize_label(task: str, value: Any) -> str:
    text = str(value).strip() if value is not None else "N/A"
    text = ALIASES.get(text, text)
    if text in LABELS[task]:
        return text
    return "No" if task == "promise_status" else "N/A"


def enforce_label_rules(pred: dict[str, Any], mode: str = "strict") -> dict[str, str]:
    fixed = {task: normalize_label(task, pred.get(task)) for task in TASKS}

    if fixed["promise_status"] == "No":
        fixed["verification_timeline"] = "N/A"
        fixed["evidence_status"] = "N/A"
        fixed["evidence_quality"] = "N/A"
        return fixed

    if mode == "strict":
        return fixed

    if fixed["verification_timeline"] == "N/A":
        # In this competition, a positive commitment should normally have an
        # expected timing. If the model abstains, use the common long-term
        # fallback rather than leaving an invalid positive/N/A combination.
        fixed["verification_timeline"] = "longer_than_5_years"

    if fixed["evidence_status"] == "N/A":
        fixed["evidence_quality"] = "N/A"
    elif fixed["evidence_status"] == "No":
        fixed["evidence_quality"] = "N/A"
    elif fixed["evidence_quality"] == "N/A":
        fixed["evidence_quality"] = "Not Clear"

    return fixed


def macro_f1(gold: list[str], pred: list[str], labels: list[str]) -> float:
    scores = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(score)
    return sum(scores) / len(scores)


def score_rows(rows: list[dict[str, Any]], pred_key: str = "pred") -> dict[str, float]:
    task_scores = {}
    for task in TASKS:
        gold = [normalize_label(task, row["gold"].get(task)) for row in rows]
        pred = [normalize_label(task, row[pred_key].get(task)) for row in rows]
        task_scores[task] = macro_f1(gold, pred, LABELS[task])
    task_scores["mean_macro_f1"] = sum(task_scores[task] for task in TASKS) / len(TASKS)
    task_scores["weighted_macro_f1"] = sum(WEIGHTS[task] * task_scores[task] for task in TASKS)
    return task_scores


def add_postprocessed_predictions(rows: list[dict[str, Any]], mode: str) -> tuple[list[dict[str, Any]], Counter]:
    out = []
    changes: Counter = Counter()
    for row in rows:
        item = dict(row)
        fixed = enforce_label_rules(row.get("pred", {}), mode=mode)
        item["pred_postprocessed"] = fixed
        for task in TASKS:
            old = normalize_label(task, row.get("pred", {}).get(task))
            if old != fixed[task]:
                changes[task] += 1
        out.append(item)
    return out, changes


def write_score_report(path: Path, before: dict[str, float], after: dict[str, float], changes: Counter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ESG Score/Postprocess Report",
        "",
        "| task | before | after | delta | changed |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in TASKS + ["mean_macro_f1", "weighted_macro_f1"]:
        changed = changes.get(task, "") if task in TASKS else ""
        lines.append(f"| {task} | {before[task]:.6f} | {after[task]:.6f} | {after[task] - before[task]:+.6f} | {changed} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score and post-process ESG predictions.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--mode", choices=["none", "strict", "full", "auto"], default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    before = score_rows(rows, pred_key="pred")

    mode = args.mode
    if mode == "auto":
        candidates = []
        for candidate_mode in ("none", "strict", "full"):
            if candidate_mode == "none":
                candidate_rows = [dict(row, pred_postprocessed={task: normalize_label(task, row.get("pred", {}).get(task)) for task in TASKS}) for row in rows]
                candidate_changes = Counter()
            else:
                candidate_rows, candidate_changes = add_postprocessed_predictions(rows, mode=candidate_mode)
            candidate_score = score_rows(candidate_rows, pred_key="pred_postprocessed")
            candidates.append((candidate_score["weighted_macro_f1"], candidate_mode, candidate_rows, candidate_changes))
        _score, mode, processed, changes = max(candidates, key=lambda item: item[0])
        print(f"auto selected mode={mode}")
    elif mode == "none":
        processed = [dict(row, pred_postprocessed={task: normalize_label(task, row.get("pred", {}).get(task)) for task in TASKS}) for row in rows]
        changes = Counter()
    else:
        processed, changes = add_postprocessed_predictions(rows, mode=mode)

    after = score_rows(processed, pred_key="pred_postprocessed")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(processed, ensure_ascii=False, indent=2), encoding="utf-8")
    write_score_report(Path(args.report), before, after, changes)
    print(f"before weighted_macro_f1={before['weighted_macro_f1']:.6f}")
    print(f"after  weighted_macro_f1={after['weighted_macro_f1']:.6f}")
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
