#!/usr/bin/env python3
"""Combine already-generated 5-fold source probabilities."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from esg_score_postprocess import LABELS, TASKS, WEIGHTS, macro_f1, normalize_label
from esg_threshold_search import apply_all_thresholds, search_task_thresholds


def read_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_probs(task: str, probs: dict[str, Any]) -> dict[str, float]:
    values = {label: max(0.0, float(probs.get(label, 0.0))) for label in LABELS[task]}
    total = sum(values.values())
    if total <= 0:
        return {label: 1.0 / len(LABELS[task]) for label in LABELS[task]}
    return {label: value / total for label, value in values.items()}


def argmax_label(task: str, probs: dict[str, float]) -> str:
    return max(LABELS[task], key=lambda label: probs.get(label, 0.0))


def average_sources(paths: list[Path]) -> list[dict[str, Any]]:
    tables = [read_json(path) for path in paths]
    ids = [str(row["id"]) for row in tables[0]]
    by_id = [{str(row["id"]): row for row in rows} for rows in tables]
    out = []
    for row_id in ids:
        base = by_id[0][row_id]
        probs = {}
        pred = {}
        for task in TASKS:
            merged = {label: 0.0 for label in LABELS[task]}
            for table in by_id:
                row_probs = normalize_probs(task, table[row_id]["probs"][task])
                for label in LABELS[task]:
                    merged[label] += row_probs[label]
            probs[task] = {label: value / len(by_id) for label, value in merged.items()}
            pred[task] = argmax_label(task, probs[task])
        out.append({"mode": "combined_5fold", "id": base["id"], "gold": base.get("gold", {}), "pred": pred, "probs": probs})
    return out


def weighted_score(rows: list[dict[str, Any]], pred_key: str) -> dict[str, float]:
    scores = {}
    for task in TASKS:
        gold = [normalize_label(task, row["gold"].get(task)) for row in rows]
        pred = [normalize_label(task, row[pred_key].get(task)) for row in rows]
        scores[task] = macro_f1(gold, pred, LABELS[task])
    scores["weighted_macro_f1"] = sum(scores[task] * WEIGHTS[task] for task in TASKS)
    return scores


def tune_thresholds(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grid = [round(0.05 + i * 0.01, 6) for i in range(91)]
    thresholds = {}
    for task in TASKS:
        thresholds[task] = search_task_thresholds(rows, task, grid)[0]
    return thresholds


def write_report(path: Path, raw: dict[str, float], tuned: dict[str, float], thresholds: dict[str, dict[str, float]]) -> None:
    lines = ["# Combined 5-Fold Report", "", "| task | raw | thresholded |", "|---|---:|---:|"]
    for task in TASKS + ["weighted_macro_f1"]:
        lines.append(f"| {task} | {raw[task]:.6f} | {tuned[task]:.6f} |")
    lines.extend(["", "```json", json.dumps(thresholds, ensure_ascii=False, indent=2), "```"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_submission(path: Path, rows: list[dict[str, Any]], pred_key: str) -> None:
    def normalize_submission(task: str, value: str) -> str:
        if task == "verification_timeline" and value == "longer_than_5_years":
            return "more_than_5_years"
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", *TASKS], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            pred = row[pred_key]
            writer.writerow({"id": row["id"], **{task: normalize_submission(task, pred[task]) for task in TASKS}})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--oof", type=Path, nargs="+", required=True)
    parser.add_argument("--test", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    oof_rows = average_sources(args.oof)
    write_json(args.output_dir / "oof_combined_predictions.json", oof_rows)
    raw = weighted_score(oof_rows, "pred")
    thresholds = tune_thresholds(oof_rows)
    tuned_rows = apply_all_thresholds(oof_rows, thresholds)
    write_json(args.output_dir / "oof_combined_thresholded.json", tuned_rows)
    tuned = weighted_score(tuned_rows, "pred_thresholded")
    write_json(args.output_dir / "oof_combined_thresholds.json", thresholds)
    write_report(args.output_dir / "oof_combined_report.md", raw, tuned, thresholds)

    test_rows = average_sources(args.test)
    thresholded_test_rows = apply_all_thresholds(test_rows, thresholds)
    write_json(args.output_dir / "test_combined_raw_predictions.json", test_rows)
    write_json(args.output_dir / "test_combined_thresholded_predictions.json", thresholded_test_rows)
    write_submission(args.output_dir / "test_combined_raw_submission.csv", test_rows, "pred")
    write_submission(args.output_dir / "test_combined_thresholded_submission.csv", thresholded_test_rows, "pred_thresholded")

    print(f"raw weighted={raw['weighted_macro_f1']:.6f}")
    print(f"thresholded weighted={tuned['weighted_macro_f1']:.6f}")
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
