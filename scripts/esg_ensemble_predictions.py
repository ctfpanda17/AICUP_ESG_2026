#!/usr/bin/env python3
"""
Ensemble ESG validation predictions with probability averaging.

Input files must use the week9.py prediction format and include `probs`.
Rows are grouped by `(id, mode)`, then model probabilities are averaged per
task. Optionally, greedy per-task model weights are selected on validation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from esg_score_postprocess import LABELS, TASKS, WEIGHTS, macro_f1, normalize_label


def load_prediction_file(path: Path, source: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    out = []
    for row in rows:
        if "probs" not in row:
            raise ValueError(f"{path} has a row without probs; run week9.py with --save-probs")
        item = dict(row)
        item["_source"] = source
        out.append(item)
    return out


def argmax_label(task: str, probs: dict[str, float]) -> str:
    return max(LABELS[task], key=lambda label: float(probs.get(label, 0.0)))


def normalize_probs(task: str, probs: dict[str, Any]) -> dict[str, float]:
    values = {label: max(0.0, float(probs.get(label, 0.0))) for label in LABELS[task]}
    total = sum(values.values())
    if total <= 0:
        return {label: 1.0 / len(LABELS[task]) for label in LABELS[task]}
    return {label: value / total for label, value in values.items()}


def average_probs(
    rows: list[dict[str, Any]],
    task: str,
    weights: dict[str, float],
) -> dict[str, float]:
    merged = {label: 0.0 for label in LABELS[task]}
    total_weight = 0.0
    for row in rows:
        source = row["_source"]
        weight = weights.get(source, 1.0)
        probs = normalize_probs(task, row["probs"].get(task, {}))
        for label in LABELS[task]:
            merged[label] += weight * probs[label]
        total_weight += weight
    if total_weight <= 0:
        total_weight = 1.0
    return {label: value / total_weight for label, value in merged.items()}


def build_ensemble_rows(
    rows_by_id: dict[Any, list[dict[str, Any]]],
    weights_by_task: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    out = []
    for row_id, rows in sorted(rows_by_id.items(), key=lambda pair: str(pair[0])):
        base = rows[0]
        probs = {}
        pred = {}
        for task in TASKS:
            task_probs = average_probs(rows, task, weights_by_task.get(task, {}))
            probs[task] = task_probs
            pred[task] = argmax_label(task, task_probs)
        out.append(
            {
                "mode": "ensemble",
                "id": row_id,
                "gold": base["gold"],
                "pred": pred,
                "probs": probs,
                "sources": [row["_source"] for row in rows],
            }
        )
    return out


def score_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    scores = {}
    for task in TASKS:
        gold = [normalize_label(task, row["gold"].get(task)) for row in rows]
        pred = [normalize_label(task, row["pred"].get(task)) for row in rows]
        scores[task] = macro_f1(gold, pred, LABELS[task])
    scores["weighted_macro_f1"] = sum(scores[task] * WEIGHTS[task] for task in TASKS)
    return scores


def tune_weights(
    rows_by_id: dict[Any, list[dict[str, Any]]],
    sources: list[str],
    grid: list[float],
) -> dict[str, dict[str, float]]:
    weights_by_task = {task: {source: 1.0 for source in sources} for task in TASKS}
    for task in TASKS:
        best_rows = build_ensemble_rows(rows_by_id, weights_by_task)
        best_score = score_rows(best_rows)[task]
        improved = True
        while improved:
            improved = False
            for source in sources:
                local_best = (best_score, weights_by_task[task][source])
                for weight in grid:
                    trial = {t: dict(w) for t, w in weights_by_task.items()}
                    trial[task][source] = weight
                    score = score_rows(build_ensemble_rows(rows_by_id, trial))[task]
                    if score > local_best[0]:
                        local_best = (score, weight)
                if local_best[0] > best_score:
                    weights_by_task[task][source] = local_best[1]
                    best_score = local_best[0]
                    improved = True
    return weights_by_task


def write_report(path: Path, scores: dict[str, float], weights: dict[str, dict[str, float]]) -> None:
    lines = [
        "# ESG Ensemble Report",
        "",
        "## Scores",
        "",
        "| task | macro F1 |",
        "|---|---:|",
    ]
    for task in TASKS + ["weighted_macro_f1"]:
        lines.append(f"| {task} | {scores[task]:.6f} |")
    lines.extend(["", "## Weights", "", "```json", json.dumps(weights, ensure_ascii=False, indent=2), "```"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average or tune ESG model prediction probabilities.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Prediction JSON files with probs.")
    parser.add_argument("--names", nargs="+", default=None, help="Optional source names matching --inputs.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--tune-weights", action="store_true")
    parser.add_argument("--weight-grid", default="0.25,0.5,0.75,1.0,1.25,1.5,2.0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(path) for path in args.inputs]
    if args.names and len(args.names) != len(input_paths):
        raise ValueError("--names must have the same length as --inputs")
    names = args.names or [path.parent.name for path in input_paths]

    rows_by_id: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for path, name in zip(input_paths, names):
        for row in load_prediction_file(path, name):
            rows_by_id[row["id"]].append(row)

    sources = list(names)
    weights = {task: {source: 1.0 for source in sources} for task in TASKS}
    if args.tune_weights:
        grid = [float(value) for value in args.weight_grid.split(",") if value.strip()]
        weights = tune_weights(rows_by_id, sources, grid)

    rows = build_ensemble_rows(rows_by_id, weights)
    scores = score_rows(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    write_report(Path(args.report), scores, weights)

    print(f"weighted_macro_f1={scores['weighted_macro_f1']:.6f}")
    print(f"wrote {output_path}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
