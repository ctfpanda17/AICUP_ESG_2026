#!/usr/bin/env python3
"""Tune/apply per-task, per-label source weights for ESG probability ensembles."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from esg_score_postprocess import LABELS, TASKS, WEIGHTS, macro_f1, normalize_label


def normalize_probs(task: str, probs: dict[str, Any]) -> dict[str, float]:
    values = {label: max(0.0, float(probs.get(label, 0.0))) for label in LABELS[task]}
    total = sum(values.values())
    if total <= 0:
        return {label: 1.0 / len(LABELS[task]) for label in LABELS[task]}
    return {label: value / total for label, value in values.items()}


def load_prediction_file(path: Path, source: str) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for row in rows:
        if "probs" not in row:
            raise ValueError(f"{path} has a row without probs")
        item = dict(row)
        item["_source"] = source
        out.append(item)
    return out


def initial_weights(sources: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    return {
        task: {label: {source: 1.0 for source in sources} for label in LABELS[task]}
        for task in TASKS
    }


def ensemble_task_probs(rows: list[dict[str, Any]], task: str, weights: dict[str, dict[str, float]]) -> dict[str, float]:
    scores = {label: 0.0 for label in LABELS[task]}
    totals = {label: 0.0 for label in LABELS[task]}
    for row in rows:
        source = row["_source"]
        probs = normalize_probs(task, row["probs"].get(task, {}))
        for label in LABELS[task]:
            weight = float(weights.get(label, {}).get(source, 1.0))
            scores[label] += weight * probs[label]
            totals[label] += weight
    return {label: scores[label] / totals[label] if totals[label] > 0 else 0.0 for label in LABELS[task]}


def argmax_label(task: str, probs: dict[str, float]) -> str:
    return max(LABELS[task], key=lambda label: float(probs.get(label, 0.0)))


def build_rows(
    rows_by_id: dict[str, list[dict[str, Any]]],
    weights: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    out = []
    for row_id, rows in sorted(rows_by_id.items(), key=lambda pair: int(pair[0]) if pair[0].isdigit() else pair[0]):
        base = rows[0]
        probs = {}
        pred = {}
        for task in TASKS:
            task_probs = ensemble_task_probs(rows, task, weights.get(task, {}))
            probs[task] = task_probs
            pred[task] = argmax_label(task, task_probs)
        out.append(
            {
                "mode": "labelwise_ensemble",
                "id": base["id"],
                "gold": base.get("gold", {}),
                "pred": pred,
                "probs": probs,
                "sources": [row["_source"] for row in rows],
            }
        )
    return out


def score_task(rows: list[dict[str, Any]], task: str) -> float:
    gold = [normalize_label(task, row["gold"].get(task)) for row in rows]
    pred = [normalize_label(task, row["pred"].get(task)) for row in rows]
    return macro_f1(gold, pred, LABELS[task])


def score_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    scores = {task: score_task(rows, task) for task in TASKS}
    scores["weighted_macro_f1"] = sum(scores[task] * WEIGHTS[task] for task in TASKS)
    return scores


def tune_weights(
    rows_by_id: dict[str, list[dict[str, Any]]],
    sources: list[str],
    grid: list[float],
    max_rounds: int,
    start_weights: dict[str, dict[str, dict[str, float]]] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    weights = start_weights or initial_weights(sources)
    for task in TASKS:
        best_score = score_task(build_rows(rows_by_id, weights), task)
        for _round in range(max_rounds):
            improved = False
            for label in LABELS[task]:
                for source in sources:
                    current = weights[task][label][source]
                    local_best = (best_score, current)
                    for value in grid:
                        trial = json.loads(json.dumps(weights))
                        trial[task][label][source] = value
                        score = score_task(build_rows(rows_by_id, trial), task)
                        if score > local_best[0]:
                            local_best = (score, value)
                    if local_best[0] > best_score:
                        weights[task][label][source] = local_best[1]
                        best_score = local_best[0]
                        improved = True
            if not improved:
                break
    return weights


def write_report(path: Path, scores: dict[str, float], weights: dict[str, Any]) -> None:
    lines = [
        "# ESG Label-Wise Ensemble Report",
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
    parser = argparse.ArgumentParser(description="Tune/apply ESG label-wise ensemble weights.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--weights-output", type=Path, default=None)
    parser.add_argument("--weights-input", type=Path, default=None)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--missing-weight-default", type=float, default=1.0)
    parser.add_argument("--common-ids-only", action="store_true")
    parser.add_argument("--weight-grid", default="0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0,3.0,4.0")
    parser.add_argument("--max-rounds", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.inputs) != len(args.names):
        raise ValueError("--inputs and --names must have the same length")

    rows_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_ids = []
    for input_path, name in zip(args.inputs, args.names):
        rows = load_prediction_file(Path(input_path), name)
        source_ids.append({str(row["id"]) for row in rows})
        for row in rows:
            rows_by_id[str(row["id"])].append(row)
    if args.common_ids_only:
        common_ids = set.intersection(*source_ids)
        rows_by_id = defaultdict(list, {row_id: rows for row_id, rows in rows_by_id.items() if row_id in common_ids})

    if args.weights_input:
        weights = json.loads(args.weights_input.read_text(encoding="utf-8"))
        for task in TASKS:
            weights.setdefault(task, {})
            for label in LABELS[task]:
                weights[task].setdefault(label, {})
                for source in args.names:
                    weights[task][label].setdefault(source, args.missing_weight_default)
    if args.tune:
        grid = [float(value) for value in args.weight_grid.split(",") if value.strip()]
        weights = tune_weights(rows_by_id, args.names, grid, args.max_rounds, weights if args.weights_input else None)
    elif not args.weights_input:
        weights = initial_weights(args.names)

    rows = build_rows(rows_by_id, weights)
    scores = score_rows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.report, scores, weights)
    if args.weights_output:
        args.weights_output.parent.mkdir(parents=True, exist_ok=True)
        args.weights_output.write_text(json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"weighted_macro_f1={scores['weighted_macro_f1']:.6f}")
    for task in TASKS:
        print(f"{task}: {scores[task]:.6f}")
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
