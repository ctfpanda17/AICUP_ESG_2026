#!/usr/bin/env python3
"""
Search per-task class thresholds for ESG predictions with probabilities.

Expected input row format:
{
  "gold": {"evidence_quality": "Clear", ...},
  "pred": {"evidence_quality": "Clear", ...},
  "probs": {"evidence_quality": {"Clear": 0.7, "Not Clear": 0.2, ...}, ...}
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from esg_score_postprocess import LABELS, TASKS, WEIGHTS, macro_f1, normalize_label


def argmax_label(task: str, probs: dict[str, float]) -> str:
    if not probs:
        return "No" if task == "promise_status" else "N/A"
    return max(LABELS[task], key=lambda label: float(probs.get(label, 0.0)))


def apply_thresholds_to_task(task: str, probs: dict[str, float], thresholds: dict[str, float]) -> str:
    # Priority goes to labels with explicit thresholds, ordered by probability.
    candidates = []
    for label, threshold in thresholds.items():
        prob = float(probs.get(label, 0.0))
        if prob >= threshold:
            candidates.append((prob, label))
    if candidates:
        return max(candidates)[1]
    return argmax_label(task, probs)


def score_task(rows: list[dict[str, Any]], task: str, thresholds: dict[str, float]) -> float:
    gold = [normalize_label(task, row["gold"].get(task)) for row in rows]
    pred = [
        apply_thresholds_to_task(task, row.get("probs", {}).get(task, {}), thresholds)
        for row in rows
    ]
    return macro_f1(gold, pred, LABELS[task])


def search_task_thresholds(rows: list[dict[str, Any]], task: str, grid: list[float]) -> tuple[dict[str, float], float]:
    base_score = score_task(rows, task, {})
    best_thresholds: dict[str, float] = {}
    best_score = base_score

    # Greedy one-vs-rest thresholds. This is intentionally simple and robust for
    # small validation sets.
    candidate_labels = [label for label in LABELS[task] if label != "N/A"]
    if task == "promise_status":
        candidate_labels = ["Yes", "No"]

    improved = True
    while improved:
        improved = False
        for label in candidate_labels:
            local_best = (best_score, None)
            for threshold in grid:
                trial = dict(best_thresholds)
                trial[label] = threshold
                score = score_task(rows, task, trial)
                if score > local_best[0]:
                    local_best = (score, threshold)
            if local_best[1] is not None and local_best[0] > best_score:
                best_thresholds[label] = local_best[1]
                best_score = local_best[0]
                improved = True
    return best_thresholds, best_score


def apply_all_thresholds(rows: list[dict[str, Any]], thresholds: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        pred = {}
        for task in TASKS:
            pred[task] = apply_thresholds_to_task(task, row.get("probs", {}).get(task, {}), thresholds.get(task, {}))
        item["pred_thresholded"] = pred
        out.append(item)
    return out


def weighted_score(rows: list[dict[str, Any]], pred_key: str) -> dict[str, float]:
    scores = {}
    for task in TASKS:
        gold = [normalize_label(task, row["gold"].get(task)) for row in rows]
        pred = [normalize_label(task, row[pred_key].get(task)) for row in rows]
        scores[task] = macro_f1(gold, pred, LABELS[task])
    scores["weighted_macro_f1"] = sum(scores[task] * WEIGHTS[task] for task in TASKS)
    return scores


def write_report(path: Path, base: dict[str, float], tuned: dict[str, float], thresholds: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ESG Threshold Search Report",
        "",
        "## Scores",
        "",
        "| task | argmax | thresholded | delta |",
        "|---|---:|---:|---:|",
    ]
    for task in TASKS + ["weighted_macro_f1"]:
        lines.append(f"| {task} | {base[task]:.6f} | {tuned[task]:.6f} | {tuned[task] - base[task]:+.6f} |")
    lines.extend(["", "## Thresholds", "", "```json", json.dumps(thresholds, ensure_ascii=False, indent=2), "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search thresholds for ESG probability predictions.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--grid-start", type=float, default=0.05)
    parser.add_argument("--grid-end", type=float, default=0.95)
    parser.add_argument("--grid-step", type=float, default=0.05)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not rows or "probs" not in rows[0]:
        raise SystemExit("Input predictions must contain a 'probs' object. Re-run week9.py with --save-probs.")

    n_steps = int(round((args.grid_end - args.grid_start) / args.grid_step)) + 1
    grid = [round(args.grid_start + i * args.grid_step, 6) for i in range(n_steps)]

    thresholds = {}
    for task in TASKS:
        task_thresholds, task_score = search_task_thresholds(rows, task, grid)
        thresholds[task] = task_thresholds
        print(f"{task}: {task_score:.6f} thresholds={task_thresholds}")

    thresholded = apply_all_thresholds(rows, thresholds)
    base = weighted_score(rows, "pred")
    tuned = weighted_score(thresholded, "pred_thresholded")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(thresholded, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(Path(args.report), base, tuned, thresholds)
    print(f"argmax weighted={base['weighted_macro_f1']:.6f}")
    print(f"tuned  weighted={tuned['weighted_macro_f1']:.6f}")
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
