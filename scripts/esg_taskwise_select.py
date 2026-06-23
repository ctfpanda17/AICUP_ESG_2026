#!/usr/bin/env python3
"""Select the best prediction source independently for each ESG task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from esg_score_postprocess import LABELS, TASKS, WEIGHTS, macro_f1, normalize_label


def row_prediction(row: dict[str, Any], pred_key: str | None) -> dict[str, str]:
    if pred_key and pred_key in row:
        return row[pred_key]
    if "pred_thresholded" in row:
        return row["pred_thresholded"]
    return row["pred"]


def load_rows(path: Path, pred_key: str | None) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    by_id = {}
    for row in rows:
        row_id = str(row["id"])
        item = dict(row)
        item["_selected_pred"] = row_prediction(row, pred_key)
        by_id[row_id] = item
    return by_id


def score_task(rows_by_id: dict[str, dict[str, Any]], task: str) -> float:
    rows = list(rows_by_id.values())
    gold = [normalize_label(task, row["gold"].get(task)) for row in rows]
    pred = [normalize_label(task, row["_selected_pred"].get(task)) for row in rows]
    return macro_f1(gold, pred, LABELS[task])


def score_all(rows: list[dict[str, Any]], pred_key: str) -> dict[str, float]:
    scores = {}
    for task in TASKS:
        gold = [normalize_label(task, row["gold"].get(task)) for row in rows]
        pred = [normalize_label(task, row[pred_key].get(task)) for row in rows]
        scores[task] = macro_f1(gold, pred, LABELS[task])
    scores["weighted_macro_f1"] = sum(scores[task] * WEIGHTS[task] for task in TASKS)
    return scores


def parse_source(value: str) -> tuple[str, Path, str | None]:
    parts = value.split("=", 2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("source must be name=path or name=path=pred_key")
    name = parts[0]
    path = Path(parts[1])
    pred_key = parts[2] if len(parts) == 3 else None
    return name, path, pred_key


def write_report(
    path: Path,
    source_scores: dict[str, dict[str, float]],
    selected: dict[str, str],
    final_scores: dict[str, float],
) -> None:
    lines = [
        "# ESG Task-Wise Selection Report",
        "",
        "## Selected Sources",
        "",
        "| task | source | Macro-F1 |",
        "|---|---|---:|",
    ]
    for task in TASKS:
        source = selected[task]
        lines.append(f"| {task} | {source} | {source_scores[source][task]:.6f} |")
    lines.extend(
        [
            "",
            "## Final Scores",
            "",
            "| task | Macro-F1 |",
            "|---|---:|",
        ]
    )
    for task in TASKS + ["weighted_macro_f1"]:
        lines.append(f"| {task} | {final_scores[task]:.6f} |")
    lines.extend(["", "## Source Scores", "", "| source | " + " | ".join(TASKS) + " | weighted |", "|---" + "|---:" * (len(TASKS) + 1) + "|"])
    for source, scores in sorted(source_scores.items()):
        values = " | ".join(f"{scores[task]:.6f}" for task in TASKS)
        lines.append(f"| {source} | {values} | {scores['weighted_macro_f1']:.6f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick best source independently for each ESG task.")
    parser.add_argument("--source", action="append", required=True, type=parse_source, help="name=path or name=path=pred_key")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    sources = {name: load_rows(path, pred_key) for name, path, pred_key in args.source}
    common_ids = set.intersection(*(set(rows) for rows in sources.values()))
    if not common_ids:
        raise SystemExit("No common ids across sources.")
    sources = {
        name: {row_id: rows[row_id] for row_id in common_ids}
        for name, rows in sources.items()
    }

    source_scores = {}
    for name, rows_by_id in sources.items():
        rows = list(rows_by_id.values())
        pred_rows = []
        for row in rows:
            item = {"gold": row["gold"], "pred": row["_selected_pred"]}
            pred_rows.append(item)
        source_scores[name] = score_all(pred_rows, "pred")

    selected = {}
    for task in TASKS:
        selected[task] = max(source_scores, key=lambda name: source_scores[name][task])

    first_source = next(iter(sources.values()))
    output_rows = []
    for row_id in sorted(common_ids, key=lambda value: int(value) if value.isdigit() else value):
        base = first_source[row_id]
        pred = {}
        source_by_task = {}
        for task in TASKS:
            source = selected[task]
            pred[task] = sources[source][row_id]["_selected_pred"][task]
            source_by_task[task] = source
        output_rows.append(
            {
                "id": base["id"],
                "gold": base["gold"],
                "pred_taskwise": pred,
                "source_by_task": source_by_task,
            }
        )

    final_scores = score_all(output_rows, "pred_taskwise")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_rows, f, ensure_ascii=False, indent=2)
    write_report(Path(args.report), source_scores, selected, final_scores)
    print(f"weighted_macro_f1={final_scores['weighted_macro_f1']:.6f}")
    for task in TASKS:
        print(f"{task}: {final_scores[task]:.6f} from {selected[task]}")
    print(f"wrote {output_path}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
