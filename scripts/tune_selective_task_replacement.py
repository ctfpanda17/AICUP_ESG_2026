#!/usr/bin/env python3
"""Tune selective replacement of one task from a candidate probability source."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from esg_score_postprocess import LABELS, macro_f1, normalize_label


def load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_pred(row: dict[str, Any], key: str) -> dict[str, str]:
    return row[key]


def confidence(row: dict[str, Any], task: str, label: str) -> tuple[float, float]:
    probs = {k: float(v) for k, v in row.get("probs", {}).get(task, {}).items()}
    prob = probs.get(label, 0.0)
    others = [v for k, v in probs.items() if k != label]
    margin = prob - max(others or [0.0])
    return prob, margin


def score(gold: list[str], pred: list[str], task: str) -> float:
    return macro_f1(gold, pred, LABELS[task])


def tune(
    base_rows: list[dict[str, Any]],
    cand_rows: list[dict[str, Any]],
    task: str,
    base_key: str,
    cand_key: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cand_by_id = {str(row["id"]): row for row in cand_rows}
    paired = [(row, cand_by_id[str(row["id"])]) for row in base_rows if str(row["id"]) in cand_by_id]
    gold = [normalize_label(task, base["gold"][task]) for base, _ in paired]
    base_pred = [normalize_label(task, get_pred(base, base_key)[task]) for base, _ in paired]
    cand_pred = [normalize_label(task, get_pred(cand, cand_key)[task]) for _, cand in paired]

    transition_stats = defaultdict(lambda: {"n": 0, "base_correct": 0, "cand_correct": 0})
    for gold_label, old, new in zip(gold, base_pred, cand_pred):
        if old == new:
            continue
        key = f"{old}->{new}"
        transition_stats[key]["n"] += 1
        transition_stats[key]["base_correct"] += int(old == gold_label)
        transition_stats[key]["cand_correct"] += int(new == gold_label)

    best = {
        "score": score(gold, base_pred, task),
        "min_prob": None,
        "min_margin": None,
        "max_replacements": 0,
        "replacements": 0,
    }
    for min_prob in [round(0.50 + i * 0.02, 2) for i in range(24)]:
        for min_margin in [round(-0.10 + i * 0.02, 2) for i in range(31)]:
            candidates = []
            for idx, ((base, cand), old, new) in enumerate(zip(paired, base_pred, cand_pred)):
                if old == new:
                    continue
                prob, margin = confidence(cand, task, new)
                if prob >= min_prob and margin >= min_margin:
                    candidates.append((prob + margin, idx))
            candidates.sort(reverse=True)
            for max_replacements in [10, 20, 40, 60, 80, 120, 160, 220, 320, 9999]:
                pred = list(base_pred)
                for _, idx in candidates[:max_replacements]:
                    pred[idx] = cand_pred[idx]
                tuned = score(gold, pred, task)
                if tuned > best["score"]:
                    best = {
                        "score": tuned,
                        "min_prob": min_prob,
                        "min_margin": min_margin,
                        "max_replacements": max_replacements,
                        "replacements": min(len(candidates), max_replacements),
                    }

    details = []
    for key, value in sorted(transition_stats.items()):
        item = {"transition": key, **value, "delta_correct": value["cand_correct"] - value["base_correct"]}
        details.append(item)
    return best, details


def apply_to_submission(
    base_csv: Path,
    cand_json: Path,
    output: Path,
    task: str,
    cand_key: str,
    min_prob: float,
    min_margin: float,
    max_replacements: int,
) -> int:
    cand_by_id = {str(row["id"]): row for row in load_json(cand_json)}
    rows = []
    candidates = []
    with base_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            rows.append(row)
            cand = cand_by_id[str(row["id"])]
            new = normalize_label(task, cand[cand_key][task])
            old = normalize_label(task, row[task])
            if old == new:
                continue
            prob, margin = confidence(cand, task, new)
            if prob >= min_prob and margin >= min_margin:
                candidates.append((prob + margin, idx, new))
    candidates.sort(reverse=True)
    for _, idx, new in candidates[:max_replacements]:
        rows[idx][task] = "more_than_5_years" if task == "verification_timeline" and new == "longer_than_5_years" else new
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "promise_status", "verification_timeline", "evidence_status", "evidence_quality"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return min(len(candidates), max_replacements)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-val", type=Path, required=True)
    parser.add_argument("--cand-val", type=Path, required=True)
    parser.add_argument("--base-key", default="pred_taskwise")
    parser.add_argument("--cand-key", default="pred_thresholded")
    parser.add_argument("--task", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--base-csv", type=Path)
    parser.add_argument("--cand-test", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    best, details = tune(load_json(args.base_val), load_json(args.cand_val), args.task, args.base_key, args.cand_key)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"best": best, "transitions": details}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False))
    if args.base_csv and args.cand_test and args.output_csv:
        changed = apply_to_submission(
            args.base_csv,
            args.cand_test,
            args.output_csv,
            args.task,
            args.cand_key,
            float(best["min_prob"]),
            float(best["min_margin"]),
            int(best["max_replacements"]),
        )
        print(f"changed={changed}")
        print(f"wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
