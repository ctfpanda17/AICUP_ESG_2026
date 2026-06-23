#!/usr/bin/env python3
"""
TF-IDF kNN retrieval baseline for ESG labels.

This implements the instance-based retrieval idea from the Week 10 references:
for each validation paragraph, retrieve similar train paragraphs and convert
their label distribution into probabilities for the four tasks.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from esg_score_postprocess import LABELS, TASKS, WEIGHTS, macro_f1, normalize_label


DROP_COLS = set()


def load_data(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    data = []
    for item in raw:
        record = dict(item)
        record["data"] = str(record.get("data", ""))
        for task in TASKS:
            record[task] = normalize_label(task, record.get(task))
        data.append(record)
    return data


def split_train_test(data: list[dict[str, Any]], seed: int, train_ratio: float = 0.8) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_label = {"Yes": [], "No": []}
    for item in data:
        by_label[item["promise_status"]].append(item)
    train_set, test_set = [], []
    for items in by_label.values():
        shuffled = items[:]
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * train_ratio)
        train_set.extend(shuffled[:cut])
        test_set.extend(shuffled[cut:])
    rng.shuffle(train_set)
    rng.shuffle(test_set)
    return train_set, test_set


def build_probs(neighbors: list[tuple[int, float]], train_set: list[dict[str, Any]], task: str, smooth: float) -> dict[str, float]:
    scores = {label: smooth for label in LABELS[task]}
    for index, sim in neighbors:
        label = train_set[index][task]
        scores[label] += max(float(sim), 0.0)
    total = sum(scores.values())
    return {label: score / total for label, score in scores.items()}


def argmax_label(task: str, probs: dict[str, float]) -> str:
    return max(LABELS[task], key=lambda label: probs.get(label, 0.0))


def score_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    scores = {}
    for task in TASKS:
        gold = [row["gold"][task] for row in rows]
        pred = [row["pred"][task] for row in rows]
        scores[task] = macro_f1(gold, pred, LABELS[task])
    scores["weighted_macro_f1"] = sum(scores[task] * WEIGHTS[task] for task in TASKS)
    return scores


def run_knn(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, float]]:
    data = load_data(Path(args.data_path))
    if args.limit:
        data = data[: args.limit]
    if args.eval_path:
        train_set = data
        test_set = load_data(Path(args.eval_path))
    else:
        train_set, test_set = split_train_test(data, seed=args.seed)
    if args.augment_train_path:
        augment = load_data(Path(args.augment_train_path))
        if args.augment_train_limit:
            augment = augment[: args.augment_train_limit]
        train_set.extend(augment)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=args.min_df,
        max_features=args.max_features,
        sublinear_tf=True,
        norm="l2",
    )
    x_train = vectorizer.fit_transform([item["data"] for item in train_set])
    x_test = vectorizer.transform([item["data"] for item in test_set])
    sims = cosine_similarity(x_test, x_train)

    rows = []
    k = min(args.k, len(train_set))
    for test_index, item in enumerate(test_set):
        top_idx = np.argpartition(-sims[test_index], k - 1)[:k]
        top = sorted(((int(idx), float(sims[test_index, idx])) for idx in top_idx), key=lambda pair: pair[1], reverse=True)
        probs = {task: build_probs(top, train_set, task, args.smooth) for task in TASKS}
        pred = {task: argmax_label(task, probs[task]) for task in TASKS}
        rows.append(
            {
                "mode": f"tfidf_knn_k{args.k}",
                "id": item.get("id"),
                "gold": {task: item[task] for task in TASKS},
                "pred": pred,
                "probs": probs,
                "neighbors": [{"id": train_set[idx].get("id"), "similarity": sim} for idx, sim in top[: args.save_neighbors]],
            }
        )
    return rows, score_rows(rows)


def write_report(path: Path, args: argparse.Namespace, scores: dict[str, float]) -> None:
    lines = [
        "# ESG kNN Retrieval Baseline",
        "",
        "## Settings",
        "",
        f"- k: `{args.k}`",
        f"- max_features: `{args.max_features}`",
        f"- min_df: `{args.min_df}`",
        f"- augment_train_path: `{args.augment_train_path or ''}`",
        "",
        "## Scores",
        "",
        "| task | macro F1 |",
        "|---|---:|",
    ]
    for task in TASKS + ["weighted_macro_f1"]:
        lines.append(f"| {task} | {scores[task]:.6f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TF-IDF kNN retrieval baseline for ESG labels.")
    parser.add_argument("--data-path", default="vpesg4k_train_1000_V1.json")
    parser.add_argument("--eval-path", default=None)
    parser.add_argument("--augment-train-path", default=None)
    parser.add_argument("--augment-train-limit", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--smooth", type=float, default=0.05)
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--save-neighbors", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, scores = run_knn(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    write_report(Path(args.report), args, scores)
    print(f"weighted_macro_f1={scores['weighted_macro_f1']:.6f}")
    print(f"wrote {output_path}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
