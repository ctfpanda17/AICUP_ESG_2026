#!/usr/bin/env python3
"""Task-specific linear text classifiers for ESG labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from esg_score_postprocess import LABELS, TASKS, WEIGHTS, macro_f1, normalize_label
from main import load_data, split_train_test


def fit_vectorizer(train_texts: list[str]):
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=2, max_features=120000, sublinear_tf=True)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_features=80000, sublinear_tf=True)
    vec = FeatureUnion([("char", char), ("word", word)])
    return vec.fit(train_texts)


def scores_to_probs(scores: np.ndarray, classes: list[str], labels: list[str]) -> list[dict[str, float]]:
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    probs_raw = softmax(scores, axis=1)
    out = []
    for row in probs_raw:
        item = {label: 0.0 for label in labels}
        for cls, prob in zip(classes, row):
            item[cls] = float(prob)
        total = sum(item.values()) or 1.0
        out.append({label: value / total for label, value in item.items()})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run linear ESG text classifiers.")
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--eval-path", type=Path, default=None)
    parser.add_argument("--augment-train-path", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--input-enrichment", default="both")
    parser.add_argument("--model", choices=["svm", "logreg"], default="svm")
    args = parser.parse_args()

    data = load_data(args.data_path, input_enrichment=args.input_enrichment)
    if args.eval_path:
        train_set = data
        test_set = load_data(args.eval_path, input_enrichment=args.input_enrichment)
    else:
        train_set, test_set = split_train_test(data, seed=args.split_seed)
    if args.augment_train_path:
        train_set.extend(load_data(args.augment_train_path, input_enrichment=args.input_enrichment))

    train_texts = [row["data"] for row in train_set]
    test_texts = [row["data"] for row in test_set]
    vec = fit_vectorizer(train_texts)
    x_train = vec.transform(train_texts)
    x_test = vec.transform(test_texts)

    probs_by_task = {}
    pred_by_task = {}
    for task in TASKS:
        y_train = [row[task] for row in train_set]
        if args.model == "logreg":
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0, n_jobs=-1)
            clf.fit(x_train, y_train)
            probs = clf.predict_proba(x_test)
            classes = clf.classes_.tolist()
            probs_by_task[task] = [
                {label: float(row[classes.index(label)]) if label in classes else 0.0 for label in LABELS[task]}
                for row in probs
            ]
        else:
            clf = LinearSVC(class_weight="balanced", C=0.5)
            clf.fit(x_train, y_train)
            scores = clf.decision_function(x_test)
            probs_by_task[task] = scores_to_probs(scores, clf.classes_.tolist(), LABELS[task])
        pred_by_task[task] = [max(LABELS[task], key=lambda label: probs[label]) for probs in probs_by_task[task]]

    rows = []
    for idx, item in enumerate(test_set):
        rows.append(
            {
                "mode": f"linear_{args.model}",
                "id": item.get("id"),
                "gold": {task: item[task] for task in TASKS},
                "pred": {task: pred_by_task[task][idx] for task in TASKS},
                "probs": {task: probs_by_task[task][idx] for task in TASKS},
            }
        )

    scores = {}
    for task in TASKS:
        gold = [normalize_label(task, row["gold"][task]) for row in rows]
        pred = [normalize_label(task, row["pred"][task]) for row in rows]
        scores[task] = macro_f1(gold, pred, LABELS[task])
    scores["weighted_macro_f1"] = sum(scores[task] * WEIGHTS[task] for task in TASKS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# ESG Linear Text Baseline", "", "| task | Macro-F1 |", "|---|---:|"]
    for task in TASKS + ["weighted_macro_f1"]:
        lines.append(f"| {task} | {scores[task]:.6f} |")
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"weighted_macro_f1={scores['weighted_macro_f1']:.6f}")
    for task in TASKS:
        print(f"{task}: {scores[task]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
