#!/usr/bin/env python3
"""Train task-wise stacking meta classifiers over prediction sources."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from esg_score_postprocess import LABELS, TASKS, WEIGHTS, macro_f1, normalize_label
from main import load_data, split_train_test


def load_source(path: Path, pred_key: str) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in rows:
        if pred_key == "columns":
            pred = {task: row.get(task) for task in TASKS}
        else:
            pred = row.get(pred_key)
        if pred is None:
            pred = row.get("pred_thresholded") or row.get("pred_taskwise") or row.get("pred")
        if pred is None:
            continue
        out[str(row["id"])] = {
            "gold": row.get("gold"),
            "pred": {task: normalize_label(task, pred.get(task)) for task in TASKS},
            "probs": row.get("probs", {}),
        }
    return out


def text_features(text: str) -> list[float]:
    text = str(text or "")
    years = re.findall(r"(?:19|20)\d{2}", text)
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    has_percent = int(bool(re.search(r"\d+(?:\.\d+)?\s*[%％]", text)))
    vague_terms = ["持續", "致力", "推動", "強化", "提升", "改善", "落實", "促進", "優化", "逐步"]
    proof_terms = ["達成", "完成", "已", "取得", "通過", "驗證", "查證", "稽核", "認證"]
    target_terms = ["目標", "承諾", "預計", "將", "未來", "規劃", "預期", "於"]
    return [
        min(len(text) / 500.0, 3.0),
        min(len(numbers) / 20.0, 2.0),
        min(len(years) / 10.0, 2.0),
        has_percent,
        int(bool(re.search(r"20(?:2[5-9]|3\d|4\d|50)", text))),
        sum(term in text for term in vague_terms) / len(vague_terms),
        sum(term in text for term in proof_terms) / len(proof_terms),
        sum(term in text for term in target_terms) / len(target_terms),
    ]


def build_features(
    row_id: str,
    task: str,
    source_names: list[str],
    sources: dict[str, dict[str, dict[str, Any]]],
    row_text: str,
) -> list[float]:
    labels = LABELS[task]
    feats: list[float] = []
    vote_counts = {label: 0 for label in labels}
    for source in source_names:
        item = sources[source][row_id]
        pred = normalize_label(task, item["pred"][task])
        vote_counts[pred] += 1
        feats.extend([1.0 if pred == label else 0.0 for label in labels])
        probs = item.get("probs", {}).get(task, {})
        if probs:
            feats.extend([float(probs.get(label, 0.0)) for label in labels])
        else:
            feats.extend([1.0 if pred == label else 0.0 for label in labels])
    total = max(len(source_names), 1)
    feats.extend([vote_counts[label] / total for label in labels])
    sorted_votes = sorted(vote_counts.values(), reverse=True)
    feats.append((sorted_votes[0] - (sorted_votes[1] if len(sorted_votes) > 1 else 0)) / total)
    feats.extend(text_features(row_text))
    return feats


def make_model(name: str, seed: int):
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=0.7, random_state=seed),
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "extra":
        return ExtraTreesClassifier(
            n_estimators=600,
            max_depth=7,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.04,
            max_leaf_nodes=10,
            l2_regularization=0.1,
            random_state=seed,
        )
    raise ValueError(name)


def score_task(gold: list[str], pred: list[str], task: str) -> float:
    return f1_score(gold, pred, labels=LABELS[task], average="macro", zero_division=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--source", action="append", nargs=3, metavar=("NAME", "PATH", "PRED_KEY"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    _, val_rows = split_train_test(load_data(args.data_path, input_enrichment="none"), seed=args.split_seed)
    text_by_id = {str(row["id"]): row.get("data", "") for row in val_rows}
    sources = {name: load_source(Path(path), key) for name, path, key in args.source}
    source_names = list(sources)
    common_ids = sorted(set(text_by_id).intersection(*(set(rows) for rows in sources.values())), key=lambda x: int(x))
    if not common_ids:
        raise SystemExit("No common ids")

    report: dict[str, Any] = {"n_common": len(common_ids), "sources": source_names, "tasks": {}}
    stacked_rows = []
    oof_by_task: dict[str, list[str]] = {}
    models = {}
    for task in TASKS:
        y = np.array([normalize_label(task, sources[source_names[0]][row_id]["gold"][task]) for row_id in common_ids])
        x = np.array([build_features(row_id, task, source_names, sources, text_by_id[row_id]) for row_id in common_ids], dtype=float)

        candidates = {}
        for model_name in ["logreg", "rf", "extra", "hgb"]:
            skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
            oof = np.empty_like(y, dtype=object)
            for train_idx, val_idx in skf.split(x, y):
                model = make_model(model_name, args.seed)
                model.fit(x[train_idx], y[train_idx])
                oof[val_idx] = model.predict(x[val_idx])
            candidates[model_name] = {
                "oof_score": score_task(list(y), list(oof), task),
                "oof_pred": list(oof),
            }

        best_name = max(candidates, key=lambda name: candidates[name]["oof_score"])
        oof_by_task[task] = candidates[best_name]["oof_pred"]
        final_model = make_model(best_name, args.seed)
        final_model.fit(x, y)
        in_sample = list(final_model.predict(x))
        models[task] = {"name": best_name, "model": final_model}
        report["tasks"][task] = {
            "best_model": best_name,
            "oof_scores": {name: value["oof_score"] for name, value in candidates.items()},
            "in_sample_score": score_task(list(y), in_sample, task),
        }
        for row_id, pred, gold in zip(common_ids, in_sample, y):
            stacked_rows.append({"id": row_id, "task": task, "pred": pred, "gold": gold})

    final_by_id = {}
    for row_id in common_ids:
        final_by_id[row_id] = {
            "id": row_id,
            "gold": {task: normalize_label(task, sources[source_names[0]][row_id]["gold"][task]) for task in TASKS},
            "pred_stacked": {},
        }
    for item in stacked_rows:
        final_by_id[item["id"]]["pred_stacked"][item["task"]] = normalize_label(item["task"], item["pred"])

    rows = list(final_by_id.values())
    oof_rows = []
    for idx, row_id in enumerate(common_ids):
        oof_rows.append(
            {
                "id": row_id,
                "gold": {task: normalize_label(task, sources[source_names[0]][row_id]["gold"][task]) for task in TASKS},
                "pred_oof": {task: normalize_label(task, oof_by_task[task][idx]) for task in TASKS},
            }
        )
    final_scores = {}
    for task in TASKS:
        gold = [row["gold"][task] for row in rows]
        pred = [row["pred_stacked"][task] for row in rows]
        final_scores[task] = macro_f1(gold, pred, LABELS[task])
    final_scores["weighted_macro_f1"] = sum(final_scores[t] * WEIGHTS[t] for t in TASKS)
    report["in_sample_final_scores"] = final_scores
    report["oof_weighted_estimate"] = sum(report["tasks"][t]["oof_scores"][report["tasks"][t]["best_model"]] * WEIGHTS[t] for t in TASKS)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "stacked_val_predictions.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "stacked_oof_predictions.json").write_text(json.dumps(oof_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "models.pkl").open("wb") as f:
        pickle.dump({"source_names": source_names, "models": models}, f)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
