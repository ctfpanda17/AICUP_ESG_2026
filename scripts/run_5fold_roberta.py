#!/usr/bin/env python3
"""Run 5-fold RoBERTa-large training, OOF scoring, and test ensemble."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from esg_score_postprocess import LABELS, TASKS, WEIGHTS, macro_f1, normalize_label
from esg_threshold_search import apply_all_thresholds, search_task_thresholds
from run_test_inference_from_checkpoints import predict_checkpoint


def read_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def make_folds(rows: list[dict[str, Any]], n_folds: int, seed: int) -> list[list[dict[str, Any]]]:
    import random

    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get("promise_status", ""))
        buckets[key].append(row)
    folds = [[] for _ in range(n_folds)]
    for items in buckets.values():
        shuffled = items[:]
        rng.shuffle(shuffled)
        for idx, row in enumerate(shuffled):
            folds[idx % n_folds].append(row)
    for fold in folds:
        rng.shuffle(fold)
    return folds


def run(cmd: list[str], cwd: Path, dry_run: bool = False) -> None:
    print("\n$ " + " ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def split_mode_predictions(pred_path: Path, mode: str) -> list[dict[str, Any]]:
    rows = read_json(pred_path)
    return [row for row in rows if row.get("mode") == mode]


def score_rows(rows: list[dict[str, Any]], pred_key: str = "pred") -> dict[str, float]:
    scores = {}
    for task in TASKS:
        gold = [normalize_label(task, row["gold"].get(task)) for row in rows]
        pred = [normalize_label(task, row[pred_key].get(task)) for row in rows]
        scores[task] = macro_f1(gold, pred, LABELS[task])
    scores["weighted_macro_f1"] = sum(scores[task] * WEIGHTS[task] for task in TASKS)
    return scores


def write_score_report(path: Path, scores: dict[str, float], title: str) -> None:
    lines = [f"# {title}", "", "| task | Macro-F1 |", "|---|---:|"]
    for task in TASKS + ["weighted_macro_f1"]:
        lines.append(f"| {task} | {scores[task]:.6f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_probs(task: str, probs: dict[str, Any]) -> dict[str, float]:
    values = {label: max(0.0, float(probs.get(label, 0.0))) for label in LABELS[task]}
    total = sum(values.values())
    if total <= 0:
        return {label: 1.0 / len(LABELS[task]) for label in LABELS[task]}
    return {label: value / total for label, value in values.items()}


def argmax_label(task: str, probs: dict[str, float]) -> str:
    return max(LABELS[task], key=lambda label: probs.get(label, 0.0))


def average_prediction_files(paths: list[Path], output: Path) -> list[dict[str, Any]]:
    rows_by_file = [read_json(path) for path in paths]
    ids = [str(row["id"]) for row in rows_by_file[0]]
    by_file_id = [{str(row["id"]): row for row in rows} for rows in rows_by_file]
    out = []
    for row_id in ids:
        base = by_file_id[0][row_id]
        probs = {}
        pred = {}
        for task in TASKS:
            merged = {label: 0.0 for label in LABELS[task]}
            for table in by_file_id:
                row_probs = normalize_probs(task, table[row_id]["probs"][task])
                for label in LABELS[task]:
                    merged[label] += row_probs[label]
            probs[task] = {label: value / len(by_file_id) for label, value in merged.items()}
            pred[task] = argmax_label(task, probs[task])
        out.append({"mode": "5fold_ensemble", "id": base["id"], "gold": base.get("gold", {}), "pred": pred, "probs": probs})
    write_json(output, out)
    return out


def write_submission(path: Path, rows: list[dict[str, Any]]) -> None:
    def normalize_submission(task: str, value: str) -> str:
        if task == "verification_timeline" and value == "longer_than_5_years":
            return "more_than_5_years"
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", *TASKS], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            pred = row["pred_thresholded"] if "pred_thresholded" in row else row["pred"]
            writer.writerow({"id": row["id"], **{task: normalize_submission(task, pred[task]) for task in TASKS}})


def tune_thresholds(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grid = [round(0.05 + i * 0.01, 6) for i in range(91)]
    thresholds = {}
    for task in TASKS:
        task_thresholds, _ = search_task_thresholds(rows, task, grid)
        thresholds[task] = task_thresholds
    return thresholds


def find_checkpoint(fold_dir: Path, mode: str) -> Path:
    matches = sorted(fold_dir.glob(f"{mode}_*_model.pt"))
    if not matches:
        raise FileNotFoundError(f"No {mode} checkpoint in {fold_dir}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 5-fold RoBERTa-large ESG pipeline.")
    parser.add_argument("--package-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=Path(r"C:\Users\User\Downloads\vpesg4k_test_2000.csv"))
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--lr", default="1e-5")
    parser.add_argument("--modes", nargs="+", choices=["baseline", "fusion"], default=["baseline"])
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    package_dir = args.package_dir.resolve()
    project_root = package_dir.parent.resolve()
    scripts_dir = package_dir / "scripts"
    data_path = (args.data_path or (package_dir / "data" / "vpesg4k_train_val_2000.json")).resolve()
    work_dir = (args.work_dir or (package_dir / "runs" / "fivefold_roberta_large")).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    rows = read_json(data_path)
    folds = make_folds(rows, args.n_folds, args.seed)
    split_dir = work_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for fold_idx, val_rows in enumerate(folds):
        val_ids = {str(row["id"]) for row in val_rows}
        train_rows = [row for row in rows if str(row["id"]) not in val_ids]
        write_json(split_dir / f"fold{fold_idx}_train.json", train_rows)
        write_json(split_dir / f"fold{fold_idx}_val.json", val_rows)

    if not args.skip_train:
        for fold_idx in range(args.n_folds):
            fold_dir = work_dir / f"fold{fold_idx}"
            pred_path = fold_dir / "week9_predictions.json"
            if args.reuse_existing and pred_path.exists():
                continue
            cmd = [
                args.python,
                str(scripts_dir / "main.py"),
                "--data-path",
                str(split_dir / f"fold{fold_idx}_train.json"),
                "--eval-path",
                str(split_dir / f"fold{fold_idx}_val.json"),
                "--model-name",
                "hfl/chinese-roberta-wwm-ext-large",
                "--output-dir",
                str(fold_dir),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--max-length",
                str(args.max_length),
                "--lr",
                args.lr,
                "--modes",
                *args.modes,
                "--seed",
                str(args.seed + fold_idx),
                "--save-probs",
                "--save-model",
                "--input-enrichment",
                "both",
                "--contrastive-alpha",
                "0.05",
                "--contrastive-task",
                "evidence_quality",
                "--final-tasks",
                "promise_status,verification_timeline,evidence_status,evidence_quality",
                "--model-role",
                f"fivefold_roberta_large_fold{fold_idx}",
            ]
            run(cmd, project_root, args.dry_run)

    for mode in args.modes:
        oof_rows = []
        for fold_idx in range(args.n_folds):
            oof_rows.extend(split_mode_predictions(work_dir / f"fold{fold_idx}" / "week9_predictions.json", mode))
        oof_path = work_dir / f"oof_{mode}_predictions.json"
        write_json(oof_path, oof_rows)
        scores = score_rows(oof_rows)
        write_score_report(work_dir / f"oof_{mode}_report.md", scores, f"5-Fold OOF {mode}")
        print(f"{mode} OOF weighted={scores['weighted_macro_f1']:.6f}")
        thresholded = work_dir / f"oof_{mode}_thresholded.json"
        run(
            [
                args.python,
                str(scripts_dir / "esg_threshold_search.py"),
                "--input",
                str(oof_path),
                "--output",
                str(thresholded),
                "--report",
                str(work_dir / f"oof_{mode}_threshold_report.md"),
            ],
            project_root,
            args.dry_run,
        )

    test_json = work_dir / "test_with_dummy_labels.json"
    if not test_json.exists():
        with args.test_csv.open("r", encoding="utf-8-sig", newline="") as f:
            test_rows = []
            for row in csv.DictReader(f):
                item = dict(row)
                item.update(
                    {
                        "promise_status": "No",
                        "verification_timeline": "N/A",
                        "evidence_status": "N/A",
                        "evidence_quality": "N/A",
                    }
                )
                test_rows.append(item)
        write_json(test_json, test_rows)

    for mode in args.modes:
        test_pred_paths = []
        for fold_idx in range(args.n_folds):
            out_path = work_dir / "test_sources" / f"fold{fold_idx}_{mode}_predictions.json"
            test_pred_paths.append(out_path)
            if args.reuse_existing and out_path.exists():
                continue
            predict_checkpoint(
                find_checkpoint(work_dir / f"fold{fold_idx}", mode),
                f"fold{fold_idx}_{mode}",
                split_dir / f"fold{fold_idx}_train.json",
                Path("__no_augment__.json"),
                test_json,
                out_path,
                args.batch_size,
            )
        avg_path = work_dir / f"test_5fold_{mode}_predictions.json"
        avg_rows = average_prediction_files(test_pred_paths, avg_path)
        raw_submission_path = work_dir / f"test_5fold_{mode}_raw_submission.csv"
        write_submission(raw_submission_path, avg_rows)

        oof_rows = read_json(work_dir / f"oof_{mode}_predictions.json")
        thresholds = tune_thresholds(oof_rows)
        write_json(work_dir / f"oof_{mode}_thresholds.json", thresholds)
        thresholded_rows = apply_all_thresholds(avg_rows, thresholds)
        thresholded_pred_path = work_dir / f"test_5fold_{mode}_thresholded_predictions.json"
        write_json(thresholded_pred_path, thresholded_rows)
        thresholded_submission_path = work_dir / f"test_5fold_{mode}_thresholded_submission.csv"
        write_submission(thresholded_submission_path, thresholded_rows)
        print(f"raw_submission={raw_submission_path}")
        print(f"thresholded_submission={thresholded_submission_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
