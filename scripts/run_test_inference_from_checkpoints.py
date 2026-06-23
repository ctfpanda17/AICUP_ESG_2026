#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PACKAGE_DIR.parent
SCRIPTS_DIR = PACKAGE_DIR / "scripts"
TASKS = ["promise_status", "verification_timeline", "evidence_status", "evidence_quality"]
LABELS = {
    "promise_status": ["Yes", "No"],
    "verification_timeline": ["already", "within_2_years", "between_2_and_5_years", "longer_than_5_years", "N/A"],
    "evidence_status": ["Yes", "No", "N/A"],
    "evidence_quality": ["Clear", "Not Clear", "Misleading", "N/A"],
}
SOURCE_CKPTS = {
    "base42_b": "trained_models/ensemble_members/base_seed42__baseline_all_ensemble_source__tasks_evidence_status_evidence_quality_model.pt",
    "base42_f": "trained_models/ensemble_members/base_seed42__fusion_all_ensemble_source__tasks_evidence_status_evidence_quality_model.pt",
    "base7_b": "trained_models/ensemble_members/base_seed7__baseline_all_ensemble_source__tasks_evidence_status_evidence_quality_model.pt",
    "base7_f": "trained_models/ensemble_members/base_seed7__fusion_all_ensemble_source__tasks_evidence_status_evidence_quality_model.pt",
    "robertaL_b": "trained_models/ensemble_members/roberta_large__baseline_promise_status_direct_and_all_ensemble_source__tasks_promise_status_evidence_status_evidence_quality_model.pt",
    "robertaL_f": "trained_models/ensemble_members/roberta_large__fusion_promise_status_direct_and_all_ensemble_source__tasks_promise_status_evidence_status_evidence_quality_model.pt",
    "macbertL_b": "trained_models/ensemble_members/macbert_large__baseline_all_ensemble_source__tasks_evidence_status_evidence_quality_model.pt",
}
ALL_MODEL_WEIGHTS = {
    "promise_status": {"base42_b": 0.25, "base42_f": 1.25, "base7_b": 1.0, "base7_f": 1.0, "robertaL_b": 1.0, "robertaL_f": 1.0, "macbertL_b": 1.0, "macbertL_f": 1.0, "knn21": 1.0},
    "verification_timeline": {"base42_b": 2.0, "base42_f": 0.75, "base7_b": 1.0, "base7_f": 1.25, "robertaL_b": 1.0, "robertaL_f": 1.25, "macbertL_b": 1.5, "macbertL_f": 1.0, "knn21": 1.0},
    "evidence_status": {"base42_b": 1.0, "base42_f": 1.5, "base7_b": 1.0, "base7_f": 1.0, "robertaL_b": 1.0, "robertaL_f": 0.75, "macbertL_b": 1.0, "macbertL_f": 1.0, "knn21": 1.0},
    "evidence_quality": {"base42_b": 0.25, "base42_f": 0.75, "base7_b": 0.25, "base7_f": 1.25, "robertaL_b": 0.75, "robertaL_f": 1.25, "macbertL_b": 2.0, "macbertL_f": 0.75, "knn21": 2.0},
}
ALL_MODEL_THRESHOLDS = {
    "promise_status": {"No": 0.3},
    "verification_timeline": {"longer_than_5_years": 0.45, "between_2_and_5_years": 0.15, "already": 0.4},
    "evidence_status": {},
    "evidence_quality": {},
}
TIMELINE_PLUS_WEIGHTS = {
    "promise_status": {"all_model": 4.0, "timeline_task": 0.25},
    "verification_timeline": {"all_model": 2.0, "timeline_task": 1.0},
    "evidence_status": {"all_model": 4.0, "timeline_task": 0.25},
    "evidence_quality": {"all_model": 4.0, "timeline_task": 0.25},
}
TIMELINE_PLUS_THRESHOLDS = {
    "promise_status": {"No": 0.4},
    "verification_timeline": {"longer_than_5_years": 0.3, "between_2_and_5_years": 0.3},
    "evidence_status": {"No": 0.4, "Yes": 0.55},
    "evidence_quality": {},
}

LATEST_RUN_CKPTS = {
    "base42_b": "base_seed42/baseline_all_ensemble_source__tasks_evidence_status_evidence_quality_model.pt",
    "base42_f": "base_seed42/fusion_all_ensemble_source__tasks_evidence_status_evidence_quality_model.pt",
    "base7_b": "base_seed7/baseline_all_ensemble_source__tasks_evidence_status_evidence_quality_model.pt",
    "base7_f": "base_seed7/fusion_all_ensemble_source__tasks_evidence_status_evidence_quality_model.pt",
    "robertaL_b": "roberta_large/baseline_promise_status_direct_and_all_ensemble_source__tasks_promise_status_evidence_status_evidence_quality_model.pt",
    "robertaL_f": "roberta_large/fusion_promise_status_direct_and_all_ensemble_source__tasks_promise_status_evidence_status_evidence_quality_model.pt",
    "macbertL_b": "macbert_large/baseline_all_ensemble_source__tasks_evidence_status_evidence_quality_model.pt",
    "macbertL_f": "macbert_large/fusion_all_ensemble_source__tasks_evidence_status_evidence_quality_model.pt",
}
LATEST_TIMELINE_CKPT = "timeline_task/baseline_timeline_task_direct_and_timeline_ensemble_source__tasks_verification_timeline_model.pt"


def read_test_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        item = dict(row)
        for task in TASKS:
            item[task] = "No" if task == "promise_status" else "N/A"
        out.append(item)
    return out


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
    return max(LABELS[task], key=lambda label: float(probs.get(label, 0.0)))


def apply_threshold(task: str, probs: dict[str, float], thresholds: dict[str, float]) -> str:
    candidates = []
    for label, threshold in thresholds.items():
        prob = float(probs.get(label, 0.0))
        if prob >= float(threshold):
            candidates.append((prob, label))
    if candidates:
        return max(candidates)[1]
    return argmax_label(task, probs)


def load_main_module():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import main as main_script

    return main_script


def predict_checkpoint(
    checkpoint_path: Path,
    source_name: str,
    train_json: Path,
    augment_json: Path,
    test_json: Path,
    output_path: Path,
    batch_size: int,
) -> list[dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    main_script = load_main_module()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    model_name = metadata["model_name"]
    input_enrichment = metadata.get("input_enrichment", "both")
    max_length = int(metadata.get("max_length", 384))
    stat_dim = int(metadata.get("stat_dim", 0))
    pooling = metadata.get("pooling", "cls")

    train_rows = main_script.load_data(train_json, input_enrichment=input_enrichment)
    if augment_json.exists():
        train_rows.extend(main_script.load_data(augment_json, input_enrichment=input_enrichment))
    test_rows = main_script.load_data(test_json, input_enrichment=input_enrichment)
    _encoded, label_maps = main_script.encode_labels(train_rows + test_rows)

    stat_test = None
    if stat_dim:
        stat_feature_names = metadata.get("stat_features", [])
        base_stat_features = [name for name in stat_feature_names if not str(name).startswith("clarity_")]
        max_features = len(base_stat_features) if base_stat_features else stat_dim
        _stat_train, stat_test, _feature_names = main_script.build_stat_features(
            train_rows,
            test_rows,
            task=metadata.get("fusion_task", "promise_status"),
            max_features=max_features,
            seed=int(metadata.get("seed", 42)),
        )
        if stat_test.shape[1] != stat_dim:
            raise RuntimeError(
                f"Stat feature dimension mismatch for {checkpoint_path}: "
                f"checkpoint expects {stat_dim}, rebuilt {stat_test.shape[1]}"
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dataset = main_script.ESGDataset(test_rows, tokenizer, label_maps, max_length, stat_features=stat_test)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = main_script.MultiTaskBertClassifier(
        model_name,
        num_labels=[len(LABELS[task]) for task in TASKS],
        stat_dim=stat_dim,
        dropout=float(metadata.get("dropout", 0.1)),
        pooling=pooling,
    ).model
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    _gold, _pred, all_probs = main_script.predict(model, loader, device, siamese=False, return_probs=True)

    rows = []
    for idx, row in enumerate(test_rows):
        probs = {task: normalize_probs(task, all_probs[task][idx]) for task in TASKS}
        pred = {task: argmax_label(task, probs[task]) for task in TASKS}
        rows.append({"mode": source_name, "id": row["id"], "gold": {task: row[task] for task in TASKS}, "pred": pred, "probs": probs})
    write_json(output_path, rows)
    print(f"wrote {output_path}")
    return rows


def average_sources(rows_by_source: dict[str, list[dict[str, Any]]], weights: dict[str, dict[str, float]], output_path: Path) -> list[dict[str, Any]]:
    ids = [row["id"] for row in next(iter(rows_by_source.values()))]
    by_source_id = {source: {str(row["id"]): row for row in rows} for source, rows in rows_by_source.items()}
    out = []
    for row_id in ids:
        probs = {}
        pred = {}
        for task in TASKS:
            merged = {label: 0.0 for label in LABELS[task]}
            total_weight = 0.0
            for source, source_rows in by_source_id.items():
                weight = weights.get(task, {}).get(source, 1.0)
                source_probs = normalize_probs(task, source_rows[str(row_id)]["probs"][task])
                for label in LABELS[task]:
                    merged[label] += weight * source_probs[label]
                total_weight += weight
            probs[task] = {label: value / total_weight for label, value in merged.items()}
            pred[task] = argmax_label(task, probs[task])
        out.append({"mode": "ensemble", "id": row_id, "gold": by_source_id[next(iter(rows_by_source))][str(row_id)]["gold"], "pred": pred, "probs": probs, "sources": list(rows_by_source)})
    write_json(output_path, out)
    return out


def threshold_rows(rows: list[dict[str, Any]], thresholds: dict[str, dict[str, float]], output_path: Path) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item["pred_thresholded"] = {
            task: apply_threshold(task, row["probs"][task], thresholds.get(task, {}))
            for task in TASKS
        }
        out.append(item)
    write_json(output_path, out)
    return out


def enforce_submission_logic(pred: dict[str, str]) -> dict[str, str]:
    fixed = dict(pred)
    if fixed["promise_status"] == "No":
        fixed["verification_timeline"] = "N/A"
        fixed["evidence_status"] = "N/A"
        fixed["evidence_quality"] = "N/A"
        return fixed
    if fixed["verification_timeline"] == "N/A":
        fixed["verification_timeline"] = "longer_than_5_years"
    if fixed["evidence_status"] == "N/A":
        fixed["evidence_status"] = "No"
    if fixed["evidence_status"] == "No":
        fixed["evidence_quality"] = "N/A"
    elif fixed["evidence_quality"] == "N/A":
        fixed["evidence_quality"] = "Not Clear"
    return fixed


def write_final_json(ids: list[str], promise_rows: list[dict[str, Any]], timeline_rows: list[dict[str, Any]], all_rows: list[dict[str, Any]], output_path: Path) -> list[dict[str, Any]]:
    promise_by_id = {str(row["id"]): row["pred_thresholded"] for row in promise_rows}
    timeline_by_id = {str(row["id"]): row["pred_thresholded"] for row in timeline_rows}
    all_by_id = {str(row["id"]): row["pred_thresholded"] for row in all_rows}
    out = []
    for row_id in ids:
        pred = {
            "promise_status": promise_by_id[row_id]["promise_status"],
            "verification_timeline": timeline_by_id[row_id]["verification_timeline"],
            "evidence_status": all_by_id[row_id]["evidence_status"],
            "evidence_quality": all_by_id[row_id]["evidence_quality"],
        }
        out.append({"id": row_id, **enforce_submission_logic(pred)})
    write_json(output_path, out)
    return out


def write_taskwise_json(
    ids: list[str],
    timeline_plus_thr: list[dict[str, Any]],
    all_thr: list[dict[str, Any]],
    all_raw: list[dict[str, Any]],
    output_path: Path,
    enforce_logic: bool,
) -> list[dict[str, Any]]:
    timeline_plus_by_id = {str(row["id"]): row["pred_thresholded"] for row in timeline_plus_thr}
    all_thr_by_id = {str(row["id"]): row["pred_thresholded"] for row in all_thr}
    all_raw_by_id = {str(row["id"]): row["pred"] for row in all_raw}
    out = []
    for row_id in ids:
        pred = {
            "promise_status": timeline_plus_by_id[row_id]["promise_status"],
            "verification_timeline": all_thr_by_id[row_id]["verification_timeline"],
            "evidence_status": all_raw_by_id[row_id]["evidence_status"],
            "evidence_quality": all_raw_by_id[row_id]["evidence_quality"],
        }
        if enforce_logic:
            pred = enforce_submission_logic(pred)
        out.append({"id": row_id, **pred})
    write_json(output_path, out)
    return out


def normalize_submission_label(task: str, value: Any) -> str:
    text = str(value).strip() if value is not None else "N/A"
    if task == "verification_timeline" and text == "longer_than_5_years":
        return "more_than_5_years"
    return text


def write_submission_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["id", *TASKS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({"id": row["id"], **{task: normalize_submission_label(task, row[task]) for task in TASKS}})
    print(f"wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run official test inference from saved ESG checkpoints.")
    parser.add_argument("--test-csv", type=Path, default=Path(r"C:\Users\User\Downloads\vpesg4k_test_2000.csv"))
    parser.add_argument("--work-dir", type=Path, default=PACKAGE_DIR / "runs/test_2000_best_saved")
    parser.add_argument("--train-json", type=Path, default=PACKAGE_DIR / "data/vpesg4k_train_val_2000.json")
    parser.add_argument("--augment-json", type=Path, default=PACKAGE_DIR / "data/vpesg4k_llm_paraphrase_aug_only_clean.json")
    parser.add_argument("--model-run-dir", type=Path, default=None, help="Use checkpoints from a run directory instead of trained_models.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    test_json = args.work_dir / "test_with_dummy_labels.json"
    if not (args.reuse_existing and test_json.exists()):
        test_rows = read_test_csv(args.test_csv)
        write_json(test_json, test_rows)
    else:
        test_rows = json.loads(test_json.read_text(encoding="utf-8"))
    ids = [str(row["id"]) for row in test_rows]

    if args.model_run_dir:
        checkpoint_root = args.model_run_dir
        source_ckpts = {source: checkpoint_root / rel_path for source, rel_path in LATEST_RUN_CKPTS.items()}
        timeline_ckpt = checkpoint_root / LATEST_TIMELINE_CKPT
    else:
        source_ckpts = {source: PACKAGE_DIR / rel_path for source, rel_path in SOURCE_CKPTS.items()}
        timeline_ckpt = PACKAGE_DIR / "trained_models/final_tasks/verification_timeline/verification_timeline_model.pt"

    source_rows = {}
    for source, checkpoint_path in source_ckpts.items():
        out_path = args.work_dir / "sources" / f"{source}_predictions.json"
        if args.reuse_existing and out_path.exists():
            source_rows[source] = json.loads(out_path.read_text(encoding="utf-8"))
            continue
        source_rows[source] = predict_checkpoint(
            checkpoint_path,
            source,
            args.train_json,
            args.augment_json,
            test_json,
            out_path,
            args.batch_size,
        )

    knn_path = args.work_dir / "sources" / "knn21_predictions.json"
    if args.reuse_existing and knn_path.exists():
        source_rows["knn21"] = json.loads(knn_path.read_text(encoding="utf-8"))
    else:
        import subprocess

        subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "esg_knn_retrieval_baseline.py"),
                "--data-path",
                str(args.train_json),
                "--eval-path",
                str(test_json),
                "--augment-train-path",
                str(args.augment_json),
                "--output",
                str(knn_path),
                "--report",
                str(args.work_dir / "sources" / "knn21_report.md"),
                "--k",
                "21",
                "--seed",
                "42",
            ],
            cwd=PROJECT_DIR,
            check=True,
        )
        source_rows["knn21"] = json.loads(knn_path.read_text(encoding="utf-8"))

    timeline_path = args.work_dir / "sources" / "timeline_task_predictions.json"
    if args.reuse_existing and timeline_path.exists():
        timeline_task_rows = json.loads(timeline_path.read_text(encoding="utf-8"))
    else:
        timeline_task_rows = predict_checkpoint(
            timeline_ckpt,
            "timeline_task",
            args.train_json,
            args.augment_json,
            test_json,
            timeline_path,
            args.batch_size,
        )

    all_raw = average_sources(source_rows, ALL_MODEL_WEIGHTS, args.work_dir / "all_model_ensemble_predictions.json")
    all_thr = threshold_rows(all_raw, ALL_MODEL_THRESHOLDS, args.work_dir / "all_model_ensemble_thresholded.json")
    timeline_plus_raw = average_sources(
        {"all_model": all_raw, "timeline_task": timeline_task_rows},
        TIMELINE_PLUS_WEIGHTS,
        args.work_dir / "timeline_plus_allmodel_ensemble.json",
    )
    timeline_plus_thr = threshold_rows(timeline_plus_raw, TIMELINE_PLUS_THRESHOLDS, args.work_dir / "timeline_plus_allmodel_ensemble_thresholded.json")
    final_no_logic = write_taskwise_json(
        ids,
        timeline_plus_thr,
        all_thr,
        all_raw,
        args.work_dir / "final_taskwise_no_cross_fix.json",
        enforce_logic=False,
    )
    final_logic = write_taskwise_json(
        ids,
        timeline_plus_thr,
        all_thr,
        all_raw,
        args.work_dir / "final_taskwise_with_cross_fix.json",
        enforce_logic=True,
    )
    write_submission_csv(args.work_dir / "submission_taskwise_no_cross_fix.csv", final_no_logic)
    write_submission_csv(args.work_dir / "submission_taskwise_with_cross_fix.csv", final_logic)
    print(f"final rows={len(final_no_logic)}")
    print(f"best public-style submission csv={args.work_dir / 'submission_taskwise_no_cross_fix.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
