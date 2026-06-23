#!/usr/bin/env python3
"""Apply trained stacking meta classifiers to test prediction sources."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from esg_score_postprocess import TASKS, normalize_label  # noqa: E402
from train_stacking_meta import build_features, load_source  # noqa: E402


def read_target_text(path: Path) -> dict[str, str]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return {str(row["id"]): str(row.get("data", "")) for row in csv.DictReader(f)}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["id"]): str(row.get("data", "")) for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target-path", type=Path, required=True)
    parser.add_argument("--source", action="append", nargs=3, metavar=("NAME", "PATH", "PRED_KEY"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    with args.model.open("rb") as f:
        bundle = pickle.load(f)
    source_names = bundle["source_names"]
    models = bundle["models"]
    sources = {name: load_source(Path(path), key) for name, path, key in args.source}
    missing = [name for name in source_names if name not in sources]
    if missing:
        raise SystemExit(f"Missing sources required by model: {missing}")

    text_by_id = read_target_text(args.target_path)
    common_ids = sorted(set(text_by_id).intersection(*(set(sources[name]) for name in source_names)), key=lambda x: int(x))
    rows = [{"id": row_id, "probs": {}} for row_id in common_ids]
    row_by_id = {row["id"]: row for row in rows}
    for task in TASKS:
        x = [build_features(row_id, task, source_names, sources, text_by_id[row_id]) for row_id in common_ids]
        model = models[task]["model"]
        predictions = model.predict(x)
        if hasattr(model, "predict_proba"):
            labels = list(model.classes_)
            probabilities = model.predict_proba(x)
        else:
            labels = None
            probabilities = None
        for idx, row_id in enumerate(common_ids):
            pred = normalize_label(task, predictions[idx])
            row_by_id[row_id][task] = pred
            if probabilities is not None:
                row_by_id[row_id]["probs"][task] = {
                    str(label): float(value) for label, value in zip(labels, probabilities[idx])
                }
            else:
                row_by_id[row_id]["probs"][task] = {pred: 1.0}

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", *TASKS], extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows={len(rows)}")
    print(f"wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

