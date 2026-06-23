#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/test_2000_best_saved"
OUT = ROOT / "outputs"
FIELDS = ["id", "promise_status", "verification_timeline", "evidence_status", "evidence_quality"]


def official_timeline(label: str) -> str:
    return "more_than_5_years" if label == "longer_than_5_years" else label


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def pred(row: dict, key: str = "pred_thresholded") -> dict[str, str]:
    return row.get(key) or row.get("pred")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def compact_from_sources(
    ids: list[str],
    promise_rows: list[dict],
    timeline_rows: list[dict],
    evidence_rows: list[dict],
    *,
    promise_key: str = "pred_thresholded",
    timeline_key: str = "pred_thresholded",
    evidence_key: str = "pred_thresholded",
    enforce_no_only: bool = False,
    enforce_full: bool = False,
) -> list[dict[str, str]]:
    promise = {str(row["id"]): pred(row, promise_key) for row in promise_rows}
    timeline = {str(row["id"]): pred(row, timeline_key) for row in timeline_rows}
    evidence = {str(row["id"]): pred(row, evidence_key) for row in evidence_rows}
    out = []
    for row_id in ids:
        row = {
            "id": row_id,
            "promise_status": promise[row_id]["promise_status"],
            "verification_timeline": official_timeline(timeline[row_id]["verification_timeline"]),
            "evidence_status": evidence[row_id]["evidence_status"],
            "evidence_quality": evidence[row_id]["evidence_quality"],
        }
        if enforce_no_only and row["promise_status"] == "No":
            row["verification_timeline"] = "N/A"
            row["evidence_status"] = "N/A"
            row["evidence_quality"] = "N/A"
        if enforce_full:
            if row["promise_status"] == "No":
                row["verification_timeline"] = "N/A"
                row["evidence_status"] = "N/A"
                row["evidence_quality"] = "N/A"
            else:
                if row["verification_timeline"] == "N/A":
                    row["verification_timeline"] = "more_than_5_years"
                if row["evidence_status"] == "N/A":
                    row["evidence_status"] = "No"
                if row["evidence_status"] == "No":
                    row["evidence_quality"] = "N/A"
                elif row["evidence_quality"] == "N/A":
                    row["evidence_quality"] = "Not Clear"
        out.append(row)
    return out


def main() -> int:
    ids = [str(row["id"]) for row in load(RUN / "test_with_dummy_labels.json")]
    promise_direct = load(RUN / "promise_status_direct_thresholded.json")
    all_model = load(RUN / "all_model_ensemble_thresholded.json")
    timeline_plus = load(RUN / "timeline_plus_allmodel_ensemble_thresholded.json")

    variants = {
        "submission_no_cross_task_fix.csv": compact_from_sources(ids, promise_direct, timeline_plus, all_model),
        "submission_only_promise_no_fix.csv": compact_from_sources(ids, promise_direct, timeline_plus, all_model, enforce_no_only=True),
        "submission_all_model_promise_no_cross_fix.csv": compact_from_sources(ids, all_model, timeline_plus, all_model),
        "submission_timeline_promise_no_cross_fix.csv": compact_from_sources(ids, timeline_plus, timeline_plus, all_model),
        "submission_timeline_raw_no_cross_fix.csv": compact_from_sources(
            ids,
            promise_direct,
            timeline_plus,
            all_model,
            timeline_key="pred",
        ),
        "submission_all_model_promise_timeline_raw_no_cross_fix.csv": compact_from_sources(
            ids,
            all_model,
            timeline_plus,
            all_model,
            timeline_key="pred",
        ),
        "submission_all_model_raw_timeline_no_cross_fix.csv": compact_from_sources(
            ids,
            promise_direct,
            all_model,
            all_model,
            timeline_key="pred",
        ),
        "submission_current_full_logic.csv": compact_from_sources(ids, promise_direct, timeline_plus, all_model, enforce_full=True),
    }
    for name, rows in variants.items():
        write_csv(OUT / name, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
