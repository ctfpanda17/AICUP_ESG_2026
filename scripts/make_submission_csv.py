#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = ["id", "promise_status", "verification_timeline", "evidence_status", "evidence_quality"]
LABELS = {
    "promise_status": {"Yes", "No"},
    "verification_timeline": {"already", "within_2_years", "between_2_and_5_years", "more_than_5_years", "N/A"},
    "evidence_status": {"Yes", "No", "N/A"},
    "evidence_quality": {"Clear", "Not Clear", "Misleading", "N/A"},
}
ALIASES = {
    "verification_timeline": {
        "longer_than_5_years": "more_than_5_years",
        "more than 5 years": "more_than_5_years",
    },
    "evidence_quality": {
        "NotClear": "Not Clear",
        "not_clear": "Not Clear",
        "not clear": "Not Clear",
    },
}


def normalize(task: str, value: Any) -> str:
    text = str(value).strip() if value is not None else "N/A"
    text = ALIASES.get(task, {}).get(text, text)
    if text in LABELS[task]:
        return text
    raise ValueError(f"invalid {task} label: {value!r}")


def enforce_logic(row: dict[str, str]) -> dict[str, str]:
    fixed = dict(row)
    if fixed["promise_status"] == "No":
        fixed["verification_timeline"] = "N/A"
        fixed["evidence_status"] = "N/A"
        fixed["evidence_quality"] = "N/A"
        return fixed
    if fixed["verification_timeline"] == "N/A":
        fixed["verification_timeline"] = "more_than_5_years"
    if fixed["evidence_status"] == "N/A":
        fixed["evidence_status"] = "No"
    if fixed["evidence_status"] == "No":
        fixed["evidence_quality"] = "N/A"
    elif fixed["evidence_quality"] == "N/A":
        fixed["evidence_quality"] = "Not Clear"
    return fixed


def read_predictions(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    return json.loads(path.read_text(encoding="utf-8"))


def convert_rows(rows: list[dict[str, Any]], expected_start: int | None, expected_count: int | None) -> list[dict[str, str]]:
    output = []
    seen = set()
    for row in rows:
        row_id = str(row.get("id", "")).strip()
        if not row_id:
            raise ValueError("missing id")
        if row_id in seen:
            raise ValueError(f"duplicate id: {row_id}")
        seen.add(row_id)
        converted = {"id": row_id}
        for task in FIELDS[1:]:
            converted[task] = normalize(task, row.get(task))
        output.append(enforce_logic(converted))

    output.sort(key=lambda item: int(item["id"]) if item["id"].isdigit() else item["id"])
    if expected_count is not None and len(output) != expected_count:
        raise ValueError(f"expected {expected_count} rows, got {len(output)}")
    if expected_start is not None and expected_count is not None:
        expected_ids = {str(i) for i in range(expected_start, expected_start + expected_count)}
        actual_ids = {row["id"] for row in output}
        missing = sorted(expected_ids - actual_ids, key=int)
        extra = sorted(actual_ids - expected_ids, key=int)
        if missing or extra:
            raise ValueError(f"id mismatch: missing={missing[:5]} extra={extra[:5]}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert ESG prediction JSON/CSV to official submission CSV.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-start", type=int, default=12001)
    parser.add_argument("--expected-count", type=int, default=2000)
    parser.add_argument("--no-expected-test-ids", action="store_true", help="Do not enforce 12001~14000 id range.")
    args = parser.parse_args()

    expected_start = None if args.no_expected_test_ids else args.expected_start
    expected_count = None if args.no_expected_test_ids else args.expected_count
    rows = convert_rows(read_predictions(args.input), expected_start, expected_count)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")
    print(f"rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
