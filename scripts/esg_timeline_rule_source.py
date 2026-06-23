#!/usr/bin/env python3
"""Create a lightweight rule probability source for verification_timeline."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from esg_score_postprocess import LABELS, TASKS, normalize_label


YEAR_RE = re.compile(r"(20[2-5][0-9])\s*年?")
WITHIN_WORDS = ("短期", "近期", "當年度", "本年度", "每年", "年度")
MID_WORDS = ("中期", "中程", "三年", "五年")
LONG_WORDS = ("長期", "2050", "2040", "2035", "2030", "淨零", "net zero", "net-zero")
ALREADY_WORDS = ("已完成", "已達成", "已達標", "已取得", "已建立", "已導入", "已通過", "自 ", "自", "起提供", "持續")


def load_data(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["id"]): row for row in rows}


def normalize_probs(probs: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in probs.values())
    if total <= 0:
        return {label: 1.0 / len(probs) for label in probs}
    return {label: max(0.0, value) / total for label, value in probs.items()}


def timeline_rule_probs(text: str, base_year: int) -> dict[str, float]:
    lower = text.lower()
    scores = {
        "already": 0.45,
        "within_2_years": 0.25,
        "between_2_and_5_years": 0.35,
        "longer_than_5_years": 0.35,
        "N/A": 0.15,
    }

    years = [int(value) for value in YEAR_RE.findall(text)]
    future_years = [year for year in years if year >= base_year]
    if future_years:
        nearest = min(future_years)
        farthest = max(future_years)
        if nearest <= base_year + 2:
            scores["within_2_years"] += 1.2
        if base_year + 2 < nearest <= base_year + 5 or base_year + 2 < farthest <= base_year + 5:
            scores["between_2_and_5_years"] += 1.4
        if farthest > base_year + 5:
            scores["longer_than_5_years"] += 1.6

    if any(word in text for word in WITHIN_WORDS):
        scores["within_2_years"] += 0.7
    if any(word in text for word in MID_WORDS):
        scores["between_2_and_5_years"] += 0.8
    if any(word.lower() in lower for word in LONG_WORDS):
        scores["longer_than_5_years"] += 1.0
    if any(word in text for word in ALREADY_WORDS):
        scores["already"] += 0.9

    if not years and not any(word in text for word in WITHIN_WORDS + MID_WORDS) and not any(word.lower() in lower for word in LONG_WORDS):
        scores["N/A"] += 0.6
    return normalize_probs(scores)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build timeline rule source prediction JSON.")
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--base-pred", required=True, type=Path, help="Prediction JSON to copy non-timeline probabilities from.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-year", type=int, default=2024)
    args = parser.parse_args()

    data_by_id = load_data(args.data_path)
    base_rows = json.loads(args.base_pred.read_text(encoding="utf-8"))
    out = []
    for row in base_rows:
        row_id = str(row["id"])
        text = str(data_by_id.get(row_id, {}).get("data", ""))
        item = dict(row)
        probs = {task: dict(row["probs"][task]) for task in TASKS}
        probs["verification_timeline"] = timeline_rule_probs(text, args.base_year)
        pred = dict(row["pred"])
        pred["verification_timeline"] = max(LABELS["verification_timeline"], key=lambda label: probs["verification_timeline"].get(label, 0.0))
        item["mode"] = "timeline_rule"
        item["pred"] = {task: normalize_label(task, pred.get(task)) for task in TASKS}
        item["probs"] = probs
        out.append(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
