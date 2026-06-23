#!/usr/bin/env python3
"""Collect trained checkpoints into task-centric folders.

The final system is task-wise:
- promise_status uses a direct RoBERTa-large checkpoint.
- verification_timeline uses a direct timeline checkpoint plus ensemble output.
- evidence_status and evidence_quality use the all-model ensemble.

This script organizes files so users see four final task folders, while shared
ensemble member checkpoints are kept separately.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


TASKS = ["promise_status", "verification_timeline", "evidence_status", "evidence_quality"]


def copy_first(patterns: list[str], search_root: Path, dest: Path) -> str | None:
    for pattern in patterns:
        matches = sorted(search_root.rglob(pattern))
        if matches:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(matches[0], dest)
            return str(matches[0])
    return None


def copy_first_from_roots(patterns: list[str], search_roots: list[Path], dest: Path) -> str | None:
    for root in search_roots:
        copied = copy_first(patterns, root, dest)
        if copied:
            return copied
    return None


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect final task model artifacts.")
    parser.add_argument("--run-dir", type=Path, default=Path("final_esg_package/runs/full_pipeline"))
    parser.add_argument("--output-dir", type=Path, default=Path("final_esg_package/trained_models"))
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    output_dir = args.output_dir
    if args.clear and output_dir.exists():
        shutil.rmtree(output_dir)

    final_dir = output_dir / "final_tasks"
    member_dir = output_dir / "ensemble_members"
    final_dir.mkdir(parents=True, exist_ok=True)
    member_dir.mkdir(parents=True, exist_ok=True)

    copied_members = []
    for ckpt in sorted(run_dir.rglob("*_model.pt")):
        dest = member_dir / f"{ckpt.parent.name}__{ckpt.name}"
        shutil.copy2(ckpt, dest)
        copied_members.append(str(dest.relative_to(output_dir)))

    manifest = {
        "run_dir": str(run_dir),
        "final_tasks": {},
        "ensemble_members": copied_members,
    }

    promise_src = copy_first_from_roots(
        [
            "*baseline*promise_status_direct*tasks_promise_status_model.pt",
            "roberta_large/*baseline*promise_status*model.pt",
        ],
        [run_dir.parent / "promise_status_task_seed42", run_dir],
        final_dir / "promise_status" / "promise_status_model.pt",
    )
    manifest["final_tasks"]["promise_status"] = {
        "type": "direct_checkpoint",
        "source": "promise_status_task_seed42_thr" if "promise_status_task_seed42" in str(promise_src) else "robertaL_b_thr",
        "checkpoint": "final_tasks/promise_status/promise_status_model.pt" if promise_src else None,
        "original_path": promise_src,
    }
    promise_threshold_src = copy_first_from_roots(
        ["baseline_threshold_report.md", "roberta_large/baseline_threshold_report.md"],
        [run_dir.parent / "promise_status_task_seed42", run_dir],
        final_dir / "promise_status" / "promise_status_threshold_report.md",
    )
    if promise_threshold_src:
        manifest["final_tasks"]["promise_status"]["threshold_report"] = "final_tasks/promise_status/promise_status_threshold_report.md"

    timeline_src = copy_first(
        ["timeline_task/*verification_timeline*model.pt"],
        run_dir,
        final_dir / "verification_timeline" / "verification_timeline_model.pt",
    )
    timeline_payload = {
        "type": "direct_checkpoint_plus_ensemble_threshold",
        "source": "timeline_plus_all_thr",
        "checkpoint": "final_tasks/verification_timeline/verification_timeline_model.pt" if timeline_src else None,
        "original_path": timeline_src,
        "thresholded_prediction_source": str(run_dir / "timeline_plus_allmodel" / "timeline_plus_allmodel_ensemble_thresholded.json"),
        "note": "If checkpoint is null, rerun timeline_task after --save-model was added.",
    }
    write_json(final_dir / "verification_timeline" / "verification_timeline_ensemble_model.json", timeline_payload)
    manifest["final_tasks"]["verification_timeline"] = timeline_payload

    all_ensemble = run_dir / "all_models" / "all_model_ensemble_thresholded.json"
    all_ensemble_members = [member for member in copied_members if not Path(member).name.startswith("timeline_task__")]
    for task in ["evidence_status", "evidence_quality"]:
        payload = {
            "type": "ensemble_model",
            "task": task,
            "source": "all_ensemble_thr",
            "thresholded_prediction_source": str(all_ensemble),
            "member_checkpoint_dir": "../ensemble_members",
            "member_checkpoints": all_ensemble_members,
            "note": "This task's best final source is an ensemble, not one standalone checkpoint.",
        }
        write_json(final_dir / task / f"{task}_ensemble_model.json", payload)
        manifest["final_tasks"][task] = payload

    write_json(output_dir / "model_manifest.json", manifest)
    readme = [
        "# Trained Models",
        "",
        "此資料夾用 task-centric 方式整理模型。",
        "",
        "- `final_tasks/`: 四個任務各一個 final model artifact。",
        "- `ensemble_members/`: all-model ensemble 需要的底層 checkpoint。",
        "- `model_manifest.json`: 總清單。",
        "",
        "注意：`evidence_status` 和 `evidence_quality` 的最佳來源是 all-model ensemble，所以 final task artifact 是 ensemble config，不是單一 `.pt`。",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(f"wrote {output_dir / 'model_manifest.json'}")
    print(f"final task folders: {', '.join(TASKS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
