#!/usr/bin/env python3
"""Run the full task-wise ESG validation pipeline.

Pipeline:
1. Train/generate multiple model predictions.
2. Threshold each source.
3. Build the all-model probability ensemble.
4. Train/generate the timeline task-specific model.
5. Ensemble timeline task probabilities with all-model ensemble.
6. Threshold the timeline/all-model ensemble.
7. Apply the fixed best task-wise source mapping.

Use --reuse-existing to skip expensive training steps when the expected
prediction files already exist.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


TASKS = ["promise_status", "verification_timeline", "evidence_status", "evidence_quality"]


def run(cmd: list[str], cwd: Path, dry_run: bool = False) -> None:
    print("\n$ " + " ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def should_run(path: Path, reuse_existing: bool) -> bool:
    return not (reuse_existing and path.exists())


def write_model_manifest(work_dir: Path, dry_run: bool = False) -> None:
    manifest_path = work_dir / "model_manifest.json"
    print(f"\n# write model manifest: {manifest_path}", flush=True)
    if dry_run:
        return
    import torch

    rows = []
    for path in sorted(work_dir.rglob("*_model.pt")):
        try:
            checkpoint = torch.load(path, map_location="cpu")
            metadata = checkpoint.get("metadata", {})
        except Exception:
            metadata = {}
        rows.append(
            {
                "path": str(path.relative_to(work_dir)),
                "filename": path.name,
                "model_name": metadata.get("model_name", ""),
                "mode": metadata.get("mode", ""),
                "model_role": metadata.get("model_role", ""),
                "final_tasks": metadata.get("final_tasks", []),
                "best_epoch": metadata.get("best_epoch"),
                "best_selection_score": metadata.get("best_selection_score"),
            }
        )
    manifest_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_mode_prediction_files(out_dir: Path, modes: list[str], reuse_existing: bool, dry_run: bool = False) -> None:
    """Split main.py week9_predictions.json into one file per mode."""
    needed = [out_dir / f"{mode}_predictions.json" for mode in modes]
    if reuse_existing and all(path.exists() for path in needed):
        return
    source = out_dir / "week9_predictions.json"
    print(f"\n# split {source} -> mode prediction files", flush=True)
    if dry_run:
        return
    rows = json.loads(source.read_text(encoding="utf-8"))
    for mode in modes:
        mode_rows = [row for row in rows if row.get("mode") == mode]
        if not mode_rows:
            raise RuntimeError(f"No rows with mode={mode!r} in {source}")
        (out_dir / f"{mode}_predictions.json").write_text(
            json.dumps(mode_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def threshold_cmd(py: str, scripts_dir: Path, input_path: Path, output_path: Path, report_path: Path) -> list[str]:
    return [
        py,
        str(scripts_dir / "esg_threshold_search.py"),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--report",
        str(report_path),
    ]


def train_main_cmd(
    py: str,
    scripts_dir: Path,
    train_path: Path,
    val_path: Path | None,
    aug_path: Path,
    output_dir: Path,
    model_name: str,
    epochs: int,
    batch_size: int,
    max_length: int,
    lr: str,
    seed: int,
    split_seed: int,
    modes: list[str],
    train_task: str = "all",
    contrastive_alpha: str = "0.05",
    contrastive_task: str = "promise_status",
    save_model: bool = True,
    final_tasks: list[str] | None = None,
    model_role: str = "ensemble_source",
) -> list[str]:
    cmd = [
        py,
        str(scripts_dir / "main.py"),
        "--data-path",
        str(train_path),
        "--augment-train-path",
        str(aug_path),
        "--model-name",
        model_name,
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--max-length",
        str(max_length),
        "--lr",
        lr,
        "--modes",
        *modes,
        "--seed",
        str(seed),
        "--split-seed",
        str(split_seed),
        "--train-task",
        train_task,
        "--save-probs",
        "--input-enrichment",
        "both",
        "--contrastive-alpha",
        contrastive_alpha,
    ]
    if val_path is not None:
        cmd.extend(["--eval-path", str(val_path)])
    if float(contrastive_alpha) > 0:
        cmd.extend(["--contrastive-task", contrastive_task])
    if save_model:
        cmd.append("--save-model")
        if final_tasks:
            cmd.extend(["--final-tasks", ",".join(final_tasks)])
        cmd.extend(["--model-role", model_role])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full ESG final validation pipeline.")
    parser.add_argument("--package-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--val-path", type=Path, default=None)
    parser.add_argument("--augment-path", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument(
        "--internal-split",
        action="store_true",
        help="Do not pass --eval-path to child models; use each training file's internal 8:2 split.",
    )
    parser.add_argument("--split-seed", type=int, default=42, help="Shared split seed for --internal-split.")
    parser.add_argument(
        "--core-models-only",
        action="store_true",
        help="Skip optional extra backbones and use the core local models only.",
    )
    parser.add_argument("--reuse-existing", action="store_true", help="Skip steps whose expected output already exists.")
    parser.add_argument("--no-save-models", action="store_true", help="Do not save Transformer checkpoints.")
    parser.add_argument(
        "--no-collect-models",
        action="store_true",
        help="Do not copy the run's selected checkpoints into final_esg_package/trained_models.",
    )
    parser.add_argument(
        "--full-taskwise-search",
        action="store_true",
        help="Also threshold every individual source and run validation taskwise selection. Default only builds final selected sources.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    package_dir = args.package_dir.resolve()
    project_root = package_dir.parent.resolve()
    scripts_dir = package_dir / "scripts"
    data_dir = package_dir / "data"
    work_dir = (args.work_dir or (package_dir / "runs" / "full_pipeline")).resolve()
    train_path = (args.train_path or (data_dir / "vpesg4k_train_1000_V1.json")).resolve()
    val_path = None if args.internal_split else (args.val_path or (data_dir / "vpesg4k_val_1000.json")).resolve()
    aug_path = (args.augment_path or (data_dir / "vpesg4k_llm_paraphrase_aug_only_clean.json")).resolve()
    py = args.python

    work_dir.mkdir(parents=True, exist_ok=True)

    model_runs = [
        {
            "name": "base_seed42",
            "model": "hfl/chinese-roberta-wwm-ext",
            "seed": 42,
            "lr": "1e-5",
            "modes": ["baseline", "fusion"],
            "contrastive_alpha": "0.05",
        },
        {
            "name": "base_seed7",
            "model": "hfl/chinese-roberta-wwm-ext",
            "seed": 7,
            "lr": "1e-5",
            "modes": ["baseline", "fusion"],
            "contrastive_alpha": "0.05",
        },
        {
            "name": "roberta_large",
            "model": "hfl/chinese-roberta-wwm-ext-large",
            "seed": 42,
            "lr": "1e-5",
            "modes": ["baseline", "fusion"],
            "contrastive_alpha": "0.05",
        },
        {
            "name": "macbert_large",
            "model": "hfl/chinese-macbert-large",
            "seed": 42,
            "lr": "1e-5",
            "modes": ["baseline", "fusion"],
            "contrastive_alpha": "0.05",
        },
        {
            "name": "erlangshen",
            "model": "IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment",
            "seed": 42,
            "lr": "1e-5",
            "modes": ["baseline", "fusion"],
            "contrastive_alpha": "0.05",
        },
        {
            "name": "mdeberta",
            "model": "microsoft/mdeberta-v3-base",
            "seed": 42,
            "lr": "5e-6",
            "modes": ["baseline"],
            "contrastive_alpha": "0",
        },
    ]
    if args.core_models_only:
        model_runs = [
            config
            for config in model_runs
            if config["name"] in {"base_seed42", "base_seed7", "roberta_large", "macbert_large"}
        ]

    source_specs: list[tuple[str, Path, str]] = []
    for config in model_runs:
        out_dir = work_dir / config["name"]
        expected = out_dir / "week9_predictions.json"
        final_tasks = ["evidence_status", "evidence_quality"]
        model_role = "all_ensemble_source"
        if config["name"] == "roberta_large":
            final_tasks = ["promise_status", "evidence_status", "evidence_quality"]
            model_role = "promise_status_direct_and_all_ensemble_source"
        if should_run(expected, args.reuse_existing):
            run(
                train_main_cmd(
                    py,
                    scripts_dir,
                    train_path,
                    val_path,
                    aug_path,
                    out_dir,
                    config["model"],
                    args.epochs,
                    args.batch_size,
                    args.max_length,
                    config["lr"],
                    config["seed"],
                    args.split_seed,
                    config["modes"],
                    contrastive_alpha=config["contrastive_alpha"],
                    save_model=not args.no_save_models,
                    final_tasks=final_tasks,
                    model_role=model_role,
                ),
                project_root,
                args.dry_run,
            )
        ensure_mode_prediction_files(out_dir, config["modes"], args.reuse_existing, args.dry_run)
        for mode in config["modes"]:
            raw = out_dir / f"{mode}_predictions.json"
            thr = out_dir / f"{mode}_thresholded.json"
            report = out_dir / f"{mode}_threshold_report.md"
            needs_individual_threshold = args.full_taskwise_search or (config["name"] == "roberta_large" and mode == "baseline")
            if needs_individual_threshold and should_run(thr, args.reuse_existing):
                run(threshold_cmd(py, scripts_dir, raw, thr, report), project_root, args.dry_run)
        name = config["name"]
        if name == "base_seed42":
            source_specs.extend([("base42_b", out_dir / "baseline_predictions.json", "pred"), ("base42_f", out_dir / "fusion_predictions.json", "pred")])
        elif name == "base_seed7":
            source_specs.extend([("base7_b", out_dir / "baseline_predictions.json", "pred"), ("base7_f", out_dir / "fusion_predictions.json", "pred")])
        elif name == "roberta_large":
            source_specs.extend([("robertaL_b", out_dir / "baseline_predictions.json", "pred"), ("robertaL_f", out_dir / "fusion_predictions.json", "pred")])
        elif name == "macbert_large":
            source_specs.extend([("macbertL_b", out_dir / "baseline_predictions.json", "pred"), ("macbertL_f", out_dir / "fusion_predictions.json", "pred")])
        elif name == "erlangshen":
            source_specs.extend([("erlang_b", out_dir / "baseline_predictions.json", "pred"), ("erlang_f", out_dir / "fusion_predictions.json", "pred")])
        elif name == "mdeberta":
            source_specs.append(("mdeberta_b", out_dir / "baseline_predictions.json", "pred"))

    knn_dir = work_dir / "knn21"
    knn_raw = knn_dir / "k21_predictions.json"
    if should_run(knn_raw, args.reuse_existing):
        knn_cmd = [
            py,
            str(scripts_dir / "esg_knn_retrieval_baseline.py"),
            "--data-path",
            str(train_path),
        ]
        if val_path is not None:
            knn_cmd.extend(["--eval-path", str(val_path)])
        knn_cmd.extend(
            [
                "--augment-train-path",
                str(aug_path),
                "--output",
                str(knn_raw),
                "--report",
                str(knn_dir / "k21_report.md"),
                "--k",
                "21",
                "--seed",
                str(args.split_seed),
            ]
        )
        run(
            knn_cmd,
            project_root,
            args.dry_run,
        )
    if should_run(knn_dir / "k21_thresholded.json", args.reuse_existing):
        run(
            threshold_cmd(py, scripts_dir, knn_raw, knn_dir / "k21_thresholded.json", knn_dir / "k21_threshold_report.md"),
            project_root,
            args.dry_run,
        )
    source_specs.append(("knn21", knn_raw, "pred"))

    all_model_raw = work_dir / "all_models" / "all_model_ensemble_predictions.json"
    if should_run(all_model_raw, args.reuse_existing):
        run(
            [
                py,
                str(scripts_dir / "esg_ensemble_predictions.py"),
                "--inputs",
                *[str(path) for _name, path, _key in source_specs],
                "--names",
                *[name for name, _path, _key in source_specs],
                "--output",
                str(all_model_raw),
                "--report",
                str(work_dir / "all_models" / "all_model_ensemble_report.md"),
                "--tune-weights",
            ],
            project_root,
            args.dry_run,
        )
    all_model_thr = work_dir / "all_models" / "all_model_ensemble_thresholded.json"
    if should_run(all_model_thr, args.reuse_existing):
        run(
            threshold_cmd(
                py,
                scripts_dir,
                all_model_raw,
                all_model_thr,
                work_dir / "all_models" / "all_model_ensemble_threshold_report.md",
            ),
            project_root,
            args.dry_run,
        )

    all_taskwise = work_dir / "all_models" / "taskwise_all_models_predictions.json"
    if args.full_taskwise_search and should_run(all_taskwise, args.reuse_existing):
        taskwise_sources = [
            f"all_ensemble={all_model_raw}=pred",
            f"all_ensemble_thr={all_model_thr}=pred_thresholded",
        ]
        for name, path, _key in source_specs:
            taskwise_sources.append(f"{name}={path}=pred")
            mode_dir = path.parent
            thresholded = mode_dir / path.name.replace("_predictions.json", "_thresholded.json")
            taskwise_sources.append(f"{name}_thr={thresholded}=pred_thresholded")
        run(
            [
                py,
                str(scripts_dir / "esg_taskwise_select.py"),
                *sum((["--source", source] for source in taskwise_sources), []),
                "--output",
                str(all_taskwise),
                "--report",
                str(work_dir / "all_models" / "taskwise_all_models_report.md"),
            ],
            project_root,
            args.dry_run,
        )

    timeline_dir = work_dir / "timeline_task"
    timeline_raw = timeline_dir / "baseline_predictions.json"
    if should_run(timeline_raw, args.reuse_existing):
        run(
            train_main_cmd(
                py,
                scripts_dir,
                train_path,
                val_path,
                aug_path,
                timeline_dir,
                "hfl/chinese-roberta-wwm-ext-large",
                args.epochs,
                args.batch_size,
                args.max_length,
                "1e-5",
                42,
                args.split_seed,
                ["baseline"],
                train_task="verification_timeline",
                contrastive_alpha="0.05",
                contrastive_task="verification_timeline",
                save_model=not args.no_save_models,
                final_tasks=["verification_timeline"],
                model_role="timeline_task_direct_and_timeline_ensemble_source",
            ),
            project_root,
            args.dry_run,
        )
    ensure_mode_prediction_files(timeline_dir, ["baseline"], args.reuse_existing, args.dry_run)
    if should_run(timeline_dir / "baseline_thresholded.json", args.reuse_existing):
        run(
            threshold_cmd(
                py,
                scripts_dir,
                timeline_raw,
                timeline_dir / "baseline_thresholded.json",
                timeline_dir / "baseline_threshold_report.md",
            ),
            project_root,
            args.dry_run,
        )

    timeline_plus_raw = work_dir / "timeline_plus_allmodel" / "timeline_plus_allmodel_ensemble.json"
    if should_run(timeline_plus_raw, args.reuse_existing):
        run(
            [
                py,
                str(scripts_dir / "esg_ensemble_predictions.py"),
                "--inputs",
                str(all_model_raw),
                str(timeline_raw),
                "--names",
                "all_model",
                "timeline_task",
                "--output",
                str(timeline_plus_raw),
                "--report",
                str(work_dir / "timeline_plus_allmodel" / "timeline_plus_allmodel_ensemble_report.md"),
                "--tune-weights",
                "--weight-grid",
                "0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0",
            ],
            project_root,
            args.dry_run,
        )
    timeline_plus_thr = work_dir / "timeline_plus_allmodel" / "timeline_plus_allmodel_ensemble_thresholded.json"
    if should_run(timeline_plus_thr, args.reuse_existing):
        run(
            threshold_cmd(
                py,
                scripts_dir,
                timeline_plus_raw,
                timeline_plus_thr,
                work_dir / "timeline_plus_allmodel" / "timeline_plus_allmodel_ensemble_threshold_report.md",
            ),
            project_root,
            args.dry_run,
        )

    final_taskwise = work_dir / "final_taskwise_ensemble_predictions.json"
    if args.full_taskwise_search and should_run(final_taskwise, args.reuse_existing):
        run(
            [
                py,
                str(scripts_dir / "esg_taskwise_select.py"),
                "--source",
                f"all_taskwise={all_taskwise}=pred_taskwise",
                "--source",
                f"timeline_task={timeline_raw}=pred",
                "--source",
                f"timeline_task_thr={timeline_dir / 'baseline_thresholded.json'}=pred_thresholded",
                "--source",
                f"timeline_plus_all={timeline_plus_raw}=pred",
                "--source",
                f"timeline_plus_all_thr={timeline_plus_thr}=pred_thresholded",
                "--output",
                str(final_taskwise),
                "--report",
                str(work_dir / "final_taskwise_ensemble_report.md"),
            ],
            project_root,
            args.dry_run,
        )

    final_predictions = work_dir / "final_task_specific_predictions.json"
    if should_run(final_predictions, args.reuse_existing):
        if args.full_taskwise_search:
            sources = [
                "--source",
                f"final_taskwise={final_taskwise}=pred_taskwise",
            ]
            task_sources = [
                "--task-source",
                "promise_status=final_taskwise",
                "--task-source",
                "verification_timeline=final_taskwise",
                "--task-source",
                "evidence_status=final_taskwise",
                "--task-source",
                "evidence_quality=final_taskwise",
            ]
        else:
            sources = [
                "--source",
                f"robertaL_b_thr={work_dir / 'roberta_large' / 'baseline_thresholded.json'}=pred_thresholded",
                "--source",
                f"all_ensemble_thr={all_model_thr}=pred_thresholded",
                "--source",
                f"timeline_plus_all_thr={timeline_plus_thr}=pred_thresholded",
            ]
            task_sources = [
                "--task-source",
                "promise_status=robertaL_b_thr",
                "--task-source",
                "verification_timeline=timeline_plus_all_thr",
                "--task-source",
                "evidence_status=all_ensemble_thr",
                "--task-source",
                "evidence_quality=all_ensemble_thr",
            ]
        run(
            [
                py,
                str(scripts_dir / "esg_apply_taskwise_sources.py"),
                *sources,
                *task_sources,
                "--output",
                str(final_predictions),
                "--submission-output",
                str(work_dir / "final_submission.json"),
            ],
            project_root,
            args.dry_run,
        )
    final_scored = work_dir / "final_task_specific_scored.json"
    if should_run(final_scored, args.reuse_existing):
        run(
            [
                py,
                str(scripts_dir / "esg_score_postprocess.py"),
                "--input",
                str(final_predictions),
                "--output",
                str(final_scored),
                "--report",
                str(work_dir / "final_score_report.md"),
                "--mode",
                "none",
            ],
            project_root,
            args.dry_run,
        )

    write_model_manifest(work_dir, args.dry_run)
    if not args.no_collect_models:
        run(
            [
                py,
                str(scripts_dir / "collect_final_task_models.py"),
                "--run-dir",
                str(work_dir),
                "--output-dir",
                str(package_dir / "trained_models"),
                "--clear",
            ],
            project_root,
            args.dry_run,
        )

    print("\nDone.")
    print(f"Work dir: {work_dir}")
    print(f"Final submission-like output: {work_dir / 'final_submission.json'}")
    print(f"Final score report: {work_dir / 'final_score_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
