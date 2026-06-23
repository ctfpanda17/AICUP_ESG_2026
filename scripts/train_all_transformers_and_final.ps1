param(
    [int]$Epochs = 8,
    [int]$BatchSize = 2,
    [int]$MaxLength = 384,
    [string]$TestCsv = "..\vpesg4k_test_2000.csv",
    [switch]$ReuseExisting,
    [switch]$CoreModelsOnly
)

$ErrorActionPreference = "Stop"

$PackageRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $PackageRoot

$Python = Join-Path $PackageRoot "..\.venv-esg-corpus\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = "python"
}

function Invoke-PythonStep {
    param([string[]]$Arguments)
    Write-Host ""
    Write-Host ">>> python $($Arguments -join ' ')"
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python step failed: $($Arguments -join ' ')"
    }
}

function Add-OptionalFlag {
    param([string[]]$Arguments, [switch]$Condition, [string]$Flag)
    if ($Condition) {
        return $Arguments + $Flag
    }
    return $Arguments
}

$TrainJson = "data_full_retrain\vpesg4k_train_val_2000.json"
$AugJson = "data_full_retrain\vpesg4k_lmstudio_aug_1000.json"
$TaskwiseRun = "runs\full_retrain_transformers\taskwise"
$FivefoldRun = "runs\full_retrain_transformers\fivefold_roberta_large"
$FivefoldCombined = Join-Path $FivefoldRun "combined"
$TestRun = "runs\full_retrain_transformers\test_taskwise"
$StackingDir = "models\stacking_meta_full_retrain"
$StackingRun = "runs\stacking_full_retrain"

New-Item -ItemType Directory -Force -Path "runs\full_retrain_transformers", $StackingDir, $StackingRun, "outputs" | Out-Null

if (-not (Test-Path -LiteralPath $TrainJson)) {
    throw "Missing training data: $TrainJson"
}
if (-not (Test-Path -LiteralPath $AugJson)) {
    throw "Missing augmentation data: $AugJson"
}
if (-not (Test-Path -LiteralPath $TestCsv)) {
    throw "Missing test csv: $TestCsv"
}

$TaskwiseArgs = @(
    "scripts\run_final_pipeline.py",
    "--package-dir", $PackageRoot,
    "--work-dir", $TaskwiseRun,
    "--train-path", $TrainJson,
    "--augment-path", $AugJson,
    "--internal-split",
    "--split-seed", "42",
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--max-length", "$MaxLength",
    "--full-taskwise-search"
)
$TaskwiseArgs = Add-OptionalFlag -Arguments $TaskwiseArgs -Condition:$ReuseExisting -Flag "--reuse-existing"
$TaskwiseArgs = Add-OptionalFlag -Arguments $TaskwiseArgs -Condition:$CoreModelsOnly -Flag "--core-models-only"
Invoke-PythonStep $TaskwiseArgs

$FivefoldArgs = @(
    "scripts\run_5fold_roberta.py",
    "--package-dir", $PackageRoot,
    "--data-path", $TrainJson,
    "--test-csv", $TestCsv,
    "--work-dir", $FivefoldRun,
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--max-length", "$MaxLength",
    "--modes", "baseline", "fusion"
)
$FivefoldArgs = Add-OptionalFlag -Arguments $FivefoldArgs -Condition:$ReuseExisting -Flag "--reuse-existing"
Invoke-PythonStep $FivefoldArgs

Invoke-PythonStep @(
    "scripts\combine_5fold_sources.py",
    "--output-dir", $FivefoldCombined,
    "--oof",
    (Join-Path $FivefoldRun "oof_baseline_predictions.json"),
    (Join-Path $FivefoldRun "oof_fusion_predictions.json"),
    "--test",
    (Join-Path $FivefoldRun "test_5fold_baseline_predictions.json"),
    (Join-Path $FivefoldRun "test_5fold_fusion_predictions.json")
)

$TestArgs = @(
    "scripts\run_test_inference_from_checkpoints.py",
    "--test-csv", $TestCsv,
    "--work-dir", $TestRun,
    "--train-json", $TrainJson,
    "--augment-json", $AugJson,
    "--model-run-dir", $TaskwiseRun,
    "--batch-size", "$BatchSize"
)
$TestArgs = Add-OptionalFlag -Arguments $TestArgs -Condition:$ReuseExisting -Flag "--reuse-existing"
Invoke-PythonStep $TestArgs

Invoke-PythonStep @(
    "scripts\train_stacking_meta.py",
    "--data-path", $TrainJson,
    "--split-seed", "42",
    "--output-dir", $StackingDir,
    "--source", "current_taskwise", (Join-Path $TaskwiseRun "final_taskwise_ensemble_predictions.json"), "pred_taskwise",
    "--source", "super_thr", (Join-Path $TaskwiseRun "all_models\all_model_ensemble_thresholded.json"), "pred_thresholded",
    "--source", "fivefold_base", (Join-Path $FivefoldRun "oof_baseline_thresholded.json"), "pred_thresholded",
    "--source", "fivefold_fusion", (Join-Path $FivefoldRun "oof_fusion_thresholded.json"), "pred_thresholded",
    "--source", "fivefold_combined", (Join-Path $FivefoldCombined "oof_combined_thresholded.json"), "pred_thresholded",
    "--source", "qwen_timeline", "source_predictions\validation\qwen_timeline.json", "pred",
    "--source", "qwen_eq", "source_predictions\validation\qwen_eq.json", "pred"
)

Invoke-PythonStep @(
    "scripts\apply_stacking_meta.py",
    "--model", (Join-Path $StackingDir "models.pkl"),
    "--target-path", (Join-Path $TestRun "test_with_dummy_labels.json"),
    "--output-json", (Join-Path $StackingRun "test_stacking_predictions.json"),
    "--output-csv", "outputs\submission_full_retrain_stacking_raw.csv",
    "--source", "current_taskwise", (Join-Path $TestRun "final_taskwise_no_cross_fix.json"), "columns",
    "--source", "super_thr", (Join-Path $TestRun "all_model_ensemble_thresholded.json"), "pred_thresholded",
    "--source", "fivefold_base", (Join-Path $FivefoldRun "test_5fold_baseline_thresholded_predictions.json"), "pred_thresholded",
    "--source", "fivefold_fusion", (Join-Path $FivefoldRun "test_5fold_fusion_thresholded_predictions.json"), "pred_thresholded",
    "--source", "fivefold_combined", (Join-Path $FivefoldCombined "test_combined_thresholded_predictions.json"), "pred_thresholded",
    "--source", "qwen_timeline", "source_predictions\test\qwen_timeline.json", "pred",
    "--source", "qwen_eq", "source_predictions\test\qwen_eq.json", "pred"
)

Invoke-PythonStep @(
    "scripts\apply_selective_transition_from_json.py",
    "--base-csv", (Join-Path $TestRun "submission_taskwise_no_cross_fix.csv"),
    "--pred-json", (Join-Path $StackingRun "test_stacking_predictions.json"),
    "--task", "evidence_quality",
    "--transition", "Clear->N/A",
    "--require-base", "evidence_status=No",
    "--output", "outputs\_tmp_full_retrain_es_no.csv"
)

Invoke-PythonStep @(
    "scripts\apply_selective_transition_from_json.py",
    "--base-csv", "outputs\_tmp_full_retrain_es_no.csv",
    "--pred-json", (Join-Path $StackingRun "test_stacking_predictions.json"),
    "--task", "evidence_quality",
    "--transition", "Clear->N/A",
    "--require-base", "evidence_status=N/A",
    "--output", "outputs\submission_full_retrain_transformer_candidate.csv"
)

Write-Host ""
Write-Host "Done."
Write-Host "Bottom Transformer run: $TaskwiseRun"
Write-Host "5-fold Transformer run: $FivefoldRun"
Write-Host "Stacking model: $StackingDir\models.pkl"
Write-Host "Candidate submission: outputs\submission_full_retrain_transformer_candidate.csv"


