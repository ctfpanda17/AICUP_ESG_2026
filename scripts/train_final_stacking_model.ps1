$ErrorActionPreference = "Stop"

$PackageRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $PackageRoot

$Python = Join-Path $PackageRoot "..\.venv-esg-corpus\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = "python"
}

function Invoke-PythonStep {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python step failed: $($Arguments -join ' ')"
    }
}

New-Item -ItemType Directory -Force -Path (Join-Path $PackageRoot "models\stacking_meta_retrained") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PackageRoot "runs\stacking_retrained") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PackageRoot "outputs") | Out-Null

Invoke-PythonStep @(
  "scripts\train_stacking_meta.py",
  "--data-path", "data\vpesg4k_train_val_2000.json",
  "--split-seed", "42",
  "--output-dir", "models\stacking_meta_retrained",
  "--source", "current_taskwise", "source_predictions\validation\current_taskwise.json", "pred_taskwise",
  "--source", "super_thr", "source_predictions\validation\super_thr.json", "pred_thresholded",
  "--source", "fivefold_base", "source_predictions\validation\fivefold_base.json", "pred_thresholded",
  "--source", "fivefold_fusion", "source_predictions\validation\fivefold_fusion.json", "pred_thresholded",
  "--source", "fivefold_combined", "source_predictions\validation\fivefold_combined.json", "pred_thresholded",
  "--source", "qwen_timeline", "source_predictions\validation\qwen_timeline.json", "pred",
  "--source", "qwen_eq", "source_predictions\validation\qwen_eq.json", "pred"
)

Invoke-PythonStep @(
  "scripts\apply_stacking_meta.py",
  "--model", "models\stacking_meta_retrained\models.pkl",
  "--target-path", "source_predictions\test\test_with_dummy_labels.json",
  "--output-json", "runs\stacking_retrained\test_stacking_predictions.json",
  "--output-csv", "outputs\submission_stacking_full_retrained.csv",
  "--source", "current_taskwise", "source_predictions\test\current_taskwise.json", "columns",
  "--source", "super_thr", "source_predictions\test\super_thr.json", "pred_thresholded",
  "--source", "fivefold_base", "source_predictions\test\fivefold_base.json", "pred_thresholded",
  "--source", "fivefold_fusion", "source_predictions\test\fivefold_fusion.json", "pred_thresholded",
  "--source", "fivefold_combined", "source_predictions\test\fivefold_combined.json", "pred_thresholded",
  "--source", "qwen_timeline", "source_predictions\test\qwen_timeline.json", "pred",
  "--source", "qwen_eq", "source_predictions\test\qwen_eq.json", "pred"
)

Invoke-PythonStep @(
  "scripts\apply_selective_transition_from_json.py",
  "--base-csv", "outputs\base_public_0_6045779.csv",
  "--pred-json", "runs\stacking_retrained\test_stacking_predictions.json",
  "--task", "evidence_quality",
  "--transition", "Clear->N/A",
  "--require-base", "evidence_status=No",
  "--output", "outputs\_tmp_retrained_es_no.csv"
)

Invoke-PythonStep @(
  "scripts\apply_selective_transition_from_json.py",
  "--base-csv", "outputs\_tmp_retrained_es_no.csv",
  "--pred-json", "runs\stacking_retrained\test_stacking_predictions.json",
  "--task", "evidence_quality",
  "--transition", "Clear->N/A",
  "--require-base", "evidence_status=N/A",
  "--output", "outputs\submission_0_6057380_retrained.csv"
)

Write-Host "Done."
Write-Host "Retrained model: models\stacking_meta_retrained\models.pkl"
Write-Host "Retrained submission: outputs\submission_0_6057380_retrained.csv"
