# ESG 最終提交版本 0.6057380

這個資料夾是 AICUP ESG 任務最終 public score `0.6057380` 對應的提交版本。

## 最終提交檔

請提交：

```text
outputs/submission_0_6057380.csv
```

已知 public leaderboard 分數：

```text
weighted_score: 0.6057380
promise_status: 0.7743
verification_timeline: 0.5803
evidence_status: 0.6843
evidence_quality: 0.4530
```

## 任務目標

本專案針對 ESG 報告文字進行四個分類任務：

```text
promise_status          判斷文字是否包含 ESG 承諾
verification_timeline   判斷承諾的完成或驗證時間
evidence_status         判斷是否有支持承諾的證據
evidence_quality        判斷證據品質
```

最後輸出格式為：

```text
id,promise_status,verification_timeline,evidence_status,evidence_quality
```

## 最終方法摘要

最終提交採用多模型整合流程，而不是只使用單一模型輸出。

流程：

1. 使用多個已訓練來源模型產生 validation/test 預測。
2. 使用 task-wise stacking meta classifier，分別為四個任務學習如何整合不同來源模型。
3. stacking 特徵包含來源模型答案、模型投票比例、預測機率與文字統計特徵。
4. 對最終答案套用 ESG 標籤一致性規則：

```text
evidence_quality: Clear -> N/A
條件：evidence_status = No 或 N/A
```

這條規則的意思是：如果沒有可用證據，證據品質就應該是不適用，而不是 Clear。

## 重要檔案

```text
outputs/submission_0_6057380.csv

models/stacking_meta/models.pkl
models/stacking_meta/report.json

runs/stacking/test_stacking_predictions.json
runs/stacking/stacked_val_predictions.json

source_predictions/test/
source_predictions/validation/

scripts/apply_stacking_meta.py
scripts/apply_selective_transition_from_json.py
scripts/train_stacking_meta.py
scripts/evaluate_selective_transition_val.py
scripts/esg_score_postprocess.py
scripts/main.py
```

## 檔案角色

`source_predictions` 保存多個來源模型在 validation/test 上的預測結果。

`models/stacking_meta/models.pkl` 是訓練好的 stacking meta model。它會根據多個來源模型的答案，為每個任務選出最終答案。

`models/stacking_meta/report.json` 記錄 stacking meta model 在 validation 上的模型選擇與分數。

`runs/stacking/test_stacking_predictions.json` 是 stacking meta model 對 test 資料產生的中間預測。

`outputs/submission_0_6057380.csv` 是已套用最終規則後的正式提交檔。

## 重現 stacking 預測

在本資料夾內執行：

```powershell
..\.venv-esg-corpus\Scripts\python.exe scripts\apply_stacking_meta.py `
  --model models\stacking_meta\models.pkl `
  --target-path source_predictions\test\test_with_dummy_labels.json `
  --output-json runs\stacking\test_stacking_predictions_reproduced.json `
  --output-csv outputs\submission_stacking_full_reproduced.csv `
  --source current_taskwise source_predictions\test\current_taskwise.json columns `
  --source super_thr source_predictions\test\super_thr.json pred_thresholded `
  --source fivefold_base source_predictions\test\fivefold_base.json pred_thresholded `
  --source fivefold_fusion source_predictions\test\fivefold_fusion.json pred_thresholded `
  --source fivefold_combined source_predictions\test\fivefold_combined.json pred_thresholded `
  --source qwen_timeline source_predictions\test\qwen_timeline.json pred `
  --source qwen_eq source_predictions\test\qwen_eq.json pred
```

這會重新產生 stacking 的 test 預測。正式提交檔 `outputs/submission_0_6057380.csv` 則是最終整理完成的版本。

## 注意

為了讓 final package 保持可交付且不要超過數 GB，這裡保留 stacking meta model、來源預測與最終提交檔，沒有保留所有深度模型 checkpoint。
