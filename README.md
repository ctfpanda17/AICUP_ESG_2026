# ESG 永續承諾驗證競賽最終方案

本資料夾整理的是最終提交版本 `submission_0_6057380.csv`，重點包含模型架構、資料增強、核心策略、重訓方式與競賽結果。最終提交檔位於：

```text
outputs/submission_0_6057380.csv
```

## 模型與訓練模態（Models & Modalities）

本方案不是使用單一模型完成四個任務，而是採用多模型、多任務來源、多階段融合的 task-wise ensemble。四個任務分別是：

```text
promise_status
verification_timeline
evidence_status
evidence_quality
```

主要底層模型如下：

| 模型                                          | 用途                                                                        |
| --------------------------------------------- | --------------------------------------------------------------------------- |
| `hfl/chinese-roberta-wwm-ext`                 | 中文 RoBERTa baseline，多 seed 訓練，提供基本語意分類來源                   |
| `hfl/chinese-roberta-wwm-ext-large`           | RoBERTa-large，作為 promise、timeline 與 5-fold ensemble 的主要強模型       |
| `hfl/chinese-macbert-large`                   | MacBERT-large，補強繁中語意與 evidence 類任務                               |
| `IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment` | 額外中文 encoder 來源，增加 ensemble 多樣性                                 |
| `microsoft/mdeberta-v3-base`                  | multilingual DeBERTa 來源，用於補充跨語言語意特徵                           |
| KNN retrieval baseline                        | 以相似訓練樣本投票，補強少量資料下的近鄰決策                                |
| Qwen/LLM teacher predictions                  | 外部大型語言模型 teacher source，主要補強 timeline 與 evidence_quality 判斷 |
| Stacking meta-selector                        | 最後一層 task-wise meta model，學習各來源在不同任務下的可靠度               |

訓練模態包含：

1. **多任務 fine-tuning**：同一個 encoder 同時輸出四個任務的 classification heads，讓模型共享 ESG 語意表示。
2. **Fusion mode**：除原始文本外，額外加入結構化特徵與統計訊號，提升 evidence 類任務判斷。
3. **Task-specific fine-tuning**：針對 `verification_timeline` 等弱項額外訓練任務專用模型。
4. **5-fold ensemble**：RoBERTa-large 以 5-fold 方式訓練，產生 OOF validation predictions 與 test ensemble predictions。
5. **Threshold tuning**：針對各任務、各 label 以 validation macro-F1 搜尋 threshold。
6. **Stacking meta-selector**：使用 validation source predictions 訓練最後的 task-wise selector，再套用到 test source predictions。

完整底層 transformer 重訓腳本：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_all_transformers_and_final.ps1 `
  -TestCsv C:\Users\User\Downloads\vpesg4k_test_2000.csv `
  -Epochs 8 `
  -BatchSize 2 `
  -MaxLength 384 `
  -CoreModelsOnly
```

只重訓最後 stacking 層：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_final_stacking_model.ps1
```

## 資料增強（Data Augmentation）

有做資料增強，目標是補強少數類別與 ESG 文本語意邊界，尤其是 `evidence_quality` 與 `verification_timeline`。

使用的資料與處理策略包括：

### 資料增強模型與來源

| 增強來源 / 模型                       | 用途                                                                                            | 保留檔案                                                                                          |
| ------------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| DeepSeek-V4-Pro                       | 批次產生主要擴增資料，模仿繁體中文 ESG 報告書語氣，補強少數類別與語意邊界                       | `data_full_retrain/vpesg4k_lmstudio_aug_1000.json`                                                |
| Qwen3-35B / Qwen teacher              | 產生少量高品質 minority samples，並作為 timeline、evidence_quality 的 teacher prediction source | `data_full_retrain/vpesg4k_llm_minority_aug_qwen35b_110.json`、`source_predictions/*/qwen_*.json` |
| Codex rule-guided generation          | 依官方標註規則產生 targeted augmentation，尤其補強 Not Clear、N/A 與 timeline 類別              | `data_full_retrain/vpesg4k_codex_targeted_aug_1154.json`                                          |
| Controlled Clear -> Not Clear rewrite | 將 Clear evidence 改寫成「方向明確但缺乏量化成效 / 追蹤機制」的 Not Clear evidence              | `data_full_retrain/vpesg4k_llm_clear_to_not_clear_250.json`                                       |

最終 full retrain 腳本預設使用：

```text
data_full_retrain/vpesg4k_train_val_2000.json
data_full_retrain/vpesg4k_lmstudio_aug_1000.json
```

其他增強檔案保留在 `data_full_retrain/`，作為實驗紀錄與可替換訓練資料來源。

| 方法                                  | 說明                                                                                          |
| ------------------------------------- | --------------------------------------------------------------------------------------------- |
| LLM paraphrase augmentation           | 對少數類別樣本改寫，保留原 label 與 ESG 報告語氣                                              |
| LLM synthetic augmentation            | 針對 Not Clear、timeline 等弱類別生成接近企業永續報告書語氣的新樣本                           |
| Clear -> Not Clear controlled rewrite | 將含具體數字、年份、百分比的 Clear evidence 改寫為方向明確但缺乏量化追蹤的 Not Clear evidence |
| 類別不平衡處理                        | 針對 Macro-F1 評分特性，避免模型只偏向多數類別                                                |
| 5-fold OOF generation                 | 用 OOF predictions 作為 stacking 訓練資料，降低 validation leakage                            |
| 外部 ESG/Promise 類資料探索           | 嘗試 ML-Promise、中文 ESG quality 資料等作為 weak supervision 或 teacher/RAG 參考             |

最後保留在 package 中的主要重訓資料：

```text
data_full_retrain/vpesg4k_train_val_2000.json
data_full_retrain/vpesg4k_lmstudio_aug_1000.json
data_full_retrain/vpesg4k_llm_paraphrase_aug_only_clean.json
```

## 核心策略與特殊設計（Strategies & Special Designs）

本競賽的分數採四個任務加權 Macro-F1，因此策略重點不是追求單一模型整體 accuracy，而是針對各任務選擇最可靠的來源。

核心設計如下：

1. **Task-wise source selection**

   不強迫四個任務使用同一個模型。不同任務分別從 all-model ensemble、timeline-specific model、5-fold RoBERTa、LLM teacher 與 stacking selector 中選擇最佳來源。

2. **All-model ensemble**

   將多個 encoder、多 seed、baseline/fusion mode 的 probability 融合，降低單一 checkpoint 對 public test 分布的敏感度。

3. **Timeline-specific model + ensemble**

   `verification_timeline` 單獨訓練 task-specific model，再與 all-model ensemble 融合，處理時間語意、年份與承諾完成區間的分類問題。

4. **5-fold RoBERTa-large ensemble**

   使用 5-fold 產生 validation OOF predictions，讓 stacking meta-selector 學到更穩定的來源可靠度，並用 5 個 fold 的 test probability 平均作為 test source。

5. **Threshold tuning**

   對各任務 label 做 threshold search，以 Macro-F1 為目標調整少數類別召回率。這對 `evidence_status` 與 `verification_timeline` 特別重要。

6. **Evidence quality 保守後處理**

   最後提交沒有大幅覆蓋 `evidence_quality`，只採用一個低風險規則：當 base prediction 的 `evidence_status` 為 `No` 或 `N/A`，且 stacking 判斷 `evidence_quality` 應由 `Clear` 轉為 `N/A` 時才套用轉換。此設計避免過度修改 Clear / Not Clear 邊界。

7. **Stacking meta-selector**

   最後一層模型不是直接讀 ESG 文本，而是讀取多個 source model 的預測結果與簡單文字特徵，學習「哪個來源在這個任務上比較可信」。這比單純 hard voting 更能利用各模型互補性。

## 競賽成果與排名（Results & Rankings）

最終保留提交檔：

```text
outputs/submission_0_6057380.csv
```

Public leaderboard 結果：

| 指標                  |      分數 |
| --------------------- | --------: |
| weighted_score        | 0.6057380 |
| promise_status        |    0.7743 |
| verification_timeline |    0.5803 |
| evidence_status       |    0.6843 |
| evidence_quality      |    0.4530 |

Public leaderboard 排名：

```text
62 / 143
```

Private leaderboard：目前資料夾內沒有官方 private final score 檔案；若主辦單位公布 Private Leaderboard，請將最終 private 分數與排名補在此處。

## 檔案結構

```text
outputs/submission_0_6057380.csv
outputs/intermediate_taskwise_before_eq_transition.csv

models/stacking_meta/models.pkl
models/stacking_meta/report.json

source_predictions/validation/
source_predictions/test/

scripts/main.py
scripts/run_final_pipeline.py
scripts/run_5fold_roberta.py
scripts/run_test_inference_from_checkpoints.py
scripts/train_stacking_meta.py
scripts/apply_stacking_meta.py
scripts/apply_selective_transition_from_json.py
scripts/train_final_stacking_model.ps1
scripts/train_all_transformers_and_final.ps1
```

## 重現最終提交

在 `final_esg_package` 目錄內執行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_final_stacking_model.ps1
```

輸出：

```text
models/stacking_meta_retrained/models.pkl
outputs/submission_final_retrained.csv
```

若要從底層 transformer 重新訓練，請執行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_all_transformers_and_final.ps1 `
  -TestCsv C:\Users\User\Downloads\vpesg4k_test_2000.csv `
  -Epochs 8 `
  -BatchSize 2 `
  -MaxLength 384 `
  -CoreModelsOnly
```

輸出：

```text
runs/full_retrain_transformers/
models/stacking_meta_full_retrain/models.pkl
outputs/submission_full_retrain_transformer_candidate.csv
```

注意：底層 transformer 重訓會受到 GPU、CUDA、PyTorch、Transformers 版本與隨機性的影響，因此不保證重新訓練後的 CSV 與最終提交檔 byte-for-byte 完全相同；但此 package 已保留完整策略、資料、程式與模型來源配置。
