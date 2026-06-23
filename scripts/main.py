import argparse
import csv
import json
import random
import re
import sys
import warnings
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")


def disable_hf_auto_conversion():
    """Avoid Hugging Face background safetensors conversion discussion checks."""
    try:
        import transformers.modeling_utils as modeling_utils
        import transformers.safetensors_conversion as safetensors_conversion

        def _disabled_auto_conversion(_pretrained_model_name_or_path, **cached_file_kwargs):
            return None, cached_file_kwargs.get("revision"), False

        modeling_utils.auto_conversion = _disabled_auto_conversion
        safetensors_conversion.auto_conversion = _disabled_auto_conversion
    except Exception:
        pass


TASKS = [
    "promise_status",
    "verification_timeline",
    "evidence_status",
    "evidence_quality",
]

LABELS = {
    "promise_status": ["Yes", "No"],
    "verification_timeline": [
        "already",
        "within_2_years",
        "between_2_and_5_years",
        "longer_than_5_years",
        "N/A",
    ],
    "evidence_status": ["Yes", "No", "N/A"],
    "evidence_quality": ["Clear", "Not Clear", "Misleading", "N/A"],
}

DROP_COLS = {
    "esg_type",
    "company",
    "ticker",
    "page_number",
    "pdf_url",
    "company_source",
}

ESG_TASK_PRIMER = (
    "ESG永續承諾驗證：ESG包含環境E、社會S、治理G。"
    "承諾是企業對未來或持續行動的目標、策略、原則或具體作為。"
    "證據是支持承諾的計畫、數據、年份、完成紀錄、認證、制度或執行成果。"
    "清楚證據通常具體、量化、可驗證；不清楚證據通常模糊、空泛或缺少衡量方式。"
    "驗證時程依承諾完成或可檢核時間分為已完成、兩年內、二到五年、五年以上或不適用。"
)

PROMISE_TERMS = [
    "承諾",
    "目標",
    "致力",
    "推動",
    "規劃",
    "預計",
    "將",
    "持續",
    "達成",
    "完成",
    "改善",
    "提升",
    "減少",
    "降低",
    "建立",
]

EVIDENCE_TERMS = [
    "已",
    "實績",
    "成果",
    "數據",
    "通過",
    "取得",
    "認證",
    "查證",
    "稽核",
    "投入",
    "完成",
    "執行",
]

VAGUE_TERMS = [
    "持續",
    "逐步",
    "積極",
    "致力",
    "努力",
    "強化",
    "優化",
    "提升",
    "促進",
]


def make_input_enrichment(raw_item, mode):
    if mode == "none":
        return ""

    text = str(raw_item.get("data", ""))
    years = sorted(set(re.findall(r"(?:19|20)\d{2}", text)))[:6]
    has_metric = bool(re.search(r"\d+(?:\.\d+)?\s*(?:%|％|公噸|噸|件|人|天|週|年|億元|萬元|kWh|度|tCO2e|CO2e)", text))
    promise_hits = [term for term in PROMISE_TERMS if term in text][:6]
    evidence_hits = [term for term in EVIDENCE_TERMS if term in text][:6]
    vague_hits = [term for term in VAGUE_TERMS if term in text][:6]
    esg_type = str(raw_item.get("esg_type", "") or "未知")
    feature_tags = (
        f"文本線索：ESG類型={esg_type}；年份={','.join(years) if years else '無'}；"
        f"量化數據={'有' if has_metric else '無'}；"
        f"承諾詞={','.join(promise_hits) if promise_hits else '無'}；"
        f"證據詞={','.join(evidence_hits) if evidence_hits else '無'}；"
        f"模糊詞={','.join(vague_hits) if vague_hits else '無'}。"
    )
    if mode == "feature_tags":
        return feature_tags
    if mode == "esg_primer":
        return ESG_TASK_PRIMER
    if mode == "both":
        return ESG_TASK_PRIMER + feature_tags
    raise ValueError(f"Unknown input enrichment mode: {mode}")

WEIGHTS = {
    "promise_status": 0.20,
    "verification_timeline": 0.15,
    "evidence_status": 0.30,
    "evidence_quality": 0.35,
}

LABEL_DESCRIPTIONS = {
    "promise_status": {
        "Yes": "這段文字包含明確的 ESG 承諾、目標或永續行動。",
        "No": "這段文字沒有 ESG 承諾、目標或永續行動。",
    },
    "verification_timeline": {
        "already": "承諾已經完成、已達成或已被驗證。",
        "within_2_years": "承諾的目標期限在兩年以內。",
        "between_2_and_5_years": "承諾的目標期限介於兩年到五年之間。",
        "longer_than_5_years": "承諾的目標期限超過五年。",
        "N/A": "文字中沒有適用的承諾時程。",
    },
    "evidence_status": {
        "Yes": "承諾的證據具體、清楚且沒有歧義。",
        "No": "承諾的證據缺失、模糊或有歧義。",
        "N/A": "沒有承諾，因此不適用證據判斷。",
    },
    "evidence_quality": {
        "Clear": "證據清楚、具體且可驗證。",
        "Not Clear": "證據存在但不夠清楚或不完整。",
        "Misleading": "證據可能造成誤導或有漂綠風險。",
        "N/A": "沒有適用的證據品質判斷。",
    },
}


def require_packages():
    missing = []
    for package in ("torch", "transformers", "sklearn", "matplotlib", "tqdm"):
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        raise SystemExit(
            "Missing packages: "
            + ", ".join(missing)
            + "\nInstall them with:\n"
            + "pip install torch transformers scikit-learn matplotlib tqdm"
        )


def normalize_label(task, value):
    if value is None:
        return "N/A"
    text = str(value).strip()
    aliases = {
        "NA": "N/A",
        "N\\A": "N/A",
        "None": "N/A",
        "yes": "Yes",
        "no": "No",
        "clear": "Clear",
        "not_clear": "Not Clear",
        "not clear": "Not Clear",
        "NotClear": "Not Clear",
        "misleading": "Misleading",
        "more_than_5_years": "longer_than_5_years",
        "more than 5 years": "longer_than_5_years",
    }
    text = aliases.get(text, text)
    return text if text in LABELS[task] else "N/A"


def load_data(path, input_enrichment="none"):
    with path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)
    data = []
    for raw_item in raw_data:
        item = {k: v for k, v in raw_item.items() if k not in DROP_COLS}
        enrichment = make_input_enrichment(raw_item, input_enrichment)
        if enrichment:
            item["data"] = enrichment + "\n原文：" + str(item.get("data", ""))
        data.append(item)
    for item in data:
        item["data"] = str(item.get("data", ""))
        for task in TASKS:
            item[task] = normalize_label(task, item.get(task, "N/A"))
    return data


def split_train_test(data, seed=42, train_ratio=0.8):
    random.seed(seed)
    by_label = {"Yes": [], "No": []}
    for item in data:
        by_label[item["promise_status"]].append(item)

    train_set, test_set = [], []
    for items in by_label.values():
        shuffled = items[:]
        random.shuffle(shuffled)
        cut = int(len(shuffled) * train_ratio)
        train_set.extend(shuffled[:cut])
        test_set.extend(shuffled[cut:])

    random.shuffle(train_set)
    random.shuffle(test_set)
    return train_set, test_set


def encode_labels(items):
    label_maps = {task: {label: i for i, label in enumerate(LABELS[task])} for task in TASKS}
    encoded = []
    for item in items:
        encoded.append([label_maps[task][item[task]] for task in TASKS])
    return encoded, label_maps


def build_label_texts():
    label_texts = []
    for task in TASKS:
        task_zh = {
            "promise_status": "ESG 承諾判斷",
            "verification_timeline": "承諾驗證時程",
            "evidence_status": "證據狀態",
            "evidence_quality": "證據品質",
        }[task]
        for label in LABELS[task]:
            label_texts.append(f"{task_zh}：{label}。{LABEL_DESCRIPTIONS[task][label]}")
    return label_texts


def clarity_feature_text(item, use_evidence_string=False):
    evidence = str(item.get("evidence_string", "") or "") if use_evidence_string else ""
    return evidence if evidence else str(item.get("data", "") or "")


def extract_clarity_features(item, use_evidence_string=False):
    text = clarity_feature_text(item, use_evidence_string=use_evidence_string)
    promise = str(item.get("promise_string", "") or "")
    data = str(item.get("data", "") or "")
    numeric_pattern = r"\d+(?:\.\d+)?\s*(?:%|％|元|萬元|億元|噸|公噸|人|件|家|次|天|週|月|年|度|kWh|tCO2e|CO2e)?"
    vague_terms = [
        "致力",
        "持續",
        "推動",
        "強化",
        "提升",
        "促進",
        "落實",
        "逐步",
        "規劃",
        "原則",
        "方向",
        "努力",
    ]
    numbers = re.findall(numeric_pattern, text)
    years = re.findall(r"(?:19|20)\d{2}", text)
    text_len = len(text)
    data_len = len(data)
    promise_len = len(promise)
    vague_count = sum(text.count(term) for term in vague_terms)
    return [
        float(bool(numbers)),
        min(text_len / 200.0, 2.0),
        min(text_len / (promise_len + 1.0), 5.0),
        min(len(numbers) / 5.0, 2.0),
        min(len(years) / 3.0, 2.0),
        min(vague_count / 5.0, 2.0),
        min(data_len / 500.0, 2.0),
        float(text_len < 80),
        float(text_len < 120),
        float((not numbers) and text_len < 120),
    ]


def compute_class_weights(train_set, device):
    import torch

    weights = []
    for task in TASKS:
        counts = torch.zeros(len(LABELS[task]), dtype=torch.float32)
        for item in train_set:
            counts[LABELS[task].index(item[task])] += 1
        counts = counts.clamp_min(1.0)
        task_weights = counts.sum() / (len(counts) * counts)
        task_weights = task_weights / task_weights.mean()
        weights.append(task_weights.to(device))
    return weights


def build_stat_features(train_set, test_set, task, max_features, seed):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler

    texts_train = [item["data"] for item in train_set]
    texts_test = [item["data"] for item in test_set]
    labels_train = [LABELS[task].index(item[task]) for item in train_set]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_features=max_features * 4,
    )
    x_train = vectorizer.fit_transform(texts_train)
    x_test = vectorizer.transform(texts_test)

    labels_train = np.array(labels_train)
    doc_presence = (x_train > 0).astype(np.float32)
    num_features = x_train.shape[1]
    delta_scores_by_class = []
    eps = 1.0

    for class_id in sorted(set(labels_train.tolist())):
        pos_mask = labels_train == class_id
        neg_mask = ~pos_mask
        pos_df = np.asarray(doc_presence[pos_mask].sum(axis=0)).ravel()
        neg_df = np.asarray(doc_presence[neg_mask].sum(axis=0)).ravel()
        delta = np.log((pos_df + eps) / (neg_df + eps))
        delta_scores_by_class.append(delta)

    delta_scores_by_class = np.stack(delta_scores_by_class, axis=0)
    global_delta = np.max(np.abs(delta_scores_by_class), axis=0)
    k = min(max_features, num_features)
    selected_idx = np.argsort(-global_delta)[:k]
    selected_idx = np.sort(selected_idx)

    selected_delta = global_delta[selected_idx]
    x_train = x_train[:, selected_idx].toarray() * selected_delta
    x_test = x_test[:, selected_idx].toarray() * selected_delta

    scaler = StandardScaler(with_mean=False)
    x_train = scaler.fit_transform(x_train).astype("float32")
    x_test = scaler.transform(x_test).astype("float32")

    feature_names = vectorizer.get_feature_names_out()[selected_idx]
    use_evidence_string = all(str(item.get("evidence_string", "") or "") for item in train_set + test_set)
    clarity_train = np.asarray(
        [extract_clarity_features(item, use_evidence_string=use_evidence_string) for item in train_set],
        dtype="float32",
    )
    clarity_test = np.asarray(
        [extract_clarity_features(item, use_evidence_string=use_evidence_string) for item in test_set],
        dtype="float32",
    )
    clarity_names = [
        "clarity_has_number",
        "clarity_text_len_norm",
        "clarity_text_promise_len_ratio",
        "clarity_number_count_norm",
        "clarity_year_count_norm",
        "clarity_vague_count_norm",
        "clarity_data_len_norm",
        "clarity_text_len_lt80",
        "clarity_text_len_lt120",
        "clarity_no_number_and_short",
    ]
    clarity_scaler = StandardScaler()
    clarity_train = clarity_scaler.fit_transform(clarity_train).astype("float32")
    clarity_test = clarity_scaler.transform(clarity_test).astype("float32")
    x_train = np.concatenate([x_train, clarity_train], axis=1)
    x_test = np.concatenate([x_test, clarity_test], axis=1)
    return x_train, x_test, feature_names.tolist() + clarity_names


class ESGDataset:
    def __init__(self, items, tokenizer, label_maps, max_length, stat_features=None):
        self.items = items
        self.tokenizer = tokenizer
        self.label_maps = label_maps
        self.max_length = max_length
        self.stat_features = stat_features

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        import torch

        item = self.items[idx]
        encoded = self.tokenizer(
            item["data"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        batch = {k: v.squeeze(0) for k, v in encoded.items()}
        batch["labels"] = torch.tensor(
            [self.label_maps[task][item[task]] for task in TASKS],
            dtype=torch.long,
        )
        if self.stat_features is not None:
            batch["stat_features"] = torch.tensor(self.stat_features[idx], dtype=torch.float32)
        return batch


class MultiTaskBertClassifier:
    def __init__(self, model_name, num_labels, stat_dim=0, dropout=0.1, pooling="cls"):
        import torch
        from torch import nn
        from transformers import AutoModel
        disable_hf_auto_conversion()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                try:
                    self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=True).float()
                except Exception:
                    self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=False).float()
                hidden = self.encoder.config.hidden_size
                self.pooling = pooling
                self.attention_pool = nn.Linear(hidden, 1) if pooling == "attention" else None
                self.dropout = nn.Dropout(dropout)
                self.stat_norm = nn.LayerNorm(stat_dim) if stat_dim else None
                in_dim = hidden + stat_dim
                self.heads = nn.ModuleList([nn.Linear(in_dim, n) for n in num_labels])

            def pool(self, last_hidden_state, attention_mask):
                if self.pooling == "mean":
                    mask = attention_mask.unsqueeze(-1).float()
                    summed = (last_hidden_state * mask).sum(dim=1)
                    denom = mask.sum(dim=1).clamp(min=1.0)
                    return (summed / denom).float()
                if self.pooling == "attention":
                    scores = self.attention_pool(last_hidden_state).squeeze(-1)
                    scores = scores.masked_fill(attention_mask == 0, -1e4)
                    weights = torch.softmax(scores, dim=1).unsqueeze(-1)
                    return (last_hidden_state * weights).sum(dim=1).float()
                return last_hidden_state[:, 0].float()

            def forward(self, input_ids, attention_mask, token_type_ids=None, stat_features=None, return_pooled=False):
                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if token_type_ids is not None:
                    kwargs["token_type_ids"] = token_type_ids
                outputs = self.encoder(**kwargs)
                pooled = self.pool(outputs.last_hidden_state, attention_mask)
                if stat_features is not None:
                    stat_features = self.stat_norm(stat_features) if self.stat_norm else stat_features
                    pooled = torch.cat([pooled, stat_features], dim=-1)
                pooled = self.dropout(pooled)
                logits = [head(pooled) for head in self.heads]
                if return_pooled:
                    return logits, pooled
                return logits

        self.model = Model()


class SiameseESGDataset:
    def __init__(self, items, tokenizer, label_maps, max_length):
        self.items = items
        self.tokenizer = tokenizer
        self.label_maps = label_maps
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        import torch

        item = self.items[idx]
        text = self.tokenizer(
            item["data"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        labels = torch.tensor([self.label_maps[task][item[task]] for task in TASKS], dtype=torch.long)
        return {
            "text": {k: v.squeeze(0) for k, v in text.items()},
            "labels": labels,
        }


class SiameseClassifier:
    def __init__(self, model_name, dropout=0.1):
        import torch
        from torch import nn
        from transformers import AutoModel
        disable_hf_auto_conversion()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                try:
                    self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=True).float()
                except Exception:
                    self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=False).float()
                hidden = self.encoder.config.hidden_size
                self.dropout = nn.Dropout(dropout)
                self.scorer = nn.Sequential(
                    nn.Linear(hidden * 3, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 1),
                )

            def encode(self, input_ids, attention_mask, token_type_ids=None):
                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if token_type_ids is not None:
                    kwargs["token_type_ids"] = token_type_ids
                return self.encoder(**kwargs).last_hidden_state[:, 0].float()

            def forward(self, text_batch, label_batch, task_label_counts):
                u = self.encode(**text_batch)
                v = self.encode(**label_batch)
                total_labels = sum(task_label_counts)
                repeated_u = u.repeat_interleave(total_labels, dim=0)
                repeated_v = v.repeat(u.size(0), 1)
                fused = torch.cat([repeated_u, repeated_v, torch.abs(repeated_u - repeated_v)], dim=-1)
                scores = self.scorer(self.dropout(fused)).view(u.size(0), sum(task_label_counts))
                chunks = torch.split(scores, task_label_counts, dim=1)
                return list(chunks)

        self.model = Model()


def collate_siamese(batch, tokenizer, max_length, label_texts):
    import torch

    text = {}
    for key in batch[0]["text"]:
        text[key] = torch.stack([item["text"][key] for item in batch])
    label_batch = tokenizer(
        label_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    labels = torch.stack([item["labels"] for item in batch])
    return {"text": text, "label_batch": label_batch, "labels": labels}


def batch_contrastive_loss(embeddings, labels, margin):
    import torch

    if embeddings.size(0) < 2:
        return embeddings.new_tensor(0.0)
    left = []
    right = []
    targets = []
    for i in range(embeddings.size(0)):
        for j in range(i + 1, embeddings.size(0)):
            left.append(embeddings[i])
            right.append(embeddings[j])
            targets.append(1.0 if labels[i].item() == labels[j].item() else -1.0)
    if not left:
        return embeddings.new_tensor(0.0)
    return torch.nn.functional.cosine_embedding_loss(
        torch.stack(left),
        torch.stack(right),
        torch.tensor(targets, device=embeddings.device),
        margin=margin,
    )


def classification_loss(logits, labels, weight=None, loss_type="ce", focal_gamma=2.0, label_smoothing=0.0):
    import torch

    if loss_type == "focal":
        ce = torch.nn.functional.cross_entropy(
            logits,
            labels,
            weight=weight,
            reduction="none",
            label_smoothing=label_smoothing,
        )
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** focal_gamma * ce).mean()
    return torch.nn.functional.cross_entropy(
        logits,
        labels,
        weight=weight,
        label_smoothing=label_smoothing,
    )


def symmetric_kl_loss(logits_a, logits_b):
    import torch

    log_probs_a = torch.nn.functional.log_softmax(logits_a, dim=-1)
    log_probs_b = torch.nn.functional.log_softmax(logits_b, dim=-1)
    probs_a = log_probs_a.exp()
    probs_b = log_probs_b.exp()
    return 0.5 * (
        torch.nn.functional.kl_div(log_probs_a, probs_b, reduction="batchmean")
        + torch.nn.functional.kl_div(log_probs_b, probs_a, reduction="batchmean")
    )


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    siamese=False,
    class_weights=None,
    train_task_index=None,
    contrastive_alpha=0.0,
    contrastive_task_index=0,
    contrastive_margin=0.5,
    loss_type="ce",
    focal_gamma=2.0,
    label_smoothing=0.0,
    rdrop_alpha=0.0,
):
    import torch

    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        if siamese:
            text = {k: v.to(device) for k, v in batch["text"].items()}
            label_batch = {k: v.to(device) for k, v in batch["label_batch"].items()}
            labels = batch["labels"].to(device)
            logits = model(text, label_batch, [len(LABELS[t]) for t in TASKS])
        else:
            inputs = {k: v.to(device) for k, v in batch.items() if k not in {"labels", "stat_features"}}
            labels = batch["labels"].to(device)
            stat_features = batch.get("stat_features")
            if stat_features is not None:
                stat_features = stat_features.to(device)
            use_pooled = contrastive_alpha > 0
            if use_pooled:
                logits, pooled = model(**inputs, stat_features=stat_features, return_pooled=True)
            else:
                logits = model(**inputs, stat_features=stat_features)
                pooled = None
            logits_rdrop = None
            if rdrop_alpha > 0:
                logits_rdrop = model(**inputs, stat_features=stat_features)

        task_indexes = range(len(logits)) if train_task_index is None else [train_task_index]
        losses = [
            classification_loss(
                logits[i],
                labels[:, i],
                weight=None if class_weights is None else class_weights[i],
                loss_type=loss_type,
                focal_gamma=focal_gamma,
                label_smoothing=label_smoothing,
            )
            for i in task_indexes
        ]
        loss = sum(losses) / len(losses)
        if logits_rdrop is not None:
            rdrop_losses = [
                classification_loss(
                    logits_rdrop[i],
                    labels[:, i],
                    weight=None if class_weights is None else class_weights[i],
                    loss_type=loss_type,
                    focal_gamma=focal_gamma,
                    label_smoothing=label_smoothing,
                )
                for i in task_indexes
            ]
            ce_pair = 0.5 * (loss + sum(rdrop_losses) / len(rdrop_losses))
            kl_pair = sum(symmetric_kl_loss(logits[i], logits_rdrop[i]) for i in task_indexes) / len(task_indexes)
            loss = ce_pair + rdrop_alpha * kl_pair
        if not siamese and contrastive_alpha > 0 and pooled is not None:
            loss = loss + contrastive_alpha * batch_contrastive_loss(
                pooled,
                labels[:, contrastive_task_index],
                contrastive_margin,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def predict(model, loader, device, siamese=False, return_probs=False):
    import torch

    model.eval()
    gold = {task: [] for task in TASKS}
    pred = {task: [] for task in TASKS}
    probs = {task: [] for task in TASKS}
    with torch.no_grad():
        for batch in loader:
            if siamese:
                text = {k: v.to(device) for k, v in batch["text"].items()}
                label_batch = {k: v.to(device) for k, v in batch["label_batch"].items()}
                labels = batch["labels"].to(device)
                logits = model(text, label_batch, [len(LABELS[t]) for t in TASKS])
            else:
                inputs = {k: v.to(device) for k, v in batch.items() if k not in {"labels", "stat_features"}}
                labels = batch["labels"].to(device)
                stat_features = batch.get("stat_features")
                if stat_features is not None:
                    stat_features = stat_features.to(device)
                logits = model(**inputs, stat_features=stat_features)

            for i, task in enumerate(TASKS):
                task_probs = torch.softmax(logits[i], dim=1).cpu().tolist()
                pred_ids = logits[i].argmax(dim=1).cpu().tolist()
                gold_ids = labels[:, i].cpu().tolist()
                pred[task].extend([LABELS[task][idx] for idx in pred_ids])
                gold[task].extend([LABELS[task][idx] for idx in gold_ids])
                if return_probs:
                    probs[task].extend(
                        [
                            {label: float(row[label_idx]) for label_idx, label in enumerate(LABELS[task])}
                            for row in task_probs
                        ]
                    )
    if return_probs:
        return gold, pred, probs
    return gold, pred


def evaluate(gold, pred):
    from sklearn.metrics import f1_score

    scores = {}
    weighted = 0.0
    for task in TASKS:
        score = f1_score(gold[task], pred[task], labels=LABELS[task], average="macro", zero_division=0)
        scores[task] = score
        weighted += score * WEIGHTS[task]
    scores["mean_macro_f1"] = sum(scores[task] for task in TASKS) / len(TASKS)
    scores["weighted_macro_f1"] = weighted
    return scores


def save_confusion_matrices(gold, pred, output_dir, mode):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    paths = {}
    for task in TASKS:
        cm = confusion_matrix(gold[task], pred[task], labels=LABELS[task])
        fig, ax = plt.subplots(figsize=(7, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS[task])
        disp.plot(ax=ax, values_format="d", colorbar=False)
        ax.set_title(f"{mode} - {task}")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        path = output_dir / f"{mode}_{task}_confusion_matrix.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths[task] = path
    return paths


def save_training_curve(history, output_dir, mode):
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history["loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["loss"], "b-o", linewidth=2, markersize=6)
    axes[0].set_title(f"{mode} Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(epochs)

    axes[1].plot(epochs, history["weighted_macro_f1"], "g-o", linewidth=2, markersize=6)
    axes[1].set_title(f"{mode} Validation Weighted Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Weighted Macro-F1")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(epochs)
    if history["weighted_macro_f1"]:
        best_epoch = history["weighted_macro_f1"].index(max(history["weighted_macro_f1"])) + 1
        axes[1].axvline(best_epoch, color="red", linestyle="--", alpha=0.5, label=f"best={best_epoch}")
        axes[1].legend()

    fig.tight_layout()
    path = output_dir / f"{mode}_training_curve.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_outputs(output_dir, rows, predictions, confusion_paths, training_curves, feature_names, args):
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "week9_scores.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["mode", "n_test", *TASKS, "mean_macro_f1", "weighted_macro_f1"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "week9_predictions.json").open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    with (output_dir / "week9_report.md").open("w", encoding="utf-8") as f:
        f.write("# Week 9 Lab Report: Advanced Sentence Representation\n\n")
        f.write("## Baseline Model\n\n")
        f.write(f"- Model: `{args.model_name}`\n")
        f.write("- Reason: it is a pretrained BERT/RoBERTa-family encoder suitable for Chinese semantic representation and standard fine-tuning.\n")
        f.write("- Method: use the `[CLS]` dense vector and four task-specific classification heads.\n\n")
        f.write("## Feature Fusion\n\n")
        f.write("- Statistical vector: Delta-TFIDF character n-gram features.\n")
        f.write("- Fusion: concatenate BERT `[CLS]` dense vector with normalized statistical features before the final classification heads.\n")
        f.write("- This tests whether class-discriminative keyword signals help fix errors that pure dense embeddings miss.\n\n")
        f.write("## Siamese / Dual Encoder\n\n")
        f.write("- Encode the paragraph as `u` and each Chinese candidate label description as `v`.\n")
        f.write("- Score each label with the fused representation `(u, v, |u - v|)`.\n")
        f.write("- This turns classification into matching text against label semantics. Class-weighted loss is used to reduce majority-label bias.\n\n")
        f.write("## Results\n\n")
        f.write("| Mode | promise_status | verification_timeline | evidence_status | evidence_quality | Mean Macro-F1 | Weighted Macro-F1 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['mode']} | {row['promise_status']:.4f} | {row['verification_timeline']:.4f} | "
                f"{row['evidence_status']:.4f} | {row['evidence_quality']:.4f} | "
                f"{row['mean_macro_f1']:.4f} | {row['weighted_macro_f1']:.4f} |\n"
            )
        f.write("\n## Confusion Matrices\n\n")
        for mode, paths in confusion_paths.items():
            for task, path in paths.items():
                f.write(f"- `{mode}` `{task}`: `{path.name}`\n")
        if training_curves:
            f.write("\n## Training Curves\n\n")
            for mode, path in training_curves.items():
                f.write(f"- `{mode}`: `{path.name}`\n")
        if feature_names:
            f.write("\n## Delta-TFIDF Features\n\n")
            f.write(", ".join(f"`{name}`" for name in feature_names[:80]) + "\n")


def run_mode(args, mode, train_set, test_set, tokenizer, label_maps, device):
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import get_linear_schedule_with_warmup

    stat_train = stat_test = feature_names = None
    if mode == "fusion":
        stat_train, stat_test, feature_names = build_stat_features(
            train_set,
            test_set,
            task=args.fusion_task,
            max_features=args.stat_features,
            seed=args.seed,
        )

    if mode == "siamese":
        train_ds = SiameseESGDataset(train_set, tokenizer, label_maps, args.max_length)
        test_ds = SiameseESGDataset(test_set, tokenizer, label_maps, args.max_length)
        label_texts = build_label_texts()
        collate = lambda batch: collate_siamese(batch, tokenizer, args.max_length, label_texts)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        model = SiameseClassifier(args.model_name, dropout=args.dropout).model.to(device)
        siamese = True
    else:
        train_ds = ESGDataset(train_set, tokenizer, label_maps, args.max_length, stat_train)
        test_ds = ESGDataset(test_set, tokenizer, label_maps, args.max_length, stat_test)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        model = MultiTaskBertClassifier(
            args.model_name,
            num_labels=[len(LABELS[task]) for task in TASKS],
            stat_dim=0 if stat_train is None else stat_train.shape[1],
            dropout=args.dropout,
            pooling=args.pooling,
        ).model.to(device)
        siamese = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    class_weights = compute_class_weights(train_set, device)
    best_score = float("-inf")
    best_state = None
    best_epoch = None
    history = {"loss": [], "weighted_macro_f1": []}
    for epoch in tqdm(range(1, args.epochs + 1), desc=f"training {mode}"):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            siamese=siamese,
            class_weights=class_weights,
            train_task_index=None if args.train_task == "all" else TASKS.index(args.train_task),
            contrastive_alpha=args.contrastive_alpha if not siamese else 0.0,
            contrastive_task_index=TASKS.index(args.contrastive_task),
            contrastive_margin=args.contrastive_margin,
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            rdrop_alpha=args.rdrop_alpha if not siamese else 0.0,
        )
        gold_epoch, pred_epoch = predict(model, test_loader, device, siamese=siamese)
        scores_epoch = evaluate(gold_epoch, pred_epoch)
        history["loss"].append(loss)
        history["weighted_macro_f1"].append(scores_epoch["weighted_macro_f1"])
        print(
            f"{mode} epoch {epoch}/{args.epochs} "
            f"loss={loss:.4f} weighted={scores_epoch['weighted_macro_f1']:.4f}"
        )
        selection_score = (
            scores_epoch["weighted_macro_f1"]
            if args.train_task == "all"
            else scores_epoch[args.train_task]
        )
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    if args.save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "metadata": {
                "model_name": args.model_name,
                "mode": mode,
                "tasks": TASKS,
                "labels": LABELS,
                "label_maps": label_maps,
                "max_length": args.max_length,
                "dropout": args.dropout,
                "pooling": args.pooling,
                "stat_dim": 0 if stat_train is None else int(stat_train.shape[1]),
                "stat_features": feature_names or [],
                "train_task": args.train_task,
                "contrastive_task": args.contrastive_task,
                "contrastive_alpha": args.contrastive_alpha,
                "loss_type": args.loss_type,
                "focal_gamma": args.focal_gamma,
                "label_smoothing": args.label_smoothing,
                "rdrop_alpha": args.rdrop_alpha,
                "best_epoch": best_epoch,
                "best_selection_score": best_score,
                "input_enrichment": args.input_enrichment,
                "fusion_task": args.fusion_task,
                "final_tasks": [task.strip() for task in args.final_tasks.split(",") if task.strip()],
                "model_role": args.model_role,
            },
        }
        final_task_suffix = "_".join(checkpoint["metadata"]["final_tasks"])
        role_suffix = checkpoint["metadata"]["model_role"]
        filename = f"{mode}_{role_suffix}"
        if final_task_suffix:
            filename += f"__tasks_{final_task_suffix}"
        torch.save(checkpoint, args.output_dir / f"{filename}_model.pt")

    if args.save_probs:
        gold, pred, probs = predict(model, test_loader, device, siamese=siamese, return_probs=True)
    else:
        gold, pred = predict(model, test_loader, device, siamese=siamese)
        probs = None
    scores = evaluate(gold, pred)
    row = {"mode": mode, "n_test": len(test_set), **scores}
    predictions = []
    for idx, item in enumerate(test_set):
        predictions.append(
            {
                "mode": mode,
                "id": item.get("id"),
                "gold": {task: gold[task][idx] for task in TASKS},
                "pred": {task: pred[task][idx] for task in TASKS},
                **({"probs": {task: probs[task][idx] for task in TASKS}} if probs is not None else {}),
            }
        )
    return row, predictions, gold, pred, feature_names, history


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Week 9: BERT fine-tuning, feature fusion, and siamese architecture")
    parser.add_argument("--data-path", type=Path, default=root / "vpesg4k_train_1000_V1.json")
    parser.add_argument("--eval-path", type=Path, default=None, help="Optional labeled validation/test data evaluated without splitting train data.")
    parser.add_argument("--augment-train-path", type=Path, default=None, help="Optional extra labeled data added to train only.")
    parser.add_argument("--augment-train-limit", type=int, default=0, help="Limit extra train-only records; 0 uses all.")
    parser.add_argument(
        "--input-enrichment",
        choices=["none", "feature_tags", "esg_primer", "both"],
        default="none",
        help="Prepend ESG task knowledge and/or extracted linguistic tags to model inputs.",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "output" / f"week9_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--model-name", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--modes", nargs="+", choices=["baseline", "fusion", "siamese"], default=["baseline", "fusion", "siamese"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", choices=["cls", "mean", "attention"], default="cls")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="Use 0 for all data, or a small number for smoke tests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None, help="Override data split seed while keeping --seed for model randomness.")
    parser.add_argument("--stat-features", type=int, default=32)
    parser.add_argument("--fusion-task", choices=TASKS, default="promise_status")
    parser.add_argument("--save-probs", action="store_true", help="Save per-class probabilities for threshold search.")
    parser.add_argument("--save-model", action="store_true", help="Save the best epoch model checkpoint for each mode.")
    parser.add_argument("--final-tasks", default="", help="Comma-separated final tasks this checkpoint is intended to serve.")
    parser.add_argument("--model-role", default="source", help="Human-readable role saved in checkpoint metadata.")
    parser.add_argument("--train-task", choices=["all", *TASKS], default="all", help="Optimize loss/checkpoint for one task or all tasks.")
    parser.add_argument("--contrastive-alpha", type=float, default=0.0, help="Add supervised cosine contrastive loss for non-siamese modes.")
    parser.add_argument("--contrastive-task", choices=TASKS, default="promise_status")
    parser.add_argument("--contrastive-margin", type=float, default=0.5)
    parser.add_argument("--loss-type", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--rdrop-alpha", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    require_packages()

    import torch
    from transformers import AutoTokenizer
    disable_hf_auto_conversion()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = load_data(args.data_path, input_enrichment=args.input_enrichment)
    if args.limit:
        data = data[: args.limit]
    split_seed = args.seed if args.split_seed is None else args.split_seed
    if args.eval_path:
        train_set = data
        test_set = load_data(args.eval_path, input_enrichment=args.input_enrichment)
    else:
        train_set, test_set = split_train_test(data, seed=split_seed)
    if args.augment_train_path:
        augment_train = load_data(args.augment_train_path, input_enrichment=args.input_enrichment)
        if args.augment_train_limit:
            augment_train = augment_train[: args.augment_train_limit]
        train_set.extend(augment_train)
        random.shuffle(train_set)
    _, label_maps = encode_labels(train_set)
    print(f"Loaded {len(data)} samples: train={len(train_set)}, test={len(test_set)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_predictions = []
    confusion_paths = {}
    training_curves = {}
    all_feature_names = []

    for mode in args.modes:
        print(f"\nRunning mode: {mode}")
        row, predictions, gold, pred, feature_names, history = run_mode(
            args,
            mode,
            train_set,
            test_set,
            tokenizer,
            label_maps,
            device,
        )
        rows.append(row)
        all_predictions.extend(predictions)
        confusion_paths[mode] = save_confusion_matrices(gold, pred, args.output_dir, mode)
        training_curves[mode] = save_training_curve(history, args.output_dir, mode)
        if feature_names:
            all_feature_names = feature_names
        print(
            f"{mode}: mean={row['mean_macro_f1']:.4f}, "
            f"weighted={row['weighted_macro_f1']:.4f}"
        )

    write_outputs(
        args.output_dir,
        rows,
        all_predictions,
        confusion_paths,
        training_curves,
        all_feature_names,
        args,
    )
    print(f"\nSaved Week 9 outputs to: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted.")
