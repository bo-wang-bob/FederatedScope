"""Compare frozen BERT pooling choices on held-out MDSent features."""

import argparse

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from federatedscope.core.configs.config import global_cfg
from federatedscope.contrib.data.mdsent_data import build_mdsent_test_sets


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--batch-size", type=int, default=128)
args = parser.parse_args()

cfg = global_cfg.clone()
cfg.merge_from_file(args.config)
datasets = build_mdsent_test_sets(cfg)
tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
model = AutoModel.from_pretrained(args.model, local_files_only=True).cuda().eval()

pooled = {"cls": [], "mean": []}
all_labels = []
with torch.no_grad():
    for dataset in datasets.values():
        for start in range(0, len(dataset), args.batch_size):
            stop = min(start + args.batch_size, len(dataset))
            encoded = tokenizer(
                dataset.texts[start:stop], padding=True, truncation=True,
                max_length=128, return_tensors="pt")
            encoded = {key: value.cuda() for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled["cls"].append(hidden[:, 0].cpu().numpy())
            pooled["mean"].append(
                ((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy())
            all_labels.extend(dataset.targets[start:stop])

y = np.asarray(all_labels)
indices = np.arange(len(y))
train_idx, test_idx = train_test_split(
    indices, test_size=.5, random_state=123, stratify=y)
for name, parts in pooled.items():
    x = np.concatenate(parts)
    scaler = StandardScaler().fit(x[train_idx])
    xa = scaler.transform(x[train_idx])
    xb = scaler.transform(x[test_idx])
    for c in (.0001, .001, .01, .1):
        classifier = LogisticRegression(
            C=c, max_iter=1000, n_jobs=-1).fit(xa, y[train_idx])
        accuracy = accuracy_score(y[test_idx], classifier.predict(xb))
        print(f"POOLING={name} C={c} ACCURACY={accuracy:.6f}")
