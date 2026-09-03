"""Measure the frozen nlptown sentiment head on the held-out MDSent split."""

import argparse

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
model = AutoModelForSequenceClassification.from_pretrained(
    args.model, local_files_only=True).cuda().eval()

# The pretrained head predicts 1--5 stars. MDSent keeps 1, 2, 4 and 5 stars.
# Resolve a predicted 3-star review by comparing its probability mass toward
# the lower pair (1/2) and upper pair (4/5).
mapping = torch.tensor([0, 1, -1, 2, 3], device="cuda")
total = correct = 0
domain_results = {}
with torch.no_grad():
    for domain, dataset in datasets.items():
        domain_total = domain_correct = 0
        for start in range(0, len(dataset), args.batch_size):
            stop = min(start + args.batch_size, len(dataset))
            texts = dataset.texts[start:stop]
            labels = torch.as_tensor(
                dataset.targets[start:stop], device="cuda", dtype=torch.long)
            encoded = tokenizer(
                texts, padding=True, truncation=True, max_length=128,
                return_tensors="pt")
            encoded = {key: value.cuda() for key, value in encoded.items()}
            logits = model(**encoded).logits
            stars = logits.argmax(dim=1)
            prediction = mapping[stars]
            neutral = stars == 2
            if neutral.any():
                lower = torch.logsumexp(logits[:, :2], dim=1)
                upper = torch.logsumexp(logits[:, 3:], dim=1)
                prediction[neutral] = torch.where(
                    lower[neutral] >= upper[neutral], 1, 2)
            hits = int((prediction == labels).sum().item())
            domain_correct += hits
            domain_total += int(labels.numel())
        domain_results[domain] = domain_correct / domain_total
        correct += domain_correct
        total += domain_total

print("DOMAIN_ACCURACY=" + repr(domain_results))
print(f"AVERAGE_ACCURACY={correct / total:.6f} CORRECT={correct} TOTAL={total}")
