#!/usr/bin/env python3
"""Train domain-personalized heads from DomainNet training feature caches."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


DOMAINS = ("clipart", "painting", "real", "sketch")
GROUPS = {
    "domainnet_vit": (512, "clip_ViT_B_16_openai_d512"),
    "domainnet_cnn": (1024, "cnn_convnext_base_d1024"),
    "domainnet_mixer": (768, "timm_mixer_b16_224_d768"),
}


def _one(cache_dir, pattern):
    matches = [
        path for path in cache_dir.glob(pattern)
        if ".part" not in path.name and "_terminal_client" not in path.name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern!r}, found {matches}")
    return matches[0]


def _normalize(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(
        np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _build_model(feature_dim, class_count, hidden_dim):
    if hidden_dim <= 0:
        return nn.Linear(feature_dim, class_count)
    return nn.Sequential(
        nn.Linear(feature_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, class_count),
    )


def _train_one(features, labels, feature_dim, class_count, hidden_dim,
               epochs, batch_size, learning_rate, seed, device):
    torch.manual_seed(seed)
    model = _build_model(feature_dim, class_count, hidden_dim).to(device)
    counts = np.bincount(labels, minlength=class_count).astype(np.float32)
    class_weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    class_weights /= class_weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(class_weights).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs))
    rng = np.random.default_rng(seed)
    losses = []
    model.train()
    for _ in range(epochs):
        order = rng.permutation(labels.size)
        total_loss = 0.0
        total = 0
        for offset in range(0, labels.size, batch_size):
            index = order[offset:offset + batch_size]
            batch_features = torch.from_numpy(features[index]).to(device)
            batch_labels = torch.from_numpy(labels[index]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * int(index.size)
            total += int(index.size)
        scheduler.step()
        losses.append(total_loss / max(1, total))
    return model, losses


def _evaluate(model, features, labels, batch_size, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for offset in range(0, labels.size, batch_size):
            batch = torch.from_numpy(
                features[offset:offset + batch_size]).to(device)
            prediction = torch.argmax(model(batch), dim=1).cpu().numpy()
            correct += int(np.sum(
                prediction == labels[offset:offset + batch_size]))
    return correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--group", choices=tuple(GROUPS), required=True)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--statistics-only", action="store_true")
    parser.add_argument("--apply-lds", action="store_true")
    parser.add_argument("--lds-alpha", type=float, default=0.1)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    feature_dim, suffix = GROUPS[args.group]
    class_count = 345
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    totals = {"correct": 0, "total": 0, "domains": {}}
    train_samples = 0
    lds_matrix = None
    if args.apply_lds:
        np.random.seed(42)
        lds_matrix = np.random.dirichlet(
            [args.lds_alpha] * len(DOMAINS), class_count).T
    for domain_index, domain in enumerate(DOMAINS):
        train_path = _one(cache_dir, f"domainnet_{domain}_{suffix}.npz")
        test_path = _one(cache_dir, f"domainnet_{domain}_test_*.npz")
        with np.load(train_path, allow_pickle=False) as data:
            train_features = _normalize(data["features"])
            train_labels = np.asarray(data["labels"], dtype=np.int64)
        if lds_matrix is not None:
            np.random.seed(42 + domain_index)
            selected = []
            for class_index in np.unique(train_labels):
                indices = np.flatnonzero(train_labels == class_index)
                np.random.shuffle(indices)
                size = int(
                    lds_matrix[domain_index, class_index] * indices.size)
                selected.extend(indices[:size].tolist())
            selected = np.asarray(selected, dtype=np.int64)
            train_features = train_features[selected]
            train_labels = train_labels[selected]
        if args.statistics_only:
            rng = np.random.default_rng(42 + domain_index)
            synthetic_parts = []
            synthetic_labels = []
            for class_index in range(class_count):
                class_features = train_features[
                    train_labels == class_index]
                mean = class_features.mean(axis=0)
                variance = class_features.var(axis=0)
                noise = rng.standard_normal(
                    class_features.shape, dtype=np.float32)
                generated = mean[None, :] + noise * np.sqrt(
                    np.maximum(variance, 1e-8))[None, :]
                synthetic_parts.append(_normalize(generated))
                synthetic_labels.append(np.full(
                    class_features.shape[0], class_index, np.int64))
            train_features = np.concatenate(synthetic_parts, axis=0)
            train_labels = np.concatenate(synthetic_labels, axis=0)
        with np.load(test_path, allow_pickle=False) as data:
            test_features = _normalize(data["features"])
            test_labels = np.asarray(data["labels"], dtype=np.int64)
        train_samples += int(train_labels.size)
        model, losses = _train_one(
            train_features, train_labels, feature_dim, class_count,
            args.hidden_dim, args.epochs, args.batch_size,
            args.learning_rate, 42 + domain_index, device)
        correct = _evaluate(
            model, test_features, test_labels, args.batch_size, device)
        total = int(test_labels.size)
        totals["correct"] += correct
        totals["total"] += total
        totals["domains"][domain] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total,
            "final_train_loss": losses[-1],
        }
        del model, train_features, train_labels, test_features, test_labels
        if device.type == "cuda":
            torch.cuda.empty_cache()
    totals["weighted_accuracy"] = totals["correct"] / totals["total"]
    totals["equal_domain_accuracy"] = float(np.mean([
        item["accuracy"] for item in totals["domains"].values()
    ]))
    result = {
        "status": "pass",
        "group": args.group,
        "cache_dir": str(cache_dir),
        "train_samples": train_samples,
        "feature_dim": feature_dim,
        "class_count": class_count,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "statistics_only": args.statistics_only,
        "apply_lds": args.apply_lds,
        "lds_alpha": args.lds_alpha,
        "device": str(device),
        "metrics": totals,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
