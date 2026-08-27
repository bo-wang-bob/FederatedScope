#!/usr/bin/env python3
"""Evaluate deterministic prototype classifiers on cached DomainNet features."""

import argparse
import json
from pathlib import Path

import numpy as np


DOMAINS = ("clipart", "painting", "real", "sketch")


def _one(cache_dir, pattern):
    matches = [
        path for path in cache_dir.glob(pattern)
        if ".part" not in path.name and "_terminal_client" not in path.name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern!r}, found {matches}")
    return matches[0]


def _normalize(values):
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)


def _parse_values(raw):
    if not raw:
        return []
    return [float(item) for item in raw.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--group", choices=("domainnet_vit", "domainnet_cnn",
                             "domainnet_mixer"), default="domainnet_cnn")
    parser.add_argument("--skip-qda", action="store_true")
    parser.add_argument("--prototype-powers", default="")
    parser.add_argument("--prior-powers", default="")
    parser.add_argument("--temperatures", default="")
    parser.add_argument("--domain-selection-temperatures", default="")
    parser.add_argument("--domain-selection-strengths", default="")
    parser.add_argument("--apply-legacy-lds", action="store_true")
    parser.add_argument("--lds-alpha", type=float, default=0.1)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    class_count = 345
    group_spec = {
        "domainnet_vit": (512, "clip_ViT_B_16_openai_d512"),
        "domainnet_cnn": (1024, "cnn_convnext_base_d1024"),
        "domainnet_mixer": (768, "timm_mixer_b16_224_d768"),
    }
    feature_dim, train_suffix = group_spec[args.group]
    sums = np.zeros((class_count, feature_dim), dtype=np.float64)
    squared_sums = np.zeros((class_count, feature_dim), dtype=np.float64)
    counts = np.zeros(class_count, dtype=np.int64)
    domain_sums = {}
    domain_squared_sums = {}
    domain_counts = {}

    train_files = {}
    test_files = {}
    lds_matrix = None
    if args.apply_legacy_lds:
        np.random.seed(42)
        lds_matrix = np.random.dirichlet(
            [args.lds_alpha] * len(DOMAINS), class_count).T
    for domain_index, domain in enumerate(DOMAINS):
        train_files[domain] = _one(
            cache_dir, f"domainnet_{domain}_{train_suffix}.npz")
        test_files[domain] = _one(cache_dir, f"domainnet_{domain}_test_*.npz")
        with np.load(train_files[domain], allow_pickle=False) as data:
            features = np.asarray(data["features"], dtype=np.float32)
            labels = np.asarray(data["labels"], dtype=np.int64)
        if lds_matrix is not None:
            np.random.seed(42 + domain_index)
            selected = []
            for class_index in np.unique(labels):
                indices = np.flatnonzero(labels == class_index)
                np.random.shuffle(indices)
                size = int(
                    lds_matrix[domain_index, class_index] * indices.size)
                selected.extend(indices[:size].tolist())
            selected = np.asarray(selected, dtype=np.int64)
            features = features[selected]
            labels = labels[selected]
        np.add.at(sums, labels, features)
        np.add.at(squared_sums, labels, features.astype(np.float64) ** 2)
        np.add.at(counts, labels, 1)
        current_sums = np.zeros_like(sums)
        current_squared_sums = np.zeros_like(squared_sums)
        current_counts = np.zeros_like(counts)
        np.add.at(current_sums, labels, features)
        np.add.at(current_squared_sums, labels,
                  features.astype(np.float64) ** 2)
        np.add.at(current_counts, labels, 1)
        domain_sums[domain] = current_sums
        domain_squared_sums[domain] = current_squared_sums
        domain_counts[domain] = current_counts

    if np.any(counts == 0):
        raise RuntimeError(
            f"missing train classes: {np.flatnonzero(counts == 0).tolist()}")
    means = (sums / counts[:, None]).astype(np.float32)
    cosine_weights = _normalize(means)
    euclidean_bias = -np.sum(means * means, axis=1)

    within_ss = squared_sums - (sums * sums / counts[:, None])
    pooled_variance = (
        within_ss.sum(axis=0) / max(1, int(counts.sum() - class_count)))
    pooled_variance = np.maximum(pooled_variance, 1e-8)
    lda_heads = {}
    for shrinkage in (0.0, 0.01, 0.1, 0.5, 1.0):
        variance = ((1.0 - shrinkage) * pooled_variance +
                    shrinkage * float(pooled_variance.mean()))
        weights = means / variance[None, :]
        bias = -0.5 * np.sum(means * weights, axis=1)
        lda_heads[f"lda_shrinkage_{shrinkage:g}"] = (weights, bias)

    domain_means = []
    domain_variances = []
    for domain in DOMAINS:
        current_counts = domain_counts[domain]
        current_means = domain_sums[domain] / current_counts[:, None]
        current_variances = (
            domain_squared_sums[domain] / current_counts[:, None] -
            current_means * current_means)
        domain_means.append(current_means)
        domain_variances.append(current_variances)
    balanced_means = np.mean(domain_means, axis=0).astype(np.float32)
    balanced_variance = np.maximum(
        np.mean(domain_variances, axis=(0, 1)), 1e-8)
    balanced_weights = balanced_means / balanced_variance[None, :]
    balanced_bias = -0.5 * np.sum(
        balanced_means * balanced_weights, axis=1)
    lda_heads["lda_domain_balanced"] = (
        balanced_weights, balanced_bias)

    class_variance = np.maximum(
        squared_sums / counts[:, None] - means.astype(np.float64) ** 2,
        1e-6)
    balanced_class_variance = np.maximum(
        np.mean(domain_variances, axis=0), 1e-6)
    qda_heads = {
        "qda_diagonal": (means.astype(np.float64), class_variance),
        "qda_domain_balanced": (
            balanced_means.astype(np.float64), balanced_class_variance),
    }
    domain_mean_stack = np.stack(domain_means, axis=0).astype(np.float32)
    domain_cosine_stack = _normalize(
        domain_mean_stack.reshape(-1, feature_dim)).reshape(
            len(DOMAINS), class_count, feature_dim)
    domain_global_means = _normalize(np.stack([
        domain_sums[domain].sum(axis=0) /
        max(1, int(domain_counts[domain].sum()))
        for domain in DOMAINS
    ], axis=0).astype(np.float32))
    domain_lda_weights = []
    domain_lda_biases = []
    for domain_index in range(len(DOMAINS)):
        means_for_domain = domain_mean_stack[domain_index].astype(np.float64)
        variance_for_domain = np.maximum(
            domain_variances[domain_index].astype(np.float64), 1e-6)
        weights_for_domain = means_for_domain / variance_for_domain
        bias_for_domain = -0.5 * np.sum(
            means_for_domain * weights_for_domain, axis=1)
        domain_lda_weights.append(weights_for_domain)
        domain_lda_biases.append(bias_for_domain)

    power_heads = {}
    prototype_powers = _parse_values(args.prototype_powers)
    prior_powers = _parse_values(args.prior_powers)
    temperatures = _parse_values(args.temperatures)
    if prototype_powers and prior_powers and temperatures:
        prior = np.stack([
            domain_counts[domain] / max(1, int(domain_counts[domain].sum()))
            for domain in DOMAINS
        ], axis=0)
        for prototype_power in prototype_powers:
            # Shrinking high-count prototypes can compensate for classes whose
            # broad training distribution otherwise dominates cosine scores.
            normalized_stack = _normalize(
                domain_mean_stack.reshape(-1, feature_dim)).reshape(
                    len(DOMAINS), class_count, feature_dim)
            normalized_stack = normalized_stack / np.stack([
                np.power(np.maximum(domain_counts[domain], 1),
                         prototype_power)
                for domain in DOMAINS
            ], axis=0)[:, :, None]
            for prior_power in prior_powers:
                prior_bias = prior_power * np.log(np.maximum(prior, 1e-12))
                for temperature in temperatures:
                    name = (f"domain_proto_t{temperature:g}_prior{prior_power:g}"
                            f"_power{prototype_power:g}")
                    power_heads[name] = (
                        normalized_stack, prior_bias, temperature)
    domain_selection_heads = {}
    for temperature in _parse_values(args.domain_selection_temperatures):
        for strength in _parse_values(args.domain_selection_strengths):
            name = f"domain_selection_t{temperature:g}_s{strength:g}"
            domain_selection_heads[name] = (temperature, strength)

    totals = {
        "cosine": {"correct": 0, "total": 0, "domains": {}},
        "euclidean": {"correct": 0, "total": 0, "domains": {}},
    }
    totals.update({
        name: {"correct": 0, "total": 0, "domains": {}}
        for name in lda_heads
    })
    for name in ("domain_prototype_max_cosine",
                 "domain_prototype_matched_cosine",
                 "domain_lda_matched",
                 "domain_prototype_mean_cosine"):
        totals[name] = {"correct": 0, "total": 0, "domains": {}}
    totals.update({
        name: {"correct": 0, "total": 0, "domains": {}}
        for name in power_heads
    })
    totals.update({
        name: {"correct": 0, "total": 0, "domains": {}}
        for name in domain_selection_heads
    })
    if not args.skip_qda:
        totals.update({
            name: {"correct": 0, "total": 0, "domains": {}}
            for name in qda_heads
        })
    for domain_index, domain in enumerate(DOMAINS):
        with np.load(test_files[domain], allow_pickle=False) as data:
            features = np.asarray(data["features"], dtype=np.float32)
            labels = np.asarray(data["labels"], dtype=np.int64)
        logits_cosine = features @ cosine_weights.T
        logits_euclidean = 2.0 * (features @ means.T) + euclidean_bias
        logits_by_method = {
            "cosine": logits_cosine,
            "euclidean": logits_euclidean,
        }
        domain_logits = np.stack([
            features @ domain_cosine_stack[index].T
            for index in range(len(DOMAINS))
        ], axis=0)
        logits_by_method["domain_prototype_max_cosine"] = np.max(
            domain_logits, axis=0)
        logits_by_method["domain_prototype_matched_cosine"] = (
            domain_logits[domain_index])
        logits_by_method["domain_lda_matched"] = (
            features.astype(np.float64) @
            domain_lda_weights[domain_index].T +
            domain_lda_biases[domain_index])
        logits_by_method["domain_prototype_mean_cosine"] = np.mean(
            domain_logits, axis=0)
        domain_scores = features @ domain_global_means.T
        for name, (temperature, strength) in domain_selection_heads.items():
            stabilized = (domain_scores / temperature)
            stabilized -= stabilized.max(axis=1, keepdims=True)
            domain_weights = np.exp(stabilized)
            domain_weights /= domain_weights.sum(axis=1, keepdims=True)
            weighted_logits = np.sum(
                domain_weights.T[:, :, None] * domain_logits, axis=0)
            max_logits = np.max(domain_logits, axis=0)
            logits_by_method[name] = (
                strength * weighted_logits + (1.0 - strength) * max_logits)
        for name, (stack, prior_bias, temperature) in power_heads.items():
            powered_logits = np.stack([
                features @ stack[index].T / temperature + prior_bias[index]
                for index in range(len(DOMAINS))
            ], axis=0)
            logits_by_method[name] = np.max(powered_logits, axis=0)
        logits_by_method.update({
            name: features @ weights.T + bias
            for name, (weights, bias) in lda_heads.items()
        })
        feature64 = features.astype(np.float64)
        if not args.skip_qda:
            for name, (head_means, head_variance) in qda_heads.items():
                weights = head_means / head_variance
                bias = -0.5 * (
                    np.sum(head_means * weights, axis=1) +
                    np.sum(np.log(head_variance), axis=1))
                # Expand the quadratic term without materializing
                # (samples, classes, dimensions).
                logits_by_method[name] = (
                    feature64 @ weights.T + bias -
                    0.5 * ((feature64 * feature64) @
                           (1.0 / head_variance).T))
        for method, logits in logits_by_method.items():
            correct = int(np.sum(np.argmax(logits, axis=1) == labels))
            total = int(labels.size)
            totals[method]["correct"] += correct
            totals[method]["total"] += total
            totals[method]["domains"][domain] = {
                "correct": correct,
                "total": total,
                "accuracy": correct / total,
            }

    result = {
        "status": "pass",
        "group": args.group,
        "apply_legacy_lds": args.apply_legacy_lds,
        "cache_dir": str(cache_dir),
        "train_samples": int(counts.sum()),
        "class_count": class_count,
        "feature_dim": feature_dim,
        "methods": totals,
    }
    for method in totals.values():
        method["weighted_accuracy"] = method["correct"] / method["total"]
        method["equal_domain_accuracy"] = float(np.mean([
            value["accuracy"] for value in method["domains"].values()
        ]))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
