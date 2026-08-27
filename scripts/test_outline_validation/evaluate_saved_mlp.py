#!/usr/bin/env python3
"""Reload saved heads and independently evaluate frozen-backbone test features."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from federatedscope.contrib.model.ggeur_text_rnn import \
    GGEURTextRNNClassifier


def load_pretrained_backbone(backbone, device):
    """Load the exact frozen encoder named by the training checkpoint."""
    extractor = str(backbone["feature_extractor"]).lower()
    if extractor == "bert":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            backbone["model"], local_files_only=True).to(device).eval()
        return model
    if extractor == "cnn":
        from federatedscope.contrib.model.ggeur_cnn_extractor import \
            CNNFeatureExtractor
        model = CNNFeatureExtractor(
            model_name=backbone["model"], pretrained=False, freeze=True,
            checkpoint_path=backbone.get("checkpoint", ""))
        return model.to(device).eval()
    if extractor == "timm":
        from federatedscope.contrib.model.ggeur_timm_extractor import \
            TimmFeatureExtractor
        model = TimmFeatureExtractor(
            model_name=backbone["model"], pretrained=False, freeze=True,
            checkpoint_path=backbone.get("checkpoint", ""))
        return model.to(device).eval()
    import open_clip
    checkpoint = str(backbone.get("checkpoint", "") or "")
    pretrained = (
        checkpoint if checkpoint and Path(checkpoint).is_file()
        else backbone.get("pretrained", "openai"))
    model, _, _ = open_clip.create_model_and_transforms(
        backbone["model"], pretrained=pretrained)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model.to(device).eval()


def build_model(architecture):
    model_type = str(architecture["model_type"]).lower()
    if model_type in {"ggeur_rnn", "ggeur_lstm"}:
        return GGEURTextRNNClassifier(
            input_dim=int(architecture["input_dim"]),
            hidden_dim=int(architecture["hidden_dim"]),
            num_classes=int(architecture["num_classes"]),
            num_layers=int(architecture["num_layers"]),
            dropout=float(architecture["dropout"]),
            rnn_type="rnn" if model_type == "ggeur_rnn" else "lstm")
    input_dim = int(architecture["input_dim"])
    hidden_dim = int(architecture["hidden_dim"])
    num_classes = int(architecture["num_classes"])
    if hidden_dim > 0:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(float(architecture["dropout"])),
            nn.Linear(hidden_dim, num_classes))
    return nn.Linear(input_dim, num_classes)


def evaluate_one(case_id, method, seed, device, loaded_backbones):
    run_dir = (
        REPO / "exp" / "test_outline_validation" / case_id /
        "seed_{}".format(seed) / method)
    checkpoint_path = run_dir / "checkpoints" / "mlp_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    feature_path = checkpoint_path.parent / checkpoint["test_feature_bundle"]
    bundle = torch.load(feature_path, map_location="cpu")
    if checkpoint["backbone"] != bundle["backbone"]:
        raise RuntimeError(
            "checkpoint and pretrained feature bundle use different backbones")
    backbone_key = json.dumps(
        checkpoint["backbone"], ensure_ascii=False, sort_keys=True)
    if backbone_key not in loaded_backbones:
        loaded_backbones[backbone_key] = load_pretrained_backbone(
            checkpoint["backbone"], device)
    model = build_model(checkpoint["architecture"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(device).eval()
    domains = {}
    with torch.no_grad():
        for domain, features in bundle["features"].items():
            labels = bundle["labels"][domain].long().to(device)
            logits = model(features.float().to(device))
            accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
            domains[str(domain)] = accuracy
    average = sum(domains.values()) / len(domains)
    result = {
        "case": case_id,
        "method": method,
        "seed": seed,
        "checkpoint": checkpoint_path.relative_to(REPO).as_posix(),
        "pretrained_backbone": checkpoint["backbone"],
        "test_feature_bundle": feature_path.relative_to(REPO).as_posix(),
        "domain_accuracy": domains,
        "average_accuracy": average,
    }
    output = run_dir / "independent_test.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(
        "INDEPENDENT_TEST_COMPLETE case={} method={} seed={} accuracy={:.4f}".
        format(case_id, method, seed, average))
    print("PRETRAINED_BACKBONE_ATTACHED={}".format(
        checkpoint["backbone"]["model"]))
    print("INDEPENDENT_TEST_RESULT={}".format(
        output.relative_to(REPO).as_posix()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", required=True,
        choices=("T-02", "T-03", "T-04", "T-05", "T-06"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--methods", nargs="+", default=["fedavg", "fedprox", "platform"],
        choices=("fedavg", "fedprox", "platform"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.20)
    args = parser.parse_args()
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu")
    loaded_backbones = {}
    results = [
        evaluate_one(args.case, method, seed, device, loaded_backbones)
        for seed in args.seeds for method in args.methods
    ]
    by_seed = {}
    for result in results:
        by_seed.setdefault(result["seed"], {})[result["method"]] = result
    comparisons = []
    for seed in args.seeds:
        methods = by_seed[seed]
        if set(methods) != {"fedavg", "fedprox", "platform"}:
            raise RuntimeError("all three methods are required for comparison")
        platform = methods["platform"]["average_accuracy"]
        gain_avg = platform - methods["fedavg"]["average_accuracy"]
        gain_prox = platform - methods["fedprox"]["average_accuracy"]
        passed = gain_avg >= args.threshold and gain_prox >= args.threshold
        comparisons.append({
            "seed": seed,
            "platform": platform,
            "fedavg": methods["fedavg"]["average_accuracy"],
            "fedprox": methods["fedprox"]["average_accuracy"],
            "platform_minus_fedavg": gain_avg,
            "platform_minus_fedprox": gain_prox,
            "passed": passed,
        })
        print(
            "SEED_COMPARISON seed={} platform={:.4f} fedavg={:.4f} "
            "fedprox={:.4f} gains={:.4f}/{:.4f} pass={}".format(
                seed, platform, methods["fedavg"]["average_accuracy"],
                methods["fedprox"]["average_accuracy"], gain_avg,
                gain_prox, passed))
    summary = {
        "case": args.case,
        "threshold": args.threshold,
        "comparisons": comparisons,
        "all_seeds_pass": all(item["passed"] for item in comparisons),
    }
    output = (
        REPO / "exp" / "test_outline_validation" / args.case /
        "independent_test_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print("THREE_SEED_INDEPENDENT_TEST=PASS" if summary["all_seeds_pass"]
          else "THREE_SEED_INDEPENDENT_TEST=FAIL")
    print("INDEPENDENT_TEST_SUMMARY={}".format(
        output.relative_to(REPO).as_posix()))
    if not summary["all_seeds_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
