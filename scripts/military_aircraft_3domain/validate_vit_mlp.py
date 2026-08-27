#!/usr/bin/env python3
"""Central sanity check for the frozen CLIP ViT-B/16 + MLP classifier."""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import open_clip
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


DOMAINS = ("aerial", "natural", "recon")
CLASSES = ("B-52", "C-130", "C-17", "F-15", "F-16")
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/root/autodl-tmp/datasets/MilitaryAircraft3D",
    )
    parser.add_argument(
        "--model-path",
        default="/root/.cache/clip/ViT-B-16.pt",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-accuracy", type=float, default=0.80)
    parser.add_argument(
        "--output",
        default="exp/military_aircraft_3domain/central_sanity.json",
    )
    return parser.parse_args()


class ImagePathDataset(Dataset):
    def __init__(self, records, preprocess):
        self.records = records
        self.preprocess = preprocess

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label, domain = self.records[index]
        with Image.open(path) as image:
            tensor = self.preprocess(image.convert("RGB"))
        return tensor, label, domain


def collect_records(root, seed):
    train_records = []
    test_records = []
    rng = random.Random(seed)
    for domain in DOMAINS:
        for label, class_name in enumerate(CLASSES):
            files = sorted(
                path for path in (root / domain / class_name).rglob("*")
                if path.is_file() and path.suffix.lower() in SUFFIXES
            )
            if len(files) < 10:
                raise RuntimeError(f"Too few images: {domain}/{class_name}={len(files)}")
            rng.shuffle(files)
            split = max(1, int(len(files) * 0.7))
            train_records.extend((path, label, domain) for path in files[:split])
            test_records.extend((path, label, domain) for path in files[split:])
    rng.shuffle(train_records)
    rng.shuffle(test_records)
    return train_records, test_records


def extract_features(model, loader, device):
    features = []
    labels = []
    domains = []
    model.eval()
    with torch.no_grad():
        for images, target, domain in loader:
            images = images.to(device)
            with torch.autocast(
                    device_type="cuda", enabled=device.type == "cuda"):
                batch_features = model.encode_image(images).float()
            features.append(batch_features.cpu())
            labels.append(target.long())
            domains.extend(domain)
    return torch.cat(features), torch.cat(labels), domains


def evaluate(classifier, features, labels, domains, device):
    classifier.eval()
    with torch.no_grad():
        predictions = classifier(features.to(device)).argmax(dim=1).cpu()
    correct = predictions.eq(labels)
    per_domain = {}
    for domain in DOMAINS:
        indices = [index for index, value in enumerate(domains) if value == domain]
        per_domain[domain] = float(correct[indices].float().mean())
    return float(correct.float().mean()), per_domain


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=args.model_path)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    train_records, test_records = collect_records(
        Path(args.data_root), args.seed)
    train_loader = DataLoader(
        ImagePathDataset(train_records, preprocess), batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(
        ImagePathDataset(test_records, preprocess), batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True)
    train_features, train_labels, _ = extract_features(model, train_loader, device)
    test_features, test_labels, test_domains = extract_features(
        model, test_loader, device)
    del model
    torch.cuda.empty_cache()

    classifier = nn.Sequential(
        nn.Linear(train_features.shape[1], 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, len(CLASSES)),
    ).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    curve = []
    for epoch in range(1, args.epochs + 1):
        classifier.train()
        for features, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(
                classifier(features.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        accuracy, per_domain = evaluate(
            classifier, test_features, test_labels, test_domains, device)
        curve.append({
            "epoch": epoch,
            "accuracy": accuracy,
            "per_domain": per_domain,
        })
        print(
            f"EPOCH={epoch} ACCURACY={accuracy:.6f} "
            + " ".join(
                f"{domain.upper()}={per_domain[domain]:.6f}"
                for domain in DOMAINS))

    final_accuracy, per_domain = evaluate(
        classifier, test_features, test_labels, test_domains, device)
    result = {
        "model": "OpenCLIP ViT-B/16 + MLP(512-256-5)",
        "train_samples": len(train_records),
        "test_samples": len(test_records),
        "epochs": args.epochs,
        "lr": args.lr,
        "final_accuracy": final_accuracy,
        "per_domain_accuracy": per_domain,
        "minimum_accuracy": args.minimum_accuracy,
        "passed": final_accuracy >= args.minimum_accuracy,
        "elapsed_seconds": time.time() - started,
        "curve": curve,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"VIT_MLP_VALID={str(result['passed']).lower()}")
    print(f"FINAL_ACCURACY={final_accuracy:.6f}")
    print(f"RESULT_FILE={output}")
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
