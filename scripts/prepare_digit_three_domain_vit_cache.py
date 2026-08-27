#!/usr/bin/env python3
"""Extract one shared frozen ViT feature cache for the three digit domains."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

class RecordDataset(Dataset):
    def __init__(self, root, records, transform):
        self.root = Path(root)
        self.records = list(records)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(self.root / record["path"]).convert("RGB")
        return self.transform(image), int(record["label"]), record["path"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",
                        default="data/digit_three_domain")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--cache-dir",
                        default="exp/distributed_feature_cache/digit3_vit")
    parser.add_argument(
        "--model",
        default="vit_tiny_patch16_224.augreg_in21k_ft_in1k")
    parser.add_argument("--backend", choices=("timm", "clip"),
                        default="timm")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else \
        data_root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    domains = list(manifest["domains"])
    if len(domains) < 3:
        raise ValueError("Expected at least three digit domains")

    requested_device = str(args.device).lower()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    if args.backend == "clip":
        import open_clip

        pretrained_source = args.checkpoint_path or args.clip_pretrained
        extractor, _, transform = open_clip.create_model_and_transforms(
            args.model, pretrained=pretrained_source)
        extractor = extractor.to(device).eval()
        embedding_dim = getattr(extractor.visual, "output_dim", None)
        if embedding_dim is None:
            embedding_dim = int(extractor.text_projection.shape[1])
        embedding_dim = int(embedding_dim)
        cache_prefix = "clip"
        cache_model = (
            f"{args.model}_{args.clip_pretrained}"
            .replace("/", "_").replace("-", "_"))
    else:
        from federatedscope.contrib.model.ggeur_timm_extractor import (
            TimmFeatureExtractor)

        extractor = TimmFeatureExtractor(
            model_name=args.model,
            pretrained=not args.no_pretrained,
            freeze=True,
            checkpoint_path=args.checkpoint_path,
            in_chans=3,
            global_pool="avg").to(device)
        extractor.eval()
        embedding_dim = extractor.get_feature_dim()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        cache_prefix = "timm"
        cache_model = str(args.model).replace("/", "_").replace("-", "_")
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": "digits-3domain",
        "domains": domains,
        "backend": args.backend,
        "model": args.model,
        "pretrained": (args.clip_pretrained
                       if args.backend == "clip" else not args.no_pretrained),
        "embedding_dim": embedding_dim,
        "device": str(device),
        "caches": {},
    }

    for domain in domains:
        records = (list(manifest["records"][domain]["train"]) +
                   list(manifest["records"][domain]["test"]))
        loader = DataLoader(RecordDataset(data_root, records, transform),
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers)
        features = []
        labels = []
        paths = []
        use_fp16 = bool(args.fp16 and device.type == "cuda")
        with torch.no_grad(), torch.amp.autocast(
                device_type=device.type, enabled=use_fp16):
            for images, batch_labels, batch_paths in loader:
                images = images.to(device)
                if args.backend == "clip":
                    batch_features = extractor.encode_image(images)
                else:
                    batch_features = extractor(images)
                batch_features = batch_features.float().cpu().numpy()
                features.append(batch_features.astype(np.float32))
                labels.extend(int(value) for value in batch_labels)
                paths.extend(str(value).replace("\\", "/").casefold()
                             for value in batch_paths)
        feature_array = np.concatenate(features, axis=0)
        cache_path = cache_dir / (
            f"digits-3domain_{domain}_{cache_prefix}_{cache_model}_"
            f"d{embedding_dim}.npz")
        np.savez(cache_path,
                 paths=np.asarray(paths),
                 features=feature_array,
                 labels=np.asarray(labels, dtype=np.int64))
        summary["caches"][domain] = {
            "path": cache_path.as_posix(),
            "samples": len(paths),
            "feature_shape": list(feature_array.shape),
        }
        print(f"{domain}: {feature_array.shape} -> {cache_path}", flush=True)

    (cache_dir / "cache_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
