#!/usr/bin/env python3
"""Build a balanced three-domain military-aircraft classification dataset.

Domains:
  aerial  - remote-sensing images from amistele/MTARSI-fixed
  natural - natural-view images from Illia56/Military-Aircraft-Detection
  recon   - disjoint natural-view samples rendered as reconnaissance imagery
"""

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image, ImageFilter, ImageOps


CLASSES = ("B-52", "C-130", "C-17", "F-15", "F-16")
DOMAIN_NAMES = ("aerial", "natural", "recon")
MTARSI_FOLDERS = {
    "B-52": "B-52",
    "C-130": "C-130",
    "C-17": "C-17",
    "F-15": "F-15",
    "F-16": "F-16",
}
NATURAL_FOLDERS = {
    "B-52": "B52",
    "C-130": "C130",
    "C-17": "C17",
    "F-15": "F15",
    "F-16": "F16",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="/root/autodl-tmp/datasets/MilitaryAircraft3D",
    )
    parser.add_argument(
        "--download-root",
        default="/root/autodl-tmp/datasets/MilitaryAircraft3D_downloads",
    )
    parser.add_argument("--samples-per-class", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def standardized(image, size):
    image = image.convert("RGB")
    return ImageOps.pad(
        image,
        (size, size),
        method=Image.Resampling.LANCZOS,
        color="white",
        centering=(0.5, 0.5),
    )


def difference_hash(image, hash_size=16):
    gray = image.convert("L").resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    bits = []
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        bits.extend(
            pixels[offset + column] > pixels[offset + column + 1]
            for column in range(hash_size)
        )
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def is_duplicate(candidate_hash, prior_hashes, threshold=2):
    return any(bin(candidate_hash ^ prior).count("1") <= threshold
               for prior in prior_hashes)


def save_candidate(image, domain, class_name, output_root, size, seen_hashes,
                   counters, provenance):
    image = standardized(image, size)
    candidate_hash = difference_hash(image)
    if is_duplicate(candidate_hash, seen_hashes[class_name]):
        return False
    seen_hashes[class_name].append(candidate_hash)
    index = counters[(domain, class_name)]
    destination = output_root / domain / class_name / f"{index:04d}.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="JPEG", quality=95, subsampling=0)
    counters[(domain, class_name)] += 1
    provenance.append({
        "domain": domain,
        "class": class_name,
        "path": destination.relative_to(output_root).as_posix(),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    })
    domain_total = sum(
        counters[(domain, item)] for item in CLASSES)
    if domain_total % 25 == 0:
        print(
            f"BUILD_PROGRESS domain={domain} images={domain_total}",
            flush=True,
        )
    return True


def domain_complete(counters, domain, required):
    return all(counters[(domain, class_name)] >= required
               for class_name in CLASSES)


def download_class_folders(repo_id, patterns, destination):
    return Path(snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=patterns,
        cache_dir=str(destination),
        resume_download=True,
        max_workers=16,
    ))


def valid_image_paths(root):
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def reconnaissance_style(image):
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    return ImageOps.colorize(image, black="#071018", white="#d9edf2")


def build_folder_domain(args, domain, repo_id, folders, download_root,
                        output_root, seen_hashes, counters, provenance,
                        partition=None, style=None):
    natural_repository = repo_id == "Illia56/Military-Aircraft-Detection"
    patterns = []
    for folder in folders.values():
        prefix = "crop/" if natural_repository else ""
        patterns.append(f"{prefix}{folder}/**")
    local_repo = download_class_folders(
        repo_id, patterns,
        download_root / ("natural" if natural_repository else domain))
    rng = random.Random(args.seed)
    for class_name, folder in folders.items():
        source_dir = local_repo / ("crop" if natural_repository else "") / folder
        candidates = valid_image_paths(source_dir)
        rng.shuffle(candidates)
        if partition is not None:
            candidates = candidates[partition::2]
        for source in candidates:
            if counters[(domain, class_name)] >= args.samples_per_class:
                break
            try:
                with Image.open(source) as image:
                    if style is not None:
                        image = style(image.convert("RGB"))
                    save_candidate(
                        image, domain, class_name, output_root,
                        args.image_size, seen_hashes, counters, provenance)
            except (OSError, ValueError):
                continue


def validate_counts(counters, required):
    missing = {
        f"{domain}/{class_name}": counters[(domain, class_name)]
        for domain in DOMAIN_NAMES
        for class_name in CLASSES
        if counters[(domain, class_name)] < required
    }
    if missing:
        raise RuntimeError(
            f"Insufficient unique samples; required={required}, counts={missing}")


def write_manifests(args, output_root, counters, provenance):
    counts = {
        domain: {
            class_name: counters[(domain, class_name)]
            for class_name in CLASSES
        }
        for domain in DOMAIN_NAMES
    }
    manifest = {
        "name": "MilitaryAircraft3D",
        "domains": list(DOMAIN_NAMES),
        "classes": list(CLASSES),
        "samples_per_domain_class": args.samples_per_class,
        "image_size": args.image_size,
        "seed": args.seed,
        "counts": counts,
        "sources": {
            "aerial": "https://huggingface.co/datasets/amistele/MTARSI-fixed",
            "natural": "https://huggingface.co/datasets/Illia56/Military-Aircraft-Detection",
            "recon": (
                "disjoint odd-index samples from "
                "Illia56/Military-Aircraft-Detection with deterministic "
                "grayscale reconnaissance rendering"
            ),
        },
        "records": provenance,
    }
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("DATASET_READY=true")
    print(f"DATASET_ROOT={output_root}")
    print(f"TOTAL_IMAGES={len(provenance)}")
    for domain in DOMAIN_NAMES:
        print(f"DOMAIN_COUNT {domain}={sum(counts[domain].values())}")


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    download_root = Path(args.download_root).resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_root}; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    download_root.mkdir(parents=True, exist_ok=True)

    seen_hashes = defaultdict(list)
    counters = Counter()
    provenance = []
    build_folder_domain(
        args, "aerial", "amistele/MTARSI-fixed", MTARSI_FOLDERS,
        download_root, output_root, seen_hashes, counters, provenance)
    build_folder_domain(
        args, "natural", "Illia56/Military-Aircraft-Detection",
        NATURAL_FOLDERS, download_root, output_root, seen_hashes, counters,
        provenance, partition=0)
    build_folder_domain(
        args, "recon", "Illia56/Military-Aircraft-Detection",
        NATURAL_FOLDERS, download_root, output_root, seen_hashes, counters,
        provenance, partition=1, style=reconnaissance_style)
    validate_counts(counters, args.samples_per_class)
    write_manifests(args, output_root, counters, provenance)


if __name__ == "__main__":
    main()
