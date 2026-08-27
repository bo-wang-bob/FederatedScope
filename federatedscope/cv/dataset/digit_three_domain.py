"""Portable three-domain digit dataset used by the formal accuracy test.

The dataset itself is prepared by ``scripts/prepare_digit_three_domain.py``.
At runtime this module only reads the portable manifest and image files.  This
keeps the client partition identical on Windows and Linux and prevents a
method from silently receiving a different Dirichlet draw.
"""

import json
import os

from PIL import Image
from torch.utils.data import Dataset


DEFAULT_DOMAINS = ("emnist_digits", "usps", "svhn")


def load_digit_three_domain_manifest(manifest_path, selected_domains=None):
    """Load and validate a global three-domain digit manifest."""
    if not manifest_path:
        raise ValueError("digit3_global_manifest_path must not be empty")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Three-domain digit manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    domains = list(manifest.get("domains", []))
    if len(domains) < 3:
        raise ValueError(
            "Three-domain digit manifest must contain at least 3 domains")
    classes = list(manifest.get("classes", []))
    if classes != [str(idx) for idx in range(10)]:
        raise ValueError(
            "Three-domain digit manifest classes must be strings '0'..'9'")

    requested = list(selected_domains or domains)
    missing = [domain for domain in requested if domain not in domains]
    if missing:
        raise ValueError(
            f"Requested digit domains are absent from manifest: {missing}")

    records = manifest.get("records", {})
    for domain in requested:
        domain_records = records.get(domain)
        if not isinstance(domain_records, dict):
            raise ValueError(
                f"Manifest records for domain {domain!r} must be a mapping")
        for split in ("train", "test"):
            split_records = domain_records.get(split)
            if not isinstance(split_records, list) or not split_records:
                raise ValueError(
                    f"Manifest domain {domain!r} has no {split!r} records")
            for record in split_records:
                label = int(record.get("label", -1))
                if label < 0 or label > 9 or not record.get("path"):
                    raise ValueError(
                        f"Invalid record in {domain}/{split}: {record}")
    return manifest, requested, classes


class DigitThreeDomain(Dataset):
    """Image dataset backed by one domain/split of the global manifest."""

    DOMAINS = list(DEFAULT_DOMAINS)
    CLASSES = [str(idx) for idx in range(10)]

    def __init__(self,
                 root,
                 domain,
                 split="train",
                 transform=None,
                 train_ratio=None,
                 val_ratio=None,
                 seed=None,
                 manifest_path="",
                 records=None):
        del train_ratio, val_ratio, seed
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unsupported digit split: {split}")
        self.root = str(root)
        self.domain = str(domain)
        self.split = str(split)
        self.transform = transform

        if records is None:
            manifest, domains, _ = load_digit_three_domain_manifest(
                manifest_path, [self.domain])
            if self.domain not in domains:
                raise ValueError(f"Unknown digit domain: {self.domain}")
            records = manifest["records"][self.domain].get(self.split, [])

        self.records = list(records or [])
        self.data = [
            record["path"] if os.path.isabs(record["path"]) else
            os.path.join(self.root, record["path"])
            for record in self.records
        ]
        self.targets = [int(record["label"]) for record in self.records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image = Image.open(self.data[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]
