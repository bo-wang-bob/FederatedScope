#!/usr/bin/env python3
"""Populate the held-out MDSent feature keys missing from formal caches."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from federatedscope.contrib.data.mdsent_data import (  # noqa: E402
    build_mdsent_test_sets,
)


DOMAINS = ("books", "dvd", "electronics", "kitchen")
CACHE_BASENAME = (
    "mdsent_{domain}_bert_"
    "nlptown_bert_base_multilingual_uncased_senti_"
    "maxlen128_cls_pre_d768.npz")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO_ROOT))
    parser.add_argument(
        "--model-id",
        default="nlptown/bert-base-multilingual-uncased-sentiment")
    parser.add_argument(
        "--model-dir",
        default="pretrained_models/nlptown_bert_base_multilingual_uncased_senti")
    parser.add_argument(
        "--hf-endpoint", default=os.environ.get(
            "HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--groups", nargs="+", default=("mdsent_rnn", "mdsent_lstm"))
    return parser.parse_args()


def normalize_key(value):
    value = str(value).replace("\\", "/").strip()
    while "//" in value:
        value = value.replace("//", "/")
    return value.lstrip("./").casefold()


def load_cache(path):
    data = np.load(path, allow_pickle=True)
    paths = [normalize_key(value) for value in data["paths"]]
    features = np.asarray(data["features"], dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 768:
        raise RuntimeError(f"invalid feature shape in {path}: {features.shape}")
    return paths, features


def save_cache(path, paths, features):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f".pid{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle,
                 paths=np.asarray(paths),
                 features=np.asarray(features, dtype=np.float32))
    os.replace(temporary, path)


def make_config(repo):
    return SimpleNamespace(
        data=SimpleNamespace(
            root=str(repo / "data" / "sentiment"),
            splits=[0.8, 0.0, 0.2],
            args=[{
                "include_unlabeled": True,
                "balance_test": True,
                "seed": 42,
                "use_cache": True,
            }],
        ))


@torch.no_grad()
def encode_missing(tokenizer, model, samples, batch_size):
    output = {}
    for start in range(0, len(samples), max(1, batch_size)):
        batch = samples[start:start + max(1, batch_size)]
        encoded = tokenizer(
            [text for _, text in batch],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt")
        hidden = model(**encoded).last_hidden_state[:, 0, :]
        for (key, _), feature in zip(batch, hidden.cpu().numpy()):
            output[key] = np.asarray(feature, dtype=np.float32)
        print(json.dumps({
            "event": "encode_progress",
            "done": min(start + len(batch), len(samples)),
            "total": len(samples),
        }), flush=True)
    return output


def main():
    args = parse_args()
    repo = Path(args.repo).resolve()
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = repo / model_dir

    tool_packages = repo / "exp" / "cache_tools" / "python_packages"
    tool_packages.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(tool_packages))
    if importlib.util.find_spec("transformers") is None:
        print(json.dumps({
            "event": "install_tool_dependency",
            "package": "transformers==4.42.4",
            "target": str(tool_packages),
        }), flush=True)
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "--disable-pip-version-check", "--no-input",
            "--no-warn-script-location", "--target", str(tool_packages),
            "transformers==4.42.4",
        ], stdout=sys.stdout, stderr=subprocess.STDOUT, check=True)
        importlib.invalidate_caches()

    from transformers import AutoModel, AutoTokenizer

    if model_dir.is_dir() and (model_dir / "config.json").is_file():
        source = str(model_dir)
        local_only = True
    else:
        source = args.model_id
        local_only = False
    print(json.dumps({
        "event": "load_model",
        "source": source,
        "local_only": local_only,
        "hf_endpoint": args.hf_endpoint,
    }), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=local_only)
    model = AutoModel.from_pretrained(source, local_files_only=local_only)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not local_only:
        model_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
        print(json.dumps({
            "event": "model_saved",
            "model_dir": str(model_dir),
        }), flush=True)

    test_sets = build_mdsent_test_sets(make_config(repo))
    summary = {}
    for domain in DOMAINS:
        dataset = test_sets[domain]
        samples = [
            (normalize_key(dataset.get_id(index)), dataset.texts[index])
            for index in range(len(dataset))
        ]
        reference_path = (repo / "exp" / "distributed_feature_cache" /
                          args.groups[0] /
                          CACHE_BASENAME.format(domain=domain))
        existing_paths, existing_features = load_cache(reference_path)
        existing = set(existing_paths)
        missing = [(key, text) for key, text in samples
                   if key not in existing]
        encoded = encode_missing(tokenizer, model, missing, args.batch_size)
        merged_paths = existing_paths + list(encoded.keys())
        if encoded:
            appended = np.stack(list(encoded.values()), axis=0)
            merged_features = np.concatenate(
                [existing_features, appended], axis=0)
        else:
            merged_features = existing_features
        test_keys = {key for key, _ in samples}
        if not test_keys.issubset(set(merged_paths)):
            raise RuntimeError(f"held-out coverage still incomplete: {domain}")
        for group in args.groups:
            target = (repo / "exp" / "distributed_feature_cache" / group /
                      CACHE_BASENAME.format(domain=domain))
            save_cache(target, merged_paths, merged_features)
        summary[domain] = {
            "before": len(existing_paths),
            "added": len(encoded),
            "after": len(merged_paths),
            "test_keys": len(test_keys),
            "test_coverage": len(test_keys & set(merged_paths)),
        }
        print(json.dumps({
            "event": "domain_complete",
            "domain": domain,
            **summary[domain],
        }), flush=True)

    marker = repo / "exp" / "distributed_feature_cache" / \
        "mdsent_eval_cache_completion.json"
    model_files = {}
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        model_files[str(path.relative_to(model_dir)).replace("\\", "/")] = {
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    marker.write_text(json.dumps({
        "model_id": args.model_id,
        "hf_endpoint": args.hf_endpoint,
        "model_dir": str(model_dir),
        "model_files": model_files,
        "groups": args.groups,
        "domains": summary,
    }, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "complete",
        "marker": str(marker),
        "domains": summary,
    }), flush=True)


if __name__ == "__main__":
    main()
