#!/usr/bin/env python3
"""Generate a portable, deterministic manifest for selected DomainNet domains."""

import argparse
import hashlib
import json
from pathlib import Path


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--domains', default='clipart,painting,real,sketch')
    parser.add_argument(
        '--split-list-dir',
        help='Directory containing <domain>_{train,test}.txt files. '
             'Each line is either an image path or an image path plus label.')
    parser.add_argument(
        '--label-list-dir',
        help='Optional parallel ground-truth lists for split-list lines '
             'that do not include labels.')
    return parser.parse_args()


def _read_list(path):
    if not path.is_file():
        raise SystemExit(f'Missing DomainNet list: {path}')
    return [line.strip() for line in path.read_text(
        encoding='utf-8').splitlines() if line.strip()]


def _split_path_and_label(line):
    parts = line.rsplit(maxsplit=1)
    if len(parts) == 2 and parts[1].lstrip('-').isdigit():
        return parts[0], int(parts[1])
    return line, None


def _class_name_from_path(relative_path, domain):
    parts = Path(relative_path).parts
    if len(parts) < 3 or parts[0] != domain:
        raise SystemExit(
            f'Invalid DomainNet path for {domain}: {relative_path}')
    return parts[1]


def _records_from_lists(root, domains, split_list_dir, label_list_dir):
    records = {}
    label_to_class = {}
    seen_paths = set()
    for domain in domains:
        domain_records = []
        for split in ('train', 'test'):
            split_path = split_list_dir / f'{domain}_{split}.txt'
            split_lines = _read_list(split_path)
            parsed = [_split_path_and_label(line) for line in split_lines]
            needs_parallel_labels = any(label is None for _, label in parsed)
            label_lines = None
            if needs_parallel_labels:
                if label_list_dir is None:
                    raise SystemExit(
                        f'Labels missing in {split_path}; provide '
                        '--label-list-dir')
                label_path = label_list_dir / f'{domain}_{split}.txt'
                label_lines = _read_list(label_path)
                if len(label_lines) != len(split_lines):
                    raise SystemExit(
                        f'List length mismatch: {split_path} has '
                        f'{len(split_lines)} lines but {label_path} has '
                        f'{len(label_lines)}')

            for index, (relative_path, label) in enumerate(parsed):
                class_name = None
                if label is None:
                    groundtruth_path, label = _split_path_and_label(
                        label_lines[index])
                    if label is None:
                        raise SystemExit(
                            f'Missing label at {label_path}:{index + 1}')
                    class_name = _class_name_from_path(
                        groundtruth_path, domain)
                else:
                    class_name = _class_name_from_path(relative_path, domain)

                image_path = root / relative_path
                if image_path.suffix.lower() not in IMAGE_SUFFIXES or \
                        not image_path.is_file():
                    raise SystemExit(
                        f'Missing or invalid image at {split_path}:'
                        f'{index + 1}: {image_path}')
                if relative_path in seen_paths:
                    raise SystemExit(
                        f'Duplicate DomainNet image path: {relative_path}')
                seen_paths.add(relative_path)

                previous = label_to_class.setdefault(label, class_name)
                if previous != class_name:
                    raise SystemExit(
                        f'Label {label} maps to both {previous} and '
                        f'{class_name}')
                domain_records.append({
                    'path': Path(relative_path).as_posix(),
                    'label': label,
                })
        if not domain_records:
            raise SystemExit(f'No images found for domain: {domain}')
        records[domain] = domain_records

    labels = sorted(label_to_class)
    if labels != list(range(len(labels))):
        raise SystemExit(
            f'DomainNet labels are not contiguous from zero: {labels[:10]}')
    classes = [label_to_class[label] for label in labels]
    return classes, records


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    domains = [item.strip() for item in args.domains.split(',')
               if item.strip()]
    missing = [domain for domain in domains
               if not (root / domain).is_dir()]
    if missing:
        raise SystemExit(f'Missing DomainNet domains under {root}: {missing}')

    if args.split_list_dir:
        split_list_dir = Path(args.split_list_dir).resolve()
        label_list_dir = (Path(args.label_list_dir).resolve()
                          if args.label_list_dir else None)
        classes, records = _records_from_lists(
            root, domains, split_list_dir, label_list_dir)
    else:
        classes = sorted({class_dir.name for domain in domains
                          for class_dir in (root / domain).iterdir()
                          if class_dir.is_dir()})
        class_to_idx = {name: idx for idx, name in enumerate(classes)}
        records = {}
        for domain in domains:
            domain_records = []
            for class_name in classes:
                class_dir = root / domain / class_name
                if not class_dir.is_dir():
                    continue
                for image_path in sorted(class_dir.iterdir()):
                    if image_path.is_file() and \
                            image_path.suffix.lower() in IMAGE_SUFFIXES:
                        domain_records.append({
                            'path': image_path.relative_to(root).as_posix(),
                            'label': class_to_idx[class_name],
                        })
            if not domain_records:
                raise SystemExit(f'No images found for domain: {domain}')
            records[domain] = domain_records

    record_digest = hashlib.sha256()
    for domain in domains:
        for record in records[domain]:
            record_digest.update(
                f"{domain}\0{record['path']}\0{record['label']}\n".encode())
    payload = {
        'version': 1,
        'source_root': str(root),
        'domains': domains,
        'classes': classes,
        'records': records,
        'record_count': sum(len(items) for items in records.values()),
        'records_sha256': record_digest.hexdigest(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                      encoding='utf-8')
    print(json.dumps({
        'output': str(output),
        'domains': domains,
        'class_count': len(classes),
        'record_count': payload['record_count'],
        'records_sha256': payload['records_sha256'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
