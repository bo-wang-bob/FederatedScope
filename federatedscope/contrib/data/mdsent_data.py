"""
Multi-Domain Sentiment data loader for GGEUR text experiments.

Expected layout:
    <data.root>/{books,dvd,electronics,kitchen}/*.review

Ratings {1.0, 2.0, 4.0, 5.0} are mapped to four classes {0, 1, 2, 3}.  Each
domain is split into train/val/test first, then the train split is partitioned
among clients inside that domain.
"""

import json
import logging
import os
import random
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from torch.utils.data import DataLoader, Dataset, Subset

from federatedscope.register import register_data

logger = logging.getLogger(__name__)

RATING_TO_CLASS = {'1.0': 0, '2.0': 1, '4.0': 2, '5.0': 3}
DEFAULT_DOMAINS = ['books', 'dvd', 'electronics', 'kitchen']


@dataclass(frozen=True)
class MDSentArgs:
    include_unlabeled: bool = True
    max_samples_per_domain: int = 0
    balance_test: bool = False
    test_samples_per_class: int = 0
    seed: int = 42
    cache_dir: str = ''
    use_cache: bool = True


class MDSentTextDataset(Dataset):
    """In-memory text dataset with stable sample ids and `.targets`."""

    def __init__(self, texts: List[str], labels: List[int], ids: List[str],
                 domain: str):
        if not (len(texts) == len(labels) == len(ids)):
            raise ValueError('texts/labels/ids length mismatch')
        self.texts = list(texts)
        self.targets = [int(label) for label in labels]
        self.ids = [str(sample_id) for sample_id in ids]
        self.domain = str(domain)
        self.data = self.texts

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.texts[idx], int(self.targets[idx])

    def get_id(self, idx):
        return self.ids[idx]


def _raw_args(cfg):
    args = cfg.data.args[0] if getattr(cfg.data, 'args', None) else {}
    return args or {}


def _bool_arg(args, key, default):
    value = args.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y'}
    return bool(value)


def _int_arg(args, key, default):
    try:
        return int(args.get(key, default))
    except Exception:
        return int(default)


def _str_arg(args, key, default):
    value = args.get(key, default)
    return str(value) if value is not None else str(default)


def get_mdsent_args(cfg) -> MDSentArgs:
    args = _raw_args(cfg)
    return MDSentArgs(
        include_unlabeled=_bool_arg(args, 'include_unlabeled', True),
        max_samples_per_domain=_int_arg(args, 'max_samples_per_domain', 0),
        balance_test=_bool_arg(args, 'balance_test', False),
        test_samples_per_class=_int_arg(args, 'test_samples_per_class', 0),
        seed=_int_arg(args, 'seed', int(getattr(cfg, 'seed', 42))),
        cache_dir=_str_arg(args, 'cache_dir', ''),
        use_cache=_bool_arg(args, 'use_cache', True),
    )


def _cache_dir(data_root, args: MDSentArgs):
    return args.cache_dir or os.path.join(data_root, 'mdsent_cache')


def _cache_path(data_root, args: MDSentArgs, domain):
    os.makedirs(_cache_dir(data_root, args), exist_ok=True)
    include_unlabeled = 1 if args.include_unlabeled else 0
    safe_domain = str(domain).lower().replace('\\', '_').replace('/', '_')
    name = f'mdsent_{safe_domain}_includeu{include_unlabeled}.jsonl'
    return os.path.join(_cache_dir(data_root, args), name)


def _domain_seed(args: MDSentArgs, domain):
    crc = zlib.crc32(str(domain).lower().encode('utf-8')) & 0xFFFFFFFF
    return int(args.seed) + int(crc % 100000)


def _load_jsonl(path):
    texts, labels, ids = [], [], []
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)
            texts.append(item['text'])
            labels.append(int(item['label']))
            ids.append(str(item['id']))
    return texts, labels, ids


def _save_jsonl(path, texts, labels, ids):
    tmp_path = f'{path}.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as fout:
        for text, label, sample_id in zip(texts, labels, ids):
            fout.write(
                json.dumps(
                    {
                        'id': str(sample_id),
                        'text': str(text),
                        'label': int(label)
                    },
                    ensure_ascii=False) + '\n')
    os.replace(tmp_path, path)


def _iter_reviews(review_path, domain):
    file_name = os.path.basename(review_path)
    review_idx = -1
    in_review = False
    tag = None
    title_lines, text_lines = [], []
    rating = None

    def flush():
        if rating is None:
            return None
        parts = []
        if title_lines:
            parts.append(' '.join(line.strip() for line in title_lines
                                  if line.strip()))
        if text_lines:
            parts.append('\n'.join(text_lines).strip())
        text = '\n\n'.join(part for part in parts if part).strip()
        if not text:
            return None
        return text, str(rating).strip(), f'{domain}/{file_name}:{review_idx}'

    with open(review_path, 'r', encoding='utf-8',
              errors='ignore') as fin:
        for raw_line in fin:
            line = raw_line.strip()
            if line == '<review>':
                in_review = True
                review_idx += 1
                tag = None
                title_lines, text_lines = [], []
                rating = None
                continue
            if not in_review:
                continue
            if line == '</review>':
                item = flush()
                if item is not None:
                    yield item
                in_review = False
                tag = None
                continue
            if line in {'<title>', '<review_text>', '<rating>'}:
                tag = line.strip('<>').lower()
                continue
            if line in {'</title>', '</review_text>', '</rating>'}:
                tag = None
                continue
            if tag == 'rating':
                if rating is None and line:
                    rating = line
            elif tag == 'title':
                if line:
                    title_lines.append(line)
            elif tag == 'review_text':
                text_lines.append(raw_line.rstrip('\n'))


def load_domain_all(data_root, args: MDSentArgs, domain):
    cache_path = _cache_path(data_root, args, domain)
    if args.use_cache and os.path.exists(cache_path):
        texts, labels, ids = _load_jsonl(cache_path)
        return MDSentTextDataset(texts, labels, ids, domain)

    domain_dir = os.path.join(data_root, str(domain))
    if not os.path.isdir(domain_dir):
        raise FileNotFoundError(f'Domain directory not found: {domain_dir}')

    files = ['positive.review', 'negative.review']
    if args.include_unlabeled and os.path.exists(
            os.path.join(domain_dir, 'unlabeled.review')):
        files.append('unlabeled.review')

    texts, labels, ids = [], [], []
    skipped = 0
    for file_name in files:
        path = os.path.join(domain_dir, file_name)
        if not os.path.exists(path):
            continue
        for text, rating, sample_id in _iter_reviews(path, str(domain)):
            label = RATING_TO_CLASS.get(rating)
            if label is None:
                skipped += 1
                continue
            texts.append(text)
            labels.append(label)
            ids.append(sample_id)

    if skipped:
        logger.warning('MDSent: skipped %d unsupported ratings in %s', skipped,
                       domain)
    if args.use_cache:
        _save_jsonl(cache_path, texts, labels, ids)
    return MDSentTextDataset(texts, labels, ids, domain)


def split_train_val_test(dataset: MDSentTextDataset, splits, seed):
    split_values = list(splits) if splits is not None else [0.8, 0.1, 0.1]
    while len(split_values) < 3:
        split_values.append(0.0)
    train_ratio, val_ratio, test_ratio = [float(v) for v in split_values[:3]]
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError(f'Invalid data.splits={splits}')
    train_ratio, val_ratio = train_ratio / total, val_ratio / total

    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_train = int(len(indices) * train_ratio)
    n_val = int(len(indices) * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def subset(idx_list):
        return MDSentTextDataset([dataset.texts[i] for i in idx_list],
                                 [dataset.targets[i] for i in idx_list],
                                 [dataset.ids[i] for i in idx_list],
                                 dataset.domain)

    return subset(train_idx), subset(val_idx) if val_idx else None, subset(
        test_idx)


def balance_dataset_by_class(dataset: MDSentTextDataset, num_classes, seed,
                             per_class=0):
    if dataset is None or len(dataset) == 0:
        return dataset
    label_indices: Dict[int, List[int]] = {i: [] for i in range(num_classes)}
    for idx, label in enumerate(dataset.targets):
        if int(label) in label_indices:
            label_indices[int(label)].append(idx)
    if any(len(v) == 0 for v in label_indices.values()):
        logger.warning('MDSent: cannot balance %s, class counts=%s',
                       dataset.domain,
                       {k: len(v)
                        for k, v in label_indices.items()})
        return dataset
    size = min(len(v) for v in label_indices.values())
    if int(per_class) > 0:
        size = min(size, int(per_class))

    rng = random.Random(seed)
    selected = []
    for indices in label_indices.values():
        indices = list(indices)
        rng.shuffle(indices)
        selected.extend(indices[:size])
    rng.shuffle(selected)
    return MDSentTextDataset([dataset.texts[i] for i in selected],
                             [dataset.targets[i] for i in selected],
                             [dataset.ids[i] for i in selected],
                             dataset.domain)


def maybe_cap_domain_samples(dataset: MDSentTextDataset, args: MDSentArgs,
                             domain):
    max_samples = int(args.max_samples_per_domain)
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    rng = random.Random(_domain_seed(args, domain) + 101)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    selected = indices[:max_samples]
    return MDSentTextDataset([dataset.texts[i] for i in selected],
                             [dataset.targets[i] for i in selected],
                             [dataset.ids[i] for i in selected],
                             dataset.domain)


def _uniform_split(indices, num_clients, seed):
    rng = random.Random(seed)
    indices = list(indices)
    rng.shuffle(indices)
    return [indices[i::num_clients] for i in range(num_clients)]


def _discover_domains(data_root):
    return [
        domain for domain in DEFAULT_DOMAINS
        if os.path.isdir(os.path.join(data_root, domain))
    ]


def build_mdsent_test_sets(config) -> Dict[str, MDSentTextDataset]:
    data_root = config.data.root
    args = get_mdsent_args(config)
    splits = tuple(getattr(config.data, 'splits', [0.8, 0.1, 0.1]))
    domains = _discover_domains(data_root)
    if not domains:
        raise FileNotFoundError(f'No MDSent domains found under {data_root}')

    test_sets = {}
    for domain in domains:
        full = load_domain_all(data_root, args, domain)
        full = maybe_cap_domain_samples(full, args, domain)
        _, _, test_set = split_train_val_test(full, splits,
                                              _domain_seed(args, domain))
        if args.balance_test or int(args.test_samples_per_class) > 0:
            test_set = balance_dataset_by_class(
                test_set,
                num_classes=len(RATING_TO_CLASS),
                seed=_domain_seed(args, domain) + 97,
                per_class=int(args.test_samples_per_class))
        test_sets[domain] = test_set
    return test_sets


def load_mdsent_data(config, client_cfgs=None):
    if str(config.data.type).lower() != 'mdsent':
        return None

    data_root = config.data.root
    args = get_mdsent_args(config)
    domains = _discover_domains(data_root)
    if not domains:
        raise FileNotFoundError(f'No MDSent domains found under {data_root}')

    configured_clients = int(getattr(config.federate, 'client_num',
                                     len(domains)))
    if configured_clients <= 0:
        configured_clients = len(domains)
    if configured_clients % len(domains) != 0:
        logger.warning('client_num=%d is not divisible by %d domains; using '
                       'one client per domain.', configured_clients,
                       len(domains))
        clients_per_domain = 1
        total_clients = len(domains)
    else:
        clients_per_domain = configured_clients // len(domains)
        total_clients = configured_clients

    splits = tuple(getattr(config.data, 'splits', [0.8, 0.1, 0.1]))
    alpha = float(getattr(config.data, 'dirichlet_alpha', 0.0))
    min_samples = int(getattr(config.data, 'min_samples_per_client', 5))
    batch_size = int(getattr(config.dataloader, 'batch_size', 32))
    num_workers = int(getattr(config.dataloader, 'num_workers', 0))

    if alpha > 0:
        from federatedscope.cv.dataloader.dataloader import \
            split_data_by_dirichlet

    data = {}
    client_id = 1
    num_classes = len(RATING_TO_CLASS)
    for domain in domains:
        logger.info('Loading MDSent domain %s', domain)
        full = load_domain_all(data_root, args, domain)
        full = maybe_cap_domain_samples(full, args, domain)
        train_set, val_set, test_set = split_train_val_test(
            full, splits, _domain_seed(args, domain))
        if args.balance_test or int(args.test_samples_per_class) > 0:
            test_set = balance_dataset_by_class(
                test_set,
                num_classes=num_classes,
                seed=_domain_seed(args, domain) + 97,
                per_class=int(args.test_samples_per_class))

        if clients_per_domain <= 1:
            client_splits = [list(range(len(train_set)))]
        elif alpha > 0:
            client_splits = split_data_by_dirichlet(
                train_set,
                num_clients=clients_per_domain,
                num_classes=num_classes,
                alpha=alpha,
                seed=_domain_seed(args, domain) + 17,
                min_samples_per_client=min_samples)
        else:
            client_splits = _uniform_split(range(len(train_set)),
                                           clients_per_domain,
                                           _domain_seed(args, domain) + 17)

        for indices in client_splits:
            train_subset = Subset(train_set, list(indices))
            data[client_id] = {
                'train':
                    DataLoader(train_subset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=num_workers,
                               drop_last=False),
                'val':
                    DataLoader(val_set,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=num_workers)
                    if val_set is not None and len(val_set) > 0 else None,
                'test':
                    DataLoader(test_set,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=num_workers)
            }
            logger.info('  Client %d (%s): train=%d, val=%d, test=%d',
                        client_id, domain, len(train_subset),
                        len(val_set) if val_set else 0, len(test_set))
            client_id += 1

    config.federate.client_num = total_clients
    config.model.num_classes = int(
        getattr(config.model, 'num_classes', num_classes))
    logger.info('MDSent loaded: clients=%d, domains=%s, alpha=%s',
                len(data), domains, alpha)

    # Registered contrib loaders return before data_builder.get_data() reaches
    # convert_data_mode().  Select the configured partition here for a real
    # distributed client; otherwise every client receives the outer
    # {client_id: ClientData} mapping and has no top-level ``train`` loader.
    role = str(getattr(config.distribute, 'role', 'client')).lower()
    data_idx = int(getattr(config.distribute, 'data_idx', -1))
    if config.federate.mode.lower() == 'distributed' and role == 'client' \
            and data_idx != -1:
        if data_idx not in data:
            raise KeyError(
                f'MDSent data_idx={data_idx} is outside the generated '
                f'client partitions 1..{len(data)}')
        logger.info('MDSent distributed client selected data_idx=%d: '
                    'train=%d', data_idx, len(data[data_idx]['train'].dataset))
        return data[data_idx], config

    return data, config


register_data('mdsent', load_mdsent_data)
