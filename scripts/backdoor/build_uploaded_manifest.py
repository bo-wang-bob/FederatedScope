"""把"已经划分好的两份数据"转成 GGEUR 可直接使用的便携 manifest。

用途
----
平台的上传数据集走的是 DomainNet 通道: 一个 manifest.json 描述类别、域和
每条记录的(train/val/test)归属。``federatedscope/cv/dataset/domainnet.py``
的 ``DomainNet`` 在记录里发现 ``split`` 字段时会**跳过随机切分**, 直接按
manifest 的划分取数据集 -- 这正是"训练集和测试集已经分好"需要的语义。

本机场景: ``<repo>/dataset/train/<类>/*.jpg`` 与 ``<repo>/dataset/test/<类>/*.jpg``
是用户上传后已划分好的两份数据, 没有 DomainNet 那样的 ``<域>/<类>/`` 层级,
所以不能直接用文件夹发现 (``discover_domainnet_metadata``), 必须先生成 manifest。

产出格式 (与 platform 的 ``DatasetRepository`` 完全一致)::

    {
      "classes": ["apple", "banana", ...],
      "domains": ["uploaded"],
      "records": {"uploaded": [
          {"path": "train/apple/xxx.jpg", "label": 0, "split": "train"},
          {"path": "test/apple/yyy.jpg",  "label": 0, "split": "test"}
      ]}
    }

用法::

    python scripts/backdoor/build_uploaded_manifest.py
    python scripts/backdoor/build_uploaded_manifest.py --root C:/path/to/dataset
"""
import argparse
import json
import os
import sys

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def _repo_root(start):
    cur = os.path.abspath(start)
    for _ in range(6):
        if os.path.isdir(os.path.join(cur, 'federatedscope')):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(os.path.join(start, '..'))


REPO_ROOT = _repo_root(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _list_images(directory):
    """Return sorted absolute image paths inside one directory."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f'目录不存在: {directory}')
    names = [
        name for name in os.listdir(directory)
        if name.lower().endswith(IMAGE_SUFFIXES)
        and os.path.isfile(os.path.join(directory, name))
    ]
    # 排序保证 manifest 可复现; 不同机器 listdir 顺序不一致会导致 label 错位。
    return sorted(names)


def _scan_split(root, split_dir, classes, class_to_idx, val_split=False):
    """Collect records for one already-partitioned split directory."""
    base = os.path.join(root, split_dir)
    if not os.path.isdir(base):
        raise FileNotFoundError(f'缺少划分目录: {base}')

    # 'split' 字段必须是 DomainNet 认得的三个值之一
    split_value = 'train' if split_dir == 'train' else (
        'test' if split_dir == 'test' else (
            'val' if val_split else None))
    if split_value is None:
        raise ValueError(f'无法识别的划分名: {split_dir}')

    unknown = []
    records = []
    for class_name in sorted(os.listdir(base)):
        class_dir = os.path.join(base, class_name)
        if not os.path.isdir(class_dir):
            continue
        if class_name not in class_to_idx:
            unknown.append(class_name)
            continue
        for name in _list_images(class_dir):
            records.append({
                # 相对 data.root, 统一正斜杠 -- Windows/Linux 都要能读
                'path': f'{split_dir}/{class_name}/{name}',
                'label': int(class_to_idx[class_name]),
                'split': split_value,
            })
    return records, unknown


def build_manifest(root, train_dir, test_dir, domain='uploaded',
                   val_dir=None):
    """Build the portable DomainNet-style manifest from pre-split folders.

    Classes are discovered from the **training** split (that is the contract
    the platform enforces on upload), so an unexpected class folder in the
    test split cannot shift any label index.
    """
    root = os.path.abspath(root)
    train_base = os.path.join(root, train_dir)
    if not os.path.isdir(train_base):
        raise FileNotFoundError(f'训练集目录不存在: {train_base}')

    train_classes = sorted(
        name for name in os.listdir(train_base)
        if os.path.isdir(os.path.join(train_base, name)))
    if len(train_classes) < 2:
        raise ValueError(f'训练集至少需要两个类别: {train_base}')

    class_to_idx = {name: idx for idx, name in enumerate(train_classes)}
    problems = []

    train_records, _ = _scan_split(root, train_dir, train_classes,
                                   class_to_idx)
    if not train_records:
        raise ValueError(f'训练集没有可用图片: {train_base}')

    counts = {'train': len(train_records)}
    test_records = []
    if test_dir:
        try:
            test_records, unknown = _scan_split(root, test_dir, train_classes,
                                                class_to_idx)
        except FileNotFoundError as error:
            problems.append(str(error))
            test_records = []
        else:
            counts['test'] = len(test_records)
            if unknown:
                problems.append(
                    f'测试集里存在训练集没有的类别, 已跳过: {sorted(unknown)}')

    val_records = []
    if val_dir and os.path.isdir(os.path.join(root, val_dir)):
        val_records, _ = _scan_split(root, val_dir, train_classes,
                                     class_to_idx, val_split=True)
        counts['val'] = len(val_records)

    # 平台要求每个训练类至少 3 张图, 这里只提示不下断言, 便于小样本调试
    per_class = {}
    for record in train_records:
        key = train_classes[record['label']]
        per_class[key] = per_class.get(key, 0) + 1
    thin = [name for name, count in per_class.items() if count < 3]
    if thin:
        problems.append(f'以下类别训练样本少于 3 张: {sorted(thin)}')

    records = train_records + test_records + val_records
    manifest = {
        'classes': train_classes,
        'domains': [domain],
        'records': {domain: records},
    }
    return manifest, counts, per_class, problems


def main():
    parser = argparse.ArgumentParser(description='生成上传数据集的 manifest')
    parser.add_argument('--root', default=os.path.join(REPO_ROOT, 'dataset'),
                        help='数据集根目录, 下面应有 train/ 与 test/')
    parser.add_argument('--train-dir', default='train')
    parser.add_argument('--test-dir', default='test')
    parser.add_argument('--val-dir', default='')
    parser.add_argument('--domain', default='uploaded')
    parser.add_argument('--output', default='',
                        help='默认 <root>/manifest.json')
    args = parser.parse_args()

    manifest, counts, per_class, problems = build_manifest(
        args.root, args.train_dir, args.test_dir, args.domain,
        args.val_dir or None)

    output = args.output or os.path.join(os.path.abspath(args.root),
                                         'manifest.json')
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=1)

    print(f'manifest -> {output}')
    print(f'  domain={args.domain} classes={len(manifest["classes"])}')
    for split, count in counts.items():
        print(f'  {split}: {count}')
    if per_class:
        smallest = min(per_class.values())
        largest = max(per_class.values())
        print(f'  train per-class: min={smallest} max={largest}')
    for problem in problems:
        print(f'  [warn] {problem}')


if __name__ == '__main__':
    main()
