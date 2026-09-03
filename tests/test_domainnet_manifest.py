import json

from federatedscope.cv.dataset.domainnet import (
    DomainNet, load_domainnet_manifest)


def test_portable_manifest_builds_deterministic_split_without_images(tmp_path):
    records = [
        {'path': f'clipart/class_a/image_{idx:03d}.png', 'label': idx % 2}
        for idx in range(10)
    ]
    manifest_path = tmp_path / 'domainnet_manifest.json'
    manifest_path.write_text(json.dumps({
        'version': 1,
        'domains': ['clipart'],
        'classes': ['class_a', 'class_b'],
        'records': {'clipart': records},
    }), encoding='utf-8')

    domains, classes, records_by_domain = load_domainnet_manifest(
        manifest_path, ['clipart'])
    assert domains == ['clipart']
    assert classes == ['class_a', 'class_b']

    first = DomainNet(root=tmp_path / 'images',
                      domain='clipart',
                      classes=classes,
                      split='train',
                      train_ratio=0.7,
                      val_ratio=0.0,
                      seed=42,
                      records=records_by_domain['clipart'])
    second = DomainNet(root=tmp_path / 'other_images',
                       domain='clipart',
                       classes=classes,
                       split='train',
                       train_ratio=0.7,
                       val_ratio=0.0,
                       seed=42,
                       records=records_by_domain['clipart'])

    assert len(first) == 7
    assert first.targets == second.targets
    assert [path.split('clipart')[-1] for path in first.data] == [
        path.split('clipart')[-1] for path in second.data
    ]


def test_manifest_rejects_missing_selected_domain(tmp_path):
    manifest_path = tmp_path / 'domainnet_manifest.json'
    manifest_path.write_text(json.dumps({
        'domains': ['clipart'],
        'classes': ['class_a'],
        'records': {'clipart': [{'path': 'clipart/class_a/a.png',
                                 'label': 0}]},
    }), encoding='utf-8')

    try:
        load_domainnet_manifest(manifest_path, ['real'])
    except ValueError as error:
        assert 'real' in str(error)
    else:
        raise AssertionError('missing selected domain must fail')
