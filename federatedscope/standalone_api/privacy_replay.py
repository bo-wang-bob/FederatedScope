"""Explicit, fail-closed alignment of historical FedMIA indexed replay."""
from .platform_config import PlatformError


def align_indexed_replay(module, runs, image_items):
    """Keep a verified common real-image prefix, never silently truncate.

    runs contains (config, client features, result) for both historical runs.
    This adapter supports only the saved test-mode indexed/global algorithm:
    its round scores retain image order and indexed mode uses a prefix. It
    rejects mixed/generated samples and any unverified round or label order.
    """
    np = module.np

    def reject():
        raise PlatformError('历史成员推理结果无法与图片安全对齐', 409)

    modes = []
    counts = []
    reference_rounds = None
    for cfg, features, result in runs:
        metadata = getattr(result, 'metadata', {})
        mode = metadata.get('shadow_stat_mode')
        rounds = metadata.get('rounds_used')
        if (mode not in {'indexed', 'global'} or not rounds
                or mode != str(cfg.attack.fedmia_shadow_stat_mode).lower()
                or str(cfg.attack.mode).lower() != 'test'
                or sorted(rounds) != sorted(features.get('train_cos', {}))):
            reject()
        if reference_rounds is not None and rounds != reference_rounds:
            reject()
        reference_rounds = rounds
        modes.append(mode)
        lengths = []
        for rnd in rounds:
            if any(features.get(key, {}).get(rnd) is not None
                   and len(features[key][rnd]) for key in
                   ('augmented_cos', 'image_aug_cos')):
                reject()
            sizes = []
            for prefix, items in zip(('train', 'test'), image_items):
                raw_labels = features.get(prefix + '_labels', {}).get(rnd)
                raw_scores = features.get(prefix + '_cos', {}).get(rnd)
                if raw_labels is None or raw_scores is None:
                    reject()
                labels = np.asarray(raw_labels).reshape(-1)
                expected = np.asarray([item[1] for item in items])
                if (not len(labels) or len(labels) != len(raw_scores)
                        or len(labels) > len(expected)
                        or not np.array_equal(labels, expected[:len(labels)])
                        or (prefix == 'test' and len(labels) != len(expected))
                        or (mode == 'global' and len(labels) != len(expected))):
                    reject()
                sizes.append(len(labels))
            lengths.append((sizes[0], min(sizes) if mode == 'indexed' else sizes[1]))
        # Different per-round sizes would require an explicit sample-id map.
        if len(set(lengths)) != 1:
            reject()
        count = (len(result.scores_member), len(result.scores_nonmember))
        if count != lengths[0]:
            reject()
        counts.append(count)
    if set(modes) != {'indexed', 'global'}:
        reject()

    common = tuple(min(count[i] for count in counts) for i in (0, 1))
    info = {
        'mode': 'verified-common-prefix',
        'memberSamples': common[0], 'nonmemberSamples': common[1],
        'originalCounts': {'noDefense': list(counts[0]), 'defense': list(counts[1])},
        'shadowStatistics': {'noDefense': modes[0], 'defense': modes[1]},
        'rounds': list(reference_rounds),
        'message': (f'共同样本：成员 {common[0]} 个，非成员 {common[1]} 个。'
                    '历史回放，图片按配置重建并核对标签；两组攻击统计口径不同，'
                    '不作为严格防御增益结论。'),
    }
    return common, info


def align_image_index_replay(module, runs, image_items):
    """Validate ordered real-image scores from an exported image-index package.

    Generated features can be stored for mix calibration, but must explicitly
    be excluded from the real-image results. Keep the legacy replay adapter
    above unchanged, including its stricter historical-format requirements.
    """
    np = module.np
    reference = None
    counts = []
    modes = []
    for cfg, features, result in runs:
        meta = getattr(result, 'metadata', {}) or {}
        mode = meta.get('shadow_stat_mode')
        rounds = meta.get('rounds_used')
        if (mode not in {'indexed', 'global'}
                or mode != str(cfg.attack.fedmia_shadow_stat_mode).lower()
                or not rounds or sorted(rounds) != sorted(features.get('train_cos', {}))
                or meta.get('use_augmented_nonmember') is not False
                or meta.get('use_image_aug_member') is not False):
            raise PlatformError('导出成员推理分数无法与真实图片安全对齐', 409)
        if reference is not None and rounds != reference:
            raise PlatformError('两组攻击特征的轮次不一致', 409)
        reference = rounds
        round_counts = []
        for rnd in rounds:
            sizes = []
            for prefix, items in zip(('train', 'test'), image_items):
                labels = features.get(prefix + '_labels', {}).get(rnd)
                scores = features.get(prefix + '_cos', {}).get(rnd)
                if labels is None or scores is None:
                    raise PlatformError('导出特征缺少真实图片的标签或分数', 409)
                labels = np.asarray(labels).reshape(-1)
                expected = np.asarray([item[1] for item in items])
                if (not len(labels) or len(labels) != len(scores)
                        or len(labels) > len(expected)
                        or not np.array_equal(labels, expected[:len(labels)])
                        or ((prefix == 'test' or mode == 'global')
                            and len(labels) != len(expected))):
                    raise PlatformError('导出特征与图片索引的标签顺序不一致', 409)
                sizes.append(len(labels))
            round_counts.append((sizes[0], min(sizes) if mode == 'indexed' else sizes[1]))
        count = (len(result.scores_member), len(result.scores_nonmember))
        if len(set(round_counts)) != 1 or count != round_counts[0]:
            raise PlatformError('真实图片分数数量与已验证的特征前缀不一致', 409)
        counts.append(count)
        modes.append(mode)
    if len(set(modes)) != 1:
        raise PlatformError('导出攻防特征的统计模式不一致', 409)
    common = tuple(min(count[i] for count in counts) for i in (0, 1))
    return common, {
        'mode': 'verified-image-index', 'metricsMode': 'mix',
        'memberSamples': common[0], 'nonmemberSamples': common[1],
        'rounds': list(reference),
        'originalCounts': {'noDefense': list(counts[0]), 'defense': list(counts[1])},
        'message': (f'真实图片展示：训练样本 {common[0]} 个，非训练样本 {common[1]} 个。'
                    '图片按导出索引核对各轮标签顺序；整体指标和分布使用 mix 校准样本，'
                    '逐图预测使用真实图片分数。'),
    }
