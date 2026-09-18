"""Opt-in training adapters. No changes to ordinary client/server behavior."""
import copy
from pathlib import Path


def operating_point(member, nonmember):
    """Tie-aware, attainable operating point (no interpolated/fake predictions)."""
    import numpy as np
    from sklearn.metrics import roc_auc_score, roc_curve
    member, nonmember = np.asarray(member), np.asarray(nonmember)
    if not len(member) or not len(nonmember) or not np.isfinite(np.r_[member, nonmember]).all():
        raise ValueError('成员/非成员攻击分数缺失或包含非有限值')
    labels = np.r_[np.ones(len(member)), np.zeros(len(nonmember))]
    scores = np.r_[member, nonmember]
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    choices = np.flatnonzero(fpr <= .01 + 1e-12)
    best = choices[np.argmax(tpr[choices])]
    threshold = float(thresholds[best])
    if not np.isfinite(threshold):
        threshold = float(np.nextafter(scores.max(), np.inf))
    return dict(auc=float(roc_auc_score(labels, scores)), tprAt1Fpr=float(tpr[best]),
                actualFpr=float(fpr[best]), threshold=threshold,
                members=len(member), nonmembers=len(nonmember))


def privacy_bases(spec, caches):
    import numpy as np
    import torch
    from federatedscope.contrib.worker.ggeur_client import GGEURClient, AugmentedFeatureDataset
    from federatedscope.contrib.worker.ggeur_fedmia_server import GGEURFedMIAServer
    from .platform_worker import sample_refs, emit

    from .privacy_cloud_augmentation import CloudAugmentation
    class PrivacyClient(CloudAugmentation, GGEURClient):
        def _train_on_augmented_data(self):
            emit('client', clientId=self.ID, stage='本地训练', round=int(self.state))
            self._adaptive_dp_global_mlp_state = copy.deepcopy(self.mlp_classifier.state_dict())
            size, state, metrics = super()._train_on_augmented_data()
            if spec['request']['defense']:
                state, stats = self._apply_adaptive_dp_to_upload(state, int(self.state))
                if not stats:
                    raise RuntimeError('防御开启但上传模型未执行自适应裁剪')
                emit('privacy_defense', clientId=self.ID, **stats)
            emit('client', clientId=self.ID, stage='已上传', round=int(self.state),
                 loss=float(metrics['train_loss']), accuracy=float(metrics['train_acc']))
            return size, state, metrics

    class PrivacyServer(GGEURFedMIAServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            from .privacy_artifacts import prepare_run
            output = Path(spec['output'])
            archive = prepare_run(Path(__file__).resolve().parents[2], spec['request'], output.name, output)
            self.attack_feature_dir = str(archive / 'ggeur_fedmia_features')

        def _evaluate_dataset_subset(self, dataset, indices, model_state, batch_size=32, global_state=None):
            if dataset is None:
                return self._empty_eval_result()
            if caches is None:
                return super()._evaluate_dataset_subset(dataset, indices, model_state, batch_size, global_state)
            refs = sample_refs(dataset)
            if indices:
                refs = [refs[int(i)] for i in indices]
            if not refs:
                return self._empty_eval_result()
            values = np.asarray([caches[d][k] for d, k, _ in refs], dtype=np.float32)
            labels = np.asarray([y for _, _, y in refs], dtype=np.int64)
            return self._evaluate_feature_arrays(values, labels, model_state, global_state=global_state)

        def _should_save_attack_round(self, round_idx):
            start = int(spec.get('cloudAttack', {}).get('fedmia_save_start_round', 0))
            # A shorter requested run still saves its final round.
            return int(round_idx) >= start and super()._should_save_attack_round(round_idx) or int(round_idx) >= self.total_round_num - 1

        def _save_attack_features_if_needed(self, round_idx, client_id, content):
            if int(round_idx) <= 0:
                return
            path = Path(self._feature_file_path(round_idx, client_id))
            super()._save_attack_features_if_needed(round_idx, client_id, content)
            if self._should_save_attack_round(round_idx):
                if not path.is_file():
                    raise RuntimeError(f'客户端 {client_id} 第 {round_idx} 轮原生攻击特征保存失败')
                emit('privacy_feature', clientId=int(client_id), round=int(round_idx), filename=path.name)

        def _finish(self):
            # Final scoring is performed once by finalize from the local files.
            self._attack_executed = True
            from federatedscope.contrib.worker.ggeur_server import GGEURServer
            GGEURServer._finish(self)

    return PrivacyClient, PrivacyServer


def finalize(server, spec, info, partition, tests):
    import numpy as np
    from federatedscope.contrib.attack.plugins.fedmia_i import FedMIAIPlugin
    from federatedscope.contrib.attack.plugins.fedmia_ii import FedMIAIIPlugin
    from .platform_worker import save, digest, emit
    from .platform_service import read
    output = Path(spec['output'])
    all_data = server._load_saved_attack_features()
    if set(all_data) != set(map(int, partition)):
        raise ValueError('并非所有客户端都有攻击特征，拒绝标记完成')
    test_refs = [(d, k, y) for d in sorted(tests) for k, y in tests[d].items()]
    report = dict(dataset=spec['request']['group'], defense=spec['request']['defense'],
        shadowMode=str(server._cfg.attack.fedmia_shadow_stat_mode), metricsMode='mix',
        thresholdPolicy='mix 校准，经验 FPR≤1%；真实图片使用 test 分数', clients={})
    for cid, target in sorted(all_data.items()):
        shadow = [v for key, v in sorted(all_data.items()) if key != cid]
        plugins, display = {}, {}
        for plugin in (FedMIAIPlugin(), FedMIAIIPlugin()):
            calibrated = plugin.compute_scores(target, shadow, config=server._cfg)
            metric = operating_point(calibrated.scores_member, calibrated.scores_nonmember)
            display_cfg = server._cfg.clone()
            display_cfg.defrost()
            display_cfg.attack.mode = 'test'
            display_cfg.attack.fedmia_use_augmented_nonmember = False
            display_cfg.attack.fedmia_use_image_aug_member = False
            actual = plugin.compute_scores(target, shadow, config=display_cfg)
            plugins[plugin.name] = metric
            display[plugin.name] = actual
        samples = {}
        for group, refs in [('member', partition[str(cid)]), ('nonmember', test_refs)]:
            arrays = {name: (result.scores_member if group == 'member' else result.scores_nonmember)
                      for name, result in display.items()}
            length = min(len(refs), *(len(values) for values in arrays.values()))
            rows = []
            for index, (domain, key, label) in enumerate(refs[:length]):
                scores = {name: float(values[index]) for name, values in arrays.items()}
                if not all(np.isfinite(list(scores.values()))):
                    raise ValueError('真实样本攻击分数包含非有限值')
                rows.append(dict(domain=domain, key=key, label=int(label), className=info['classes'][int(label)],
                    scores=scores, predictions={name: ('member' if score >= plugins[name]['threshold'] else 'nonmember')
                        for name, score in scores.items()}))
            samples[group] = rows
        # Verify the label ordering that binds scores to original images.
        for round_idx, labels in target['train_labels'].items():
            actual = np.asarray(labels).reshape(-1)
            expected = np.asarray([r[2] for r in partition[str(cid)]])[:len(actual)]
            if not np.array_equal(actual, expected):
                raise ValueError('保存的成员特征与原图清单顺序不一致')
        for labels in target['test_labels'].values():
            if not np.array_equal(np.asarray(labels).reshape(-1), [r[2] for r in test_refs]):
                raise ValueError('保存的非成员特征与原图清单顺序不一致')
        report['clients'][str(cid)] = dict(metrics=plugins, samples=samples,
            rounds=sorted(map(int, target['train_losses'])),
            indexedWarning='indexed 截断至影子客户端最短有效长度；不同客户端索引并非同一图片' if str(server._cfg.attack.fedmia_shadow_stat_mode) != 'global' else None)
        emit('stage', stage=f'客户端 {cid} 攻击结果与原图分数已保存')
    archive = Path(server.attack_feature_dir).parent
    from .platform_paths import relative_path
    files = [dict(path=relative_path(archive, p), sha256=digest(p), bytes=p.stat().st_size)
             for p in sorted(Path(server.attack_feature_dir).glob('*.pt'))]
    if not files:
        raise ValueError('没有保存任何攻击特征')
    save(output / 'privacy_feature_manifest.json', dict(files=files, protocol='native-client-round-data'))
    report['completeTraining'] = bool(spec.get('trainingComplete', True))
    save(archive / 'privacy_feature_manifest.json', dict(files=files, protocol='native-client-round-data'))
    save(archive / 'privacy_results.json', report)
    save(output / 'privacy_results.json', report)
    result = read(output / 'result.json', {})
    result['privacy'] = dict(featureFiles=len(files), clients=len(report['clients']),
        defense=report['defense'], report='privacy_results.json', thresholdPolicy=report['thresholdPolicy'])
    save(output / 'result.json', result)
    emit('stage', stage='训练、原生攻击特征和逐客户端评测全部完成' if report['completeTraining'] else '已从中断训练保存的特征恢复逐客户端评测结果')
