"""Launch the existing FederatedScope standalone runner as a child process."""

from __future__ import annotations

import os
import queue
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List

from federatedscope.core.monitoring.events import parse_training_event_line
from federatedscope.standalone_api.preflight import inspect_runtime


EventCallback = Callable[[str, Dict[str, Any]], None]
MetricCallback = Callable[[Dict[str, Any]], None]


class RunnerPreflightError(RuntimeError):
    def __init__(self, message: str, result: Dict[str, Any] | None = None):
        super().__init__(message)
        self.result = result or {}


class StandaloneProcessRunner:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def _template(self, config: Dict[str, Any]) -> Path:
        base = self.repo_root / 'scripts' / 'standalone_configs'
        experiment_type = config['type']
        if experiment_type == 'heterogeneity':
            name = 'heterogeneity.yaml'
        elif experiment_type == 'privacy':
            attack = config['privacy']['attack']
            suffix = '_protected' if config['privacy']['defenseEnabled'] else ''
            name = f'privacy_{attack}_attack{suffix}.yaml'
        else:
            suffix = '_defended' if config['backdoor']['defenseEnabled'] else ''
            name = f'backdoor_attack{suffix}.yaml'
        return base / name

    def preflight(self, config: Dict[str, Any]) -> Dict[str, Any]:
        template = self._template(config)
        data_root = Path(os.environ.get(
            'FEDERATEDSCOPE_DATA_ROOT',
            '/root/autodl-tmp/datasets/OfficeHomeDataset_10072016'))
        model_path = Path(os.environ.get(
            'FEDERATEDSCOPE_MODEL_PATH',
            '/root/autodl-tmp/models/open_clip_vitb16.bin'))
        result = inspect_runtime(
            config, template, data_root, model_path, self.repo_root)
        if not result['ready']:
            errors = [item['message'] for item in result['checks']
                      if not item['ready']]
            raise RunnerPreflightError('；'.join(errors), result)
        return result

    @staticmethod
    def _attacker_ids(client_ids: List[str], client_num: int,
                      ratio: float) -> List[int]:
        domain_offsets = {'DT': 0, 'TS': 15, 'ED': 30, 'FR': 45}
        result = []
        for client_id in client_ids:
            match = re.fullmatch(r'OH-(DT|TS|ED|FR)-C(\d{2})', client_id)
            if not match:
                continue
            value = domain_offsets[match.group(1)] + int(match.group(2))
            if 1 <= value <= client_num:
                result.append(value)
        if result:
            return sorted(set(result))
        count = max(1, round(client_num * ratio))
        return list(range(1, count + 1))

    @staticmethod
    def _verify_effective_config(command: List[str],
                                 experiment_type: str) -> None:
        """Freeze the exact child configuration before a process is created."""
        from federatedscope.core.configs.config import global_cfg
        from federatedscope.core.configs.cfg_security import \
            resolve_security_mode

        cfg = global_cfg.clone()
        cfg.merge_from_file(command[4])
        cfg.merge_from_list(command[5:])
        cfg.freeze(inform=False, save=False)
        actual_mode = resolve_security_mode(cfg)
        expected_mode = ('none' if experiment_type == 'heterogeneity'
                         else experiment_type)
        if actual_mode != expected_mode:
            raise RunnerPreflightError(
                f'最终配置安全模式不一致：期望 {expected_mode}，'
                f'实际 {actual_mode}')
        if experiment_type == 'backdoor' and (
                bool(cfg.dp.enabled) or bool(cfg.adaptive_dp.use)):
            raise RunnerPreflightError('后门任务不得启用隐私保护流程')

    def build_command(self, config: Dict[str, Any], output_dir: Path) -> List[str]:
        preflight = self.preflight(config)
        common = config['common']
        scenario = config['_scenario']
        client_num = scenario['request']['clientsPerDomain'] * 4
        sample_num = max(1, round(client_num * common['participationRate']))
        command = [
            sys.executable, '-m', 'federatedscope.main', '--cfg',
            preflight['template'],
            'data.root', preflight['dataRoot'],
            'federate.client_num', str(client_num),
            'federate.sample_client_num', str(sample_num),
            'federate.total_round_num', str(common['rounds']),
            'train.local_update_steps', str(common['localEpochs']),
            'dataloader.batch_size', str(common['batchSize']),
            'train.optimizer.lr', str(common['learningRate']),
            'seed', str(common['seed']),
            'use_gpu', str(common['device'] == 'cuda'),
            'outdir', str(output_dir),
            'expname', config['experimentId'],
            'ggeur.use_lds', 'True',
            'ggeur.lds_alpha', str(scenario['request']['partition']['alpha']),
            'ggeur.lds_seed', str(scenario['request']['partition']['seed']),
            'ggeur.officehome_manifest_path',
            preflight['partitionManifest'],
        ]
        if preflight.get('modelPath'):
            command.extend(['ggeur.clip_model_path', preflight['modelPath']])
        method = common['method']
        command.extend(['fedprox.use', str(method == 'fedprox')])
        if method == 'fedprox':
            command.extend(['fedprox.mu', str(common['fedproxMu'])])
        if config['type'] == 'heterogeneity' and \
                method == 'heterogeneous_solution':
            block = config['heterogeneity']
            command.extend([
                'ggeur.target_size_per_class', str(block['expansionTarget']),
                'ggeur.num_generated_per_sample',
                str(block['expansionTarget']),
                'ggeur.extract_batch_size',
                str(block['featureBatchSize']),
            ])
        elif method != 'heterogeneous_solution':
            command.extend([
                'ggeur.target_size_per_class', '0',
                'ggeur.num_generated_per_sample', '0',
                'ggeur.num_generated_per_prototype', '0',
            ])
        if config['type'] == 'privacy':
            block = config['privacy']
            protected = block['defenseEnabled']
            command.extend([
                'dp.enabled', str(protected),
                'dp.protect_ggeur_update', str(protected),
                'adaptive_dp.use', 'False',
            ])
            if protected:
                command.extend([
                    'dp.clipping.type', 'adaptive',
                    'dp.clipping.initial_clip', str(block['initialClip']),
                    'dp.clipping.target_quantile',
                    str(block['targetQuantile']),
                    'dp.noise_multiplier', str(block['noiseMultiplier']),
                    'dp.epsilon', str(block['epsilon']),
                ])
        elif config['type'] == 'backdoor':
            block = config['backdoor']
            # Backdoor tasks are strictly isolated from both current and
            # backward-compatible privacy protection paths.
            command.extend([
                'dp.enabled', 'False',
                'dp.protect_ggeur_update', 'False',
                'adaptive_dp.use', 'False',
                'ggeur.multi_metrics_stats_defense',
                str(block['defenseEnabled'] and
                    block['featureStageDefense']),
                'ggeur.defense_method',
                ('multi_metrics' if block['defenseEnabled'] and
                 block['trainingStageDefense'] else ''),
            ])
            attacker_ids = self._attacker_ids(
                block.get('maliciousClients', []), client_num,
                block['maliciousRatio'])
            command.extend([
                'attack.attacker_id', str(attacker_ids),
                'attack.target_label_ind', str(block['targetLabel']),
                'attack.poison_ratio', str(block['poisonRatio']),
            ])
            if block['attack'] == 'trigger_injection':
                command.extend([
                    'attack.attack_method', 'sabre',
                    'attack.sabre.start_round', str(block['startRound']),
                ])
            else:
                command.extend([
                    'attack.attack_method', 'label_flip',
                    'attack.label_flip.target_label_ind',
                    str(block['targetLabel']),
                    'attack.label_flip.all_to_target', 'True',
                    'attack.label_flip.poison_ratio',
                    str(block['poisonRatio']),
                    'attack.label_flip.start_round', str(block['startRound']),
                ])
                if block['attack'] == 'model_update_poisoning':
                    command.extend([
                        'attack.label_flip.poison_statistics', 'False',
                        'attack.label_flip.update_reversal', 'True',
                    ])
        else:
            command.extend([
                'dp.enabled', 'False',
                'dp.protect_ggeur_update', 'False',
                'adaptive_dp.use', 'False',
            ])
        self._verify_effective_config(command, config['type'])
        return command

    @staticmethod
    def _parse_line(line: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {'message': line.rstrip()[-2000:]}
        round_match = re.search(r'(?i)(?:round|轮次)[^0-9]{0,8}(\d+)', line)
        if round_match:
            payload['round'] = int(round_match.group(1))
        client_match = re.search(r'(?i)client[^0-9]{0,5}(\d+)', line)
        if client_match:
            payload['clientIndex'] = int(client_match.group(1))
        metrics: Dict[str, float] = {}
        metric_patterns = {
            'accuracy': r'(?i)(?:test_acc|accuracy|准确率)[^0-9]{0,8}([0-9]*\.?[0-9]+)',
            'loss': r'(?i)(?:test_loss|loss|损失)[^0-9]{0,8}([0-9]*\.?[0-9]+)',
            'attackSuccess': r'(?i)(?:attack_success|asr)[^0-9]{0,8}([0-9]*\.?[0-9]+)',
            'privacyRisk': r'(?i)(?:privacy_risk|attack_auc|\bauc\b)[^0-9]{0,8}([0-9]*\.?[0-9]+)',
            'truePositiveRate': r'(?i)(?:\btpr(?:@[^=:\s]+)?|true_positive_rate)\s*[:=]\s*([0-9]*\.?[0-9]+)',
            'falsePositiveRate': r'(?i)(?:\bfpr(?:@[^=:\s]+)?|false_positive_rate)\s*[:=]\s*([0-9]*\.?[0-9]+)',
            'noiseMultiplier': r'(?i)(?:noise_multiplier|noise_scale)[^0-9]{0,8}([0-9]*\.?[0-9]+)',
            'noiseStd': r'(?i)noise_std\s*[:=]\s*([0-9]*\.?[0-9]+)',
        }
        for name, pattern in metric_patterns.items():
            match = re.search(pattern, line)
            if match:
                metrics[name] = float(match.group(1))
        domain_values = {
            name: float(value)
            for name, value in re.findall(
                r'\b(Art|Clipart|Product|Real_World|average)\s*:\s*'
                r'([0-9]*\.?[0-9]+)', line)
        }
        domain_scores = [domain_values[name] for name in (
            'Art', 'Clipart', 'Product', 'Real_World')
                         if name in domain_values]
        if 'Test Accuracy -' in line and domain_scores:
            metrics.update({
                'accuracy': domain_values.get(
                    'average', sum(domain_scores) / len(domain_scores)),
                'worstDomain': min(domain_scores),
                'domainGap': max(domain_scores) - min(domain_scores),
            })
            for name in ('Art', 'Clipart', 'Product', 'Real_World'):
                if name in domain_values:
                    metrics[f'domain{name}'] = domain_values[name]
        if re.search(r'(?i)\bASR\b', line) and domain_scores:
            metrics['attackSuccess'] = domain_values.get(
                'average', sum(domain_scores) / len(domain_scores))
        if metrics:
            payload['metrics'] = metrics
        return payload

    @staticmethod
    def _client_payload(client_index: int, round_index: int,
                        line: str) -> Dict[str, Any]:
        zero_based = max(0, client_index - 1) % 60
        domain_index = zero_based // 15
        domain_keys = ['Art', 'Clipart', 'Product', 'Real_World']
        prefixes = ['OH-DT', 'OH-TS', 'OH-ED', 'OH-FR']
        lowered = line.lower()
        if 'filter' in lowered or '过滤' in line:
            status = '已过滤'
            progress = 100
        elif 'upload' in lowered or '上传' in line:
            status = '上传中'
            progress = 85
        elif 'complete' in lowered or '完成' in line:
            status = '已完成'
            progress = 100
        elif 'train' in lowered or '训练' in line:
            status = '本地训练'
            progress = 55
        else:
            status = '接收中'
            progress = 20
        return {
            'clientId': f'{prefixes[domain_index]}-C{zero_based % 15 + 1:02d}',
            'domainKey': domain_keys[domain_index],
            'status': status,
            'progress': progress,
            'round': max(0, round_index),
        }

    def run(self, config: Dict[str, Any], output_dir: Path,
            stop_event: threading.Event, emit: EventCallback,
            on_metric: MetricCallback) -> int:
        command = self.build_command(config, output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / 'runner.log'
        emit('stage.changed', {'phaseIndex': 0, 'round': 0,
                               'stage': '中央下发'})
        process = subprocess.Popen(
            command,
            cwd=str(self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        lines: queue.Queue[str | None] = queue.Queue()

        def collect_output() -> None:
            assert process.stdout is not None
            with log_path.open('a', encoding='utf-8') as log_stream:
                for output_line in process.stdout:
                    log_stream.write(output_line)
                    log_stream.flush()
                    lines.put(output_line)
            lines.put(None)

        reader = threading.Thread(target=collect_output, daemon=True)
        reader.start()
        last_round = -1
        client_states: Dict[str, tuple] = {}
        output_closed = False
        stop_requested = False
        while process.poll() is None or not output_closed:
            if stop_event.is_set() and not stop_requested:
                stop_requested = True
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                continue
            if line is None:
                output_closed = True
                continue
            structured = parse_training_event_line(line)
            if structured is not None:
                event_type = structured['type']
                event_payload = dict(structured['payload'])
                if event_type in {'client.status.changed',
                                  'client.metric.updated'} and \
                        'clientIndex' in event_payload:
                    client_index = int(event_payload.pop('clientIndex'))
                    identity = self._client_payload(
                        client_index,
                        int(event_payload.get('round', last_round)), '')
                    event_payload = {**identity, **event_payload}
                if event_type == 'metric.updated':
                    metric = dict(event_payload)
                    metric.setdefault('round', max(0, last_round))
                    on_metric(metric)
                if event_type == 'round.started' and \
                        'round' in event_payload:
                    last_round = int(event_payload['round'])
                emit(event_type, event_payload)
                continue
            if '[AdaptiveDP]' in line:
                # The client emits the same values as a structured event after
                # applying protection; do not create a duplicate global point.
                continue
            parsed = self._parse_line(line)
            current_round = parsed.get('round')
            if current_round is not None and current_round != last_round:
                last_round = current_round
                emit('round.started', {
                    'phaseIndex': 2,
                    'round': current_round,
                })
            if 'clientIndex' in parsed:
                client_payload = self._client_payload(
                    int(parsed['clientIndex']), last_round, line)
                client_key = client_payload['clientId']
                state_key = (client_payload['status'],
                             client_payload['progress'],
                             client_payload['round'])
                if client_states.get(client_key) != state_key:
                    client_states[client_key] = state_key
                    emit('client.status.changed', client_payload)
            if parsed.get('metrics'):
                metric = {'round': max(0, last_round), **parsed['metrics']}
                on_metric(metric)
                emit('metric.updated', metric)
            lowered = line.lower()
            if any(token in lowered for token in ('warning', 'error', 'exception', 'traceback')):
                level = 'error' if any(token in lowered for token in (
                    'error', 'exception', 'traceback')) else 'warning'
                emit('warning.raised', {'level': level, **parsed})
        reader.join(timeout=1)
        return -15 if stop_requested else int(process.returncode or 0)
