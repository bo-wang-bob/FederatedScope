"""Thread-safe lifecycle manager for standalone experiment processes."""

from __future__ import annotations

import copy
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from federatedscope.standalone_api.repository import JsonRepository
from federatedscope.standalone_api.runner import StandaloneProcessRunner


TERMINAL_STATES = {'completed', 'failed', 'stopped'}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


class TaskNotFound(KeyError):
    pass


class TaskConflict(RuntimeError):
    pass


class ExperimentTaskManager:
    def __init__(self, repository: JsonRepository,
                 runner: StandaloneProcessRunner,
                 output_root: Path):
        self.repository = repository
        self.runner = runner
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conditions: Dict[str, threading.Condition] = {}
        self._stop_events: Dict[str, threading.Event] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._idempotency: Dict[str, str] = {}
        self._recover_interrupted()

    def _recover_interrupted(self) -> None:
        for record in self.repository.list_experiments():
            idempotency_key = str(
                record.get('config', {}).get('idempotencyKey', '')).strip()
            if idempotency_key:
                self._idempotency[idempotency_key] = record['experimentId']
            if record.get('status') in {'created', 'validating', 'queued',
                                        'running', 'stopping'}:
                record['status'] = 'failed'
                record['endedAt'] = utc_now()
                record['error'] = {
                    'code': 'API_RESTARTED',
                    'message': '控制服务重启，原训练进程状态无法恢复',
                }
                self.repository.save_experiment(record)

    def preflight(self, config: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
        runtime_config = {**config, '_scenario': scenario,
                          'experimentId': 'preflight'}
        return self.runner.preflight(runtime_config)

    def create(self, config: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
        key = str(config.get('idempotencyKey', '')).strip()
        with self._lock:
            if key and key in self._idempotency:
                return self.get(self._idempotency[key])
            requested_device = config['common']['device']
            active = [
                record for record in self.repository.list_experiments()
                if record.get('status') in {'queued', 'running', 'stopping'}
                and record.get('config', {}).get('common', {}).get('device') ==
                requested_device
            ]
            if active:
                raise TaskConflict(
                    f'{requested_device.upper()} 已有实验正在运行：'
                    f"{active[0]['experimentId']}")
        self.preflight(config, scenario)
        experiment_id = f"EXP-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6].upper()}"
        created_at = utc_now()
        public_config = copy.deepcopy(config)
        record = {
            'experimentId': experiment_id,
            'name': public_config['name'],
            'type': public_config['type'],
            'method': public_config['common']['method'],
            'scenarioId': public_config['scenarioId'],
            'scenarioSummary': {
                'dataset': scenario['request']['dataset'],
                'alpha': scenario['request']['partition']['alpha'],
                'seed': scenario['request']['partition']['seed'],
                'partitionVersion': scenario['preview']['partitionVersion'],
            },
            'config': public_config,
            'status': 'queued',
            'createdAt': created_at,
            'startedAt': None,
            'endedAt': None,
            'updatedAt': created_at,
            'sequence': 0,
            'phaseIndex': 0,
            'round': 0,
            'totalRounds': public_config['common']['rounds'],
            'clients': {},
            'metrics': [],
            'events': [],
            'finalMetrics': {},
            'error': None,
        }
        runtime_config = {
            **copy.deepcopy(public_config),
            '_scenario': scenario,
            'experimentId': experiment_id,
        }
        condition = threading.Condition(self._lock)
        stop_event = threading.Event()
        with self._lock:
            self._conditions[experiment_id] = condition
            self._stop_events[experiment_id] = stop_event
            if key:
                self._idempotency[key] = experiment_id
            self.repository.save_experiment(record)
            thread = threading.Thread(
                target=self._execute,
                name=f'experiment-{experiment_id}',
                args=(runtime_config,),
                daemon=True,
            )
            self._threads[experiment_id] = thread
            thread.start()
        return self.get(experiment_id)

    def _execute(self, runtime_config: Dict[str, Any]) -> None:
        experiment_id = runtime_config['experimentId']
        self._set_status(experiment_id, 'running', startedAt=utc_now())
        self._emit(experiment_id, 'experiment.started', {
            'status': 'running',
            'type': runtime_config['type'],
        })
        try:
            return_code = self.runner.run(
                runtime_config,
                self.output_root / experiment_id,
                self._stop_events[experiment_id],
                lambda event_type, payload: self._emit(
                    experiment_id, event_type, payload),
                lambda metric: self._append_metric(experiment_id, metric),
            )
            if self._stop_events[experiment_id].is_set() or return_code == -15:
                self._finalize(experiment_id, 'stopped',
                               'experiment.stopped')
            elif return_code == 0:
                self._finalize(experiment_id, 'completed',
                               'experiment.completed')
            else:
                raise RuntimeError(f'训练进程退出码为 {return_code}')
        except Exception as error:  # worker boundary: persist all failures
            self._finalize(
                experiment_id, 'failed', 'experiment.failed',
                error={'code': 'RUNNER_FAILED', 'message': str(error)},
                message=str(error))

    def _finalize(self, experiment_id: str, status: str,
                  event_type: str, error: Any = None,
                  message: str = '') -> None:
        """Persist the terminal event and state in one atomic transition."""
        with self._lock:
            record = self._get_record(experiment_id)
            sequence = int(record.get('sequence', 0)) + 1
            timestamp = utc_now()
            payload = {'status': status}
            if message:
                payload['message'] = message
            event = {
                'id': f'{experiment_id}-{sequence}',
                'sequence': sequence,
                'experimentId': experiment_id,
                'type': event_type,
                'timestamp': timestamp,
                'source': 'backend',
                'payload': payload,
            }
            record.update({
                'status': status,
                'endedAt': timestamp,
                'updatedAt': timestamp,
                'sequence': sequence,
                'error': error,
            })
            record['events'].append(event)
            record['events'] = record['events'][-2000:]
            self.repository.save_experiment(record)
            condition = self._conditions.get(experiment_id)
            if condition:
                condition.notify_all()

    def _set_status(self, experiment_id: str, status: str, **updates: Any) -> None:
        with self._lock:
            record = self._get_record(experiment_id)
            record.update(updates)
            record['status'] = status
            record['updatedAt'] = utc_now()
            self.repository.save_experiment(record)

    def _append_metric(self, experiment_id: str,
                       metric: Dict[str, Any]) -> None:
        with self._lock:
            record = self._get_record(experiment_id)
            record['metrics'].append(metric)
            record['metrics'] = record['metrics'][-2000:]
            record['finalMetrics'].update({
                key: value for key, value in metric.items() if key != 'round'
            })
            if 'round' in metric:
                record['round'] = int(metric['round'])
            record['updatedAt'] = utc_now()
            self.repository.save_experiment(record)

    def _emit(self, experiment_id: str, event_type: str,
              payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            record = self._get_record(experiment_id)
            sequence = int(record.get('sequence', 0)) + 1
            timestamp = utc_now()
            event = {
                'id': f'{experiment_id}-{sequence}',
                'sequence': sequence,
                'experimentId': experiment_id,
                'type': event_type,
                'timestamp': timestamp,
                'source': 'backend',
                'payload': payload,
            }
            record['sequence'] = sequence
            record['updatedAt'] = timestamp
            if 'phaseIndex' in payload:
                record['phaseIndex'] = int(payload['phaseIndex'])
            if 'round' in payload:
                record['round'] = int(payload['round'])
            if event_type == 'client.status.changed' and payload.get('clientId'):
                record['clients'][payload['clientId']] = {
                    **payload,
                    'source': 'backend',
                }
            record['events'].append(event)
            record['events'] = record['events'][-2000:]
            self.repository.save_experiment(record)
            condition = self._conditions.get(experiment_id)
            if condition:
                condition.notify_all()
            return event

    def stop(self, experiment_id: str) -> Dict[str, Any]:
        with self._lock:
            record = self._get_record(experiment_id)
            if record['status'] in TERMINAL_STATES:
                return self._public_record(record)
            if record['status'] == 'stopping':
                return self._public_record(record)
            record['status'] = 'stopping'
            record['updatedAt'] = utc_now()
            self.repository.save_experiment(record)
            stop_event = self._stop_events.get(experiment_id)
            if stop_event:
                stop_event.set()
        self._emit(experiment_id, 'experiment.stopping', {
            'status': 'stopping',
        })
        return self.get(experiment_id)

    def _get_record(self, experiment_id: str) -> Dict[str, Any]:
        record = self.repository.get_experiment(experiment_id)
        if not record:
            raise TaskNotFound(experiment_id)
        return record

    @staticmethod
    def _public_record(record: Dict[str, Any]) -> Dict[str, Any]:
        result = copy.deepcopy(record)
        result.pop('events', None)
        return result

    def get(self, experiment_id: str) -> Dict[str, Any]:
        with self._lock:
            return self._public_record(self._get_record(experiment_id))

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [self._public_record(record)
                    for record in self.repository.list_experiments()]

    def snapshot(self, experiment_id: str) -> Dict[str, Any]:
        with self._lock:
            record = self._get_record(experiment_id)
            return {
                'experimentId': experiment_id,
                'sequence': record['sequence'],
                'phaseIndex': record['phaseIndex'],
                'round': record['round'],
                'totalRounds': record['totalRounds'],
                'status': record['status'],
                'clients': record['clients'],
                'metrics': record['metrics'],
                'recentEvents': record['events'][-100:],
                'source': 'backend',
                'updatedAt': record['updatedAt'],
                'type': record['type'],
                'method': record['method'],
                'name': record['name'],
                'finalMetrics': record['finalMetrics'],
                'error': record['error'],
            }

    def metrics(self, experiment_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._get_record(experiment_id)['metrics'])

    def events_after(self, experiment_id: str,
                     sequence: int) -> List[Dict[str, Any]]:
        with self._lock:
            record = self._get_record(experiment_id)
            return [copy.deepcopy(event) for event in record['events']
                    if event['sequence'] > sequence]

    def wait_for_events(self, experiment_id: str, sequence: int,
                        timeout: float = 15.0) -> List[Dict[str, Any]]:
        with self._lock:
            events = self.events_after(experiment_id, sequence)
            if events:
                return events
            condition = self._conditions.setdefault(
                experiment_id, threading.Condition(self._lock))
            condition.wait(timeout)
            return self.events_after(experiment_id, sequence)
