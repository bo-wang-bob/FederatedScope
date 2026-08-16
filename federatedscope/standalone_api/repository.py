"""Small durable repository for scenario and experiment metadata."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional


class JsonRepository:
    def __init__(self, root: Path):
        self.root = root
        self.scenario_dir = root / 'scenarios'
        self.experiment_dir = root / 'experiments'
        self.scenario_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def _read(path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        with path.open('r', encoding='utf-8') as stream:
            return json.load(stream)

    @staticmethod
    def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f'.{path.name}.', suffix='.tmp', dir=str(path.parent))
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def save_scenario(self, scenario: Dict[str, Any]) -> None:
        with self._lock:
            self._atomic_write(
                self.scenario_dir / f"{scenario['scenarioId']}.json", scenario)

    def get_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._read(self.scenario_dir / f'{scenario_id}.json')

    def save_experiment(self, experiment: Dict[str, Any]) -> None:
        with self._lock:
            self._atomic_write(
                self.experiment_dir / f"{experiment['experimentId']}.json",
                experiment)

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._read(self.experiment_dir / f'{experiment_id}.json')

    def list_experiments(self) -> List[Dict[str, Any]]:
        with self._lock:
            records = [self._read(path)
                       for path in self.experiment_dir.glob('*.json')]
        return sorted(
            [record for record in records if record],
            key=lambda record: record.get('createdAt', ''),
            reverse=True,
        )
