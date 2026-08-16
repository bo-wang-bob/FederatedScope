"""Small JSON event protocol emitted by standalone training workers."""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, Optional


EVENT_MARKER = '__FS_EVENT_V1__'
EVENT_TYPES = {
    'stage.changed',
    'round.started',
    'round.completed',
    'client.status.changed',
    'client.metric.updated',
    'metric.updated',
    'defense.decision',
    'warning.raised',
}
_output_lock = threading.Lock()


def _json_default(value: Any) -> Any:
    if hasattr(value, 'item'):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def emit_training_event(event_type: str, **payload: Any) -> None:
    """Write one machine-readable event to stdout for the parent runner."""
    if event_type not in EVENT_TYPES:
        raise ValueError(f'unsupported training event type: {event_type}')
    message = json.dumps({'type': event_type, 'payload': payload},
                         ensure_ascii=False, separators=(',', ':'),
                         default=_json_default)
    with _output_lock:
        print(f'{EVENT_MARKER}{message}', flush=True)


def parse_training_event_line(line: str) -> Optional[Dict[str, Any]]:
    marker_index = line.find(EVENT_MARKER)
    if marker_index < 0:
        return None
    raw = line[marker_index + len(EVENT_MARKER):].strip()
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict) or event.get('type') not in EVENT_TYPES or \
            not isinstance(event.get('payload'), dict):
        return None
    return event
