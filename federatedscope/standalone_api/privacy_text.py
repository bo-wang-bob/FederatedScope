"""Read historical Windows logs without changing recorded training results."""
import json


def decode_log(data):
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return data.decode('gb18030', errors='replace')


def restore_status_text(job, data):
    """Restore only exact corrupted stage strings verified against raw events."""
    mapping = {}
    for line in data.splitlines():
        if not line.startswith(b'__PLATFORM__'):
            continue
        raw = line[len(b'__PLATFORM__'):]
        try:
            damaged = json.loads(raw.decode('utf-8', errors='replace')).get('stage')
            original = json.loads(decode_log(raw)).get('stage')
        except (ValueError, AttributeError):
            continue
        if isinstance(damaged, str) and '\ufffd' in damaged and isinstance(original, str) and '\ufffd' not in original:
            mapping[damaged] = original
    for entry in [job, *job.get('clients', {}).values(), *job.get('events', [])]:
        if entry.get('stage') in mapping:
            entry['stage'] = mapping[entry['stage']]
    return job
