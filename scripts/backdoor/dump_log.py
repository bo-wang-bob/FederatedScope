"""Tiny helper to inspect FederatedScope run logs (UTF-16 or UTF-8)."""
import io
import re
import sys
from pathlib import Path


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ('utf-8', 'utf-16', 'utf-16-le', 'gbk', 'latin-1'):
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        if '\x00' in text[:200]:
            continue
        return text
    return raw.decode('utf-16-le', errors='replace')


ANSI = re.compile(r'\x1b\[[0-9;]*m')


def main():
    if len(sys.argv) < 2:
        print('usage: dump_log.py <log> [pattern ...]')
        return 1
    path = Path(sys.argv[1])
    text = ANSI.sub('', read_text(path))
    lines = text.splitlines()
    patterns = sys.argv[2:]
    print(f'== {path.name}: {len(lines)} lines')
    rounds = re.findall(r'Round\s+#(\d+)', text)
    if rounds:
        print(f'rounds seen: {len(rounds)} max={max(int(r) for r in rounds)}')
    hits = [l for l in lines
            if re.search(r'(Traceback|Error|error|Exception)', l)]
    if hits:
        print(f'-- {len(hits)} suspicious lines, last 8:')
        for line in hits[-8:]:
            print('   ', line.strip()[:200])
    for pat in patterns:
        matched = [l.strip() for l in lines if re.search(pat, l)]
        print(f'-- pattern /{pat}/ -> {len(matched)} hits, last 12:')
        for line in matched[-12:]:
            print('   ', line[:260])
    return 0


if __name__ == '__main__':
    sys.exit(main())
