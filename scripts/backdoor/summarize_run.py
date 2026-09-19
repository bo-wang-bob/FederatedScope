"""Summarize one FederatedScope backdoor run log (UTF-16 or UTF-8).

Prints: clean-accuracy trajectory, SABRE ASR trajectory, and final verdict.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_log import read_text  # noqa: E402

ANSI = re.compile(r'\x1b\[[0-9;]*m')


def summary(path: Path, tag: str):
    text = ANSI.sub('', read_text(path))
    lines = text.splitlines()

    clean = {}
    for line in lines:
        m = re.search(r'Round (\d+) MLP Test Accuracy - uploaded: ([0-9.]+)',
                      line)
        if m:
            clean[int(m.group(1))] = float(m.group(2))

    asr, clean_target = {}, {}
    skips = 0
    for line in lines:
        m = re.search(r'Round (\d+) SABRE ASR \(attacker (\d+)\) - (.+)$',
                      line)
        if m:
            rnd = int(m.group(1))
            vals = [float(v) for v in re.findall(r'([0-9.]+)', m.group(3))]
            if vals:
                asr[rnd] = vals[0]
            continue
        m = re.search(
            r'Round (\d+) SABRE clean target rate \(target (\d+)\) - (.+)$',
            line)
        if m:
            nums = re.findall(r'([0-9.]+)', m.group(3))
            if nums:
                clean_target[int(m.group(1))] = float(nums[0])
            continue
        if 'skipped SABRE ASR logging' in line:
            skips += 1

    final = re.search(r'Classifier Best Average Accuracy: ([0-9.]+)', text)
    print(f'===== {tag}: {path.name} ({len(lines)} lines)')
    print(f'  best clean acc logged : {final.group(1) if final else "n/a"}')
    if clean:
        keys = sorted(clean)
        print(f'  clean acc rounds      : {min(keys)}..{max(keys)} '
              f'({len(keys)} pts)')
        print(f'    r{min(keys)}={clean[keys[0]]:.4f}  '
              f'r{keys[len(keys)//2]}={clean[keys[len(keys)//2]]:.4f}  '
              f'r{max(keys)}={clean[max(keys)]:.4f}')
    if asr:
        keys = sorted(asr)
        print(f'  SABRE ASR rounds      : {min(keys)}..{max(keys)} '
              f'({len(keys)} pts)')
        head = ', '.join(f'{r}:{asr[r]:.3f}' for r in keys[:6])
        tail = ', '.join(f'{r}:{asr[r]:.3f}' for r in keys[-6:])
        print(f'    first -> {head}')
        print(f'    last  -> {tail}')
        print(f'    peak  -> {max(asr.values()):.4f}')
    else:
        print('  SABRE ASR numeric values: NONE')
    print(f'  SABRE ASR skip lines  : {skips}')
    if clean_target:
        keys = sorted(clean_target)
        print(f'  clean target rate     : r{max(keys)}='
              f'{clean_target[max(keys)]:.4f}')
    return clean, asr


if __name__ == '__main__':
    for arg in sys.argv[1:]:
        if '=' in arg:
            tag, p = arg.split('=', 1)
        else:
            tag, p = Path(arg).stem, arg
        summary(Path(p), tag)
