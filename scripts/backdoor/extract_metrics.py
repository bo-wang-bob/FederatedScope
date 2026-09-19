"""Extract comparable metrics from the three uploaded-dataset runs."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_log import read_text  # noqa: E402

ANSI = re.compile(r'\x1b\[[0-9;]*m')
RUNS = Path('C:/Users/pc/Desktop/logs/FederatedScope/_runs')
RECORD_START = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+\s')


def load(name):
    """Return logical log records.

    The logger hard-wraps long messages at ~110 characters, so physical
    lines must be re-joined before they can be parsed with line regexes.
    """
    text = ANSI.sub('', read_text(RUNS / name))
    records, current = [], None
    for raw in text.splitlines():
        if RECORD_START.match(raw):
            if current is not None:
                records.append(current)
            current = raw
        elif current is not None:
            current += raw.lstrip()
        else:
            current = raw
    if current is not None:
        records.append(current)
    return records


def parse(lines):
    clean, asr, drops = {}, {}, {}
    cur_round = None
    for line in lines:
        # The filter report does not repeat the round id, so track the most
        # recent round mentioned before it (aggregation logs emit it first).
        m_round = re.search(r'Round (\d+)', line)
        if m_round:
            cur_round = int(m_round.group(1))
        m = re.search(r'Round (\d+) MLP Test Accuracy - uploaded: ([0-9.]+)',
                      line)
        if m:
            clean[int(m.group(1))] = float(m.group(2))
            continue
        m = re.search(r'Round (\d+) SABRE ASR \(attacker (\d+)\) - (.+)$',
                      line)
        if m:
            nums = re.findall(r'([0-9.]+)', m.group(3))
            if nums:
                asr[int(m.group(1))] = float(nums[0])
            continue
        m = re.search(r'Multi-metrics \w+ client_ids=\[(.*?)\], '
                      r'kept=\[(.*?)\], dropped=\[(.*?)\]', line)
        if m and cur_round is not None:
            dropped = [int(x) for x in m.group(3).split(',') if x.strip()]
            drops.setdefault(cur_round, []).extend(dropped)
    return clean, asr, drops


def main():
    out = {}
    for tag, name in (('baseline', '1_baseline.log'),
                      ('sabre', '2_sabre.log'),
                      ('sabre_defense', '3_defense.log')):
        clean, asr, drops = parse(load(name))
        entry = {'clean_acc': clean, 'asr': asr}
        if drops:
            attackers = {1, 2, 3}
            attack_rounds = [r for r in sorted(drops) if r >= 40]
            hit = sum(1 for r in attack_rounds
                      if attackers & set(drops[r]))
            anydrop = sum(1 for r in attack_rounds if attackers & set(drops[r]))
            full = sum(1 for r in attack_rounds
                       if attackers.issubset(set(drops[r])))
            benign_dropped = 0
            for r in attack_rounds:
                benign_dropped += len(set(drops[r]) - attackers)
            entry['filter'] = {
                'rounds_scored': len(attack_rounds),
                'rounds_dropping_any_attacker': anydrop,
                'rounds_dropping_all_attackers': full,
                'attacker_drop_rate': round(hit / max(1, len(attack_rounds)), 4),
                'avg_benign_dropped_per_round': round(
                    benign_dropped / max(1, len(attack_rounds)), 3),
            }
        out[tag] = entry

    target = RUNS / 'metrics.json'
    target.write_text(json.dumps(out, indent=2, sort_keys=True),
                      encoding='utf-8')
    print(f'wrote {target}')
    for tag, entry in out.items():
        acc = entry['clean_acc']
        acc = {int(k): float(v) for k, v in acc.items()}
        print(f"{tag:14s} clean: r0={acc.get(0, float('nan')):.4f} "
              f"r50={acc.get(50, float('nan')):.4f} "
              f"r99={acc.get(99, float('nan')):.4f} "
              f"best={max(acc.values()):.4f}")
        if entry['asr']:
            asr = entry['asr']
            r = sorted(asr)
            print(f"{'':14s} asr  : first={asr[r[0]]:.4f} "
                  f"last={asr[r[-1]]:.4f} peak={max(asr.values()):.4f}")
        if 'filter' in entry:
            print(f"{'':14s} filt : {entry['filter']}")


if __name__ == '__main__':
    main()
