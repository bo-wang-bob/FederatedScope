#!/usr/bin/env python
import argparse
import ast
import csv
import re
from pathlib import Path


CASES = [
    ('mdsent_rnn', 'fedavg', ['ggeur_final_5models/mdsent_rnn/fedavg']),
    ('mdsent_rnn', 'fedprox', ['ggeur_final_5models/mdsent_rnn/fedprox']),
    ('mdsent_rnn', 'fedproto', ['ggeur_final_5models/mdsent_rnn/fedproto']),
    ('mdsent_rnn', 'fedopt', ['ggeur_final_5models/mdsent_rnn/fedopt']),
    ('mdsent_rnn', 'ggeur', [
        'ggeur_final_5models/mdsent_rnn/ggeur',
        'ggeur_rnn_sentiment/ggeur_mdsent_rating4_rnn',
    ]),
    ('mdsent_lstm', 'fedavg', ['ggeur_final_5models/mdsent_lstm/fedavg']),
    ('mdsent_lstm', 'fedprox', ['ggeur_final_5models/mdsent_lstm/fedprox']),
    ('mdsent_lstm', 'fedproto', ['ggeur_final_5models/mdsent_lstm/fedproto']),
    ('mdsent_lstm', 'fedopt', ['ggeur_final_5models/mdsent_lstm/fedopt']),
    ('mdsent_lstm', 'ggeur', [
        'ggeur_final_5models/mdsent_lstm/ggeur',
        'ggeur_lstm_sentiment/ggeur_mdsent_rating4_lstm',
    ]),
]


ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


def iter_eval_logs(root, candidates):
    for rel in candidates:
        base = root / rel
        if base.exists():
            yield from sorted(base.glob('**/eval_results.log'))


def iter_stdout_logs(root, model, method):
    log_dir = root / 'ggeur_remote_logs'
    if not log_dir.exists():
        return
    yield from sorted(log_dir.glob(f'{model}_{method}_*.log'))
    if model == 'mdsent_rnn' and method == 'ggeur':
        yield from sorted(log_dir.glob('rnn_ggeur_*.log'))
    if model == 'mdsent_lstm' and method == 'ggeur':
        yield from sorted(log_dir.glob('lstm_ggeur_*.log'))


def parse_line(line):
    clean = ANSI_RE.sub('', line)
    if "'Round': 'Final'" not in clean and '"Round": "Final"' not in clean:
        return None
    start = clean.find("{'Role'")
    if start < 0:
        start = clean.find('{"Role"')
    end = clean.rfind('}')
    if start < 0 or end < start:
        return None
    try:
        return ast.literal_eval(clean[start:end + 1])
    except (SyntaxError, ValueError):
        return None


def parse_final(log_path):
    final = None
    final_acc = None
    best_acc = None
    last_round_acc = None
    completed = False
    for line in log_path.read_text(errors='ignore').splitlines():
        parsed = parse_line(line)
        if parsed is not None:
            final = parsed
            completed = True
        clean = ANSI_RE.sub('', line)
        if 'Training finished after' in clean:
            completed = True
        round_match = re.search(
            r'MLP Test Accuracy - .* average: ([0-9.]+)', clean)
        if round_match:
            last_round_acc = float(round_match.group(1))
        avg_match = re.search(
            r'average: final=([0-9.]+), best=([0-9.]+)', clean)
        if avg_match:
            final_acc = float(avg_match.group(1))
            best_acc = float(avg_match.group(2))
        best_match = re.search(
            r'Classifier Best Average Accuracy: ([0-9.]+)', clean)
        if best_match:
            best_acc = float(best_match.group(1))
            completed = True

    if completed and (
            best_acc is not None or final_acc is not None
            or last_round_acc is not None):
        return {
            'test_acc': best_acc
            if best_acc is not None else final_acc
            if final_acc is not None else last_round_acc,
            'final_acc': final_acc if final_acc is not None else last_round_acc,
            'best_acc': best_acc if best_acc is not None else '',
            'test_loss': '',
            'test_avg_loss': '',
            'test_total': '',
            'log': str(log_path),
        }

    if final is None:
        return None
    raw = final.get('Results_raw', {})
    metrics = raw.get('server_global_eval', raw)
    if 'test_acc' not in metrics:
        for value in raw.values():
            if isinstance(value, dict) and 'test_acc' in value:
                metrics = value
                break
    return {
        'test_acc': metrics.get('test_acc', ''),
        'final_acc': metrics.get('test_acc', ''),
        'best_acc': metrics.get('test_acc', ''),
        'test_loss': metrics.get('test_loss', ''),
        'test_avg_loss': metrics.get('test_avg_loss', ''),
        'test_total': metrics.get('test_total', ''),
        'log': str(log_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='exp')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for model, method, candidates in CASES:
        parsed = None
        log_paths = list(iter_eval_logs(root, candidates))
        log_paths.extend(iter_stdout_logs(root, model, method) or [])
        for log_path in log_paths:
            parsed = parse_final(log_path)
            if parsed:
                break
        row = {
            'model': model,
            'method': method,
            'status': 'done' if parsed else 'missing',
            'test_acc': parsed['test_acc'] if parsed else '',
            'final_acc': parsed['final_acc'] if parsed else '',
            'best_acc': parsed['best_acc'] if parsed else '',
            'test_loss': parsed['test_loss'] if parsed else '',
            'test_avg_loss': parsed['test_avg_loss'] if parsed else '',
            'test_total': parsed['test_total'] if parsed else '',
            'log': parsed['log'] if parsed else '',
        }
        rows.append(row)

    acc_by_model = {}
    for row in rows:
        if row['status'] == 'done' and row['test_acc'] != '':
            acc_by_model.setdefault(row['model'], {})[row['method']] = \
                float(row['test_acc'])

    for row in rows:
        accs = acc_by_model.get(row['model'], {})
        baselines = [
            value for method, value in accs.items() if method != 'ggeur'
        ]
        best_baseline = max(baselines) if baselines else None
        row['best_baseline_acc'] = best_baseline if best_baseline is not None else ''
        if row['method'] == 'ggeur' and row['test_acc'] != '' \
                and best_baseline is not None:
            ggeur_acc = float(row['test_acc'])
            row['abs_improve'] = ggeur_acc - best_baseline
            row['rel_improve_pct'] = (
                (ggeur_acc - best_baseline) / best_baseline * 100.0
                if best_baseline else ''
            )
        else:
            row['abs_improve'] = ''
            row['rel_improve_pct'] = ''

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'model',
        'method',
        'status',
        'test_acc',
        'final_acc',
        'best_acc',
        'best_baseline_acc',
        'abs_improve',
        'rel_improve_pct',
        'test_loss',
        'test_avg_loss',
        'test_total',
        'log',
    ]
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(','.join(str(row[name]) for name in fieldnames))


if __name__ == '__main__':
    main()
