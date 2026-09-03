#!/usr/bin/env python
"""Run grouped GGEUR five-model experiment configs."""

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / 'scripts' / 'example_configs' / 'ggeur_final_5models'
MAIN = REPO_ROOT / 'federatedscope' / 'main.py'


def iter_cases():
    for group_dir in sorted(CONFIG_ROOT.iterdir()):
        if not group_dir.is_dir():
            continue
        for cfg_path in sorted(group_dir.glob('*.yaml')):
            yield group_dir.name, cfg_path.stem, cfg_path


def resolve_cases(group, method, run_all):
    cases = list(iter_cases())
    if run_all:
        return cases
    selected = [
        case for case in cases
        if case[0] == group and case[1] == method
    ]
    if not selected:
        available = ', '.join(f'{g}/{m}' for g, m, _ in cases)
        raise SystemExit(
            f'Unknown case {group}/{method}. Available cases: {available}')
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', action='store_true',
                        help='List available group/method cases.')
    parser.add_argument('--group', help='Experiment group folder name.')
    parser.add_argument('--method', help='Method config name without .yaml.')
    parser.add_argument('--all', action='store_true',
                        help='Run every config sequentially.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without running them.')
    parser.add_argument('opts', nargs=argparse.REMAINDER,
                        help='Extra FederatedScope opts after --.')
    args = parser.parse_args()

    if args.list:
        for group, method, cfg_path in iter_cases():
            print(f'{group}/{method}\t{cfg_path.relative_to(REPO_ROOT)}')
        return

    if not args.all and (not args.group or not args.method):
        parser.error('use --group and --method, or use --all')

    extra_opts = list(args.opts)
    if extra_opts and extra_opts[0] == '--':
        extra_opts = extra_opts[1:]

    for group, method, cfg_path in resolve_cases(args.group, args.method,
                                                 args.all):
        cmd = [
            sys.executable,
            str(MAIN),
            '--cfg',
            str(cfg_path),
        ] + extra_opts
        print(f'[{group}/{method}] ' + ' '.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


if __name__ == '__main__':
    main()
